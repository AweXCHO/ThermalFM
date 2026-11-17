"""Sequential tar-shard dataset for large ThermalFM image corpora."""

from __future__ import annotations

import io
import json
import math
import tarfile
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .data import checked_grayscale


class PackedThermalTarDataset(IterableDataset):
    """Read uncompressed image tar shards sequentially with DDP-safe sharding.

    Every archive member is named ``0/<id>.<ext>`` or ``1/<id>.<ext>``.  Shards
    are built to have near-identical record counts, so rank-level partitioning
    gives both DDP ranks an equal number of full batches without random seeks
    across millions of loose PNG files.
    """

    def __init__(self, shard_dir: str | Path, transform: Optional[Callable] = None):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        metadata_path = self.shard_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.shards = [self.shard_dir / item["name"] for item in metadata["shards"]]
        self.total_samples = int(metadata["total_samples"])
        self.cooled_count = int(metadata["cooled_count"])
        self.uncooled_count = int(metadata["uncooled_count"])
        self.transform = transform
        if not self.shards or self.total_samples <= 0:
            raise RuntimeError(f"invalid shard metadata: {metadata_path}")

    @staticmethod
    def _rank_and_world() -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def detector_class_counts(self) -> tuple[int, int]:
        return self.cooled_count, self.uncooled_count

    def __len__(self) -> int:
        _, world_size = self._rank_and_world()
        return math.ceil(self.total_samples / world_size)

    def __iter__(self):
        rank, world_size = self._rank_and_world()
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers

        rank_shard_indices = range(rank, len(self.shards), world_size)
        for rank_position, shard_index in enumerate(rank_shard_indices):
            if rank_position % worker_count != worker_id:
                continue
            with tarfile.open(self.shards[shard_index], mode="r") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    label_text, _, _ = member.name.partition("/")
                    if label_text not in {"0", "1"}:
                        raise RuntimeError(
                            f"invalid packed label member {member.name!r} in "
                            f"{self.shards[shard_index]}"
                        )
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise RuntimeError(
                            f"cannot read member {member.name!r} in "
                            f"{self.shards[shard_index]}"
                        )
                    with handle:
                        payload = handle.read()
                    with Image.open(io.BytesIO(payload)) as image:
                        image = checked_grayscale(image)
                        if self.transform is not None:
                            image = self.transform(image)
                    yield image, torch.tensor(int(label_text), dtype=torch.float32)


class MixedPackedThermalTarBatchDataset(IterableDataset):
    """Read labelled tar shards as two streams and emit class-mixed batches.

    This preserves sequential reads within each class (and therefore avoids
    random tar seeks), while preventing long runs of a single detector class
    from dominating consecutive Adam updates.  A small sidecar index supplies
    the number of members of each class in every shard; it is generated from
    the exact manifest used to create the archives.
    """

    emits_batches = True

    def __init__(
        self,
        shard_dir: str | Path,
        label_index: str | Path,
        transform: Optional[Callable],
        cool_per_batch: int,
        uncooled_per_batch: int,
        loader_workers: int,
    ) -> None:
        super().__init__()
        # Capture rank in the parent: spawned DataLoader workers have no DDP group.
        self.rank, self.world_size = PackedThermalTarDataset._rank_and_world()
        self.shard_dir = Path(shard_dir)
        metadata = json.loads(
            (self.shard_dir / "metadata.json").read_text(encoding="utf-8")
        )
        index = json.loads(Path(label_index).read_text(encoding="utf-8"))
        metadata_shards = metadata["shards"]
        indexed_shards = index["shards"]
        if len(metadata_shards) != len(indexed_shards):
            raise ValueError("tar label index does not match metadata shard count")
        self.shards: list[Path] = []
        self.shard_counts: list[tuple[int, int]] = []
        for metadata_item, index_item in zip(metadata_shards, indexed_shards):
            if metadata_item["name"] != index_item["name"]:
                raise ValueError("tar label index shard order does not match metadata")
            cooled = int(index_item["cooled"])
            uncooled = int(index_item["uncooled"])
            if cooled < 0 or uncooled < 0 or cooled + uncooled != int(metadata_item["samples"]):
                raise ValueError(f"invalid class counts for {metadata_item['name']}")
            self.shards.append(self.shard_dir / metadata_item["name"])
            self.shard_counts.append((cooled, uncooled))
        self.cooled_count = sum(item[0] for item in self.shard_counts)
        self.uncooled_count = sum(item[1] for item in self.shard_counts)
        if self.cooled_count != int(metadata["cooled_count"]) or self.uncooled_count != int(metadata["uncooled_count"]):
            raise ValueError("tar label index global counts do not match metadata")
        if cool_per_batch <= 0 or uncooled_per_batch <= 0:
            raise ValueError("both mixed batch class quotas must be positive")
        self.cool_per_batch = cool_per_batch
        self.uncooled_per_batch = uncooled_per_batch
        self.batch_size = cool_per_batch + uncooled_per_batch
        self.transform = transform
        self.loader_workers = loader_workers
        self._rank_shard_cache: dict[int, list[list[int]]] = {}
        # Each worker owns two disjoint, class-specific shard streams.  A
        # single whole-shard assignment is incorrect for label-sorted tar
        # archives: with many loader workers, most workers otherwise receive
        # only pure cooled or pure uncooled shards and can emit no mixed batch.
        self._worker_shard_cache: dict[
            tuple[int, int, int], list[tuple[list[int], list[int]]]
        ] = {}
        self.epoch = 0
        if loader_workers <= 0:
            raise ValueError("mixed packed tar reader requires loader_workers >= 1")

    def set_epoch(self, epoch: int) -> None:
        """Rotate shard ownership so quota remainders are covered over epochs.

        A fixed class quota necessarily leaves a small tail from one class in
        a single epoch.  Rotating before the capacity-balanced partition keeps
        every tar read sequential but prevents that tail from being the same
        records in every epoch.
        """
        self.epoch = int(epoch)
        self._rank_shard_cache.clear()
        self._worker_shard_cache.clear()

    def detector_class_counts(self) -> tuple[int, int]:
        return self.cooled_count, self.uncooled_count

    def training_sample_count(self) -> int:
        return self.steps_per_rank() * self.batch_size

    def _balanced_partition(
        self, shard_indices: list[int], group_count: int
    ) -> list[list[int]]:
        """Partition shards while balancing both class capacities per group."""
        if group_count <= 0:
            raise ValueError("group_count must be positive")
        total_cool = sum(self.shard_counts[index][0] for index in shard_indices)
        total_uncool = sum(self.shard_counts[index][1] for index in shard_indices)
        target_cool = total_cool / group_count
        target_uncool = total_uncool / group_count
        groups = [[] for _ in range(group_count)]
        counts = [[0, 0] for _ in range(group_count)]
        # Boundary shards can contain both labels.  Assign large and mixed
        # shards first; pure shards then fill the lower-count group.
        ordering = sorted(
            shard_indices,
            key=lambda index: (
                max(
                    self.shard_counts[index][0] / max(total_cool, 1),
                    self.shard_counts[index][1] / max(total_uncool, 1),
                ),
                sum(self.shard_counts[index]),
            ),
            reverse=True,
        )
        for shard_index in ordering:
            shard_cool, shard_uncool = self.shard_counts[shard_index]
            destination = min(
                range(group_count),
                key=lambda group: (
                    max(
                        (counts[group][0] + shard_cool) / max(target_cool, 1),
                        (counts[group][1] + shard_uncool) / max(target_uncool, 1),
                    ),
                    (counts[group][0] + shard_cool) / max(target_cool, 1)
                    + (counts[group][1] + shard_uncool) / max(target_uncool, 1),
                    len(groups[group]),
                    group,
                ),
            )
            groups[destination].append(shard_index)
            counts[destination][0] += shard_cool
            counts[destination][1] += shard_uncool
        return [sorted(group) for group in groups]

    def _rank_shards(self, rank: int, world_size: int) -> list[int]:
        if world_size not in self._rank_shard_cache:
            shard_indices = list(range(len(self.shards)))
            if shard_indices:
                offset = self.epoch % len(shard_indices)
                shard_indices = shard_indices[offset:] + shard_indices[:offset]
            self._rank_shard_cache[world_size] = self._balanced_partition(
                shard_indices, world_size
            )
        return self._rank_shard_cache[world_size][rank]

    def _partition_class_shards(
        self,
        shard_indices: list[int],
        label: int,
        group_count: int,
    ) -> list[list[int]]:
        """Greedily balance one class's sequential tar capacity per worker."""
        if group_count <= 0:
            raise ValueError("group_count must be positive")
        eligible = [index for index in shard_indices if self.shard_counts[index][label]]
        groups = [[] for _ in range(group_count)]
        counts = [0 for _ in range(group_count)]
        for shard_index in sorted(
            eligible,
            key=lambda index: (self.shard_counts[index][label], -index),
            reverse=True,
        ):
            destination = min(range(group_count), key=lambda group: (counts[group], group))
            groups[destination].append(shard_index)
            counts[destination] += self.shard_counts[shard_index][label]
        return [sorted(group) for group in groups]

    def _worker_shards(
        self, rank: int, world_size: int, worker_count: int
    ) -> list[tuple[list[int], list[int]]]:
        key = (rank, world_size, worker_count)
        if key not in self._worker_shard_cache:
            rank_shards = self._rank_shards(rank, world_size)
            cooled_shards = [
                index for index in rank_shards if self.shard_counts[index][0]
            ]
            uncooled_shards = [
                index for index in rank_shards if self.shard_counts[index][1]
            ]
            # A worker needs both streams to form a batch.  Extra DataLoader
            # workers are intentionally idle when there are fewer cooled
            # shards than workers; this is preferable to duplicating samples.
            active_workers = min(worker_count, len(cooled_shards), len(uncooled_shards))
            if active_workers <= 0:
                self._worker_shard_cache[key] = [([], []) for _ in range(worker_count)]
            else:
                cool_groups = self._partition_class_shards(
                    rank_shards, 0, active_workers
                )
                uncool_groups = self._partition_class_shards(
                    rank_shards, 1, active_workers
                )
                self._worker_shard_cache[key] = [
                    (cool_groups[index], uncool_groups[index])
                    if index < active_workers
                    else ([], [])
                    for index in range(worker_count)
                ]
        return self._worker_shard_cache[key]

    def _worker_capacity(
        self,
        rank: int,
        world_size: int,
        worker_id: int,
        worker_count: int,
    ) -> int:
        cool_shards, uncool_shards = self._worker_shards(
            rank, world_size, worker_count
        )[worker_id]
        cooled = sum(self.shard_counts[index][0] for index in cool_shards)
        uncooled = sum(self.shard_counts[index][1] for index in uncool_shards)
        return min(cooled // self.cool_per_batch, uncooled // self.uncooled_per_batch)

    def _rank_capacity(self, rank: int, world_size: int, worker_count: int) -> int:
        return sum(
            self._worker_capacity(rank, world_size, worker_id, worker_count)
            for worker_id in range(worker_count)
        )

    def steps_per_rank(self) -> int:
        _, world_size = self._rank_and_world()
        return min(
            self._rank_capacity(rank, world_size, self.loader_workers)
            for rank in range(world_size)
        )

    def __len__(self) -> int:
        # The DataLoader receives already formed batches from this iterable.
        return self.steps_per_rank()

    def _rank_and_world(self) -> tuple[int, int]:
        return self.rank, self.world_size

    @staticmethod
    def _allocate(capacities: list[int], total: int) -> list[int]:
        if total > sum(capacities):
            raise ValueError("requested worker allocation exceeds capacity")
        allocation = [total * capacity // sum(capacities) for capacity in capacities]
        remainder = total - sum(allocation)
        for index in sorted(
            range(len(capacities)),
            key=lambda item: (capacities[item] - allocation[item], -item),
            reverse=True,
        ):
            if remainder == 0:
                break
            if allocation[index] < capacities[index]:
                allocation[index] += 1
                remainder -= 1
        if remainder:
            raise RuntimeError("could not allocate mixed batches to tar workers")
        return allocation

    def _class_stream(self, shard_indices: list[int], label: int):
        label_text = str(label)
        for shard_index in shard_indices:
            if self.shard_counts[shard_index][label] == 0:
                continue
            with tarfile.open(self.shards[shard_index], mode="r") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.startswith(label_text + "/"):
                        continue
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise RuntimeError(
                            f"cannot read member {member.name!r} in {self.shards[shard_index]}"
                        )
                    with handle:
                        payload = handle.read()
                    with Image.open(io.BytesIO(payload)) as image:
                        image = checked_grayscale(image)
                        if self.transform is not None:
                            image = self.transform(image)
                    yield image

    def __iter__(self):
        rank, world_size = self._rank_and_world()
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        if worker_count != self.loader_workers:
            raise RuntimeError(
                f"mixed tar loader was configured for {self.loader_workers} workers, "
                f"but started with {worker_count}"
            )
        cool_shards, uncool_shards = self._worker_shards(
            rank, world_size, worker_count
        )[worker_id]
        if cool_shards:
            # Rotate the sequential starting shard for both class streams.
            # This changes which quota remainder is deferred without random
            # seeks within an archive.
            offset = self.epoch % len(cool_shards)
            cool_shards = cool_shards[offset:] + cool_shards[:offset]
        if uncool_shards:
            offset = self.epoch % len(uncool_shards)
            uncool_shards = uncool_shards[offset:] + uncool_shards[:offset]
        capacities = [
            self._worker_capacity(rank, world_size, index, worker_count)
            for index in range(worker_count)
        ]
        steps = self._allocate(capacities, self.steps_per_rank())[worker_id]
        cool_stream = self._class_stream(cool_shards, 0)
        uncooled_stream = self._class_stream(uncool_shards, 1)
        for _ in range(steps):
            images = [next(cool_stream) for _ in range(self.cool_per_batch)]
            labels = [0] * self.cool_per_batch
            images.extend(next(uncooled_stream) for _ in range(self.uncooled_per_batch))
            labels.extend([1] * self.uncooled_per_batch)
            # Prevent the network from observing a class-contiguous tensor
            # despite the deterministic two-stream source order.
            order = torch.randperm(self.batch_size)
            yield (
                torch.stack([images[index] for index in order.tolist()]),
                torch.tensor([labels[index] for index in order.tolist()], dtype=torch.float32),
            )
