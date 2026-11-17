"""TSV images, safe grayscale conversion and exact per-rank class mixtures."""

from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
ANNOTATION_DIRS = {"mask", "masks", "label", "labels", "annotation", "annotations"}


def checked_grayscale(image):
    if image.mode in {"I", "F"} or image.mode.startswith("I;"):
        raise ValueError("High-bit-depth input needs an explicit intensity mapping before training")
    return image.convert("L")


def image_transform(size=224, train=False):
    spatial = [T.RandomResizedCrop(size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
               T.RandomHorizontalFlip()] if train else [
        T.Resize(round(size * 256 / 224), interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(size),
    ]
    return T.Compose(spatial + [T.Grayscale(3), T.ToTensor(), T.Normalize(MEAN, STD)])


class ImageManifest(Dataset):
    """Each non-comment line: path<TAB>integer_label; paths relative to root."""

    def __init__(self, manifest, root=".", transform=None, num_classes=2):
        self.manifest = Path(manifest)
        self.root = Path(root)
        self.transform = transform
        self.paths, self.labels = [], []
        for number, line in enumerate(self.manifest.read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                path, label = line.rsplit("\t", 1)
                label = int(label)
            except ValueError as exc:
                raise ValueError(f"{manifest}:{number}: expected path<TAB>integer_label") from exc
            if not 0 <= label < num_classes:
                raise ValueError(f"{manifest}:{number}: invalid label {label}")
            if ANNOTATION_DIRS.intersection(part.lower() for part in Path(path).parts):
                raise ValueError(f"Annotation path in image manifest: {path}; create a cleaned manifest")
            self.paths.append(Path(path) if Path(path).is_absolute() else self.root / path)
            self.labels.append(label)
        if not self.paths:
            raise ValueError(f"Empty manifest: {manifest}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = checked_grayscale(image)
            if self.transform is not None:
                image = self.transform(image)
        return image, self.labels[index]


class MixedBatchSampler(Sampler):
    """Disjoint class-wise permutations, then fixed quotas within every batch.

    No oversampling. Remainders are deferred; shuffling changes them each epoch.
    All ranks have identical batch counts. Workers fetch indices, not classes.
    """

    def __init__(self, labels, cooled, uncooled, rank=0, world_size=1, seed=0):
        self.indices = [torch.tensor([i for i, y in enumerate(labels) if y == label])
                        for label in (0, 1)]
        self.quotas = (cooled, uncooled)
        self.rank, self.world_size, self.seed, self.epoch = rank, world_size, seed, 0
        if min(self.quotas) <= 0 or not 0 <= rank < world_size:
            raise ValueError("Both class quotas must be positive and rank must be valid")
        self.steps = min(len(x) // (quota * world_size)
                         for x, quota in zip(self.indices, self.quotas))
        if self.steps == 0:
            raise ValueError("No full mixed batch per rank: reduce batch/quotas or world size")

    def __len__(self):
        return self.steps

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed + self.epoch)
        streams = [x[torch.randperm(len(x), generator=rng)][self.rank::self.world_size]
                   for x in self.indices]
        for step in range(self.steps):
            batch = torch.cat([x[step * quota:(step + 1) * quota]
                               for x, quota in zip(streams, self.quotas)])
            yield batch[torch.randperm(len(batch), generator=rng)].tolist()


class EvaluationSampler(Sampler):
    """No padding/duplicates, including when some ranks have zero samples."""

    def __init__(self, dataset, rank=0, world_size=1):
        self.indices = range(rank, len(dataset), world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def class_quotas(cooled_count, uncooled_count, batch_size, cooled=None):
    if min(cooled_count, uncooled_count) <= 0 or batch_size < 2:
        raise ValueError("Pretraining requires both cooled and uncooled samples and batch_size >= 2")
    if cooled is None:
        cooled = max(1, min(batch_size - 1, round(batch_size * cooled_count / (cooled_count + uncooled_count))))
    if not 0 < cooled < batch_size:
        raise ValueError("Require 0 < cooled_per_batch < batch_size")
    return cooled, batch_size - cooled
