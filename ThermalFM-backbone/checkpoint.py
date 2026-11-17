"""Checkpoint loading with an explicit, inspectable load report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Tuple, Union

import torch
import torch.nn as nn


CheckpointSource = Union[str, Path, Mapping[str, object]]
PHYSICS_PREFIXES = (
    "detector_router.",
    "radiation.",
    "atmosphere.",
    "tim.",
)


@dataclass(frozen=True)
class CheckpointLoadReport:
    loaded_keys: Tuple[str, ...]
    missing_keys: Tuple[str, ...]
    unexpected_keys: Tuple[str, ...]
    shape_mismatches: Tuple[str, ...]

    @property
    def loaded_count(self) -> int:
        return len(self.loaded_keys)

    def summary(self) -> str:
        return (
            f"loaded={len(self.loaded_keys)}, missing={len(self.missing_keys)}, "
            f"unexpected={len(self.unexpected_keys)}, "
            f"shape_mismatches={len(self.shape_mismatches)}"
        )


def load_checkpoint(source: CheckpointSource) -> Mapping[str, object]:
    if isinstance(source, Mapping):
        return source
    path = str(source)
    try:
        return torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_model_state(checkpoint: Mapping[str, object]) -> Dict[str, torch.Tensor]:
    """Accept AAA_phys training checkpoints and plain model state dicts."""
    candidate = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(candidate, Mapping):
        raise TypeError("checkpoint model/state_dict must be a mapping")
    state: Dict[str, torch.Tensor] = {}
    for raw_key, value in candidate.items():
        if not isinstance(raw_key, str) or not isinstance(value, torch.Tensor):
            continue
        key = raw_key
        for prefix in ("module.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        state[key] = value
    if not state:
        raise ValueError("checkpoint contains no tensor model state dict")
    return state


def load_matching_weights(
    module: nn.Module,
    source: CheckpointSource,
) -> CheckpointLoadReport:
    source_state = extract_model_state(load_checkpoint(source))
    target_state = module.state_dict()
    matching: MutableMapping[str, torch.Tensor] = {}
    unexpected = []
    mismatches = []
    for key, value in source_state.items():
        if key not in target_state:
            unexpected.append(key)
        elif tuple(value.shape) != tuple(target_state[key].shape):
            mismatches.append(
                f"{key}: checkpoint{tuple(value.shape)} != "
                f"model{tuple(target_state[key].shape)}"
            )
        else:
            matching[key] = value
    missing = [key for key in target_state if key not in matching]
    module.load_state_dict(matching, strict=False)
    return CheckpointLoadReport(
        loaded_keys=tuple(sorted(matching)),
        missing_keys=tuple(sorted(missing)),
        unexpected_keys=tuple(sorted(unexpected)),
        shape_mismatches=tuple(sorted(mismatches)),
    )
