"""Small shared training runtime: DDP, AMP, epoch checkpoints and RNG state."""

from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

FORMAT = "thermalfm-minimal-v1"


def distributed():
    return dist.is_available() and dist.is_initialized()


def setup(device_name, seed):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --device cpu only for small smoke tests")
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    rank = dist.get_rank() if distributed() else 0
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    return device, rank, world


def cleanup():
    if distributed():
        dist.destroy_process_group()


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def autocast(device, enabled):
    return torch.autocast("cuda", dtype=torch.float16) if enabled and device.type == "cuda" else nullcontext()


def scaler_for(device, enabled):
    return torch.amp.GradScaler("cuda", enabled=enabled and device.type == "cuda")


def optimizer_for(model, lr, weight_decay):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim == 1 or name.endswith(".bias") else decay).append(parameter)
    return torch.optim.AdamW([
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ], lr=lr, betas=(0.9, 0.95))


def schedule(optimizer, progress, lr, min_lr, warmup, epochs):
    if warmup and progress < warmup:
        value = lr * progress / warmup
    else:
        phase = min(1.0, max(0.0, (progress - warmup) / max(epochs - warmup, 1e-12)))
        value = min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * phase))
    for group in optimizer.param_groups:
        group["lr"] = value * group.get("lr_scale", 1.0)
    return value


def finite_loss(loss, device):
    flag = torch.tensor(int(bool(torch.isfinite(loss).all())), device=device)
    if distributed():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise FloatingPointError("Non-finite loss on at least one rank")


def reduce_sum(tensor):
    if distributed():
        dist.all_reduce(tensor)
    return tensor


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_directory(path, resume, rank, start=0):
    path = Path(path)
    if path.exists() and list(path.glob("checkpoint-*.pth")) and not resume:
        raise FileExistsError(f"{path} already contains checkpoints; use a new --output or --resume")
    if resume and path.exists():
        future = [p for p in path.glob("checkpoint-*.pth")
                  if p.stem.removeprefix("checkpoint-").isdigit()
                  and int(p.stem.removeprefix("checkpoint-")) >= start]
        if future:
            raise FileExistsError(f"Resume would overwrite completed epochs in {path}; use a new --output")
    if rank == 0:
        path.mkdir(parents=True, exist_ok=True)
    if distributed():
        dist.barrier()
    return path


def log_record(output, record, rank):
    if rank == 0:
        text = json.dumps(record, allow_nan=False)
        print(text, flush=True)
        with (output / "log.jsonl").open("a") as stream:
            stream.write(text + "\n")


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def save_epoch(output, model, optimizer, scaler, epoch, config, args, signature, device, rank, task):
    state = rng_state(device)
    states = [None] * dist.get_world_size() if distributed() else [state]
    if distributed():
        dist.all_gather_object(states, state)
    if rank == 0:
        path = output / f"checkpoint-{epoch}.pth"
        temporary = path.with_suffix(".pth.tmp")
        torch.save(dict(
            format=FORMAT, task=task, pipeline_version="aaa_phys_detector_gated_tim5_v1",
            epoch=epoch, model=model.state_dict(), model_config=config,
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            args=vars(args), signature=signature, rng_states=states,
        ), temporary)
        temporary.replace(path)
    if distributed():
        dist.barrier()


def resume_epoch(saved, model, optimizer, scaler, signature, device, rank, world, task):
    if saved.get("format") != FORMAT or saved.get("task") != task:
        raise ValueError("Strict --resume accepts this package's same-task checkpoints only; use --init for old pth")
    if saved["signature"] != signature or len(saved["rng_states"]) != world:
        raise ValueError("Resume configuration/data/world size changed; use a new run with --init instead")
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    scaler.load_state_dict(saved["scaler"])
    restore_rng(saved["rng_states"][rank], device)
    return saved["epoch"] + 1
