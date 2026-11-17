"""Full ThermalFM pretraining: python -m thermalfm.pretrain --help."""

import argparse
from contextlib import nullcontext
import time

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .api import PIPELINE_VERSION, create_pretrainer, model_config
from .checkpoint import extract_model_state, load_checkpoint
from .data import ImageManifest, MixedBatchSampler, class_quotas, image_transform
from . import runtime as R
from .tar_data import MixedPackedThermalTarBatchDataset


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="TSV: image_path<TAB>0=cooled/1=uncooled")
    source.add_argument("--tar-dir", help="Existing labelled tar cache with metadata.json")
    p.add_argument("--root", default=".")
    p.add_argument("--tar-index", help="Per-shard cooled/uncooled label index JSON")
    p.add_argument("--output", required=True)
    weight = p.add_mutually_exclusive_group()
    weight.add_argument("--init", help="Strict full-model weight initialization, resetting optimizer/LR/epoch")
    weight.add_argument("--resume", help="Epoch-boundary resume of this package's checkpoints")
    p.add_argument("--epochs", type=int, default=500, help="Total target epochs, not additional epochs")
    p.add_argument("--batch-size", type=int, default=100, help="Per GPU/rank")
    p.add_argument("--cooled-per-batch", type=int, help="Default: rounded corpus fraction, at least one per class")
    p.add_argument("--workers", type=int, default=1, help="Start with 1 for sequential tar; benchmark before increasing")
    p.add_argument("--lr", type=float, default=1.5e-3, help="Explicit actual LR; use e.g. 1e-5 for low-LR continuation")
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=float, default=20)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--accum-iter", type=int, default=1)
    p.add_argument("--frequency-loss-weight", type=float, default=0.05)
    p.add_argument("--detector-loss-weight", type=float, default=0.01)
    p.add_argument("--detector-pos-weight", type=float, help="Default: cooled/uncooled batch quota ratio")
    p.add_argument("--mask-ratio", type=float, default=0.75)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--print-freq", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=0, help="Debug only; 0 traverses the planned epoch")
    p.add_argument("--tiny", action="store_true", help="Small random model for smoke tests")
    return p


def run(args):
    if min(args.epochs, args.batch_size, args.accum_iter, args.print_freq) < 1 or args.workers < 0 or args.max_steps < 0:
        raise ValueError("Invalid epoch/batch/accumulation/worker/step configuration")
    if not 0 <= args.warmup_epochs < args.epochs or not 0 <= args.min_lr <= args.lr or args.lr <= 0:
        raise ValueError("Require 0 <= warmup_epochs < epochs and 0 <= min_lr <= lr, with lr > 0")
    if not 0 < args.mask_ratio < 1:
        raise ValueError("mask_ratio must lie strictly between zero and one")
    device, rank, world = R.setup(args.device, args.seed)
    config = model_config(args.tiny, mask_ratio=args.mask_ratio,
                          frequency_loss_weight=args.frequency_loss_weight,
                          detector_loss_weight=args.detector_loss_weight)
    transform = image_transform(config["img_size"], train=True)
    if args.tar_dir:
        if not args.tar_index:
            raise ValueError("--tar-index is required with --tar-dir")
        # Metadata supplies counts; actual archive members remain sequential.
        import json
        from pathlib import Path
        metadata = json.loads((Path(args.tar_dir) / "metadata.json").read_text())
        counts = metadata["cooled_count"], metadata["uncooled_count"]
        cooled, uncooled = class_quotas(*counts, args.batch_size, args.cooled_per_batch)
        data = MixedPackedThermalTarBatchDataset(
            args.tar_dir, args.tar_index, transform, cooled, uncooled, max(args.workers, 1)
        )
        loader = DataLoader(data, batch_size=None, num_workers=args.workers,
                            pin_memory=device.type == "cuda", persistent_workers=False,
                            worker_init_fn=R.seed_worker)
        sampler = data
        data_signature = dict(tar_dir=str(Path(args.tar_dir).resolve()),
                              metadata=R.file_sha256(Path(args.tar_dir) / "metadata.json"),
                              index=R.file_sha256(args.tar_index))
    else:
        data = ImageManifest(args.manifest, args.root, transform)
        counts = data.labels.count(0), data.labels.count(1)
        cooled, uncooled = class_quotas(*counts, args.batch_size, args.cooled_per_batch)
        sampler = MixedBatchSampler(data.labels, cooled, uncooled, rank, world, args.seed)
        loader = DataLoader(data, batch_sampler=sampler, num_workers=args.workers,
                            pin_memory=device.type == "cuda", persistent_workers=False,
                            worker_init_fn=R.seed_worker)
        from pathlib import Path
        data_signature = dict(manifest=R.file_sha256(args.manifest), root=str(Path(args.root).resolve()))
    if len(loader) == 0:
        raise ValueError("Mixed plan is empty; reduce batch/world size or inspect per-class shard capacity")
    config["detector_pos_weight"] = (cooled / uncooled if args.detector_pos_weight is None
                                      else args.detector_pos_weight)
    model = create_pretrainer(**config).to(device)
    if args.init:
        saved = load_checkpoint(args.init)
        if saved.get("pipeline_version", PIPELINE_VERSION) != PIPELINE_VERSION:
            raise ValueError("--init expects the full model, not an ablation checkpoint")
        model.load_state_dict(extract_model_state(saved), strict=True)
        del saved
    optimizer = R.optimizer_for(model, args.lr, args.weight_decay)
    scaler = R.scaler_for(device, args.amp)
    signature = dict(data=data_signature, quotas=(cooled, uncooled), model=config,
                     world=world, workers=args.workers, seed=args.seed, epochs=args.epochs,
                     lr=args.lr, min_lr=args.min_lr, warmup=args.warmup_epochs,
                     accum=args.accum_iter, amp=args.amp, weight_decay=args.weight_decay,
                     max_steps=args.max_steps)
    start = 0
    if args.resume:
        saved = load_checkpoint(args.resume)
        start = R.resume_epoch(saved, model, optimizer, scaler, signature, device, rank, world, "pretrain")
        del saved
    if start >= args.epochs:
        raise ValueError("Resume already reached target epochs; use --init for a new continuation schedule")
    output = R.output_directory(args.output, args.resume, rank, start=start)
    train_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None) if world > 1 else model
    if rank == 0:
        print(f"Images={sum(counts):,}; cooled={counts[0]:,}, uncooled={counts[1]:,}; "
              f"steps/rank={len(loader)}; batch/rank={cooled} cooled + {uncooled} uncooled; "
              f"pos_weight={config['detector_pos_weight']:.8g}; parameters="
              f"{sum(p.numel() for p in model.parameters()):,}", flush=True)
    steps = min(len(loader), args.max_steps) if args.max_steps else len(loader)
    for epoch in range(start, args.epochs):
        sampler.set_epoch(epoch)
        # New workers each epoch see updated tar assignments and deterministic seeds.
        loader.generator = torch.Generator().manual_seed(args.seed + epoch * world + rank)
        train_model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(5, dtype=torch.float64, device=device)
        began = time.monotonic()
        for step, (images, labels) in enumerate(loader):
            if step >= steps:
                break
            if int((labels == 0).sum()) != cooled or int((labels == 1).sum()) != uncooled:
                raise RuntimeError("Batch class quota mismatch")
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.float32, non_blocking=True)
            group_start = step // args.accum_iter * args.accum_iter
            divisor = min(args.accum_iter, steps - group_start)
            update = (step + 1) % args.accum_iter == 0 or step + 1 == steps
            lr = R.schedule(optimizer, epoch + step / steps, args.lr, args.min_lr,
                            args.warmup_epochs, args.epochs)
            sync = train_model.no_sync() if world > 1 and not update else nullcontext()
            with sync:
                with R.autocast(device, args.amp):
                    loss, _, _, parts = train_model(images, detector_labels=labels,
                                                   return_loss_components=True)
                R.finite_loss(loss, device)
                scaler.scale(loss / divisor).backward()
            if update:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            values = [loss, parts["pixel_reconstruction_loss"],
                      parts["low_frequency_residual_loss"], parts["detector_classification_loss"]]
            totals[:4] += torch.stack([v.detach().double() for v in values]) * len(images)
            totals[4] += len(images)
            if rank == 0 and (step % args.print_freq == 0 or step + 1 == steps):
                print(f"epoch={epoch} step={step + 1}/{steps} loss={loss.item():.6f} lr={lr:.3g}", flush=True)
        R.reduce_sum(totals)
        names = ("loss", "pixel_mse_normalized", "low_frequency_loss", "detector_bce")
        record = dict(epoch=epoch, **{name: float(totals[i] / totals[4]) for i, name in enumerate(names)},
                      images=int(totals[4]), seconds=time.monotonic() - began, lr=lr)
        R.log_record(output, record, rank)
        R.save_epoch(output, model, optimizer, scaler, epoch, config, args, signature, device, rank, "pretrain")


def main(argv=None):
    try:
        run(parser().parse_args(argv))
    finally:
        R.cleanup()


if __name__ == "__main__":
    main()
