"""Frozen linear probe or full backbone fine-tuning on image classification."""

import argparse

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from .api import ThermalClassifier, create_backbone, model_config
from .checkpoint import load_checkpoint
from .data import EvaluationSampler, ImageManifest, image_transform
from . import runtime as R


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-manifest")
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--output", required=True)
    weights = p.add_mutually_exclusive_group(required=True)
    weights.add_argument("--init", help="Full MAE or encoder checkpoint used to initialize the backbone")
    weights.add_argument("--resume", help="Same-task package checkpoint including trained head")
    p.add_argument("--mode", choices=("linear-probe", "finetune"), default="linear-probe")
    p.add_argument("--eval-only", action="store_true", help="Requires --resume, retaining the trained head")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32, help="Per rank")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3, help="Head LR")
    p.add_argument("--backbone-lr-scale", type=float, default=0.1)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=float, default=0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-steps", type=int, default=0, help="Training debug limit only")
    p.add_argument("--tiny", action="store_true", help="Testing architecture only; incompatible with checkpoint")
    return p


@torch.inference_mode()
def evaluate(model, loader, device, num_classes, amp):
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.float64, device=device)
    loss_sum = torch.zeros(1, dtype=torch.float64, device=device)
    # Use the bare module, not DDP: no collectives per forward or padded duplicates.
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device, dtype=torch.long)
        with R.autocast(device, amp):
            logits = model(images)
        loss_sum += F.cross_entropy(logits.float(), labels, reduction="sum").double()
        confusion += torch.bincount(labels * num_classes + logits.argmax(1),
                                    minlength=num_classes**2).view(num_classes, num_classes)
    R.reduce_sum(confusion)
    R.reduce_sum(loss_sum)
    support = confusion.sum(1)
    valid = support > 0
    return dict(loss=float(loss_sum[0] / support.sum()),
                accuracy=float(confusion.diag().sum() / support.sum()),
                balanced_accuracy=float((confusion.diag()[valid] / support[valid]).mean()),
                evaluated_images=int(support.sum()), classes_with_support=int(valid.sum()),
                confusion=confusion.long().cpu().tolist())


def run(args):
    if min(args.epochs, args.batch_size) < 1 or args.workers < 0 or args.max_steps < 0:
        raise ValueError("Invalid epoch/batch/worker/step configuration")
    if not 0 <= args.warmup_epochs < args.epochs or not 0 <= args.min_lr <= args.lr or args.lr <= 0:
        raise ValueError("Invalid learning-rate schedule")
    if args.backbone_lr_scale <= 0:
        raise ValueError("backbone_lr_scale must be positive")
    if args.eval_only and not args.resume:
        raise ValueError("--eval-only requires --resume with a trained classification head")
    if not args.eval_only and not args.train_manifest:
        raise ValueError("--train-manifest required for head training")
    device, rank, world = R.setup(args.device, args.seed)
    saved = load_checkpoint(args.resume or args.init)
    config = saved.get("model_config", model_config(args.tiny))
    if args.resume:
        if saved.get("format") != R.FORMAT or saved.get("task") != "classification":
            raise ValueError("--resume must be this package's classification checkpoint")
        if saved["args"]["num_classes"] != args.num_classes or saved["args"]["mode"] != args.mode:
            raise ValueError("Classification mode/num_classes differs from checkpoint")
        backbone = create_backbone(**config)
    else:
        backbone = create_backbone(checkpoint=saved, **config)
    model = ThermalClassifier(backbone, args.num_classes, frozen=args.mode == "linear-probe").to(device)
    if args.resume:
        model.load_state_dict(saved["model"], strict=True)
    validation = ImageManifest(args.val_manifest, args.root,
                               image_transform(config["img_size"]), args.num_classes)
    val_loader = DataLoader(validation, batch_size=args.batch_size,
                            sampler=EvaluationSampler(validation, rank, world),
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    if args.eval_only:
        metrics = evaluate(model, val_loader, device, args.num_classes, args.amp)
        if rank == 0:
            import json
            print(json.dumps(metrics), flush=True)
        return
    # Linear probes use deterministic eval preprocessing. Fine-tuning augments.
    train_data = ImageManifest(args.train_manifest, args.root,
                               image_transform(config["img_size"], train=args.mode == "finetune"),
                               args.num_classes)
    if set(path.resolve() for path in train_data.paths) & set(path.resolve() for path in validation.paths):
        raise ValueError("Exact image-path overlap between train and validation manifests")
    sampler = DistributedSampler(train_data, num_replicas=world, rank=rank, shuffle=True,
                                  seed=args.seed) if world > 1 else RandomSampler(train_data)
    loader = DataLoader(train_data, sampler=sampler, batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        worker_init_fn=R.seed_worker, persistent_workers=False)
    groups = [{"params": model.head.parameters(), "lr_scale": 1.0}]
    if args.mode == "finetune":
        groups.append({"params": model.backbone.parameters(), "lr_scale": args.backbone_lr_scale})
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = R.scaler_for(device, args.amp)
    signature = dict(model=config, train=R.file_sha256(args.train_manifest),
                     val=R.file_sha256(args.val_manifest), root=str(train_data.root.resolve()),
                     mode=args.mode, num_classes=args.num_classes, world=world,
                     epochs=args.epochs, batch=args.batch_size, workers=args.workers,
                     seed=args.seed, lr=args.lr, backbone_lr_scale=args.backbone_lr_scale,
                     min_lr=args.min_lr, warmup=args.warmup_epochs, amp=args.amp,
                     weight_decay=args.weight_decay, max_steps=args.max_steps)
    start = 0
    if args.resume:
        start = R.resume_epoch(saved, model, optimizer, scaler, signature, device, rank, world, "classification")
    del saved
    if start >= args.epochs:
        raise ValueError("Target epochs already completed; use --eval-only to evaluate")
    output = R.output_directory(args.output, args.resume, rank, start=start)
    train_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                      find_unused_parameters=args.mode == "finetune") if world > 1 else model
    steps = min(len(loader), args.max_steps) if args.max_steps else len(loader)
    for epoch in range(start, args.epochs):
        if world > 1:
            sampler.set_epoch(epoch)
        generator = torch.Generator().manual_seed(args.seed + epoch * world + rank)
        loader.generator = generator
        if world == 1:
            sampler.generator = generator
        train_model.train()
        totals = torch.zeros(2, dtype=torch.float64, device=device)
        for step, (images, labels) in enumerate(loader):
            if step >= steps:
                break
            images, labels = images.to(device), labels.to(device, dtype=torch.long)
            lr = R.schedule(optimizer, epoch + step / steps, args.lr, args.min_lr,
                            args.warmup_epochs, args.epochs)
            optimizer.zero_grad(set_to_none=True)
            with R.autocast(device, args.amp):
                loss = F.cross_entropy(train_model(images), labels)
            R.finite_loss(loss, device)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            totals[0] += loss.detach().double() * len(images)
            totals[1] += len(images)
        R.reduce_sum(totals)
        metrics = evaluate(model, val_loader, device, args.num_classes, args.amp)
        R.log_record(output, dict(epoch=epoch, train_loss=float(totals[0] / totals[1]),
                                  lr=lr, validation=metrics), rank)
        R.save_epoch(output, model, optimizer, scaler, epoch, config, args, signature,
                      device, rank, "classification")


def main(argv=None):
    try:
        run(parser().parse_args(argv))
    finally:
        R.cleanup()


if __name__ == "__main__":
    main()
