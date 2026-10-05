"""
Training loop for saree design embedding model.

Training strategy:
- ArcFace loss with PK sampling (P designs × K views per batch)
- Heavy colour augmentations make repeated samples of the same image
  appear as different "colorways" → teaches colour invariance
- AdamW optimizer with cosine LR schedule + warmup
- AMP (mixed precision) for efficiency
- Best checkpoint saved by validation mAP
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from saree_reid.data.dataset import SareeDataset
from saree_reid.data.samplers import PKSampler
from saree_reid.data.transforms import build_eval_transform, build_train_transform
from saree_reid.eval.identification import extract_embeddings, run_identification
from saree_reid.losses.metric_losses import ArcFaceLoss
from saree_reid.models.backbone import build_model
from saree_reid.models.head import build_head
from saree_reid.utils import ensure_dir, get_device, seed_everything, setup_logging

log = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler | None = None,
    epoch: int = 0,
) -> dict[str, float]:
    """Train for one epoch, return metrics."""
    model.train()
    head.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels, _) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                embeddings = model(images)
                logits = head(embeddings, labels)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            embeddings = model(images)
            logits = head(embeddings, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # Track metrics
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)

    return {"loss": round(avg_loss, 4), "accuracy": round(accuracy, 4)}


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    """Full training pipeline."""
    setup_logging()
    train_cfg = cfg.get("training", {})
    seed_everything(train_cfg.get("seed", 42))

    device = get_device()
    log.info(f"Device: {device}")

    data_cfg = cfg.get("data", {})
    data_root = Path(data_cfg["data_root"])
    manifest_path = Path(data_cfg["manifest_path"])

    # Build datasets
    train_transform = build_train_transform(cfg)
    eval_transform = build_eval_transform(cfg)

    max_images = train_cfg.get("max_images", None)

    train_ds = SareeDataset(
        manifest_path, data_root, split="train",
        transform=train_transform, max_images=max_images,
    )
    val_ds = SareeDataset(
        manifest_path, data_root, split="val",
        transform=eval_transform,
    )

    # PK sampler for training
    sampler = PKSampler(
        labels=train_ds.get_labels(),
        p=train_cfg.get("pk_p", 8),
        k=train_cfg.get("pk_k", 4),
        seed=train_cfg.get("seed", 42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("eval", {}).get("gallery_batch_size", 32),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )

    # Build model + head + loss
    model = build_model(cfg).to(device)
    head = build_head(cfg, num_classes=train_ds.num_classes).to(device)
    criterion = ArcFaceLoss()

    # Optimizer
    params = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    # Scheduler
    epochs = train_cfg.get("epochs", 30)
    warmup = train_cfg.get("warmup_epochs", 2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup, eta_min=1e-6,
    )

    # AMP scaler
    use_amp = train_cfg.get("amp", False) and device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    # Checkpoint directory
    ckpt_dir = ensure_dir(cfg.get("output", {}).get("checkpoint_dir", "checkpoints"))
    results_dir = ensure_dir(cfg.get("output", {}).get("results_dir", "results"))

    best_metric = -1.0
    history = []

    log.info(f"Starting training: {epochs} epochs, {len(train_loader)} batches/epoch")

    for epoch in range(epochs):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, head, train_loader, optimizer, criterion, device, scaler, epoch
        )

        # Step scheduler (after warmup)
        if epoch >= warmup:
            scheduler.step()

        # Validate (identification metrics)
        if len(val_ds) > 1:
            val_results = run_identification(model, val_loader, device)
            val_metric = val_results.get("mAP", 0.0)
        else:
            val_results = {}
            val_metric = 0.0

        epoch_time = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        epoch_log = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_results,
            "lr": round(lr, 6),
            "time_s": round(epoch_time, 1),
        }
        history.append(epoch_log)

        log.info(
            f"Epoch {epoch}/{epochs-1}: "
            f"loss={train_metrics['loss']:.4f}, "
            f"acc={train_metrics['accuracy']:.4f}, "
            f"val_mAP={val_metric:.4f}, "
            f"lr={lr:.6f}, "
            f"time={epoch_time:.1f}s"
        )

        # Save best checkpoint
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "head_state_dict": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metric": val_metric,
                "cfg": cfg,
            }, ckpt_dir / "best.pth")
            log.info(f"  → New best checkpoint (mAP={val_metric:.4f})")

    # Save final checkpoint
    torch.save({
        "epoch": epochs - 1,
        "model_state_dict": model.state_dict(),
        "head_state_dict": head.state_dict(),
        "cfg": cfg,
    }, ckpt_dir / "last.pth")

    # Save training history
    history_path = results_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    log.info(f"Training complete. Best val mAP: {best_metric:.4f}")
    return {"best_val_mAP": best_metric, "history": history}
