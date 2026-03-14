"""
train.py — Training loop for TB Bacilli Segmentation.

- AdamW differential LR: encoder=6e-5, decoder=6e-4
- CosineAnnealingWarmRestarts
- Mixed precision + gradient clipping
- Early stopping patience=15 on val_f1
- val_fp_rate_on_negatives logged every epoch (CRITICAL)
- Checkpointing: best (val_f1) + last (every epoch)
- Auto-resume from last checkpoint
- TensorBoard logging
"""

import argparse
import logging
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

from augmentations import get_train_transforms, get_val_transforms
from config import TBConfig
from dataset import get_dataloaders
from loss import BinaryDice, TBSegLoss
from models import get_model
from utils import (
    AverageMeter,
    EarlyStopping,
    compute_fp_rate_on_negatives,
    compute_pixel_metrics,
    load_checkpoint,
    plot_training_curves,
    save_checkpoint,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, criterion, optimizer, scaler, config, device):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    focal_meter = AverageMeter()
    pp_meter = AverageMeter()

    for batch_idx, (images, masks, _meta) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        if config.mixed_precision:
            with autocast():
                logits = model(images)
                loss_dict = criterion(logits, masks)
                loss = loss_dict["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss_dict = criterion(logits, masks)
            loss = loss_dict["total"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_max_norm)
            optimizer.step()

        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        dice_meter.update(loss_dict["dice_loss"].item(), bs)
        focal_meter.update(loss_dict["focal_loss"].item(), bs)
        pp_meter.update(loss_dict["precision_penalty"].item(), bs)

        if (batch_idx + 1) % config.log_interval == 0:
            logger.info(
                f"  Batch [{batch_idx+1}/{len(loader)}] "
                f"Loss: {loss_meter.avg:.4f} "
                f"Dice: {dice_meter.avg:.4f} "
                f"Focal: {focal_meter.avg:.4f} "
                f"PP: {pp_meter.avg:.4f}"
            )

    return {
        "train_loss": loss_meter.avg,
        "train_dice_loss": dice_meter.avg,
        "train_focal_loss": focal_meter.avg,
        "train_precision_penalty": pp_meter.avg,
    }


@torch.no_grad()
def validate(model, loader, criterion, config, device, threshold=0.65):
    """Validate on val split."""
    model.eval()
    loss_meter = AverageMeter()
    dice_metric = BinaryDice()

    all_preds = []
    all_targets = []

    for images, masks, _meta in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if config.mixed_precision:
            with autocast():
                logits = model(images)
                loss_dict = criterion(logits, masks)
        else:
            logits = model(images)
            loss_dict = criterion(logits, masks)

        loss_meter.update(loss_dict["total"].item(), images.size(0))

        # Collect predictions for metrics
        probs = torch.sigmoid(logits)
        for i in range(images.size(0)):
            pred_np = probs[i, 0].cpu().numpy()
            target_np = masks[i, 0].cpu().numpy()
            all_preds.append(pred_np)
            all_targets.append(target_np)

    # Compute metrics
    metrics_sum = {"dice": 0, "iou": 0, "precision": 0, "recall": 0, "f1": 0}
    for pred, target in zip(all_preds, all_targets):
        m = compute_pixel_metrics(pred, target, threshold=threshold)
        for k in metrics_sum:
            metrics_sum[k] += m[k]
    n = len(all_preds)
    metrics_avg = {k: v / n for k, v in metrics_sum.items()}

    # FP rate on negatives — CRITICAL metric
    fp_rate = compute_fp_rate_on_negatives(all_preds, all_targets, threshold=threshold)

    return {
        "val_loss": loss_meter.avg,
        "val_dice": metrics_avg["dice"],
        "val_iou": metrics_avg["iou"],
        "val_precision": metrics_avg["precision"],
        "val_recall": metrics_avg["recall"],
        "val_f1": metrics_avg["f1"],
        "val_fp_rate_on_negatives": fp_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="Train TB Bacilli Segmentation")
    parser.add_argument("--model", type=str, default=None, help="Model name override")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs override")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    parser.add_argument("--resume", action="store_true", help="Force resume from checkpoint")
    args = parser.parse_args()

    config = TBConfig()
    if args.model:
        config.model_name = args.model
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size

    # Set seed — NON-NEGOTIABLE
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Model
    model = get_model(config)
    model = model.to(device)

    # Loss
    criterion = TBSegLoss(
        dice_weight=config.dice_weight,
        focal_weight=config.focal_weight,
        precision_penalty_weight=config.precision_penalty_weight,
        focal_gamma=config.focal_gamma,
        focal_alpha=config.focal_alpha,
        dice_smooth=config.dice_smooth,
        precision_smooth=config.precision_smooth,
    )

    # Optimizer — Differential LR
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(
            key in name
            for key in ["encoder", "patch_embed", "downsample"]
        ):
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    optimizer = AdamW(
        [
            {"params": encoder_params, "lr": config.encoder_lr},
            {"params": decoder_params, "lr": config.decoder_lr},
        ],
        weight_decay=config.weight_decay,
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=config.cosine_T_0, T_mult=config.cosine_T_mult
    )

    scaler = GradScaler() if config.mixed_precision else None

    # Data
    train_transforms = get_train_transforms(config.input_size)
    val_transforms = get_val_transforms(config.input_size)
    dataloaders = get_dataloaders(config, train_transforms, val_transforms)

    if "train" not in dataloaders:
        logger.error("No training data found. Run prepare_data.py first.")
        return
    if "val" not in dataloaders:
        logger.error("No validation data found. Run prepare_data.py first.")
        return

    # Resume
    start_epoch = 0
    best_metric = 0.0
    last_ckpt = os.path.join(config.checkpoint_dir, "last.pth")
    best_ckpt = os.path.join(config.checkpoint_dir, "best.pth")

    if os.path.isfile(last_ckpt):
        logger.info("Auto-resuming from last checkpoint...")
        start_epoch, best_metric = load_checkpoint(
            last_ckpt, model, optimizer, scheduler
        )
        start_epoch += 1  # Start from next epoch

    # TensorBoard
    writer = SummaryWriter(log_dir=config.log_dir)
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        monitor=config.early_stopping_monitor,
        mode="max",
    )
    log_history = defaultdict(list)

    logger.info(f"Starting training from epoch {start_epoch} to {config.epochs}")
    logger.info(f"Encoder LR: {config.encoder_lr}, Decoder LR: {config.decoder_lr}")

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        logger.info(f"Epoch [{epoch+1}/{config.epochs}]")

        # Train
        train_metrics = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, scaler, config, device
        )

        # Validate
        val_metrics = validate(
            model, dataloaders["val"], criterion, config, device,
            threshold=config.confidence_threshold,
        )

        # Scheduler step
        scheduler.step()

        # Get current LRs
        encoder_lr = optimizer.param_groups[0]["lr"]
        decoder_lr = optimizer.param_groups[1]["lr"]

        # Log
        all_metrics = {**train_metrics, **val_metrics, "encoder_lr": encoder_lr, "decoder_lr": decoder_lr}
        for k, v in all_metrics.items():
            log_history[k].append(v)
            writer.add_scalar(k, v, epoch)

        elapsed = time.time() - epoch_start
        logger.info(
            f"  Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val Dice: {val_metrics['val_dice']:.4f} | "
            f"Val F1: {val_metrics['val_f1']:.4f} | "
            f"Val IoU: {val_metrics['val_iou']:.4f} | "
            f"Val Prec: {val_metrics['val_precision']:.4f} | "
            f"Val Rec: {val_metrics['val_recall']:.4f} | "
            f"FP_neg: {val_metrics['val_fp_rate_on_negatives']:.4f} | "
            f"Enc LR: {encoder_lr:.2e} | Dec LR: {decoder_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # Checkpoint
        is_best = val_metrics["val_f1"] > best_metric
        if is_best:
            best_metric = val_metrics["val_f1"]

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_metric": best_metric,
                "config": config,
            },
            filepath=last_ckpt,
            is_best=is_best,
            best_path=best_ckpt,
        )

        # Sample overlays (every N epochs)
        if (epoch + 1) % config.overlay_interval == 0:
            _log_sample_overlay(model, dataloaders["val"], writer, epoch, device, config)

        # Early stopping
        if early_stopping.step(val_metrics["val_f1"]):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Save training curves
    plot_training_curves(
        dict(log_history),
        os.path.join(config.output_dir, "training_curves.png"),
    )
    writer.close()
    logger.info(f"Training complete. Best val_f1: {best_metric:.4f}")


@torch.no_grad()
def _log_sample_overlay(model, val_loader, writer, epoch, device, config):
    """Log sample overlays to TensorBoard."""
    model.eval()
    images, masks, _meta = next(iter(val_loader))
    images = images.to(device)
    logits = model(images)
    probs = torch.sigmoid(logits)

    # Take first 4 samples
    n = min(4, images.size(0))
    for i in range(n):
        pred = probs[i, 0].cpu().numpy()
        target = masks[i, 0].numpy()
        pred_bin = (pred >= config.confidence_threshold).astype(np.float32)

        # Stack: image, GT, pred
        img_grid = np.stack([pred, target, pred_bin], axis=0)  # (3, H, W)
        writer.add_image(
            f"sample_{i}/pred_gt_bin", img_grid, epoch, dataformats="CHW"
        )


if __name__ == "__main__":
    main()
