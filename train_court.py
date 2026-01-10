#!/usr/bin/env python3
"""
Court Detection Training Script
Trains TrackNet model with BCEWithLogitsLoss on sparse heatmap targets
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
import yaml
from datetime import datetime
import shutil

# Import project modules
from app.models.tracknet import TrackNet
from app.src.postprocess import court_postprocess
from app.src.datasets import CocoCourtDataset
from app.src.steps import safe_collate

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_config(config_path="configs/train.yaml"):
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_metrics(preds, targets):
    """
    Compute evaluation metrics for keypoint detection

    Args:
        preds: (B, K, 2) predicted keypoints
        targets: (B, K, 2) ground truth keypoints

    Returns:
        dict with metrics
    """
    # Euclidean distance per keypoint
    distances = torch.sqrt(((preds - targets) ** 2).sum(dim=-1))  # (B, K)

    mean_error = distances.mean().item()
    max_error = distances.max().item()

    # Percentage of keypoints within threshold
    pck_5 = (distances < 5).float().mean().item() * 100
    pck_10 = (distances < 10).float().mean().item() * 100

    return {
        "mean_error_px": mean_error,
        "max_error_px": max_error,
        "pck@5px": pck_5,
        "pck@10px": pck_10,
    }


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss,
                   metrics, config, checkpoint_path):
    """Save model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "metrics": metrics,
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"  💾 Saved: {checkpoint_path}")


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, config, epoch):
    """Train for one epoch using direct coordinate regression with gradient accumulation"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    # Gradient accumulation setup
    accumulation_steps = config["training"].get("accumulation_steps", 1)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Epoch {epoch+1} [Train]")

    for step, (imgs, target_heatmaps, keypoints, _) in pbar:
        imgs = imgs.to(device, dtype=torch.float32)
        target_heatmaps = target_heatmaps.to(device, dtype=torch.float32)
        keypoints = keypoints.to(device, dtype=torch.float32)

        # Forward: model outputs predicted heatmaps
        outputs = model(imgs)  # (B, 14, H, W)

        # HEATMAP LOSS - This is the correct approach!
        # MSE between predicted heatmaps and target Gaussian heatmaps
        loss = criterion(outputs, target_heatmaps) / accumulation_steps
        loss.backward()

        # Only step optimizer every accumulation_steps batches
        if (step + 1) % accumulation_steps == 0:
            if config["training"]["grad_clip"] > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["training"]["grad_clip"]
                )
                # Warning if gradients are exploding
                if grad_norm > 10.0:
                    print(f"\n⚠️  WARNING: Large gradient norm: {grad_norm:.2f}")
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps

        # For metrics: convert heatmaps to coordinates (not used in loss!)
        with torch.no_grad():
            preds = court_postprocess(outputs)  # (B, 14, 2)

            # Keypoints are already in output resolution (512x512)
            # No scaling needed when scale=1!
            all_preds.append(preds)
            all_targets.append(keypoints)

        pbar.set_postfix(loss=f"{loss.item() * accumulation_steps:.3f}")

    # Compute epoch metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_preds, all_targets)

    avg_loss = running_loss / len(dataloader)

    return avg_loss, metrics


@torch.no_grad()
def validate(model, dataloader, criterion, config, epoch):
    """Validation loop using heatmap-based loss"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")

    for imgs, target_heatmaps, keypoints, _ in pbar:
        imgs = imgs.to(device, dtype=torch.float32)
        target_heatmaps = target_heatmaps.to(device, dtype=torch.float32)
        keypoints = keypoints.to(device, dtype=torch.float32)

        # Forward: model outputs predicted heatmaps
        outputs = model(imgs)  # (B, 14, H, W)

        # HEATMAP LOSS
        loss = criterion(outputs, target_heatmaps)
        running_loss += loss.item()

        # For metrics: convert to coordinates
        preds = court_postprocess(outputs)  # (B, 14, 2)

        # Keypoints already in output resolution (512x512)
        # No scaling needed when scale=1!
        all_preds.append(preds)
        all_targets.append(keypoints)

        pbar.set_postfix(loss=f"{loss.item():.3f}")

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_preds, all_targets)

    avg_loss = running_loss / len(dataloader)

    return avg_loss, metrics


def train():
    """Main training function"""

    # Load config
    config = load_config("configs/train.yaml")

    # Create output directories
    checkpoint_dir = Path(config["checkpoint"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config["output"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("COURT DETECTION TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Initial LR: {config['training']['initial_lr']}")
    print(f"Mixed precision: {config['training']['mixed_precision']}")
    print(f"Target size: {config['data']['target_size']}")
    print("="*60)
    print()

    # Load dataset
    print("📊 Loading dataset...")
    # BUG FIX: Create separate datasets for train/val
    # Previously we created one dataset with train=True, then split it
    # This caused validation data to be augmented!

    # Load all data first to get indices for splitting
    temp_dataset = CocoCourtDataset(
        ann_file=config["data"]["ann_file"],
        img_dir=config["data"]["img_dir"],
        target_size=tuple(config["data"]["target_size"]),
        train=False,  # No augmentation for splitting
    )

    # Get train/val indices
    total_size = len(temp_dataset)
    train_size = int(config["data"]["train_split"] * total_size)
    val_size = total_size - train_size

    indices = list(range(total_size))
    torch.manual_seed(config["system"]["seed"])
    torch.Generator().manual_seed(config["system"]["seed"])
    import random as pyrandom
    pyrandom.seed(config["system"]["seed"])
    pyrandom.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create separate datasets with different transforms
    train_dataset_full = CocoCourtDataset(
        ann_file=config["data"]["ann_file"],
        img_dir=config["data"]["img_dir"],
        target_size=tuple(config["data"]["target_size"]),
        train=True,  # Augmentation enabled (currently disabled in transform.py)
    )

    val_dataset_full = CocoCourtDataset(
        ann_file=config["data"]["ann_file"],
        img_dir=config["data"]["img_dir"],
        target_size=tuple(config["data"]["target_size"]),
        train=False,  # NO augmentation for validation
    )

    # Create subset datasets
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    print(f"  Total: {total_size}")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["system"]["num_workers"],
        collate_fn=safe_collate,
        pin_memory=config["system"]["pin_memory"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["system"]["num_workers"],
        collate_fn=safe_collate,
        pin_memory=config["system"]["pin_memory"],
    )

    # Initialize model
    print("🏗️  Initializing model...")
    model = TrackNet(out_channels=config["model"]["out_channels"]).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print()

    # Loss, optimizer, scheduler
    print("⚙️  Setting up training...")
    if config["loss"]["type"] == "mse":
        criterion = nn.MSELoss()
    elif config["loss"]["type"] == "smoothl1":
        criterion = nn.SmoothL1Loss(beta=config["loss"]["beta"])
    else:
        raise ValueError(f"Unknown loss type: {config['loss']['type']}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["initial_lr"],
        betas=config["optimizer"]["betas"],
        eps=config["optimizer"]["eps"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Scheduler with warmup
    warmup_epochs = config["training"].get("warmup_epochs", 0)
    total_epochs = config["training"]["num_epochs"]

    if warmup_epochs > 0:
        # Warmup: gradually increase LR from 0.1x to 1.0x
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        # Decay: decrease LR from 1.0x to 0.1x
        decay_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_epochs - warmup_epochs,
        )
        # Combine warmup + decay
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_epochs]
        )
        print(f"  Scheduler: Warmup ({warmup_epochs} epochs) + Linear Decay")
    else:
        # No warmup, just linear decay
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_epochs,
        )
        print(f"  Scheduler: LinearLR (1.0 -> 0.1)")

    scaler = GradScaler() if config["training"]["mixed_precision"] else None

    print(f"  Loss: {config['loss']['type'].upper()}")
    print(f"  Optimizer: AdamW")
    print()

    # Training loop
    print("🚀 Starting training...")
    print()

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    training_history = []

    for epoch in range(config["training"]["num_epochs"]):
        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, config, epoch
        )

        # Validate
        if (epoch + 1) % config["validation"]["val_interval"] == 0:
            val_loss, val_metrics = validate(
                model, val_loader, criterion, config, epoch
            )
        else:
            val_loss, val_metrics = None, None

        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']

        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch+1}/{config['training']['num_epochs']}]")
        print(f"{'='*60}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"  Mean Error: {train_metrics['mean_error_px']:.2f} px")
        print(f"  PCK@5px: {train_metrics['pck@5px']:.1f}%")
        print(f"  PCK@10px: {train_metrics['pck@10px']:.1f}%")

        if val_loss is not None:
            print(f"\nVal Loss: {val_loss:.4f}")
            print(f"  Mean Error: {val_metrics['mean_error_px']:.2f} px")
            print(f"  PCK@5px: {val_metrics['pck@5px']:.1f}%")
            print(f"  PCK@10px: {val_metrics['pck@10px']:.1f}%")

        print(f"\nLR: {current_lr:.6f} → {new_lr:.6f}")
        print(f"{'='*60}\n")

        # Save training history
        training_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_metrics": train_metrics,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "lr": new_lr,
        })

        # Save checkpoints
        if val_loss is not None:
            # Check if best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0

                best_path = checkpoint_dir / "best_model.pt"
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1,
                    train_loss, val_loss, val_metrics, config, best_path
                )
                print(f"✨ New best model! Val Loss: {val_loss:.4f}\n")
            else:
                epochs_without_improvement += 1

        # Periodic checkpoint (every 10 epochs instead of 5)
        if (epoch + 1) % config["checkpoint"]["save_interval"] == 0:
            periodic_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1,
                train_loss, val_loss, val_metrics if val_loss else train_metrics,
                config, periodic_path
            )

        # CRITICAL: Check for gradient explosion / training collapse
        if val_metrics and val_metrics['mean_error_px'] > 100:
            print(f"\n🛑 STOPPING TRAINING - Gradient explosion detected!")
            print(f"   Val Mean Error: {val_metrics['mean_error_px']:.2f}px (threshold: 100px)")
            print(f"   Restoring best model from epoch with Val Loss: {best_val_loss:.4f}")
            # Load best model
            best_checkpoint = torch.load(checkpoint_dir / "best_model.pt")
            model.load_state_dict(best_checkpoint['model_state_dict'])
            print(f"   ✓ Best model restored")
            break

        # Early stopping (no improvement)
        if epochs_without_improvement >= config["validation"]["early_stopping_patience"]:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            print(f"   No improvement for {epochs_without_improvement} epochs")
            break

    # Save final model
    if config["checkpoint"]["save_final"]:
        final_path = checkpoint_dir / "final_model.pt"
        save_checkpoint(
            model, optimizer, scheduler, epoch + 1,
            train_loss, val_loss if val_loss else train_loss,
            val_metrics if val_metrics else train_metrics,
            config, final_path
        )

    # Save training history
    history_path = Path(config["output"]["log_file"])
    with open(history_path, 'w') as f:
        json.dump({
            "config": config,
            "history": training_history,
            "best_val_loss": best_val_loss,
        }, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    print(f"   Training history: {history_path}")

    # Copy best model to standard location
    best_model_path = Path("models/court_model_best.pt")
    best_model_path.parent.mkdir(exist_ok=True)
    shutil.copy2(checkpoint_dir / "best_model.pt", best_model_path)
    print(f"   Best model: {best_model_path}")


if __name__ == "__main__":
    train()
