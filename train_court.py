#!/usr/bin/env python3
"""
Court Detection Training Script
Optimized for RTX 3070 (8GB) with batch_size=4
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Epoch {epoch+1} [Train]")

    for step, (imgs, keypoints, _) in pbar:
        imgs = imgs.to(device, dtype=torch.float32)
        keypoints = keypoints.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        # Mixed precision training
        if config["training"]["mixed_precision"]:
            with autocast():
                outputs = model(imgs)
                preds = court_postprocess(outputs)

                # Scale GT to output resolution
                H_out, W_out = outputs.shape[2:]
                H_in, W_in = imgs.shape[2:]
                scale_x, scale_y = W_out / W_in, H_out / H_in
                target_scaled = keypoints.clone()
                target_scaled[..., 0] *= scale_x
                target_scaled[..., 1] *= scale_y

                loss = criterion(preds, target_scaled)

            scaler.scale(loss).backward()

            if config["training"]["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["training"]["grad_clip"]
                )

            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            preds = court_postprocess(outputs)

            H_out, W_out = outputs.shape[2:]
            H_in, W_in = imgs.shape[2:]
            scale_x, scale_y = W_out / W_in, H_out / H_in
            target_scaled = keypoints.clone()
            target_scaled[..., 0] *= scale_x
            target_scaled[..., 1] *= scale_y

            loss = criterion(preds, target_scaled)
            loss.backward()

            if config["training"]["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["training"]["grad_clip"]
                )

            optimizer.step()

        running_loss += loss.item()
        all_preds.append(preds.detach())
        all_targets.append(target_scaled.detach())

        pbar.set_postfix(loss=f"{loss.item():.3f}")

    # Compute epoch metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_preds, all_targets)

    avg_loss = running_loss / len(dataloader)

    return avg_loss, metrics


@torch.no_grad()
def validate(model, dataloader, criterion, config, epoch):
    """Validation loop"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")

    for imgs, keypoints, _ in pbar:
        imgs = imgs.to(device, dtype=torch.float32)
        keypoints = keypoints.to(device, dtype=torch.float32)

        outputs = model(imgs)
        preds = court_postprocess(outputs)

        H_out, W_out = outputs.shape[2:]
        H_in, W_in = imgs.shape[2:]
        scale_x, scale_y = W_out / W_in, H_out / H_in
        target_scaled = keypoints.clone()
        target_scaled[..., 0] *= scale_x
        target_scaled[..., 1] *= scale_y

        loss = criterion(preds, target_scaled)
        running_loss += loss.item()

        all_preds.append(preds)
        all_targets.append(target_scaled)

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
    full_dataset = CocoCourtDataset(
        ann_file=config["data"]["ann_file"],
        img_dir=config["data"]["img_dir"],
        target_size=tuple(config["data"]["target_size"]),
        train=True,
    )

    # Split into train/val
    train_size = int(config["data"]["train_split"] * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config["system"]["seed"])
    )

    print(f"  Total: {len(full_dataset)}")
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
    criterion = nn.SmoothL1Loss(beta=config["loss"]["beta"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["initial_lr"],
        betas=config["optimizer"]["betas"],
        eps=config["optimizer"]["eps"],
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config["scheduler"]["T_0"],
        T_mult=config["scheduler"]["T_mult"],
        eta_min=config["scheduler"]["eta_min"],
    )

    scaler = GradScaler() if config["training"]["mixed_precision"] else None

    print(f"  Loss: SmoothL1Loss")
    print(f"  Optimizer: AdamW")
    print(f"  Scheduler: CosineAnnealingWarmRestarts")
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

                best_path = checkpoint_dir / "best_model.pth"
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1,
                    train_loss, val_loss, val_metrics, config, best_path
                )
                print(f"✨ New best model! Val Loss: {val_loss:.4f}\n")
            else:
                epochs_without_improvement += 1

        # Periodic checkpoint (every 10 epochs instead of 5)
        if (epoch + 1) % config["checkpoint"]["save_interval"] == 0:
            periodic_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1,
                train_loss, val_loss, val_metrics if val_loss else train_metrics,
                config, periodic_path
            )

        # Early stopping
        if epochs_without_improvement >= config["validation"]["early_stopping_patience"]:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            print(f"   No improvement for {epochs_without_improvement} epochs")
            break

    # Save final model
    if config["checkpoint"]["save_final"]:
        final_path = checkpoint_dir / "final_model.pth"
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
    shutil.copy2(checkpoint_dir / "best_model.pth", best_model_path)
    print(f"   Best model: {best_model_path}")


if __name__ == "__main__":
    train()
