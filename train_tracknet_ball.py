#!/usr/bin/env python3
"""
Tennis Computer Vision - TrackNet Ball Detection Training
File: train_tracknet_ball.py

Train TrackNet model for ball detection using COCO-style dataset
"""
import sys
import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from app.models.tracknet import BallTrackerNet


class TennisBallCOCODataset(Dataset):
    """
    COCO-style dataset for tennis ball detection training
    """
    
    def __init__(self, 
                 data_dir: str,
                 annotation_file: str,
                 image_size: Tuple[int, int] = (512, 512),
                 heatmap_size: Tuple[int, int] = (128, 128),
                 transform: Optional[A.Compose] = None,
                 is_training: bool = True):
        """
        Initialize COCO-style tennis ball dataset
        
        Args:
            data_dir: Directory containing images
            annotation_file: Path to COCO annotation JSON file
            image_size: Input image size (height, width)
            heatmap_size: Output heatmap size (height, width)
            transform: Albumentations transforms
            is_training: Whether this is training dataset
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.transform = transform
        self.is_training = is_training
        
        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create mappings
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = {ann['id']: ann for ann in self.coco_data['annotations']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # Filter for tennis ball category (assuming category_id = 1 for tennis ball)
        self.ball_category_id = 1
        
        # Create image-annotation mapping
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            if ann['category_id'] == self.ball_category_id:
                image_id = ann['image_id']
                if image_id not in self.image_annotations:
                    self.image_annotations[image_id] = []
                self.image_annotations[image_id].append(ann)
        
        # Get valid image IDs (images with ball annotations)
        self.valid_image_ids = list(self.image_annotations.keys())
        
        print(f"Dataset loaded: {len(self.valid_image_ids)} images with ball annotations")
        print(f"Categories: {[cat['name'] for cat in self.coco_data['categories']]}")
    
    def __len__(self):
        return len(self.valid_image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get dataset item
        
        Returns:
            Dictionary with 'image', 'heatmap', 'bbox', 'keypoints'
        """
        image_id = self.valid_image_ids[idx]
        image_info = self.images[image_id]
        annotations = self.image_annotations[image_id]
        
        # Load image
        image_path = self.data_dir / image_info['file_name']
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get ball annotations
        ball_bboxes = []
        ball_keypoints = []
        
        for ann in annotations:
            # Bounding box [x, y, width, height]
            bbox = ann['bbox']
            ball_bboxes.append(bbox)
            
            # Keypoints (if available) - center of ball
            if 'keypoints' in ann and ann['keypoints']:
                # COCO keypoints format: [x1, y1, v1, x2, y2, v2, ...]
                # For ball detection, we typically use center point
                keypoints = ann['keypoints']
                if len(keypoints) >= 3:
                    center_x = keypoints[0]
                    center_y = keypoints[1]
                    visibility = keypoints[2]
                    ball_keypoints.append([center_x, center_y, visibility])
            else:
                # Calculate center from bounding box
                center_x = bbox[0] + bbox[2] / 2
                center_y = bbox[1] + bbox[3] / 2
                ball_keypoints.append([center_x, center_y, 1.0])
        
        # Create heatmap
        heatmap = self.create_heatmap(image.shape[:2], ball_keypoints)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(
                image=image,
                heatmap=heatmap,
                bboxes=ball_bboxes,
                keypoints=ball_keypoints
            )
            image = transformed['image']
            heatmap = transformed['heatmap']
            ball_bboxes = transformed['bboxes']
            ball_keypoints = transformed['keypoints']
        
        # Convert to tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if not isinstance(heatmap, torch.Tensor):
            heatmap = torch.from_numpy(heatmap).float()
        
        return {
            'image': image,
            'heatmap': heatmap,
            'bboxes': ball_bboxes,
            'keypoints': ball_keypoints,
            'image_id': image_id,
            'image_info': image_info
        }
    
    def create_heatmap(self, image_shape: Tuple[int, int], keypoints: List[List[float]]) -> np.ndarray:
        """
        Create Gaussian heatmap for ball keypoints
        
        Args:
            image_shape: (height, width) of original image
            keypoints: List of [x, y, visibility] keypoints
            
        Returns:
            Heatmap array of shape (1, heatmap_height, heatmap_width)
        """
        heatmap_height, heatmap_width = self.heatmap_size
        heatmap = np.zeros((1, heatmap_height, heatmap_width), dtype=np.float32)
        
        # Scale factor from original image to heatmap
        scale_x = heatmap_width / image_shape[1]
        scale_y = heatmap_height / image_shape[0]
        
        for keypoint in keypoints:
            x, y, visibility = keypoint
            
            if visibility > 0:  # Only process visible keypoints
                # Scale coordinates
                heatmap_x = int(x * scale_x)
                heatmap_y = int(y * scale_y)
                
                # Ensure coordinates are within bounds
                heatmap_x = max(0, min(heatmap_width - 1, heatmap_x))
                heatmap_y = max(0, min(heatmap_height - 1, heatmap_y))
                
                # Create Gaussian kernel
                sigma = 2.0
                kernel_size = int(6 * sigma) + 1
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                # Create Gaussian kernel
                kernel = self.create_gaussian_kernel(kernel_size, sigma)
                
                # Place kernel on heatmap
                half_kernel = kernel_size // 2
                y_start = max(0, heatmap_y - half_kernel)
                y_end = min(heatmap_height, heatmap_y + half_kernel + 1)
                x_start = max(0, heatmap_x - half_kernel)
                x_end = min(heatmap_width, heatmap_x + half_kernel + 1)
                
                kernel_y_start = max(0, half_kernel - heatmap_y)
                kernel_y_end = kernel_y_start + (y_end - y_start)
                kernel_x_start = max(0, half_kernel - heatmap_x)
                kernel_x_end = kernel_x_start + (x_end - x_start)
                
                heatmap[0, y_start:y_end, x_start:x_end] = np.maximum(
                    heatmap[0, y_start:y_end, x_start:x_end],
                    kernel[kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end] * visibility
                )
        
        return heatmap
    
    def create_gaussian_kernel(self, kernel_size: int, sigma: float) -> np.ndarray:
        """Create 2D Gaussian kernel"""
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x*x + y*y) / (2 * sigma*sigma))
        
        # Normalize
        kernel = kernel / np.max(kernel)
        return kernel


class TrackNetBallTrainer:
    """
    TrackNet trainer for ball detection
    """
    
    def __init__(self, 
                 model: BallTrackerNet,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 config: Dict[str, Any]):
        """
        Initialize trainer
        
        Args:
            model: TrackNet model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Training device
            config: Training configuration
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Loss function - MSE for heatmap regression
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config['patience'],
            verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            images = batch['image'].to(self.device)
            heatmaps = batch['heatmap'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.criterion(outputs, heatmaps)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch in pbar:
                images = batch['image'].to(self.device)
                heatmaps = batch['heatmap'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss = self.criterion(outputs, heatmaps)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, num_epochs: int, save_dir: str) -> None:
        """
        Train the model
        
        Args:
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(current_lr)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': self.config
                }
                
                torch.save(checkpoint, save_dir / 'best_model.pth')
                print(f"✓ New best model saved (val_loss: {val_loss:.4f})")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': self.config
                }
                torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        # Save final model
        torch.save(self.best_model_state, save_dir / 'final_model.pth')
        
        # Plot training history
        self.plot_training_history(save_dir)
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Models saved to: {save_dir}")
    
    def plot_training_history(self, save_dir: Path) -> None:
        """Plot and save training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Learning rate plot
        ax2.plot(self.learning_rates, label='Learning Rate', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()


def create_data_transforms(image_size: Tuple[int, int], is_training: bool = True) -> A.Compose:
    """
    Create data augmentation transforms
    
    Args:
        image_size: Target image size (height, width)
        is_training: Whether to apply training augmentations
        
    Returns:
        Albumentations compose transform
    """
    if is_training:
        transforms_list = [
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ]
    else:
        transforms_list = [
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ]
    
    return A.Compose(
        transforms_list,
        additional_targets={'heatmap': 'mask'}
    )


def create_coco_dataset_structure(data_dir: str, output_dir: str) -> None:
    """
    Create COCO-style dataset structure for tennis ball detection
    
    Args:
        data_dir: Directory containing tennis images
        output_dir: Output directory for COCO dataset
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Create directory structure
    (output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_dir / 'annotations').mkdir(parents=True, exist_ok=True)
    
    print(f"Creating COCO dataset structure in: {output_dir}")
    print("Directory structure:")
    print("  - images/train/")
    print("  - images/val/")
    print("  - annotations/")
    print("\nNext steps:")
    print("1. Place tennis images in images/train/ and images/val/")
    print("2. Create annotations_train.json and annotations_val.json")
    print("3. Use annotation tools like labelme or CVAT")
    print("4. Run training script")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train TrackNet for ball detection")
    parser.add_argument("--data_dir", required=True, help="Dataset directory")
    parser.add_argument("--train_annotations", required=True, help="Training annotations JSON")
    parser.add_argument("--val_annotations", required=True, help="Validation annotations JSON")
    parser.add_argument("--output_dir", default="outputs/tracknet_ball", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 512], help="Image size")
    parser.add_argument("--heatmap_size", type=int, nargs=2, default=[128, 128], help="Heatmap size")
    parser.add_argument("--create_dataset", action="store_true", help="Create dataset structure")
    
    args = parser.parse_args()
    
    if args.create_dataset:
        create_coco_dataset_structure(args.data_dir, args.output_dir)
        return 0
    
    # Training configuration
    config = {
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-4,
        'patience': 10,
        'save_interval': 10,
        'image_size': tuple(args.image_size),
        'heatmap_size': tuple(args.heatmap_size),
        'batch_size': args.batch_size
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("Loading datasets...")
    train_transform = create_data_transforms(config['image_size'], is_training=True)
    val_transform = create_data_transforms(config['image_size'], is_training=False)
    
    train_dataset = TennisBallCOCODataset(
        data_dir=args.data_dir,
        annotation_file=args.train_annotations,
        image_size=config['image_size'],
        heatmap_size=config['heatmap_size'],
        transform=train_transform,
        is_training=True
    )
    
    val_dataset = TennisBallCOCODataset(
        data_dir=args.data_dir,
        annotation_file=args.val_annotations,
        image_size=config['image_size'],
        heatmap_size=config['heatmap_size'],
        transform=val_transform,
        is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Create model
    model = BallTrackerNet(out_channels=1)  # 1 channel for ball detection
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer = TrackNetBallTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    # Train
    trainer.train(num_epochs=args.epochs, save_dir=args.output_dir)
    
    print("Training completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())


