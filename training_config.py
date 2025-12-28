#!/usr/bin/env python3
"""
Tennis Computer Vision - TrackNet Training Configuration
File: training_config.py

Configuration settings for TrackNet ball detection training
"""
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple


class TrainingConfig:
    """
    Configuration class for TrackNet training
    """
    
    def __init__(self, config_name: str = "default"):
        """
        Initialize training configuration
        
        Args:
            config_name: Name of the configuration preset
        """
        self.config_name = config_name
        self.configs = self._get_config_presets()
        self.config = self.configs[config_name]
    
    def _get_config_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration presets"""
        return {
            "default": {
                # Model settings
                "model": {
                    "architecture": "BallTrackerNet",
                    "out_channels": 1,  # 1 channel for ball detection
                    "input_size": (512, 512),
                    "heatmap_size": (128, 128)
                },
                
                # Training settings
                "training": {
                    "epochs": 100,
                    "batch_size": 8,
                    "learning_rate": 1e-4,
                    "weight_decay": 1e-4,
                    "patience": 10,
                    "save_interval": 10,
                    "gradient_clip": 1.0
                },
                
                # Data settings
                "data": {
                    "image_size": (512, 512),
                    "heatmap_size": (128, 128),
                    "num_workers": 4,
                    "pin_memory": True,
                    "augmentation": True
                },
                
                # Augmentation settings
                "augmentation": {
                    "horizontal_flip": 0.5,
                    "brightness_contrast": 0.5,
                    "gamma": 0.3,
                    "gaussian_noise": 0.3,
                    "blur": 0.2,
                    "rotation": 0.2,
                    "scale": 0.2
                },
                
                # Loss settings
                "loss": {
                    "type": "mse",
                    "focal_loss_alpha": 0.25,
                    "focal_loss_gamma": 2.0,
                    "heatmap_weight": 1.0,
                    "bbox_weight": 0.1
                },
                
                # Optimizer settings
                "optimizer": {
                    "type": "adam",
                    "lr": 1e-4,
                    "weight_decay": 1e-4,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8
                },
                
                # Scheduler settings
                "scheduler": {
                    "type": "reduce_on_plateau",
                    "factor": 0.5,
                    "patience": 10,
                    "min_lr": 1e-6,
                    "mode": "min"
                },
                
                # Paths
                "paths": {
                    "data_dir": "data/tennis_ball_dataset",
                    "train_annotations": "data/tennis_ball_dataset/annotations/annotations_train.json",
                    "val_annotations": "data/tennis_ball_dataset/annotations/annotations_val.json",
                    "output_dir": "outputs/tracknet_ball_training",
                    "checkpoint_dir": "checkpoints",
                    "log_dir": "logs"
                },
                
                # Logging
                "logging": {
                    "log_interval": 10,
                    "save_images": True,
                    "save_predictions": True,
                    "tensorboard": True
                }
            },
            
            "fast": {
                # Fast training for testing
                "model": {
                    "architecture": "BallTrackerNet",
                    "out_channels": 1,
                    "input_size": (256, 256),
                    "heatmap_size": (64, 64)
                },
                
                "training": {
                    "epochs": 10,
                    "batch_size": 16,
                    "learning_rate": 1e-3,
                    "weight_decay": 1e-4,
                    "patience": 5,
                    "save_interval": 5,
                    "gradient_clip": 1.0
                },
                
                "data": {
                    "image_size": (256, 256),
                    "heatmap_size": (64, 64),
                    "num_workers": 2,
                    "pin_memory": True,
                    "augmentation": True
                },
                
                "augmentation": {
                    "horizontal_flip": 0.5,
                    "brightness_contrast": 0.3,
                    "gamma": 0.2,
                    "gaussian_noise": 0.2,
                    "blur": 0.1,
                    "rotation": 0.1,
                    "scale": 0.1
                },
                
                "loss": {
                    "type": "mse",
                    "heatmap_weight": 1.0
                },
                
                "optimizer": {
                    "type": "adam",
                    "lr": 1e-3,
                    "weight_decay": 1e-4
                },
                
                "scheduler": {
                    "type": "reduce_on_plateau",
                    "factor": 0.5,
                    "patience": 5,
                    "min_lr": 1e-5
                },
                
                "paths": {
                    "data_dir": "data/tennis_ball_dataset",
                    "train_annotations": "data/tennis_ball_dataset/annotations/annotations_train.json",
                    "val_annotations": "data/tennis_ball_dataset/annotations/annotations_val.json",
                    "output_dir": "outputs/tracknet_ball_fast",
                    "checkpoint_dir": "checkpoints",
                    "log_dir": "logs"
                },
                
                "logging": {
                    "log_interval": 5,
                    "save_images": True,
                    "save_predictions": False,
                    "tensorboard": False
                }
            },
            
            "high_res": {
                # High resolution training
                "model": {
                    "architecture": "BallTrackerNet",
                    "out_channels": 1,
                    "input_size": (1024, 1024),
                    "heatmap_size": (256, 256)
                },
                
                "training": {
                    "epochs": 200,
                    "batch_size": 4,
                    "learning_rate": 5e-5,
                    "weight_decay": 1e-4,
                    "patience": 20,
                    "save_interval": 20,
                    "gradient_clip": 1.0
                },
                
                "data": {
                    "image_size": (1024, 1024),
                    "heatmap_size": (256, 256),
                    "num_workers": 8,
                    "pin_memory": True,
                    "augmentation": True
                },
                
                "augmentation": {
                    "horizontal_flip": 0.5,
                    "brightness_contrast": 0.4,
                    "gamma": 0.3,
                    "gaussian_noise": 0.3,
                    "blur": 0.2,
                    "rotation": 0.3,
                    "scale": 0.3
                },
                
                "loss": {
                    "type": "mse",
                    "heatmap_weight": 1.0
                },
                
                "optimizer": {
                    "type": "adam",
                    "lr": 5e-5,
                    "weight_decay": 1e-4
                },
                
                "scheduler": {
                    "type": "reduce_on_plateau",
                    "factor": 0.5,
                    "patience": 20,
                    "min_lr": 1e-7
                },
                
                "paths": {
                    "data_dir": "data/tennis_ball_dataset",
                    "train_annotations": "data/tennis_ball_dataset/annotations/annotations_train.json",
                    "val_annotations": "data/tennis_ball_dataset/annotations/annotations_val.json",
                    "output_dir": "outputs/tracknet_ball_high_res",
                    "checkpoint_dir": "checkpoints",
                    "log_dir": "logs"
                },
                
                "logging": {
                    "log_interval": 20,
                    "save_images": True,
                    "save_predictions": True,
                    "tensorboard": True
                }
            }
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values
        
        Args:
            updates: Dictionary with configuration updates
        """
        def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = deep_update(self.config, updates)
    
    def save_config(self, file_path: str) -> None:
        """
        Save configuration to JSON file
        
        Args:
            file_path: Path to save configuration
        """
        import json
        
        config_to_save = {
            "config_name": self.config_name,
            "config": self.config
        }
        
        with open(file_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        print(f"Configuration saved to: {file_path}")
    
    def load_config(self, file_path: str) -> None:
        """
        Load configuration from JSON file
        
        Args:
            file_path: Path to configuration file
        """
        import json
        
        with open(file_path, 'r') as f:
            loaded_config = json.load(f)
        
        self.config_name = loaded_config["config_name"]
        self.config = loaded_config["config"]
        
        print(f"Configuration loaded from: {file_path}")
    
    def print_config(self) -> None:
        """Print current configuration"""
        print(f"Configuration: {self.config_name}")
        print("=" * 50)
        
        def print_dict(d: Dict[str, Any], indent: int = 0) -> None:
            for key, value in d.items():
                if isinstance(value, dict):
                    print("  " * indent + f"{key}:")
                    print_dict(value, indent + 1)
                else:
                    print("  " * indent + f"{key}: {value}")
        
        print_dict(self.config)
    
    def create_directories(self) -> None:
        """Create necessary directories for training"""
        paths = self.config["paths"]
        
        directories = [
            paths["output_dir"],
            paths["checkpoint_dir"],
            paths["log_dir"],
            paths["data_dir"]
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        print("Training directories created:")
        for directory in directories:
            print(f"  - {directory}")
    
    def validate_config(self) -> bool:
        """
        Validate configuration
        
        Returns:
            True if configuration is valid
        """
        required_sections = ["model", "training", "data", "paths"]
        
        for section in required_sections:
            if section not in self.config:
                print(f"Missing required section: {section}")
                return False
        
        # Check required paths
        paths = self.config["paths"]
        required_paths = ["data_dir", "train_annotations", "val_annotations", "output_dir"]
        
        for path_key in required_paths:
            if path_key not in paths:
                print(f"Missing required path: {path_key}")
                return False
        
        print("Configuration validation passed!")
        return True


def main():
    """Main function for configuration management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TrackNet Training Configuration Manager")
    parser.add_argument("--config", default="default", help="Configuration preset")
    parser.add_argument("--print", action="store_true", help="Print configuration")
    parser.add_argument("--save", help="Save configuration to file")
    parser.add_argument("--load", help="Load configuration from file")
    parser.add_argument("--validate", action="store_true", help="Validate configuration")
    parser.add_argument("--create_dirs", action="store_true", help="Create training directories")
    
    args = parser.parse_args()
    
    config = TrainingConfig(args.config)
    
    if args.print:
        config.print_config()
    
    elif args.save:
        config.save_config(args.save)
    
    elif args.load:
        config.load_config(args.load)
    
    elif args.validate:
        config.validate_config()
    
    elif args.create_dirs:
        config.create_directories()
    
    else:
        print("No action specified. Use --help for available options.")
        print(f"Available presets: {list(config.configs.keys())}")
    
    return 0


if __name__ == "__main__":
    exit(main())


