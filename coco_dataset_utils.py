#!/usr/bin/env python3
"""
Tennis Computer Vision - COCO Dataset Utilities
File: coco_dataset_utils.py

Utilities for creating and managing COCO-style datasets for tennis ball detection
"""
import sys
import os
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import shutil


class COCODatasetManager:
    """
    Manager for COCO-style tennis ball detection datasets
    """
    
    def __init__(self, dataset_dir: str):
        """
        Initialize COCO dataset manager
        
        Args:
            dataset_dir: Root directory of the dataset
        """
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.annotations_dir = self.dataset_dir / "annotations"
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # COCO format structure
        self.coco_format = {
            "info": {
                "description": "Tennis Ball Detection Dataset",
                "version": "1.0",
                "year": 2024,
                "contributor": "Tennis Computer Vision",
                "date_created": "2024-01-01"
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "MIT License",
                    "url": "https://opensource.org/licenses/MIT"
                }
            ],
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": "tennis_ball",
                    "supercategory": "ball"
                }
            ]
        }
    
    def create_dataset_structure(self, split_ratio: float = 0.8) -> None:
        """
        Create COCO dataset directory structure
        
        Args:
            split_ratio: Ratio of training data (0.8 = 80% train, 20% val)
        """
        # Create split directories
        (self.images_dir / "train").mkdir(exist_ok=True)
        (self.images_dir / "val").mkdir(exist_ok=True)
        
        print(f"Created dataset structure in: {self.dataset_dir}")
        print("Directory structure:")
        print("  - images/train/")
        print("  - images/val/")
        print("  - annotations/")
        print(f"\nSplit ratio: {split_ratio:.1%} train, {1-split_ratio:.1%} validation")
    
    def add_image(self, image_path: str, image_id: int, split: str = "train") -> Dict[str, Any]:
        """
        Add image to COCO format
        
        Args:
            image_path: Path to image file
            split: Dataset split ("train" or "val")
            
        Returns:
            Image info dictionary
        """
        image_path = Path(image_path)
        
        # Copy image to appropriate directory
        dest_dir = self.images_dir / split
        dest_path = dest_dir / image_path.name
        
        if not dest_path.exists():
            shutil.copy2(image_path, dest_path)
        
        # Get image dimensions
        with Image.open(image_path) as img:
            width, height = img.size
        
        # Create image info
        image_info = {
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
            "date_captured": "2024-01-01T00:00:00+00:00",
            "license": 1,
            "coco_url": "",
            "flickr_url": ""
        }
        
        return image_info
    
    def add_annotation(self, 
                       annotation_id: int,
                       image_id: int,
                       bbox: List[float],
                       keypoints: Optional[List[float]] = None,
                       area: Optional[float] = None) -> Dict[str, Any]:
        """
        Add annotation to COCO format
        
        Args:
            annotation_id: Unique annotation ID
            image_id: ID of the image
            bbox: Bounding box [x, y, width, height]
            keypoints: Keypoints [x1, y1, v1, x2, y2, v2, ...]
            area: Area of the bounding box
            
        Returns:
            Annotation dictionary
        """
        if area is None:
            area = bbox[2] * bbox[3]  # width * height
        
        if keypoints is None:
            # Default keypoint: center of bounding box
            center_x = bbox[0] + bbox[2] / 2
            center_y = bbox[1] + bbox[3] / 2
            keypoints = [center_x, center_y, 2]  # 2 = visible
            num_keypoints = 1
        else:
            num_keypoints = len(keypoints) // 3
        
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,  # tennis_ball
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
            "keypoints": keypoints,
            "num_keypoints": num_keypoints
        }
        
        return annotation
    
    def save_annotations(self, split: str, annotations_file: str = None) -> None:
        """
        Save annotations to JSON file
        
        Args:
            split: Dataset split ("train" or "val")
            annotations_file: Output annotations file path
        """
        if annotations_file is None:
            annotations_file = self.annotations_dir / f"annotations_{split}.json"
        
        # Filter annotations for this split
        split_images = [img for img in self.coco_format["images"] 
                       if img["file_name"].startswith(split) or split in str(img["file_name"])]
        split_image_ids = {img["id"] for img in split_images}
        
        split_annotations = [ann for ann in self.coco_format["annotations"] 
                           if ann["image_id"] in split_image_ids]
        
        # Create split-specific COCO format
        split_coco = {
            "info": self.coco_format["info"],
            "licenses": self.coco_format["licenses"],
            "images": split_images,
            "annotations": split_annotations,
            "categories": self.coco_format["categories"]
        }
        
        # Save to JSON
        with open(annotations_file, 'w') as f:
            json.dump(split_coco, f, indent=2)
        
        print(f"Saved {len(split_images)} images and {len(split_annotations)} annotations to {annotations_file}")
    
    def load_annotations(self, annotations_file: str) -> None:
        """
        Load annotations from JSON file
        
        Args:
            annotations_file: Path to annotations JSON file
        """
        with open(annotations_file, 'r') as f:
            loaded_data = json.load(f)
        
        # Merge with existing data
        self.coco_format["images"].extend(loaded_data["images"])
        self.coco_format["annotations"].extend(loaded_data["annotations"])
        
        print(f"Loaded {len(loaded_data['images'])} images and {len(loaded_data['annotations'])} annotations")
    
    def visualize_annotations(self, 
                             image_id: int, 
                             output_path: str = None,
                             show_keypoints: bool = True,
                             show_bboxes: bool = True) -> None:
        """
        Visualize annotations for an image
        
        Args:
            image_id: ID of the image to visualize
            output_path: Path to save visualization
            show_keypoints: Whether to show keypoints
            show_bboxes: Whether to show bounding boxes
        """
        # Find image info
        image_info = None
        for img in self.coco_format["images"]:
            if img["id"] == image_id:
                image_info = img
                break
        
        if image_info is None:
            print(f"Image with ID {image_id} not found")
            return
        
        # Load image
        image_path = self.images_dir / image_info["file_name"]
        if not image_path.exists():
            print(f"Image file not found: {image_path}")
            return
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find annotations for this image
        annotations = [ann for ann in self.coco_format["annotations"] 
                     if ann["image_id"] == image_id]
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        for ann in annotations:
            bbox = ann["bbox"]
            keypoints = ann["keypoints"]
            
            # Draw bounding box
            if show_bboxes:
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2], bbox[3],
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
            
            # Draw keypoints
            if show_keypoints and keypoints:
                for i in range(0, len(keypoints), 3):
                    x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
                    if v > 0:  # Visible keypoint
                        color = 'green' if v == 2 else 'yellow'
                        ax.plot(x, y, 'o', color=color, markersize=8)
        
        ax.set_title(f"Image ID: {image_id} - {len(annotations)} annotations")
        ax.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get dataset statistics
        
        Returns:
            Dictionary with dataset statistics
        """
        total_images = len(self.coco_format["images"])
        total_annotations = len(self.coco_format["annotations"])
        
        # Count annotations per image
        annotations_per_image = {}
        for ann in self.coco_format["annotations"]:
            img_id = ann["image_id"]
            annotations_per_image[img_id] = annotations_per_image.get(img_id, 0) + 1
        
        # Calculate statistics
        avg_annotations_per_image = np.mean(list(annotations_per_image.values()))
        max_annotations_per_image = max(annotations_per_image.values()) if annotations_per_image else 0
        
        # Bounding box statistics
        bbox_areas = [ann["area"] for ann in self.coco_format["annotations"]]
        avg_bbox_area = np.mean(bbox_areas) if bbox_areas else 0
        
        stats = {
            "total_images": total_images,
            "total_annotations": total_annotations,
            "avg_annotations_per_image": avg_annotations_per_image,
            "max_annotations_per_image": max_annotations_per_image,
            "avg_bbox_area": avg_bbox_area,
            "categories": len(self.coco_format["categories"])
        }
        
        return stats
    
    def print_stats(self) -> None:
        """Print dataset statistics"""
        stats = self.get_dataset_stats()
        
        print("Dataset Statistics:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Total annotations: {stats['total_annotations']}")
        print(f"  Average annotations per image: {stats['avg_annotations_per_image']:.2f}")
        print(f"  Max annotations per image: {stats['max_annotations_per_image']}")
        print(f"  Average bbox area: {stats['avg_bbox_area']:.2f}")
        print(f"  Categories: {stats['categories']}")


def create_sample_dataset(dataset_dir: str, num_images: int = 10) -> None:
    """
    Create a sample dataset for testing
    
    Args:
        dataset_dir: Directory to create dataset
        num_images: Number of sample images to create
    """
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    manager = COCODatasetManager(dataset_dir)
    manager.create_dataset_structure()
    
    print(f"Creating sample dataset with {num_images} images...")
    
    # Create sample images and annotations
    for i in range(num_images):
        # Create a sample image (random tennis court scene)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some tennis court-like features
        cv2.rectangle(image, (50, 50), (590, 430), (255, 255, 255), 2)  # Court outline
        cv2.line(image, (320, 50), (320, 430), (255, 255, 255), 2)  # Center line
        
        # Add random tennis ball
        ball_x = np.random.randint(100, 540)
        ball_y = np.random.randint(100, 380)
        cv2.circle(image, (ball_x, ball_y), 15, (255, 255, 0), -1)  # Yellow ball
        
        # Save image
        image_path = dataset_dir / "images" / "train" / f"sample_{i:03d}.jpg"
        cv2.imwrite(str(image_path), image)
        
        # Add to COCO format
        image_info = manager.add_image(str(image_path), i+1, "train")
        manager.coco_format["images"].append(image_info)
        
        # Add annotation (bounding box around ball)
        bbox = [ball_x - 15, ball_y - 15, 30, 30]  # [x, y, width, height]
        annotation = manager.add_annotation(i+1, i+1, bbox)
        manager.coco_format["annotations"].append(annotation)
    
    # Save annotations
    manager.save_annotations("train")
    
    print(f"Sample dataset created in: {dataset_dir}")
    print("Sample images:")
    print("  - Random tennis court scenes")
    print("  - Yellow tennis balls")
    print("  - Bounding box annotations")
    print("  - Center point keypoints")


def main():
    """Main function for COCO dataset utilities"""
    parser = argparse.ArgumentParser(description="COCO Dataset Utilities for Tennis Ball Detection")
    parser.add_argument("--dataset_dir", required=True, help="Dataset directory")
    parser.add_argument("--create_structure", action="store_true", help="Create dataset structure")
    parser.add_argument("--create_sample", action="store_true", help="Create sample dataset")
    parser.add_argument("--load_annotations", help="Load annotations from JSON file")
    parser.add_argument("--save_annotations", help="Save annotations to JSON file")
    parser.add_argument("--visualize", type=int, help="Visualize annotations for image ID")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    
    args = parser.parse_args()
    
    manager = COCODatasetManager(args.dataset_dir)
    
    if args.create_structure:
        manager.create_dataset_structure()
        print("Dataset structure created!")
    
    elif args.create_sample:
        create_sample_dataset(args.dataset_dir)
    
    elif args.load_annotations:
        manager.load_annotations(args.load_annotations)
    
    elif args.save_annotations:
        manager.save_annotations("train", args.save_annotations)
    
    elif args.visualize:
        manager.visualize_annotations(args.visualize)
    
    elif args.stats:
        manager.print_stats()
    
    else:
        print("No action specified. Use --help for available options.")
    
    return 0


if __name__ == "__main__":
    exit(main())


