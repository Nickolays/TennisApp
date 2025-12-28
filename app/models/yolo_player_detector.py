"""
Tennis Computer Vision - YOLO Player Detection
File: app/models/yolo_player_detector.py

YOLO v11 player detection for tennis video analysis
"""
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from app.core.data_models import ProcessingConfig


class YOLOPlayerDetector:
    """
    YOLO v11 player detection using Ultralytics
    
    Supports loading by model name (yolo11n.pt, yolo11s.pt, etc.) or weights path
    Filters detections for class 0 (person)
    """
    
    def __init__(self, config: ProcessingConfig, model_name_or_path: str = "yolo11n.pt"):
        """
        Initialize YOLO player detector
        
        Args:
            config: Processing configuration
            model_name_or_path: YOLO model name (e.g., 'yolo11n.pt') or path to weights file
        """
        self.config = config
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.device = None
        self.is_loaded = False
        
        # Person class ID in COCO dataset
        self.person_class_id = 0
        
        # Confidence threshold for person detection
        self.conf_threshold = 0.25
        
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load YOLO model
        
        Args:
            model_path: Optional path to model file (if None, uses model_name_or_path)
        
        Returns:
            True if successful
        """
        try:
            from ultralytics import YOLO
            
            # Determine model path
            if model_path is None:
                model_path = self.model_name_or_path
            
            # Check if it's a model name (like 'yolo11n.pt') or a file path
            if not Path(model_path).exists() and not model_path.startswith('yolo'):
                # Try to find in models directory
                full_path = Path(self.config.model_path) / model_path
                if full_path.exists():
                    model_path = str(full_path)
            
            # Load YOLO model (will download if using model name)
            self.model = YOLO(model_path)
            
            # Get device
            import torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            self.is_loaded = True
            print(f"✓ YOLO player detection model loaded: {model_path}")
            print(f"  Device: {self.device}")
            
            return True
            
        except ImportError:
            print("⚠️ Ultralytics not installed. Install with: pip install ultralytics")
            return False
        except Exception as e:
            print(f"✗ Failed to load YOLO model: {e}")
            return False
    
    def __call__(self, frames: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect players in frames
        
        Args:
            frames: Array of frames (N, H, W, 3) in BGR format
        
        Returns:
            List of detection dictionaries with player bounding boxes
        """
        if not self.is_loaded:
            print("⚠️ YOLO model not loaded, returning empty detections")
            return [{'player_boxes': [], 'player_confs': [], 'player_class_ids': []} for _ in range(len(frames))]
        
        results = []
        
        try:
            # Process each frame
            for i, frame in enumerate(frames):
                # YOLO expects BGR format (OpenCV default)
                # Run inference
                yolo_results = self.model(frame, conf=self.conf_threshold, verbose=False)
                
                # Extract person detections (class_id = 0)
                player_boxes = []
                player_confs = []
                player_class_ids = []
                
                if len(yolo_results) > 0:
                    result = yolo_results[0]
                    
                    # Get boxes, confidences, and class IDs
                    boxes = result.boxes
                    
                    if boxes is not None and len(boxes) > 0:
                        # Filter for person class (class_id = 0)
                        person_mask = boxes.cls.cpu().numpy() == self.person_class_id
                        
                        if np.any(person_mask):
                            # Get person boxes
                            person_boxes = boxes.xyxy[person_mask].cpu().numpy()  # (x1, y1, x2, y2)
                            person_confs = boxes.conf[person_mask].cpu().numpy()
                            
                            # Convert to list of tuples
                            for box, conf in zip(person_boxes, person_confs):
                                # box format: [x1, y1, x2, y2]
                                player_boxes.append((float(box[0]), float(box[1]), float(box[2]), float(box[3])))
                                player_confs.append(float(conf))
                                player_class_ids.append(int(self.person_class_id))
                
                result_dict = {
                    'player_boxes': player_boxes,
                    'player_confs': player_confs,
                    'player_class_ids': player_class_ids
                }
                
                results.append(result_dict)
                
                if i == 0 and len(player_boxes) > 0:
                    print(f"[DEBUG YOLOPlayerDetector] Frame {i}: Detected {len(player_boxes)} players")
                    print(f"  First player box: {player_boxes[0]}, conf: {player_confs[0]:.2f}")
        
        except Exception as e:
            print(f"YOLO player detection error: {e}")
            import traceback
            traceback.print_exc()
            # Return empty detections
            results = [{'player_boxes': [], 'player_confs': [], 'player_class_ids': []} for _ in range(len(frames))]
        
        return results

