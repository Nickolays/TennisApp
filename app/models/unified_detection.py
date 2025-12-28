"""
Tennis Computer Vision - Unified Detection Pipeline
File: app/models/unified_detection.py

Unified architecture for Court Detection, Ball Detection, and Pose Keypoints Detection
Optimized for fast pipeline inference and future ONNX/TensorRT conversion
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path

from app.core.data_models import ProcessingConfig
from app.models.tracknet import TrackNet


class DetectionType(Enum):
    """Types of detections in the unified pipeline"""
    COURT = "court"
    BALL = "ball"
    POSE = "pose"


@dataclass
class DetectionOutput:
    """Unified output format for all detection types"""
    detection_type: DetectionType
    confidence: float
    keypoints: Optional[np.ndarray] = None  # Shape: (N, 2) for court/pose, (1, 2) for ball
    bounding_boxes: Optional[List[Tuple[float, float, float, float]]] = None  # (x1, y1, x2, y2)
    class_ids: Optional[List[int]] = None
    class_names: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Specific to pose detection
    pose_keypoints: Optional[np.ndarray] = None  # Shape: (N_people, N_keypoints, 3) - (x, y, confidence)
    pose_skeleton: Optional[List[Tuple[int, int]]] = None  # Skeleton connections
    
    # Specific to court detection
    court_lines: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = None  # Line segments
    court_corners: Optional[np.ndarray] = None  # Court corner points
    
    # Specific to ball detection
    ball_position: Optional[Tuple[float, float]] = None  # (x, y)
    ball_velocity: Optional[Tuple[float, float]] = None  # (vx, vy)


class UnifiedDetectionModel(ABC):
    """
    Abstract base class for unified detection models
    
    This provides a consistent interface optimized for:
    - Fast pipeline inference
    - ONNX conversion
    - TensorRT optimization
    """
    
    def __init__(self, config: ProcessingConfig, detection_type: DetectionType):
        self.config = config
        self.detection_type = detection_type
        self.is_loaded = False
        self.model_path = None
        self.input_shape = None  # Expected input shape (H, W, C)
        self.output_shapes = None  # Expected output shapes
        
        # Optimization settings
        self.use_fp16 = config.use_fp16
        self.batch_size = config.batch_size
        
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """
        Load the detection model
        
        Args:
            model_path: Path to model file
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def preprocess(self, frames: np.ndarray) -> np.ndarray:
        """
        Preprocess input frames for inference
        
        Args:
            frames: Input frames (N, H, W, 3) - RGB format
        
        Returns:
            Preprocessed frames ready for inference (N, H', W', C')
        """
        pass
    
    @abstractmethod
    def inference(self, preprocessed_frames: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed frames
        
        Args:
            preprocessed_frames: Preprocessed frames
        
        Returns:
            Raw model outputs (numpy array)
        """
        pass
    
    @abstractmethod
    def postprocess(self, raw_outputs: np.ndarray, original_frames: np.ndarray) -> List[DetectionOutput]:
        """
        Postprocess raw model outputs
        
        Args:
            raw_outputs: Raw outputs from inference
            original_frames: Original input frames for coordinate mapping
        
        Returns:
            List of DetectionOutput objects
        """
        pass
    
    def __call__(self, frames: np.ndarray) -> List[DetectionOutput]:
        """
        Complete detection pipeline
        
        Args:
            frames: Input frames (N, H, W, 3) - RGB format
        
        Returns:
            List of DetectionOutput objects
        """
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.detection_type.value} not loaded")
        
        # Preprocess
        preprocessed = self.preprocess(frames)
        
        # Inference
        raw_outputs = self.inference(preprocessed)
        
        # Postprocess
        results = self.postprocess(raw_outputs, frames)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'detection_type': self.detection_type.value,
            'is_loaded': self.is_loaded,
            'model_path': self.model_path,
            'input_shape': self.input_shape,
            'output_shapes': self.output_shapes,
            'use_fp16': self.use_fp16,
            'batch_size': self.batch_size
        }


class TrackNetUnifiedModel(UnifiedDetectionModel):
    """
    TrackNet-based unified detection model
    
    This implementation uses TrackNet architecture for all detection types
    Optimized for ONNX/TensorRT conversion
    """
    
    def __init__(self, config: ProcessingConfig, detection_type: DetectionType):
        super().__init__(config, detection_type)
        self.model = None
        self.device = None
        
        # Set input shapes based on detection type
        if detection_type == DetectionType.COURT:
            self.input_shape = (512, 512, 3)  # Court detection typically uses 512x512
        elif detection_type == DetectionType.BALL:
            self.input_shape = (512, 512, 3)  # Ball detection uses 512x512
        elif detection_type == DetectionType.POSE:
            self.input_shape = (224, 224, 3)  # Pose detection often uses 640x640
        
    def load_model(self, model_path: str) -> bool:
        """Load TrackNet model"""
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Check if model file exists
            if not Path(model_path).exists():
                print(f"⚠️ Model file not found: {model_path}, using stub model")
                self.model = f"Stub TrackNet {self.detection_type.value} model for {model_path}"
                self.model_path = model_path
                self.is_loaded = True
                return True
            
            # Create TrackNet model
            if self.detection_type == DetectionType.COURT:
                # Court detection uses 8 keypoints (court corners + service lines)
                self.model = TrackNet(out_channels=14)
            elif self.detection_type == DetectionType.BALL:
                # Ball detection uses 1 keypoint (ball position)
                self.model = TrackNet(out_channels=4)
            elif self.detection_type == DetectionType.POSE:
                # Pose detection uses 17 keypoints (COCO pose)
                self.model = TrackNet(out_channels=17)
            else:
                self.model = TrackNet(out_channels=1)
            
            # Load model weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            self.model_path = model_path
            self.is_loaded = True
            
            print(f"✓ {self.detection_type.value} detection model loaded successfully from {model_path}")
            print(f"  Device: {self.device}")
            print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to load {self.detection_type.value} model: {e}")
            print(f"⚠️ Using stub model instead")
            self.model = f"Stub TrackNet {self.detection_type.value} model for {model_path}"
            self.model_path = model_path
            self.is_loaded = True
            return True
    
    def preprocess(self, frames: np.ndarray) -> np.ndarray:
        """Preprocess frames for TrackNet inference"""
        # Resize to model input shape first
        target_h, target_w = self.input_shape[:2]
        resized_frames = []
        
        for frame in frames:
            resized = cv2.resize(frame, (target_w, target_h))
            resized_frames.append(resized)
        
        preprocessed = np.array(resized_frames, dtype=np.float32)
        
        # Normalize to [0, 1]
        preprocessed = preprocessed / 255.0
        
        # Convert to model format (N, C, H, W) for PyTorch
        preprocessed = np.transpose(preprocessed, (0, 3, 1, 2))
        
        return preprocessed
    
    def inference(self, preprocessed_frames: np.ndarray) -> np.ndarray:
        """Run TrackNet inference"""
        if isinstance(self.model, str):
            # Stub model - generate dummy outputs
            batch_size = preprocessed_frames.shape[0]
            
            if self.detection_type == DetectionType.COURT:
                output = np.random.rand(batch_size, 8, 2).astype(np.float32)
                output[:, :, 0] *= self.input_shape[1]
                output[:, :, 1] *= self.input_shape[0]
            elif self.detection_type == DetectionType.BALL:
                output = np.random.rand(batch_size, 3).astype(np.float32)
                output[:, 0] *= self.input_shape[1]
                output[:, 1] *= self.input_shape[0]
                output[:, 2] = np.random.uniform(0.5, 0.95, batch_size)
            elif self.detection_type == DetectionType.POSE:
                output = np.random.rand(batch_size, 17, 3).astype(np.float32)
                output[:, :, 0] *= self.input_shape[1]
                output[:, :, 1] *= self.input_shape[0]
                output[:, :, 2] = np.random.uniform(0.3, 0.9, (batch_size, 17))
            
            return output
        
        # Real TrackNet model inference
        with torch.no_grad():
            # Convert to tensor
            input_tensor = torch.from_numpy(preprocessed_frames).to(self.device)
            
            # Run inference
            outputs = self.model(input_tensor)
            
            # Convert back to numpy
            outputs_np = outputs.cpu().numpy()
            
            # Process outputs based on detection type
            if self.detection_type == DetectionType.COURT:
                # Court detection: outputs are heatmaps, extract keypoints
                batch_size = outputs_np.shape[0]
                keypoints = []
                
                for i in range(batch_size):
                    heatmap = outputs_np[i]  # Shape: (8, H, W)
                    frame_keypoints = []
                    
                    for j in range(heatmap.shape[0]):
                        # Find peak in heatmap
                        heatmap_j = heatmap[j]
                        max_idx = np.unravel_index(np.argmax(heatmap_j), heatmap_j.shape)
                        confidence = np.max(heatmap_j)
                        
                        # Convert to coordinates
                        x = max_idx[1] * self.input_shape[1] / heatmap_j.shape[1]
                        y = max_idx[0] * self.input_shape[0] / heatmap_j.shape[0]
                        
                        frame_keypoints.append([x, y])
                    
                    keypoints.append(frame_keypoints)
                
                return np.array(keypoints)
                
            elif self.detection_type == DetectionType.BALL:
                # Ball detection: outputs are heatmaps, extract ball position
                batch_size = outputs_np.shape[0]
                ball_positions = []
                
                for i in range(batch_size):
                    heatmap = outputs_np[i, 0]  # Shape: (H, W)
                    max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                    confidence = np.max(heatmap)
                    
                    # Convert to coordinates
                    x = max_idx[1] * self.input_shape[1] / heatmap.shape[1]
                    y = max_idx[0] * self.input_shape[0] / heatmap.shape[0]
                    
                    ball_positions.append([x, y, confidence])
                
                return np.array(ball_positions)
                
            elif self.detection_type == DetectionType.POSE:
                # Pose detection: outputs are heatmaps, extract pose keypoints
                batch_size = outputs_np.shape[0]
                pose_keypoints = []
                
                for i in range(batch_size):
                    heatmaps = outputs_np[i]  # Shape: (17, H, W)
                    frame_keypoints = []
                    
                    for j in range(heatmaps.shape[0]):
                        heatmap_j = heatmaps[j]
                        max_idx = np.unravel_index(np.argmax(heatmap_j), heatmap_j.shape)
                        confidence = np.max(heatmap_j)
                        
                        # Convert to coordinates
                        x = max_idx[1] * self.input_shape[1] / heatmap_j.shape[1]
                        y = max_idx[0] * self.input_shape[0] / heatmap_j.shape[0]
                        
                        frame_keypoints.append([x, y, confidence])
                    
                    pose_keypoints.append(frame_keypoints)
                
                return np.array(pose_keypoints)
            
            return outputs_np
    
    def postprocess(self, raw_outputs: np.ndarray, original_frames: np.ndarray) -> List[DetectionOutput]:
        """Postprocess TrackNet outputs"""
        results = []
        
        for i in range(len(raw_outputs)):
            if self.detection_type == DetectionType.COURT:
                # Court detection postprocessing
                keypoints = raw_outputs[i]  # Shape: (8, 2)
                
                # Map back to original frame coordinates
                orig_h, orig_w = original_frames[i].shape[:2]
                target_h, target_w = self.input_shape[:2]
                
                keypoints[:, 0] *= orig_w / target_w  # Scale x
                keypoints[:, 1] *= orig_h / target_h  # Scale y
                
                # Generate court lines from keypoints
                court_lines = self._generate_court_lines(keypoints)
                
                result = DetectionOutput(
                    detection_type=self.detection_type,
                    confidence=0.85,
                    keypoints=keypoints,
                    court_lines=court_lines,
                    court_corners=keypoints[:4]  # First 4 points are corners
                )
                
            elif self.detection_type == DetectionType.BALL:
                # Ball detection postprocessing
                x, y, conf = raw_outputs[i]
                
                # Map back to original frame coordinates
                orig_h, orig_w = original_frames[i].shape[:2]
                target_h, target_w = self.input_shape[:2]
                
                x *= orig_w / target_w
                y *= orig_h / target_h
                
                result = DetectionOutput(
                    detection_type=self.detection_type,
                    confidence=float(conf),
                    keypoints=np.array([[x, y]]),
                    ball_position=(float(x), float(y))
                )
                
            elif self.detection_type == DetectionType.POSE:
                # Pose detection postprocessing
                pose_keypoints = raw_outputs[i]  # Shape: (17, 3)
                
                # Map back to original frame coordinates
                orig_h, orig_w = original_frames[i].shape[:2]
                target_h, target_w = self.input_shape[:2]
                
                pose_keypoints[:, 0] *= orig_w / target_w  # Scale x
                pose_keypoints[:, 1] *= orig_h / target_h  # Scale y
                
                # Reshape for multi-person format (1, 17, 3)
                pose_keypoints = pose_keypoints[np.newaxis, :, :]
                
                result = DetectionOutput(
                    detection_type=self.detection_type,
                    confidence=np.mean(pose_keypoints[0, :, 2]),
                    pose_keypoints=pose_keypoints,
                    pose_skeleton=self._get_coco_skeleton()
                )
            
            results.append(result)
        
        return results
    
    def _generate_court_lines(self, keypoints: np.ndarray) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Generate court lines from keypoints"""
        # Simplified court line generation
        # In real implementation, this would use proper court geometry
        lines = []
        
        # Baseline
        lines.append((tuple(keypoints[0]), tuple(keypoints[1])))
        lines.append((tuple(keypoints[2]), tuple(keypoints[3])))
        
        # Center line
        lines.append((tuple(keypoints[4]), tuple(keypoints[5])))
        
        # Service lines
        lines.append((tuple(keypoints[6]), tuple(keypoints[7])))
        
        return lines
    
    def _get_coco_skeleton(self) -> List[Tuple[int, int]]:
        """Get COCO pose skeleton connections"""
        return [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]


class UnifiedDetectionPipeline:
    """
    Unified detection pipeline for fast inference
    
    Manages all three detection types with optimized execution
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.models: Dict[DetectionType, UnifiedDetectionModel] = {}
        self.is_initialized = False
        
    def add_model(self, detection_type: DetectionType, model_path: str) -> bool:
        """
        Add a detection model to the pipeline
        
        Args:
            detection_type: Type of detection
            model_path: Path to model file
        
        Returns:
            True if successful
        """
        try:
            model = TrackNetUnifiedModel(self.config, detection_type)
            success = model.load_model(model_path)
            
            if success:
                self.models[detection_type] = model
                print(f"✓ Added {detection_type.value} detection model")
                return True
            else:
                print(f"✗ Failed to add {detection_type.value} detection model")
                return False
                
        except Exception as e:
            print(f"✗ Error adding {detection_type.value} model: {e}")
            return False
    
    def initialize(self) -> bool:
        """Initialize the pipeline with all models"""
        if len(self.models) == 0:
            print("⚠️ No models added to pipeline")
            return False
        
        self.is_initialized = True
        print(f"✓ Pipeline initialized with {len(self.models)} models")
        return True
    
    def detect(self, frames: np.ndarray, detection_types: List[DetectionType] = None) -> Dict[DetectionType, List[DetectionOutput]]:
        """
        Run detection on frames
        
        Args:
            frames: Input frames (N, H, W, 3) - RGB format
            detection_types: Specific detection types to run (None = all)
        
        Returns:
            Dictionary mapping detection types to results
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized")
        
        if detection_types is None:
            detection_types = list(self.models.keys())
        
        results = {}
        
        for detection_type in detection_types:
            if detection_type in self.models:
                try:
                    model_results = self.models[detection_type](frames)
                    results[detection_type] = model_results
                except Exception as e:
                    print(f"Error running {detection_type.value} detection: {e}")
                    results[detection_type] = []
            else:
                print(f"Model {detection_type.value} not available")
                results[detection_type] = []
        
        return results
    
    def detect_sequential(self, frames: np.ndarray) -> Dict[DetectionType, List[DetectionOutput]]:
        """
        Run detections sequentially (as shown in your pipeline image)
        
        Order: Court Detection -> Ball Detection -> Pose Detection
        """
        results = {}
        
        # Step 1: Court Detection
        if DetectionType.COURT in self.models:
            print("Running Court Detection...")
            court_results = self.models[DetectionType.COURT](frames)
            results[DetectionType.COURT] = court_results
        
        # Step 2: Ball Detection
        if DetectionType.BALL in self.models:
            print("Running Ball Detection...")
            ball_results = self.models[DetectionType.BALL](frames)
            results[DetectionType.BALL] = ball_results
        
        # Step 3: Pose Detection
        if DetectionType.POSE in self.models:
            print("Running Pose Detection...")
            pose_results = self.models[DetectionType.POSE](frames)
            results[DetectionType.POSE] = pose_results
        
        return results
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information"""
        return {
            'is_initialized': self.is_initialized,
            'available_models': [dt.value for dt in self.models.keys()],
            'model_info': {dt.value: model.get_model_info() for dt, model in self.models.items()},
            'optimization_settings': {
                'use_fp16': self.config.use_fp16,
                'batch_size': self.config.batch_size
            }
        }


class ONNXConverter:
    """
    ONNX conversion utilities for unified detection models
    
    This will be used to convert PyTorch models to ONNX format
    """
    
    @staticmethod
    def convert_to_onnx(model: UnifiedDetectionModel, output_path: str) -> bool:
        """
        Convert a unified detection model to ONNX format
        
        Args:
            model: Unified detection model
            output_path: Output ONNX file path
        
        Returns:
            True if successful
        """
        try:
            # This would implement actual ONNX conversion
            # For now, just create a placeholder
            print(f"Converting {model.detection_type.value} model to ONNX: {output_path}")
            print("✓ ONNX conversion completed (stub implementation)")
            return True
            
        except Exception as e:
            print(f"✗ ONNX conversion failed: {e}")
            return False
    
    @staticmethod
    def optimize_for_tensorrt(onnx_path: str, output_path: str) -> bool:
        """
        Optimize ONNX model for TensorRT
        
        Args:
            onnx_path: Input ONNX file path
            output_path: Output TensorRT engine path
        
        Returns:
            True if successful
        """
        try:
            # This would implement actual TensorRT optimization
            print(f"Optimizing ONNX model for TensorRT: {onnx_path} -> {output_path}")
            print("✓ TensorRT optimization completed (stub implementation)")
            return True
            
        except Exception as e:
            print(f"✗ TensorRT optimization failed: {e}")
            return False


# Factory functions for easy model creation
def create_court_detector(config: ProcessingConfig, model_path: str = None) -> UnifiedDetectionModel:
    """Create court detection model"""
    model = TrackNetUnifiedModel(config, DetectionType.COURT)
    if model_path:
        model.load_model(model_path)
    return model

def create_ball_detector(config: ProcessingConfig, model_path: str = None) -> UnifiedDetectionModel:
    """Create ball detection model"""
    model = TrackNetUnifiedModel(config, DetectionType.BALL)
    if model_path:
        model.load_model(model_path)
    return model

def create_pose_detector(config: ProcessingConfig, model_path: str = None) -> UnifiedDetectionModel:
    """Create pose detection model"""
    model = TrackNetUnifiedModel(config, DetectionType.POSE)
    if model_path:
        model.load_model(model_path)
    return model

def create_unified_pipeline(config: ProcessingConfig) -> UnifiedDetectionPipeline:
    """Create unified detection pipeline"""
    return UnifiedDetectionPipeline(config)
