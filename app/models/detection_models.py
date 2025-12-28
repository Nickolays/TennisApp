"""
Tennis Computer Vision - Detection Models Architecture
File: app/models/detection_models.py

Flexible architecture for different detection models
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from app.core.data_models import ProcessingConfig


class ModelType(Enum):
    """Types of detection models"""
    COURT_DETECTION = "court_detection"
    BALL_DETECTION = "ball_detection"
    PLAYER_DETECTION = "player_detection"
    MULTI_OBJECT = "multi_object"


@dataclass
class DetectionResult:
    """Generic detection result"""
    model_type: ModelType
    confidence: float
    bounding_boxes: Optional[List[Tuple[float, float, float, float]]] = None  # (x1, y1, x2, y2)
    keypoints: Optional[np.ndarray] = None  # Shape: (N, 2)
    class_ids: Optional[List[int]] = None
    class_names: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseDetectionModel(ABC):
    """
    Abstract base class for all detection models
    
    This provides a consistent interface for different detection models
    that can be easily swapped and extended.
    """
    
    def __init__(self, config: ProcessingConfig, model_type: ModelType):
        self.config = config
        self.model_type = model_type
        self.is_loaded = False
        self.model_path = None
        
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
            frames: Input frames (N, H, W, 3)
        
        Returns:
            Preprocessed frames ready for inference
        """
        pass
    
    @abstractmethod
    def inference(self, preprocessed_frames: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on preprocessed frames
        
        Args:
            preprocessed_frames: Preprocessed frames
        
        Returns:
            List of raw detection results
        """
        pass
    
    @abstractmethod
    def postprocess(self, raw_results: List[Dict[str, Any]]) -> List[DetectionResult]:
        """
        Postprocess raw inference results
        
        Args:
            raw_results: Raw results from inference
        
        Returns:
            List of DetectionResult objects
        """
        pass
    
    def __call__(self, frames: np.ndarray) -> List[DetectionResult]:
        """
        Complete detection pipeline
        
        Args:
            frames: Input frames (N, H, W, 3)
        
        Returns:
            List of DetectionResult objects
        """
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.model_type.value} not loaded")
        
        # Preprocess
        preprocessed = self.preprocess(frames)
        
        # Inference
        raw_results = self.inference(preprocessed)
        
        # Postprocess
        results = self.postprocess(raw_results)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': self.model_type.value,
            'is_loaded': self.is_loaded,
            'model_path': self.model_path,
            'config': {
                'batch_size': self.config.batch_size,
                'use_fp16': self.config.use_fp16
            }
        }


class TrackNetDetectionModel(BaseDetectionModel):
    """
    TrackNet-based detection model implementation
    
    This is a concrete implementation for TrackNet models
    """
    
    def __init__(self, config: ProcessingConfig, model_type: ModelType):
        super().__init__(config, model_type)
        self.model = None
        self.device = None
        
    def load_model(self, model_path: str) -> bool:
        """Load TrackNet model"""
        try:
            # Import PyTorch (will be available when dependencies are installed)
            import torch
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load model (stub implementation)
            # In real implementation, this would load the actual TrackNet model
            self.model = f"TrackNet model loaded from {model_path}"
            self.model_path = model_path
            self.is_loaded = True
            
            print(f"✓ {self.model_type.value} model loaded successfully")
            return True
            
        except ImportError:
            print("⚠️ PyTorch not available, using stub model")
            self.model = f"Stub TrackNet model for {model_path}"
            self.model_path = model_path
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"✗ Failed to load {self.model_type.value} model: {e}")
            return False
    
    def preprocess(self, frames: np.ndarray) -> np.ndarray:
        """Preprocess frames for TrackNet"""
        # Normalize to [0, 1]
        preprocessed = frames.astype(np.float32) / 255.0
        
        # Resize if needed (TrackNet typically expects specific input size)
        if frames.shape[1:3] != (512, 512):  # Example size
            # In real implementation, resize to model input size
            pass
        
        return preprocessed
    
    def inference(self, preprocessed_frames: np.ndarray) -> List[Dict[str, Any]]:
        """Run TrackNet inference"""
        # Stub implementation
        results = []
        for i in range(len(preprocessed_frames)):
            # Generate dummy results based on model type
            if self.model_type == ModelType.COURT_DETECTION:
                result = {
                    'keypoints': np.array([[100, 100], [400, 100], [400, 300], [100, 300]], dtype=np.float32),
                    'confidence': 0.85
                }
            elif self.model_type == ModelType.BALL_DETECTION:
                result = {
                    'position': (320 + i*5, 240 + i*3),
                    'confidence': 0.75
                }
            elif self.model_type == ModelType.PLAYER_DETECTION:
                result = {
                    'boxes': [(100, 100, 200, 300), (400, 150, 500, 350)],
                    'confidences': [0.92, 0.88],
                    'class_ids': [0, 1]
                }
            else:
                result = {'confidence': 0.5}
            
            results.append(result)
        
        return results
    
    def postprocess(self, raw_results: List[Dict[str, Any]]) -> List[DetectionResult]:
        """Postprocess TrackNet results"""
        detection_results = []
        
        for i, raw_result in enumerate(raw_results):
            if self.model_type == ModelType.COURT_DETECTION:
                detection_result = DetectionResult(
                    model_type=self.model_type,
                    confidence=raw_result['confidence'],
                    keypoints=raw_result['keypoints']
                )
            elif self.model_type == ModelType.BALL_DETECTION:
                detection_result = DetectionResult(
                    model_type=self.model_type,
                    confidence=raw_result['confidence'],
                    keypoints=np.array([raw_result['position']]) if 'position' in raw_result else None
                )
            elif self.model_type == ModelType.PLAYER_DETECTION:
                detection_result = DetectionResult(
                    model_type=self.model_type,
                    confidence=np.mean(raw_result['confidences']) if 'confidences' in raw_result else raw_result['confidence'],
                    bounding_boxes=raw_result.get('boxes', []),
                    class_ids=raw_result.get('class_ids', []),
                    class_names=['Player1', 'Player2'] if raw_result.get('class_ids') else None
                )
            else:
                detection_result = DetectionResult(
                    model_type=self.model_type,
                    confidence=raw_result['confidence']
                )
            
            detection_results.append(detection_result)
        
        return detection_results


class DetectionModelFactory:
    """
    Factory for creating detection models
    
    This allows easy creation and management of different detection models
    """
    
    @staticmethod
    def create_model(model_type: ModelType, config: ProcessingConfig, 
                    model_path: Optional[str] = None) -> BaseDetectionModel:
        """
        Create a detection model
        
        Args:
            model_type: Type of model to create
            config: Processing configuration
            model_path: Optional path to model file
        
        Returns:
            Detection model instance
        """
        if model_type == ModelType.COURT_DETECTION:
            model = TrackNetDetectionModel(config, model_type)
        elif model_type == ModelType.BALL_DETECTION:
            model = TrackNetDetectionModel(config, model_type)
        elif model_type == ModelType.PLAYER_DETECTION:
            model = TrackNetDetectionModel(config, model_type)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load model if path provided
        if model_path:
            model.load_model(model_path)
        
        return model
    
    @staticmethod
    def create_court_detector(config: ProcessingConfig, model_path: str = None) -> BaseDetectionModel:
        """Create court detection model"""
        return DetectionModelFactory.create_model(ModelType.COURT_DETECTION, config, model_path)
    
    @staticmethod
    def create_ball_detector(config: ProcessingConfig, model_path: str = None) -> BaseDetectionModel:
        """Create ball detection model"""
        return DetectionModelFactory.create_model(ModelType.BALL_DETECTION, config, model_path)
    
    @staticmethod
    def create_player_detector(config: ProcessingConfig, model_path: str = None) -> BaseDetectionModel:
        """Create player detection model"""
        return DetectionModelFactory.create_model(ModelType.PLAYER_DETECTION, config, model_path)


class DetectionPipeline:
    """
    Pipeline for running multiple detection models
    
    This allows running different detection models in sequence or parallel
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.models: Dict[ModelType, BaseDetectionModel] = {}
    
    def add_model(self, model_type: ModelType, model_path: str = None) -> bool:
        """
        Add a detection model to the pipeline
        
        Args:
            model_type: Type of model to add
            model_path: Path to model file
        
        Returns:
            True if successful
        """
        try:
            model = DetectionModelFactory.create_model(model_type, self.config, model_path)
            self.models[model_type] = model
            return True
        except Exception as e:
            print(f"Failed to add {model_type.value} model: {e}")
            return False
    
    def detect(self, frames: np.ndarray, model_types: List[ModelType] = None) -> Dict[ModelType, List[DetectionResult]]:
        """
        Run detection on frames
        
        Args:
            frames: Input frames (N, H, W, 3)
            model_types: Specific models to run (None = all)
        
        Returns:
            Dictionary mapping model types to detection results
        """
        if model_types is None:
            model_types = list(self.models.keys())
        
        results = {}
        for model_type in model_types:
            if model_type in self.models:
                try:
                    model_results = self.models[model_type](frames)
                    results[model_type] = model_results
                except Exception as e:
                    print(f"Error running {model_type.value}: {e}")
                    results[model_type] = []
            else:
                print(f"Model {model_type.value} not available")
                results[model_type] = []
        
        return results
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information"""
        return {
            'available_models': [mt.value for mt in self.models.keys()],
            'model_info': {mt.value: model.get_model_info() for mt, model in self.models.items()}
        }
