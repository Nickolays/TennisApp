"""
Model Registry - Universal model loader with caching

File: app/models/model_registry.py
"""
from typing import Dict, Any, Optional
import torch
from pathlib import Path


class ModelRegistry:
    """
    Singleton registry for loading and caching models.

    Supports:
    - Automatic model caching (load once, reuse everywhere)
    - Multiple model types (TrackNet, YOLO, custom)
    - Config-driven model swapping

    Usage:
        # In detection step
        model = ModelRegistry.load_model(
            model_type="tracknet",
            model_path="models/ball_model_best.pt",
            out_channels=14
        )

        # Model is cached - subsequent calls return same instance
        same_model = ModelRegistry.load_model(
            model_type="tracknet",
            model_path="models/ball_model_best.pt"
        )
        assert model is same_model  # True!
    """

    # Class-level cache (singleton pattern)
    _models: Dict[str, Any] = {}
    _model_classes: Dict[str, type] = {}

    @classmethod
    def register_model_class(cls, model_type: str, model_class: type):
        """
        Register a model class for a given type.

        Args:
            model_type: Model type identifier (e.g., "tracknet", "yolo")
            model_class: Model class to instantiate
        """
        cls._model_classes[model_type] = model_class
        print(f"[ModelRegistry] Registered model type: {model_type} → {model_class.__name__}")

    @classmethod
    def load_model(
        cls,
        model_type: str,
        model_path: str,
        device: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Load model (with caching).

        Args:
            model_type: Model type ("tracknet", "yolo", etc.)
            model_path: Path to model weights
            device: Device ("cuda", "cpu", or None for auto)
            **kwargs: Model-specific parameters

        Returns:
            Loaded model instance

        Raises:
            ValueError: If model_type not registered
            FileNotFoundError: If model_path doesn't exist
        """
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create cache key
        cache_key = f"{model_type}:{model_path}:{device}"

        # Return cached model if available
        if cache_key in cls._models:
            print(f"[ModelRegistry] Using cached model: {model_type} ({model_path})")
            return cls._models[cache_key]

        # Check if model type is registered
        if model_type not in cls._model_classes:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available types: {list(cls._model_classes.keys())}"
            )

        # Check if model file exists (skip for YOLO - it auto-downloads)
        if model_type != 'yolo' and not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model
        print(f"[ModelRegistry] Loading model: {model_type} from {model_path} (device: {device})")

        model_class = cls._model_classes[model_type]
        model = model_class(model_path=model_path, device=device, **kwargs)

        # Cache model
        cls._models[cache_key] = model

        print(f"[ModelRegistry] ✓ Model loaded and cached: {model_type}")
        return model

    @classmethod
    def clear_cache(cls):
        """Clear all cached models (frees GPU/CPU memory)"""
        cls._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[ModelRegistry] Cache cleared")

    @classmethod
    def get_cached_models(cls) -> Dict[str, Any]:
        """Get all cached models"""
        return cls._models.copy()


# ========== MODEL WRAPPERS ==========

class TrackNetModelWrapper:
    """
    Wrapper for TrackNet model (court & ball detection)

    Provides consistent interface for ModelRegistry
    """

    def __init__(self, model_path: str, device: str = "cuda", out_channels: int = 14):
        """
        Initialize TrackNet wrapper

        Args:
            model_path: Path to TrackNet weights (.pt file)
            device: Device to load model on
            out_channels: Number of output channels (14 for court, 1 for ball)
        """
        from app.models.tracknet import TrackNet

        self.device = device

        # Load weights first to infer actual out_channels
        checkpoint = torch.load(model_path, map_location=device)

        # Infer out_channels from checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Check actual output channels from last conv layer
            if 'conv18.block.0.weight' in state_dict:
                actual_out_channels = state_dict['conv18.block.0.weight'].shape[0]
                print(f"[TrackNet] Inferred out_channels={actual_out_channels} from checkpoint")
                out_channels = actual_out_channels

        self.out_channels = out_channels

        # Create model with correct out_channels
        self.model = TrackNet(out_channels=out_channels)
        self.model = self.model.to(device)

        # Load weights
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume checkpoint IS the state_dict
                self.model.load_state_dict(checkpoint)
        else:
            # Checkpoint is the model itself
            self.model = checkpoint.to(device)

        self.model.eval()

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Run inference

        Args:
            frame: Input frame tensor (B, C, H, W) or (C, H, W)

        Returns:
            Output heatmap (B, out_channels, H, W) or (out_channels, H, W)
        """
        with torch.no_grad():
            return self.model(frame)


class YOLOModelWrapper:
    """
    Wrapper for YOLO model (player detection)

    Provides consistent interface for ModelRegistry
    """

    def __init__(self, model_path: str, device: str = "cuda", conf_threshold: float = 0.25):
        """
        Initialize YOLO wrapper

        Args:
            model_path: Path to YOLO weights
            device: Device to load model on
            conf_threshold: Confidence threshold for detections
        """
        try:
            from ultralytics import YOLO
            from pathlib import Path
        except ImportError:
            raise ImportError(
                "Ultralytics not installed. Install with: pip install ultralytics"
            )

        self.device = device
        self.conf_threshold = conf_threshold

        # Convert to absolute path to avoid YOLO auto-download
        model_path_abs = str(Path(model_path).resolve())

        # Load YOLO model (absolute path prevents auto-download)
        self.model = YOLO(model_path_abs)
        self.model.to(device)

        # Person class ID in COCO
        self.person_class_id = 0

    def __call__(self, frame, **kwargs):
        """
        Run inference

        Args:
            frame: Input frame (NumPy array or path)
            **kwargs: Additional YOLO parameters

        Returns:
            YOLO results object
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            classes=[self.person_class_id],  # Only detect persons
            **kwargs
        )
        return results


# ========== AUTO-REGISTER AVAILABLE MODELS ==========

# Register TrackNet
ModelRegistry.register_model_class("tracknet", TrackNetModelWrapper)

# Register YOLO
ModelRegistry.register_model_class("yolo", YOLOModelWrapper)

# Future models can be registered here:
# ModelRegistry.register_model_class("faster_rcnn", FasterRCNNWrapper)
# ModelRegistry.register_model_class("ball_hit_classifier", BallHitClassifier)
