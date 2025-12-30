"""
Ball Detection Step - TrackNet-based ball position detection

File: app/steps/detection/ball_detection.py
"""
import numpy as np
import cv2
import torch
from typing import Optional, Tuple

from app.steps.base import PipelineStep
from app.core.context import ProcessingContext
from app.models.model_registry import ModelRegistry


class BallDetectionStep(PipelineStep):
    """
    Detect tennis ball position using TrackNet.

    Features:
    - Runs detection every frame (or every N frames for speed)
    - Uses ModelRegistry for automatic model caching
    - Stores results in context.detections

    Configuration:
        ball:
          enabled: true
          model_path: "models/ball_model_best.pt"
          model_type: "tracknet"
          interval: 1  # Detect every frame
          confidence_threshold: 0.3
          input_size: [640, 360]
    """

    def __init__(self, config: dict):
        """
        Initialize ball detection step

        Args:
            config: Configuration dictionary with ball detection parameters
        """
        super().__init__(config)

        self.model_path = config.get('model_path', 'models/ball_model_best.pt')
        self.model_type = config.get('model_type', 'tracknet')
        self.interval = config.get('interval', 1)  # Every frame by default
        self.confidence_threshold = config.get('confidence_threshold', 0.3)
        self.input_size = tuple(config.get('input_size', [640, 360]))  # (width, height)

        # Model will be loaded lazily via ModelRegistry
        self.model = None

    def _load_model(self):
        """Lazy load model via ModelRegistry"""
        if self.model is None:
            self.model = ModelRegistry.load_model(
                model_type=self.model_type,
                model_path=self.model_path,
                out_channels=1  # TrackNet ball has single channel output
            )

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for TrackNet

        Args:
            frame: Input frame (H, W, 3) in RGB

        Returns:
            Preprocessed tensor (1, 3, H', W')
        """
        # Resize to input size
        frame_resized = cv2.resize(frame, self.input_size)

        # Convert to tensor (C, H, W)
        frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float()

        # Normalize to [0, 1]
        frame_tensor = frame_tensor / 255.0

        # Add batch dimension (1, C, H, W)
        frame_tensor = frame_tensor.unsqueeze(0)

        # Move to device
        device = next(self.model.model.parameters()).device
        frame_tensor = frame_tensor.to(device)

        return frame_tensor

    def _postprocess_output(self, output: torch.Tensor, original_size: tuple) -> Tuple[Optional[Tuple[float, float]], float]:
        """
        Convert model output to ball position

        Args:
            output: Model output (1, C, H, W) - heatmap(s) for ball (C can be 1 or 4)
            original_size: (orig_width, orig_height)

        Returns:
            ((x, y), confidence) or (None, 0.0) if no detection
        """
        # Handle different number of channels
        if output.shape[1] == 1:
            # Single channel output
            heatmap = output[0, 0].cpu().numpy()  # (H, W)
        else:
            # Multi-channel output - take max across channels
            heatmap = output[0].cpu().numpy().max(axis=0)  # (H, W)

        # Find maximum activation
        max_val = heatmap.max()

        # Check confidence threshold
        if max_val < self.confidence_threshold:
            return None, 0.0

        # Find position
        max_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
        y_norm, x_norm = max_idx

        # Convert to original image coordinates
        x = x_norm / heatmap.shape[1] * original_size[0]
        y = y_norm / heatmap.shape[0] * original_size[1]

        return (x, y), max_val

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Detect ball position in frames

        Updates context.detections with ball_position_px and ball_confidence

        Args:
            context: Processing context with frames

        Returns:
            Updated context
        """
        # Load model if not already loaded
        self._load_model()

        # Ensure detections list is initialized
        if not context.detections:
            from app.core.data_models import FrameDetection
            context.detections = [
                FrameDetection(
                    frame_id=fid,
                    timestamp=fid / context.fps
                )
                for fid in context.frame_ids
            ]

        # Process frames at specified interval
        detected_count = 0
        for i in range(0, len(context.frames), self.interval):
            frame = context.frames[i]
            frame_id = context.frame_ids[i]

            # Preprocess
            frame_tensor = self._preprocess_frame(frame)

            # Inference
            output = self.model(frame_tensor)

            # Postprocess
            position, confidence = self._postprocess_output(
                output,
                original_size=(context.width, context.height)
            )

            # Store in detection
            detection_idx = context.frame_ids.index(frame_id)
            context.detections[detection_idx].ball_position_px = position
            context.detections[detection_idx].ball_confidence = confidence

            if position is not None:
                detected_count += 1

        detection_rate = detected_count / len(context.frames) * 100
        print(f"  Ball detected in {detected_count}/{len(context.frames)} frames ({detection_rate:.1f}%)")

        return context
