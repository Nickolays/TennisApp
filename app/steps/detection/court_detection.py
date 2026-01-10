"""
Court Detection Step - TrackNet-based court keypoint detection

File: app/steps/detection/court_detection.py
"""
import numpy as np
import cv2
import torch
from typing import Optional

from app.steps.base import PipelineStep
from app.core.context import ProcessingContext
from app.models.model_registry import ModelRegistry
from app.src.court_refinement import CourtKeypointRefinement
from app.core.data_models import COURT_TEMPLATE_KEYPOINTS


class CourtDetectionStep(PipelineStep):
    """
    Detect tennis court keypoints using TrackNet.

    Features:
    - Runs detection every N frames (court doesn't move much)
    - Uses ModelRegistry for automatic model caching
    - Stores results in context.detections

    Configuration:
        court:
          enabled: true
          model_path: "models/court_model_best.pt"
          model_type: "tracknet"
          interval: 30  # Detect every 30 frames
          confidence_threshold: 0.5
          input_size: [640, 360]
    """

    def __init__(self, config: dict):
        """
        Initialize court detection step

        Args:
            config: Configuration dictionary with court detection parameters
        """
        super().__init__(config)

        self.model_path = config.get('model_path', 'models/court_model_best.pt')
        self.model_type = config.get('model_type', 'tracknet')
        self.interval = config.get('interval', 30)  # Detect every N frames
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.input_size = tuple(config.get('input_size', [640, 360]))  # (width, height)

        # Model will be loaded lazily via ModelRegistry
        self.model = None

        # Initialize geometric refinement module
        self.use_refinement = config.get('use_refinement', True)
        if self.use_refinement:
            self.refinement = CourtKeypointRefinement(
                court_template=COURT_TEMPLATE_KEYPOINTS,
                crop_size=config.get('refinement_crop_size', 50),
                max_local_shift=config.get('refinement_max_local_shift', 15.0),
                max_regularization_shift=config.get('refinement_max_reg_shift', 10.0),
                ransac_threshold=config.get('refinement_ransac_threshold', 5.0)
            )
        else:
            self.refinement = None

    def _load_model(self):
        """Lazy load model via ModelRegistry"""
        if self.model is None:
            self.model = ModelRegistry.load_model(
                model_type=self.model_type,
                model_path=self.model_path,
                out_channels=14  # TrackNet court has 14 keypoints
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

    def _postprocess_output(
        self,
        output: torch.Tensor,
        original_size: tuple,
        frame: Optional[np.ndarray] = None
    ) -> tuple:
        """
        Convert model output to keypoints with optional geometric refinement

        Args:
            output: Model output (1, 14, H, W) - heatmaps for 14 keypoints
            original_size: (orig_width, orig_height)
            frame: Original frame (H, W, 3) for local refinement (optional)

        Returns:
            (keypoints, confidence):
                keypoints: np.ndarray of shape (14, 2) with (x, y) coordinates
                confidence: float, average confidence across keypoints
        """
        # Remove batch dimension
        heatmaps = output[0].cpu().numpy()  # (14, H, W)

        keypoints = []
        confidences = []

        for i in range(14):
            heatmap = heatmaps[i]

            # Find maximum activation
            max_val = heatmap.max()
            max_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)

            # Convert to original image coordinates
            y_norm, x_norm = max_idx
            x = x_norm / heatmap.shape[1] * original_size[0]
            y = y_norm / heatmap.shape[0] * original_size[1]

            keypoints.append([x, y])
            confidences.append(max_val)

        keypoints = np.array(keypoints)  # (14, 2)
        confidences = np.array(confidences)  # (14,)

        # Apply geometric refinement if enabled
        if self.use_refinement and self.refinement is not None and frame is not None:
            try:
                keypoints, refinement_stats = self.refinement.refine(
                    keypoints, frame, confidences
                )

                # Optional: Log refinement statistics
                if self.config.get('log_refinement_stats', False):
                    self._log_refinement_stats(refinement_stats)

            except Exception as e:
                # Fallback to original keypoints if refinement fails
                print(f"    Warning: Refinement failed ({e}), using original keypoints")

        avg_confidence = float(confidences.mean())
        return keypoints, avg_confidence

    def _log_refinement_stats(self, stats: dict):
        """
        Log refinement statistics for debugging.

        Args:
            stats: Refinement statistics dictionary
        """
        for stage, shifts in stats['shifts'].items():
            if shifts is not None and len(shifts) > 0:
                mean_shift = shifts.mean()
                max_shift = shifts.max()
                print(f"    {stage}: mean={mean_shift:.2f}px, max={max_shift:.2f}px")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Detect court keypoints in frames

        Updates context.detections with court_keypoints and court_confidence

        Args:
            context: Processing context with frames

        Returns:
            Updated context
        """
        # Load model if not already loaded
        self._load_model()

        # Ensure detections list is initialized
        if not context.detections:
            # Initialize detections for all frames
            from app.core.data_models import FrameDetection
            context.detections = [
                FrameDetection(
                    frame_id=fid,
                    timestamp=fid / context.fps
                )
                for fid in context.frame_ids
            ]

        # Process frames at specified interval
        for i in range(0, len(context.frames), self.interval):
            frame_original = context.frames[i]  # Keep original frame for refinement
            frame_id = context.frame_ids[i]

            # Preprocess
            frame_tensor = self._preprocess_frame(frame_original)

            # Inference
            output = self.model(frame_tensor)

            # Postprocess with refinement (pass original frame)
            keypoints, confidence = self._postprocess_output(
                output,
                original_size=(context.width, context.height),
                frame=frame_original  # NEW: Pass frame for geometric refinement
            )

            # Store in detection
            detection_idx = context.frame_ids.index(frame_id)
            context.detections[detection_idx].court_keypoints = keypoints
            context.detections[detection_idx].court_confidence = confidence

            print(f"  Frame {frame_id}: Court detected (conf={confidence:.3f})")

        # Fill intermediate frames (use nearest detection)
        self._fill_intermediate_frames(context)

        return context

    def _fill_intermediate_frames(self, context: ProcessingContext):
        """
        Fill court keypoints for frames between detections.

        Uses nearest neighbor (court doesn't move between frames).

        Args:
            context: Processing context
        """
        # Get frames with court detections
        detected_frames = [
            (i, det) for i, det in enumerate(context.detections)
            if det.court_keypoints is not None
        ]

        if not detected_frames:
            return

        # Fill missing frames
        for i, detection in enumerate(context.detections):
            if detection.court_keypoints is None:
                # Find nearest detected frame
                nearest_idx, nearest_det = min(
                    detected_frames,
                    key=lambda x: abs(x[0] - i)
                )

                # Copy court data
                detection.court_keypoints = nearest_det.court_keypoints.copy()
                detection.court_confidence = nearest_det.court_confidence
