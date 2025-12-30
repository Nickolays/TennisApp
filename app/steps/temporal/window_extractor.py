"""
Temporal Window Extractor - Extract frame windows for event detection

File: app/steps/temporal/window_extractor.py
"""
import numpy as np
from typing import List, Optional, Tuple

from app.steps.base import PipelineStep
from app.core.context import ProcessingContext
from app.core.data_models import TemporalWindow


class TemporalWindowExtractorStep(PipelineStep):
    """
    Extract temporal windows (±N frames) for event detection models.

    Features:
    - Extracts windows of frames around each candidate frame
    - Stores windows in context for hit detection model
    - Handles edge cases (start/end of video)
    - Configurable window size and stride

    Configuration:
        temporal:
          window_extraction:
            enabled: true
            window_size: 5         # Extract ±5 frames (11 total)
            stride: 1              # Extract window every N frames
            only_with_ball: true   # Only extract windows where center frame has ball
            min_ball_confidence: 0.1  # Minimum confidence for center frame
    """

    def __init__(self, config: dict):
        """
        Initialize window extractor step

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        self.window_size = config.get('window_size', 5)
        self.stride = config.get('stride', 1)
        self.only_with_ball = config.get('only_with_ball', True)
        self.min_ball_confidence = config.get('min_ball_confidence', 0.1)

    def _extract_window(
        self,
        context: ProcessingContext,
        center_idx: int
    ) -> Optional[TemporalWindow]:
        """
        Extract temporal window around a center frame

        Args:
            context: Processing context
            center_idx: Index of center frame in context.frame_ids

        Returns:
            TemporalWindow object or None if window is incomplete
        """
        # Calculate window bounds
        start_idx = max(0, center_idx - self.window_size)
        end_idx = min(len(context.frame_ids) - 1, center_idx + self.window_size)

        # Check if we have a full window
        expected_size = 2 * self.window_size + 1
        actual_size = end_idx - start_idx + 1

        if actual_size < expected_size:
            # Incomplete window (near start/end of video)
            return None

        # Extract frame IDs
        frame_ids = context.frame_ids[start_idx:end_idx + 1]

        # Extract frames (if available in context)
        frames = None
        if context.frames:
            frames = context.frames[start_idx:end_idx + 1]

        # Extract ball positions
        ball_positions = []
        for i in range(start_idx, end_idx + 1):
            det = context.detections[i]
            if det.ball_position_px is not None:
                ball_positions.append(det.ball_position_px)
            else:
                ball_positions.append(None)

        # Create window
        window = TemporalWindow(
            center_frame_id=context.frame_ids[center_idx],
            frame_ids=frame_ids,
            frames=frames,
            ball_positions=ball_positions,
            center_ball_position=context.detections[center_idx].ball_position_px,
            center_ball_confidence=context.detections[center_idx].ball_confidence
        )

        return window

    def _should_extract_window(
        self,
        context: ProcessingContext,
        center_idx: int
    ) -> bool:
        """
        Check if we should extract a window for this frame

        Args:
            context: Processing context
            center_idx: Index of center frame

        Returns:
            True if window should be extracted
        """
        det = context.detections[center_idx]

        # Check if ball is present
        if self.only_with_ball:
            if det.ball_position_px is None:
                return False

            # Check ball confidence
            if det.ball_confidence < self.min_ball_confidence:
                return False

        return True

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Extract temporal windows for event detection

        Updates context.temporal_windows

        Args:
            context: Processing context with detections

        Returns:
            Updated context
        """
        if not context.detections:
            print("  No detections to process")
            return context

        # Extract windows
        windows = []
        extracted_count = 0
        skipped_count = 0

        for i in range(0, len(context.frame_ids), self.stride):
            # Check if we should extract
            if not self._should_extract_window(context, i):
                skipped_count += 1
                continue

            # Extract window
            window = self._extract_window(context, i)

            if window is not None:
                windows.append(window)
                extracted_count += 1
            else:
                skipped_count += 1

        # Store windows in context
        context.temporal_windows = windows

        # Report statistics
        total_frames = len(context.frame_ids)
        print(f"  Extracted {extracted_count} temporal windows (size=±{self.window_size} frames)")
        print(f"  Skipped {skipped_count} frames (no ball or incomplete window)")
        print(f"  Coverage: {extracted_count}/{total_frames} frames ({extracted_count/total_frames*100:.1f}%)")

        return context
