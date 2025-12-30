"""
Gap Filling Step - Interpolate missing ball positions

File: app/steps/temporal/gap_filling.py
"""
import numpy as np
from typing import List, Optional, Tuple
from scipy.interpolate import interp1d

from app.steps.base import PipelineStep
from app.core.context import ProcessingContext


class GapFillingStep(PipelineStep):
    """
    Fill gaps in ball trajectory using interpolation.

    Features:
    - Linear interpolation for small gaps (< threshold)
    - Polynomial interpolation for larger gaps
    - Preserves original detections (doesn't smooth)
    - Configurable gap size thresholds
    - Reports gap statistics

    Configuration:
        temporal:
          gap_filling:
            enabled: true
            max_gap_linear: 5      # Use linear for gaps <= 5 frames
            max_gap_poly: 15       # Use polynomial for gaps <= 15 frames
            poly_order: 2          # Polynomial order (2 = quadratic)
            min_points_for_poly: 4 # Need at least 4 points for polynomial
    """

    def __init__(self, config: dict):
        """
        Initialize gap filling step

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        self.max_gap_linear = config.get('max_gap_linear', 5)
        self.max_gap_poly = config.get('max_gap_poly', 15)
        self.poly_order = config.get('poly_order', 2)
        self.min_points_for_poly = config.get('min_points_for_poly', 4)

    def _find_gaps(self, detections: List) -> List[Tuple[int, int]]:
        """
        Find all gaps (consecutive NaN positions) in ball trajectory

        Args:
            detections: List of FrameDetection objects

        Returns:
            List of (start_idx, end_idx) tuples for each gap
        """
        gaps = []
        in_gap = False
        gap_start = None

        for i, det in enumerate(detections):
            has_ball = det.ball_position_px is not None

            if not has_ball and not in_gap:
                # Start of new gap
                gap_start = i
                in_gap = True
            elif has_ball and in_gap:
                # End of gap
                gaps.append((gap_start, i - 1))
                in_gap = False

        # Handle gap at end
        if in_gap:
            gaps.append((gap_start, len(detections) - 1))

        return gaps

    def _get_surrounding_points(
        self,
        detections: List,
        gap_start: int,
        gap_end: int,
        window: int = 10
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get valid ball positions before and after gap

        Args:
            detections: List of FrameDetection objects
            gap_start: First frame of gap
            gap_end: Last frame of gap
            window: How many frames to look before/after gap

        Returns:
            (frames, positions) arrays or (None, None) if insufficient data
        """
        frames_before = []
        positions_before = []

        frames_after = []
        positions_after = []

        # Look before gap
        for i in range(max(0, gap_start - window), gap_start):
            if detections[i].ball_position_px is not None:
                frames_before.append(i)
                positions_before.append(detections[i].ball_position_px)

        # Look after gap
        for i in range(gap_end + 1, min(len(detections), gap_end + 1 + window)):
            if detections[i].ball_position_px is not None:
                frames_after.append(i)
                positions_after.append(detections[i].ball_position_px)

        # Need at least 1 point on each side for interpolation
        if not frames_before or not frames_after:
            return None, None

        # Combine
        frames = np.array(frames_before + frames_after)
        positions = np.array(positions_before + positions_after)

        return frames, positions

    def _fill_gap_linear(
        self,
        detections: List,
        gap_start: int,
        gap_end: int
    ) -> int:
        """
        Fill gap using linear interpolation

        Args:
            detections: List of FrameDetection objects
            gap_start: First frame of gap
            gap_end: Last frame of gap

        Returns:
            Number of frames filled
        """
        frames, positions = self._get_surrounding_points(
            detections, gap_start, gap_end, window=3
        )

        if frames is None:
            return 0

        # Linear interpolation for x and y separately
        interp_x = interp1d(frames, positions[:, 0], kind='linear', fill_value='extrapolate')
        interp_y = interp1d(frames, positions[:, 1], kind='linear', fill_value='extrapolate')

        filled_count = 0
        for i in range(gap_start, gap_end + 1):
            x = float(interp_x(i))
            y = float(interp_y(i))

            detections[i].ball_position_px = (x, y)
            detections[i].ball_confidence = 0.0  # Mark as interpolated
            filled_count += 1

        return filled_count

    def _fill_gap_polynomial(
        self,
        detections: List,
        gap_start: int,
        gap_end: int
    ) -> int:
        """
        Fill gap using polynomial interpolation

        Args:
            detections: List of FrameDetection objects
            gap_start: First frame of gap
            gap_end: Last frame of gap

        Returns:
            Number of frames filled
        """
        frames, positions = self._get_surrounding_points(
            detections, gap_start, gap_end, window=10
        )

        if frames is None or len(frames) < self.min_points_for_poly:
            # Fall back to linear if not enough points
            return self._fill_gap_linear(detections, gap_start, gap_end)

        # Polynomial interpolation for x and y separately
        try:
            interp_x = interp1d(
                frames, positions[:, 0],
                kind=min(self.poly_order, len(frames) - 1),
                fill_value='extrapolate'
            )
            interp_y = interp1d(
                frames, positions[:, 1],
                kind=min(self.poly_order, len(frames) - 1),
                fill_value='extrapolate'
            )
        except Exception:
            # Fall back to linear if polynomial fails
            return self._fill_gap_linear(detections, gap_start, gap_end)

        filled_count = 0
        for i in range(gap_start, gap_end + 1):
            x = float(interp_x(i))
            y = float(interp_y(i))

            detections[i].ball_position_px = (x, y)
            detections[i].ball_confidence = 0.0  # Mark as interpolated
            filled_count += 1

        return filled_count

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Fill gaps in ball trajectory

        Updates context.detections with interpolated ball positions

        Args:
            context: Processing context with detections

        Returns:
            Updated context
        """
        if not context.detections:
            print("  No detections to process")
            return context

        # Find gaps
        gaps = self._find_gaps(context.detections)

        if not gaps:
            print("  No gaps found in ball trajectory")
            return context

        print(f"  Found {len(gaps)} gap(s) in ball trajectory")

        # Fill each gap
        filled_linear = 0
        filled_poly = 0
        skipped = 0

        for gap_start, gap_end in gaps:
            gap_size = gap_end - gap_start + 1

            if gap_size <= self.max_gap_linear:
                # Small gap - use linear interpolation
                count = self._fill_gap_linear(context.detections, gap_start, gap_end)
                filled_linear += count
                method = "linear"
            elif gap_size <= self.max_gap_poly:
                # Larger gap - use polynomial interpolation
                count = self._fill_gap_polynomial(context.detections, gap_start, gap_end)
                filled_poly += count
                method = "polynomial"
            else:
                # Gap too large - skip
                count = 0
                skipped += gap_size
                method = "skipped (too large)"

            print(f"    Gap frames {gap_start}-{gap_end} (size={gap_size}): {method} ({count} filled)")

        # Summary
        total_filled = filled_linear + filled_poly
        print(f"  Filled {total_filled} frames ({filled_linear} linear, {filled_poly} polynomial)")

        if skipped > 0:
            print(f"  Skipped {skipped} frames (gaps too large)")

        return context
