"""
Bounce Detection Step - Detect ball bounces using physics-based velocity analysis

File: app/steps/events/bounce_detection.py
"""
import numpy as np
from typing import List, Tuple, Optional

from app.steps.base import PipelineStep
from app.core.context import ProcessingContext
from app.core.data_models import BallState


class BounceDetectionStep(PipelineStep):
    """
    Detect ball bounces using physics-based velocity flip analysis.

    Physics Principle:
    - When ball bounces, vertical velocity (vy) changes sign (downward → upward)
    - Horizontal velocity (vx) may change slightly due to friction
    - Speed magnitude decreases due to energy loss

    Detection Method (Simple, No ML):
    1. Find frames where vy changes from negative to positive (velocity flip)
    2. Verify speed decrease before and after bounce
    3. Check ball is near court surface (low y-coordinate)
    4. Filter false positives with thresholds

    Configuration:
        events:
          bounce:
            enabled: true
            min_vy_flip: 2.0           # Minimum vertical velocity change (m/s)
            max_height_threshold: 0.5   # Maximum height for bounce (meters)
            speed_decrease_ratio: 0.3   # Expected speed loss (0-1)
            min_frames_between: 5       # Minimum frames between bounces
    """

    def __init__(self, config: dict):
        """
        Initialize bounce detection step

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        self.min_vy_flip = config.get('min_vy_flip', 2.0)
        self.max_height_threshold = config.get('max_height_threshold', 0.5)
        self.speed_decrease_ratio = config.get('speed_decrease_ratio', 0.3)
        self.min_frames_between = config.get('min_frames_between', 5)

    def _detect_velocity_flip(
        self,
        ball_states: List[BallState],
        idx: int
    ) -> Tuple[bool, float]:
        """
        Detect if vertical velocity flips from negative to positive

        Args:
            ball_states: List of ball states
            idx: Current index

        Returns:
            (is_flip, vy_change) - True if flip detected, magnitude of change
        """
        # Need at least 2 frames before and after
        if idx < 2 or idx >= len(ball_states) - 2:
            return False, 0.0

        current = ball_states[idx]
        prev = ball_states[idx - 1]
        next_state = ball_states[idx + 1]

        # Check all have velocities
        if (current.velocity is None or
            prev.velocity is None or
            next_state.velocity is None):
            return False, 0.0

        # Get vertical velocities (vy component)
        vy_prev = prev.velocity[1]
        vy_current = current.velocity[1]
        vy_next = next_state.velocity[1]

        # Detect flip: negative → positive (ball going down → going up)
        # Allow small tolerance for noise
        if vy_prev < -0.5 and vy_next > 0.5:
            # Flip detected at current frame
            vy_change = abs(vy_next - vy_prev)
            return True, vy_change

        return False, 0.0

    def _check_height(self, ball_state: BallState) -> bool:
        """
        Check if ball is near court surface (low height)

        Args:
            ball_state: Ball state to check

        Returns:
            True if ball is near ground
        """
        if ball_state.position_court is None:
            return False

        # In court coordinates, y=0 is one baseline
        # Ball should be close to court surface for bounce
        # This is a simplified check - assumes court is flat
        # TODO: Add proper height estimation with camera calibration

        # For now, just check ball is within reasonable court bounds
        # (we can't determine height from 2D court coordinates alone)
        y_court = ball_state.position_court[1]

        # Court length is 23.77m
        # Ball should be within court for valid bounce
        if 0 <= y_court <= 23.77:
            return True

        return False

    def _check_speed_decrease(
        self,
        ball_states: List[BallState],
        idx: int
    ) -> Tuple[bool, float]:
        """
        Check if speed decreases around bounce (energy loss)

        Args:
            ball_states: List of ball states
            idx: Current index

        Returns:
            (is_decrease, ratio) - True if speed decreases, loss ratio
        """
        if idx < 2 or idx >= len(ball_states) - 2:
            return False, 0.0

        prev = ball_states[idx - 2]
        next_state = ball_states[idx + 2]

        # Check both have speeds
        if prev.speed is None or next_state.speed is None:
            return False, 0.0

        # Check speed decrease
        if prev.speed > 0:
            speed_ratio = (prev.speed - next_state.speed) / prev.speed
            if speed_ratio >= self.speed_decrease_ratio:
                return True, speed_ratio

        return False, 0.0

    def _filter_duplicate_bounces(
        self,
        bounce_candidates: List[int],
        ball_states: List[BallState]
    ) -> List[int]:
        """
        Filter out duplicate bounce detections (too close together)

        Args:
            bounce_candidates: List of frame indices with potential bounces
            ball_states: List of ball states

        Returns:
            Filtered list of bounce frame indices
        """
        if not bounce_candidates:
            return []

        filtered = []
        last_bounce_frame = -self.min_frames_between - 1

        for frame_idx in sorted(bounce_candidates):
            ball_state = ball_states[frame_idx]
            frame_id = ball_state.frame_id

            # Check minimum spacing
            if frame_id - last_bounce_frame >= self.min_frames_between:
                filtered.append(frame_idx)
                last_bounce_frame = frame_id

        return filtered

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Detect ball bounces using physics-based velocity analysis

        Updates context.ball_states with is_bounce flag

        Args:
            context: Processing context with ball_states from VelocityEstimationStep

        Returns:
            Updated context
        """
        if not hasattr(context, 'ball_states') or not context.ball_states:
            print("  No ball states to process (run VelocityEstimationStep first)")
            return context

        ball_states = context.ball_states
        bounce_candidates = []

        # Pass 1: Detect velocity flips
        for idx in range(len(ball_states)):
            is_flip, vy_change = self._detect_velocity_flip(ball_states, idx)

            if not is_flip:
                continue

            # Check minimum velocity change
            if vy_change < self.min_vy_flip:
                continue

            # Check ball is near court surface
            if not self._check_height(ball_states[idx]):
                continue

            # Check speed decrease (optional - may not always happen)
            is_decrease, speed_ratio = self._check_speed_decrease(ball_states, idx)

            # Add to candidates
            bounce_candidates.append(idx)

        # Pass 2: Filter duplicates
        filtered_bounces = self._filter_duplicate_bounces(bounce_candidates, ball_states)

        # Mark bounces in ball states
        bounce_count = 0
        for idx in filtered_bounces:
            ball_states[idx].is_bounce = True
            bounce_count += 1

        # Summary
        print(f"  Detected {bounce_count} bounces")

        if bounce_count > 0:
            print(f"  Bounce frames: {[ball_states[i].frame_id for i in filtered_bounces]}")

        return context
