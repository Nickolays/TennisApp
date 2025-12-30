"""
Hit Detection Step - Detect ball hits using velocity spike analysis

File: app/steps/events/hit_detection.py
"""
import numpy as np
from typing import List, Tuple, Optional

from app.steps.base import PipelineStep
from app.core.context import ProcessingContext
from app.core.data_models import BallState


class HitDetectionStep(PipelineStep):
    """
    Detect ball hits using velocity spike analysis.

    Simple Physics-Based Method (No ML):
    - When player hits ball, velocity changes dramatically
    - Speed increases suddenly (acceleration spike)
    - Direction may change (velocity vector change)

    Detection Strategy:
    1. Compute acceleration from velocity changes
    2. Find frames with high acceleration (spikes)
    3. Filter false positives with temporal constraints
    4. Mark hits in ball states

    Future Enhancement:
    - Add ML model (CatBoost or Logistic Regression)
    - Train on features: acceleration, speed change, direction change, ball position
    - Compare with simple method and keep better one

    Configuration:
        events:
          hit:
            enabled: true
            acceleration_threshold: 15.0  # m/s² threshold for hit
            min_speed_increase: 3.0       # Minimum speed increase (m/s)
            min_frames_between: 8         # Minimum frames between hits
            use_temporal_windows: false   # Use extracted windows (future)
    """

    def __init__(self, config: dict):
        """
        Initialize hit detection step

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        self.acceleration_threshold = config.get('acceleration_threshold', 15.0)
        self.min_speed_increase = config.get('min_speed_increase', 3.0)
        self.min_frames_between = config.get('min_frames_between', 8)
        self.use_temporal_windows = config.get('use_temporal_windows', False)

    def _compute_acceleration(
        self,
        ball_states: List[BallState],
        idx: int,
        fps: float
    ) -> Optional[float]:
        """
        Compute acceleration magnitude at given frame

        Args:
            ball_states: List of ball states
            idx: Current index
            fps: Video frame rate

        Returns:
            Acceleration magnitude (m/s²) or None
        """
        # Need at least 2 frames before and after
        if idx < 2 or idx >= len(ball_states) - 2:
            return None

        prev = ball_states[idx - 2]
        next_state = ball_states[idx + 2]

        # Check both have velocities
        if prev.velocity is None or next_state.velocity is None:
            return None

        # Compute acceleration (central difference)
        dt = 4.0 / fps  # Time between prev and next (4 frames)

        dvx = next_state.velocity[0] - prev.velocity[0]
        dvy = next_state.velocity[1] - prev.velocity[1]

        ax = dvx / dt
        ay = dvy / dt

        # Acceleration magnitude
        acceleration = np.sqrt(ax**2 + ay**2)

        return float(acceleration)

    def _compute_speed_change(
        self,
        ball_states: List[BallState],
        idx: int
    ) -> Optional[float]:
        """
        Compute speed change at given frame

        Args:
            ball_states: List of ball states
            idx: Current index

        Returns:
            Speed change (m/s) or None
        """
        if idx < 2 or idx >= len(ball_states) - 2:
            return None

        prev = ball_states[idx - 2]
        next_state = ball_states[idx + 2]

        # Check both have speeds
        if prev.speed is None or next_state.speed is None:
            return None

        # Speed increase
        speed_change = next_state.speed - prev.speed

        return float(speed_change)

    def _filter_duplicate_hits(
        self,
        hit_candidates: List[int],
        ball_states: List[BallState]
    ) -> List[int]:
        """
        Filter out duplicate hit detections (too close together)

        Args:
            hit_candidates: List of frame indices with potential hits
            ball_states: List of ball states

        Returns:
            Filtered list of hit frame indices
        """
        if not hit_candidates:
            return []

        filtered = []
        last_hit_frame = -self.min_frames_between - 1

        for frame_idx in sorted(hit_candidates):
            ball_state = ball_states[frame_idx]
            frame_id = ball_state.frame_id

            # Check minimum spacing
            if frame_id - last_hit_frame >= self.min_frames_between:
                filtered.append(frame_idx)
                last_hit_frame = frame_id

        return filtered

    def _detect_hits_simple(
        self,
        ball_states: List[BallState],
        fps: float
    ) -> List[int]:
        """
        Detect hits using simple velocity spike method

        Args:
            ball_states: List of ball states
            fps: Video frame rate

        Returns:
            List of frame indices with detected hits
        """
        hit_candidates = []

        for idx in range(len(ball_states)):
            # Compute acceleration
            acceleration = self._compute_acceleration(ball_states, idx, fps)

            if acceleration is None:
                continue

            # Check acceleration threshold
            if acceleration < self.acceleration_threshold:
                continue

            # Compute speed change
            speed_change = self._compute_speed_change(ball_states, idx)

            if speed_change is None:
                continue

            # Check speed increase
            if speed_change < self.min_speed_increase:
                continue

            # Add to candidates
            hit_candidates.append(idx)

        # Filter duplicates
        filtered_hits = self._filter_duplicate_hits(hit_candidates, ball_states)

        return filtered_hits

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Detect ball hits using velocity spike analysis

        Updates context.ball_states with is_hit flag

        Args:
            context: Processing context with ball_states from VelocityEstimationStep

        Returns:
            Updated context
        """
        if not hasattr(context, 'ball_states') or not context.ball_states:
            print("  No ball states to process (run VelocityEstimationStep first)")
            return context

        ball_states = context.ball_states
        fps = context.fps

        # Detect hits using simple method
        hit_indices = self._detect_hits_simple(ball_states, fps)

        # Mark hits in ball states
        hit_count = 0
        for idx in hit_indices:
            ball_states[idx].is_hit = True
            hit_count += 1

        # Summary
        print(f"  Detected {hit_count} hits using velocity spike method")

        if hit_count > 0:
            hit_frames = [ball_states[i].frame_id for i in hit_indices]
            print(f"  Hit frames: {hit_frames}")

            # Show acceleration values
            accelerations = []
            for idx in hit_indices:
                acc = self._compute_acceleration(ball_states, idx, fps)
                if acc is not None:
                    accelerations.append(acc)

            if accelerations:
                print(f"  Acceleration range: {min(accelerations):.1f} - {max(accelerations):.1f} m/s²")

        return context
