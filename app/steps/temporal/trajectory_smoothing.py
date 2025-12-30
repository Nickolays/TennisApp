"""
Trajectory Smoothing Step - Kalman filter for ball trajectory

File: app/steps/temporal/trajectory_smoothing.py
"""
import numpy as np
from typing import List, Optional, Tuple
from filterpy.kalman import KalmanFilter

from app.steps.base import PipelineStep
from app.core.context import ProcessingContext


class TrajectorySmoothingStep(PipelineStep):
    """
    Smooth ball trajectory using Kalman filter.

    Features:
    - Handles missing measurements gracefully
    - Tracks position and velocity
    - Accounts for gravity (constant acceleration model)
    - Only smooths original detections (preserves interpolated positions)
    - Reports smoothing statistics

    Configuration:
        temporal:
          smoothing:
            enabled: true
            process_noise: 0.1      # Process noise (how much we trust the model)
            measurement_noise: 10.0 # Measurement noise (how much we trust detections)
            smooth_interpolated: false  # Whether to smooth interpolated positions
            min_confidence: 0.1     # Only smooth detections above this confidence
    """

    def __init__(self, config: dict):
        """
        Initialize trajectory smoothing step

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        self.process_noise = config.get('process_noise', 0.1)
        self.measurement_noise = config.get('measurement_noise', 10.0)
        self.smooth_interpolated = config.get('smooth_interpolated', False)
        self.min_confidence = config.get('min_confidence', 0.1)

    def _create_kalman_filter(self, fps: float) -> KalmanFilter:
        """
        Create Kalman filter for ball tracking

        State vector: [x, y, vx, vy, ax, ay]
        - (x, y): position
        - (vx, vy): velocity
        - (ax, ay): acceleration (constant, for gravity)

        Args:
            fps: Video frame rate

        Returns:
            Configured Kalman filter
        """
        dt = 1.0 / fps  # Time step between frames

        kf = KalmanFilter(dim_x=6, dim_z=2)

        # State transition matrix (constant acceleration model)
        kf.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],        # x = x + vx*dt + 0.5*ax*dt^2
            [0, 1, 0, dt, 0, 0.5*dt**2],        # y = y + vy*dt + 0.5*ay*dt^2
            [0, 0, 1, 0, dt, 0],                # vx = vx + ax*dt
            [0, 0, 0, 1, 0, dt],                # vy = vy + ay*dt
            [0, 0, 0, 0, 1, 0],                 # ax = ax (constant)
            [0, 0, 0, 0, 0, 1],                 # ay = ay (constant, gravity)
        ])

        # Measurement matrix (we only observe position)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],  # Measure x
            [0, 1, 0, 0, 0, 0],  # Measure y
        ])

        # Process noise covariance (how much we trust the model)
        kf.Q = np.eye(6) * self.process_noise

        # Measurement noise covariance (how much we trust measurements)
        kf.R = np.eye(2) * self.measurement_noise

        # Initial state covariance (high uncertainty)
        kf.P = np.eye(6) * 1000.0

        return kf

    def _initialize_filter(
        self,
        kf: KalmanFilter,
        first_position: Tuple[float, float]
    ):
        """
        Initialize Kalman filter state with first detection

        Args:
            kf: Kalman filter
            first_position: (x, y) of first detection
        """
        x, y = first_position

        # Initial state: [x, y, vx=0, vy=0, ax=0, ay=0]
        kf.x = np.array([x, y, 0.0, 0.0, 0.0, 0.0])

    def _get_trajectory_segments(
        self,
        detections: List
    ) -> List[Tuple[int, int]]:
        """
        Split trajectory into continuous segments (no large gaps)

        We run a separate Kalman filter for each segment to avoid
        filter divergence across large gaps.

        Args:
            detections: List of FrameDetection objects

        Returns:
            List of (start_idx, end_idx) for each segment
        """
        segments = []
        segment_start = None

        for i, det in enumerate(detections):
            has_ball = det.ball_position_px is not None

            if has_ball and segment_start is None:
                # Start new segment
                segment_start = i
            elif not has_ball and segment_start is not None:
                # End segment
                segments.append((segment_start, i - 1))
                segment_start = None

        # Handle segment at end
        if segment_start is not None:
            segments.append((segment_start, len(detections) - 1))

        return segments

    def _smooth_segment(
        self,
        detections: List,
        segment_start: int,
        segment_end: int,
        fps: float,
        velocity_storage: dict = None
    ) -> int:
        """
        Smooth a single trajectory segment using Kalman filter

        Args:
            detections: List of FrameDetection objects
            segment_start: First frame of segment
            segment_end: Last frame of segment
            fps: Video frame rate
            velocity_storage: Optional dict to store velocities (frame_id -> (vx, vy))

        Returns:
            Number of frames smoothed
        """
        # Create and initialize Kalman filter
        kf = self._create_kalman_filter(fps)

        # Find first valid position in segment
        first_pos = None
        for i in range(segment_start, segment_end + 1):
            if detections[i].ball_position_px is not None:
                first_pos = detections[i].ball_position_px
                break

        if first_pos is None:
            return 0

        self._initialize_filter(kf, first_pos)

        # Run filter through segment
        smoothed_count = 0

        for i in range(segment_start, segment_end + 1):
            det = detections[i]

            # Predict step
            kf.predict()

            # Update step (if we have a measurement)
            if det.ball_position_px is not None:
                # Check if we should smooth this detection
                is_original = det.ball_confidence > self.min_confidence
                is_interpolated = det.ball_confidence == 0.0

                if is_original or (is_interpolated and self.smooth_interpolated):
                    # Update filter with measurement
                    z = np.array([det.ball_position_px[0], det.ball_position_px[1]])
                    kf.update(z)

                    # Replace position with filtered estimate
                    det.ball_position_px = (float(kf.x[0]), float(kf.x[1]))

                    # Store velocity from Kalman state (vx, vy from state vector)
                    if velocity_storage is not None:
                        velocity_px = (float(kf.x[2]), float(kf.x[3]))  # vx, vy
                        velocity_storage[det.frame_id] = velocity_px

                    smoothed_count += 1

        return smoothed_count

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Smooth ball trajectory using Kalman filter

        Updates context.detections with smoothed ball positions
        Stores Kalman velocities in context for VelocityEstimationStep

        Args:
            context: Processing context with detections

        Returns:
            Updated context
        """
        if not context.detections:
            print("  No detections to process")
            return context

        # Split into continuous segments
        segments = self._get_trajectory_segments(context.detections)

        if not segments:
            print("  No trajectory segments found")
            return context

        print(f"  Found {len(segments)} trajectory segment(s)")

        # Create velocity storage dict
        velocity_storage = {}

        # Smooth each segment
        total_smoothed = 0

        for seg_idx, (start, end) in enumerate(segments):
            segment_length = end - start + 1
            smoothed = self._smooth_segment(
                context.detections,
                start,
                end,
                context.fps,
                velocity_storage=velocity_storage  # Pass storage dict
            )

            total_smoothed += smoothed

            print(f"    Segment {seg_idx + 1}: frames {start}-{end} (length={segment_length}, smoothed={smoothed})")

        print(f"  Smoothed {total_smoothed} ball positions using Kalman filter")

        # Store velocities in context for later use by VelocityEstimationStep
        context.kalman_velocities = velocity_storage

        return context
