"""
Velocity Estimation Step - Extract velocities from Kalman filter and compute speeds

File: app/steps/geometry/velocity_estimation.py
"""
import numpy as np
from typing import Optional, Tuple, Dict

from app.steps.base import PipelineStep
from app.core.context import ProcessingContext


class VelocityEstimationStep(PipelineStep):
    """
    Estimate ball velocities and speeds from Kalman filter state.

    Features:
    - Extracts velocity from Kalman filter (already computed in smoothing)
    - Transforms velocity vectors to court coordinates
    - Computes speed magnitude (m/s and km/h)
    - Detects acceleration events (potential hits)
    - Handles missing velocities gracefully

    Configuration:
        geometry:
          velocity:
            enabled: true
            use_kalman: true              # Use Kalman velocities (recommended)
            fallback_to_finite_diff: true # Use finite difference if Kalman not available
            acceleration_threshold: 20.0  # m/s² for hit detection
    """

    def __init__(self, config: dict):
        """
        Initialize velocity estimation step

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        self.use_kalman = config.get('use_kalman', True)
        self.fallback_to_finite_diff = config.get('fallback_to_finite_diff', True)
        self.acceleration_threshold = config.get('acceleration_threshold', 20.0)

        # Store Kalman velocities from smoothing step (if available)
        self.kalman_velocities: Dict[int, Tuple[float, float]] = {}

    def _transform_velocity(
        self,
        velocity_px: Tuple[float, float],
        homography: np.ndarray,
        position_px: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        Transform velocity vector from pixel/s to m/s using homography

        Args:
            velocity_px: Velocity in pixels/second (vx, vy)
            homography: 3x3 homography matrix
            position_px: Current position in pixels (needed for velocity transformation)

        Returns:
            Velocity in m/s (vx, vy) or None if transformation fails
        """
        if homography is None:
            return None

        try:
            # Transform position
            p1 = np.array([position_px[0], position_px[1], 1.0])
            p1_court_h = homography @ p1
            if abs(p1_court_h[2]) < 1e-8:
                return None

            # Transform position + velocity (small time step)
            dt = 0.001  # Small time step for numerical derivative
            p2_px = (position_px[0] + velocity_px[0] * dt,
                    position_px[1] + velocity_px[1] * dt)
            p2 = np.array([p2_px[0], p2_px[1], 1.0])
            p2_court_h = homography @ p2
            if abs(p2_court_h[2]) < 1e-8:
                return None

            # Compute velocity in court coordinates
            p1_court = p1_court_h[:2] / p1_court_h[2]
            p2_court = p2_court_h[:2] / p2_court_h[2]

            velocity_court = (p2_court - p1_court) / dt

            return (float(velocity_court[0]), float(velocity_court[1]))

        except Exception:
            return None

    def _compute_finite_difference_velocity(
        self,
        frame_id: int,
        context: ProcessingContext
    ) -> Optional[Tuple[float, float]]:
        """
        Compute velocity using finite difference (fallback method)

        Args:
            frame_id: Current frame ID
            context: Processing context

        Returns:
            Velocity in pixels/second (vx, vy) or None
        """
        # Find current frame index
        try:
            current_idx = context.frame_ids.index(frame_id)
        except ValueError:
            return None

        # Need previous and next frame for central difference
        if current_idx == 0 or current_idx >= len(context.frame_ids) - 1:
            return None

        prev_det = context.detections[current_idx - 1]
        next_det = context.detections[current_idx + 1]

        # Check both have ball positions
        if prev_det.ball_position_px is None or next_det.ball_position_px is None:
            return None

        # Compute velocity (central difference)
        dt = 2.0 / context.fps  # Time between prev and next frame
        dx = next_det.ball_position_px[0] - prev_det.ball_position_px[0]
        dy = next_det.ball_position_px[1] - prev_det.ball_position_px[1]

        vx = dx / dt
        vy = dy / dt

        return (vx, vy)

    def register_kalman_velocity(
        self,
        frame_id: int,
        velocity_px: Tuple[float, float]
    ):
        """
        Register Kalman velocity for a frame (called by TrajectorySmoothingStep)

        Args:
            frame_id: Frame ID
            velocity_px: Velocity from Kalman filter in pixels/second
        """
        self.kalman_velocities[frame_id] = velocity_px

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Estimate velocities and speeds for all ball states

        Updates context.ball_states with velocity and speed information

        Args:
            context: Processing context with ball_states from CoordinateTransformStep

        Returns:
            Updated context
        """
        if not hasattr(context, 'ball_states') or not context.ball_states:
            print("  No ball states to process (run CoordinateTransformStep first)")
            return context

        # Get Kalman velocities from context (stored by TrajectorySmoothingStep)
        kalman_velocities = getattr(context, 'kalman_velocities', {})

        # Process each ball state
        velocity_count = 0
        failed_count = 0
        speeds = []

        for ball_state in context.ball_states:
            # Get velocity in pixel coordinates
            velocity_px = None

            # Try Kalman first
            if self.use_kalman and ball_state.frame_id in kalman_velocities:
                velocity_px = kalman_velocities[ball_state.frame_id]

            # Fallback to finite difference
            elif self.fallback_to_finite_diff:
                velocity_px = self._compute_finite_difference_velocity(
                    ball_state.frame_id,
                    context
                )

            if velocity_px is None:
                failed_count += 1
                continue

            # Get homography for transformation
            H = context.get_homography_for_frame(ball_state.frame_id)

            if H is None:
                failed_count += 1
                continue

            # Transform velocity to court coordinates
            velocity_court = self._transform_velocity(
                velocity_px,
                H,
                ball_state.position_px
            )

            if velocity_court is None:
                failed_count += 1
                continue

            # Compute speed
            speed_ms = np.sqrt(velocity_court[0]**2 + velocity_court[1]**2)
            speed_kmh = speed_ms * 3.6  # Convert m/s to km/h

            # Update ball state
            ball_state.velocity = velocity_court
            ball_state.speed = float(speed_ms)

            velocity_count += 1
            speeds.append(speed_kmh)

        # Summary statistics
        print(f"  Computed {velocity_count} velocities ({failed_count} failed)")

        if speeds:
            avg_speed = np.mean(speeds)
            max_speed = np.max(speeds)
            min_speed = np.min(speeds)

            print(f"  Speed statistics:")
            print(f"    Average: {avg_speed:.1f} km/h ({avg_speed/3.6:.1f} m/s)")
            print(f"    Max: {max_speed:.1f} km/h ({max_speed/3.6:.1f} m/s)")
            print(f"    Min: {min_speed:.1f} km/h ({min_speed/3.6:.1f} m/s)")

        return context
