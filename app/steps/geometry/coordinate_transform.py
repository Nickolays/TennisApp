"""
Coordinate Transform Step - Transform pixel coordinates to court coordinates

File: app/steps/geometry/coordinate_transform.py
"""
import numpy as np
from typing import Optional, Tuple

from app.steps.base import PipelineStep
from app.core.context import ProcessingContext
from app.core.data_models import BallState


class CoordinateTransformStep(PipelineStep):
    """
    Transform ball positions from pixel to court coordinates.

    Features:
    - Transforms ball positions using cached homography matrices
    - Uses nearest homography for frames without direct computation
    - Handles edge cases (no homography, ball outside court)
    - Stores results in context.ball_trajectories

    Configuration:
        geometry:
          transform:
            enabled: true
            transform_ball: true
            transform_players: false  # Future: transform player positions
    """

    def __init__(self, config: dict):
        """
        Initialize coordinate transform step

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        self.transform_ball = config.get('transform_ball', True)
        self.transform_players = config.get('transform_players', False)

    def _transform_point(
        self,
        point_px: Tuple[float, float],
        homography: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """
        Transform a single point from pixel to court coordinates

        Args:
            point_px: Point in pixel coordinates (x, y)
            homography: 3x3 homography matrix

        Returns:
            Point in court coordinates (x, y) in meters, or None if transform fails
        """
        if homography is None:
            return None

        try:
            # Convert to homogeneous coordinates
            point_h = np.array([point_px[0], point_px[1], 1.0])

            # Apply homography
            transformed_h = homography @ point_h

            # Convert back to cartesian
            if abs(transformed_h[2]) < 1e-8:
                return None

            x_court = transformed_h[0] / transformed_h[2]
            y_court = transformed_h[1] / transformed_h[2]

            return (float(x_court), float(y_court))

        except Exception as e:
            return None

    def _is_valid_court_position(
        self,
        position_court: Tuple[float, float],
        margin: float = 2.0
    ) -> bool:
        """
        Check if court position is reasonable (inside court + margin)

        Args:
            position_court: Position in court coordinates (x, y) meters
            margin: Margin around court (meters)

        Returns:
            True if position is reasonable
        """
        x, y = position_court

        # Tennis court dimensions (from COURT_DIMENSIONS)
        court_width = 10.97  # meters
        court_length = 23.77  # meters

        # Check if within court bounds + margin
        if x < -margin or x > court_width + margin:
            return False

        if y < -margin or y > court_length + margin:
            return False

        return True

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Transform ball positions to court coordinates

        Updates context.detections with court coordinates and creates BallState objects

        Args:
            context: Processing context with detections and homography cache

        Returns:
            Updated context
        """
        if not context.detections:
            print("  No detections to process")
            return context

        if not context.homography_cache:
            print("  No homography matrices available!")
            return context

        # Transform ball positions
        if self.transform_ball:
            transformed_count = 0
            failed_count = 0
            ball_states = []

            for det in context.detections:
                # Skip if no ball
                if det.ball_position_px is None:
                    continue

                # Get homography for this frame
                H = context.get_homography_for_frame(det.frame_id)

                if H is None:
                    failed_count += 1
                    continue

                # Transform position
                position_court = self._transform_point(det.ball_position_px, H)

                if position_court is None:
                    failed_count += 1
                    continue

                # Validate position
                if not self._is_valid_court_position(position_court, margin=3.0):
                    # Position is far outside court - likely error, but keep it
                    # (ball can go out of bounds in real game)
                    pass

                # Create BallState
                ball_state = BallState(
                    frame_id=det.frame_id,
                    position_px=det.ball_position_px,
                    position_court=position_court,
                    velocity=None,  # Will be filled by VelocityEstimationStep
                    speed=None,
                    is_bounce=False,
                    is_hit=False
                )

                ball_states.append(ball_state)
                transformed_count += 1

            # Store ball states in context (will be used by velocity step)
            context.ball_states = ball_states

            print(f"  Transformed {transformed_count} ball positions to court coordinates")
            if failed_count > 0:
                print(f"  Failed: {failed_count} transformations")

            # Sample positions
            if transformed_count > 0:
                sample = ball_states[0]
                print(f"  Sample: Frame {sample.frame_id}")
                print(f"    Pixel: ({sample.position_px[0]:.1f}, {sample.position_px[1]:.1f})")
                print(f"    Court: ({sample.position_court[0]:.2f}, {sample.position_court[1]:.2f}) meters")

        return context
