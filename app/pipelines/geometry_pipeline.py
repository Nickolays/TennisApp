"""
Geometry Pipeline - Homography, coordinate transform, and velocity estimation

File: app/pipelines/geometry_pipeline.py
"""
from app.core.pipeline import Pipeline
from app.steps.geometry.homography_estimation import HomographyEstimationStep
from app.steps.geometry.coordinate_transform import CoordinateTransformStep
from app.steps.geometry.velocity_estimation import VelocityEstimationStep


class GeometryPipeline(Pipeline):
    """
    Geometry pipeline for transforming pixel coordinates to court coordinates.

    This pipeline MUST run AFTER temporal processing (to have smooth positions)
    and BEFORE event detection (which needs court coordinates and velocities).

    Pipeline Steps:
    1. HomographyEstimationStep - Compute homography matrices from court keypoints
    2. CoordinateTransformStep - Transform ball positions to court coordinates
    3. VelocityEstimationStep - Extract velocities from Kalman filter and compute speeds

    Configuration Example:
        geometry:
          homography:
            enabled: true
            interval: 30              # Compute every 30 frames
            min_keypoints: 8
            ransac_threshold: 5.0

          transform:
            enabled: true
            transform_ball: true
            transform_players: false

          velocity:
            enabled: true
            use_kalman: true          # Use Kalman velocities (recommended)
            fallback_to_finite_diff: true

    Usage:
        # Via config
        config = yaml.safe_load(open('configs/default.yaml'))
        geometry_config = config['geometry']

        geometry_pipeline = GeometryPipeline.from_config(geometry_config)
        context = geometry_pipeline.run(context)

        # Manual construction
        geometry_pipeline = GeometryPipeline(
            homography_config={'enabled': True, 'interval': 30},
            transform_config={'enabled': True, 'transform_ball': True},
            velocity_config={'enabled': True, 'use_kalman': True}
        )
    """

    def __init__(
        self,
        homography_config: dict = None,
        transform_config: dict = None,
        velocity_config: dict = None
    ):
        """
        Initialize geometry pipeline

        Args:
            homography_config: Configuration for homography estimation step
            transform_config: Configuration for coordinate transform step
            velocity_config: Configuration for velocity estimation step
        """
        # Default configurations
        homography_config = homography_config or {
            'enabled': True,
            'interval': 30,
            'min_keypoints': 8,
            'ransac_threshold': 5.0,
            'min_inliers': 6
        }

        transform_config = transform_config or {
            'enabled': True,
            'transform_ball': True,
            'transform_players': False
        }

        velocity_config = velocity_config or {
            'enabled': True,
            'use_kalman': True,
            'fallback_to_finite_diff': True,
            'acceleration_threshold': 20.0
        }

        # Create steps
        steps = [
            HomographyEstimationStep(homography_config),
            CoordinateTransformStep(transform_config),
            VelocityEstimationStep(velocity_config)
        ]

        # Initialize parent Pipeline
        super().__init__(name="GeometryPipeline", steps=steps)

    @classmethod
    def from_config(cls, config: dict):
        """
        Create GeometryPipeline from configuration dictionary

        Args:
            config: Configuration dictionary with keys:
                - homography: dict
                - transform: dict
                - velocity: dict

        Returns:
            GeometryPipeline instance
        """
        return cls(
            homography_config=config.get('homography', {}),
            transform_config=config.get('transform', {}),
            velocity_config=config.get('velocity', {})
        )
