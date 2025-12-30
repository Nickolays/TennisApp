"""
Temporal Pipeline - Gap filling, smoothing, and window extraction

File: app/pipelines/temporal_pipeline.py
"""
from app.core.pipeline import Pipeline
from app.steps.temporal.gap_filling import GapFillingStep
from app.steps.temporal.trajectory_smoothing import TrajectorySmoothingStep
from app.steps.temporal.window_extractor import TemporalWindowExtractorStep


class TemporalPipeline(Pipeline):
    """
    Temporal processing pipeline for ball trajectory.

    This pipeline MUST run AFTER detection but BEFORE geometry to handle
    missing ball positions (NaNs) that would break homography computation.

    Pipeline Steps:
    1. GapFillingStep - Interpolate missing ball positions
    2. TrajectorySmoothingStep - Apply Kalman filter for smoothing
    3. TemporalWindowExtractorStep - Extract ±N frame windows for hit detection

    Configuration Example:
        temporal:
          gap_filling:
            enabled: true
            max_gap_linear: 5
            max_gap_poly: 15
            poly_order: 2

          smoothing:
            enabled: true
            process_noise: 0.1
            measurement_noise: 10.0
            smooth_interpolated: false

          window_extraction:
            enabled: true
            window_size: 5        # ±5 frames = 11 total
            stride: 1
            only_with_ball: true

    Usage:
        # Via config
        config = yaml.safe_load(open('configs/default.yaml'))
        temporal_config = config['temporal']

        temporal_pipeline = TemporalPipeline.from_config(temporal_config)
        context = temporal_pipeline.run(context)

        # Manual construction
        temporal_pipeline = TemporalPipeline(
            gap_filling_config={'enabled': True, 'max_gap_linear': 5},
            smoothing_config={'enabled': True, 'process_noise': 0.1},
            window_config={'enabled': True, 'window_size': 5}
        )
    """

    def __init__(
        self,
        gap_filling_config: dict = None,
        smoothing_config: dict = None,
        window_config: dict = None
    ):
        """
        Initialize temporal pipeline

        Args:
            gap_filling_config: Configuration for gap filling step
            smoothing_config: Configuration for smoothing step
            window_config: Configuration for window extraction step
        """
        # Default configurations
        gap_filling_config = gap_filling_config or {
            'enabled': True,
            'max_gap_linear': 5,
            'max_gap_poly': 15,
            'poly_order': 2,
            'min_points_for_poly': 4
        }

        smoothing_config = smoothing_config or {
            'enabled': True,
            'process_noise': 0.1,
            'measurement_noise': 10.0,
            'smooth_interpolated': False,
            'min_confidence': 0.1
        }

        window_config = window_config or {
            'enabled': True,
            'window_size': 5,
            'stride': 1,
            'only_with_ball': True,
            'min_ball_confidence': 0.1
        }

        # Create steps
        steps = [
            GapFillingStep(gap_filling_config),
            TrajectorySmoothingStep(smoothing_config),
            TemporalWindowExtractorStep(window_config)
        ]

        # Initialize parent Pipeline
        super().__init__(name="TemporalPipeline", steps=steps)

    @classmethod
    def from_config(cls, config: dict):
        """
        Create TemporalPipeline from configuration dictionary

        Args:
            config: Configuration dictionary with keys:
                - gap_filling: dict
                - smoothing: dict
                - window_extraction: dict

        Returns:
            TemporalPipeline instance
        """
        return cls(
            gap_filling_config=config.get('gap_filling', {}),
            smoothing_config=config.get('smoothing', {}),
            window_config=config.get('window_extraction', {})
        )
