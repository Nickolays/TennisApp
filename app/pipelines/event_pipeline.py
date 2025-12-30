"""
Event Pipeline - Ball hit, bounce, and in/out detection

File: app/pipelines/event_pipeline.py
"""
from app.core.pipeline import Pipeline
from app.steps.events.bounce_detection import BounceDetectionStep
from app.steps.events.inout_decision import InOutDecisionStep
from app.steps.events.hit_detection import HitDetectionStep


class EventPipeline(Pipeline):
    """
    Event detection pipeline for identifying tennis events.

    This pipeline MUST run AFTER geometry processing (needs velocities and court coordinates).

    Pipeline Steps:
    1. HitDetectionStep - Detect ball hits using velocity spikes
    2. BounceDetectionStep - Detect bounces using velocity flip analysis
    3. InOutDecisionStep - Determine if bounces are in or out

    Configuration Example:
        events:
          hit:
            enabled: true
            acceleration_threshold: 15.0  # m/s²
            min_speed_increase: 3.0       # m/s
            min_frames_between: 8

          bounce:
            enabled: true
            min_vy_flip: 2.0              # m/s
            max_height_threshold: 0.5     # meters
            speed_decrease_ratio: 0.3
            min_frames_between: 5

          inout:
            enabled: true
            line_margin: 0.02             # meters
            court_type: 'auto'            # 'singles', 'doubles', or 'auto'
            check_only_bounces: true

    Usage:
        # Via config
        config = yaml.safe_load(open('configs/default.yaml'))
        event_config = config['events']

        event_pipeline = EventPipeline.from_config(event_config)
        context = event_pipeline.run(context)

        # Manual construction
        event_pipeline = EventPipeline(
            hit_config={'enabled': True, 'acceleration_threshold': 15.0},
            bounce_config={'enabled': True, 'min_vy_flip': 2.0},
            inout_config={'enabled': True, 'court_type': 'auto'}
        )
    """

    def __init__(
        self,
        hit_config: dict = None,
        bounce_config: dict = None,
        inout_config: dict = None
    ):
        """
        Initialize event pipeline

        Args:
            hit_config: Configuration for hit detection step
            bounce_config: Configuration for bounce detection step
            inout_config: Configuration for in/out decision step
        """
        # Default configurations
        hit_config = hit_config or {
            'enabled': True,
            'acceleration_threshold': 15.0,
            'min_speed_increase': 3.0,
            'min_frames_between': 8,
            'use_temporal_windows': False
        }

        bounce_config = bounce_config or {
            'enabled': True,
            'min_vy_flip': 2.0,
            'max_height_threshold': 0.5,
            'speed_decrease_ratio': 0.3,
            'min_frames_between': 5
        }

        inout_config = inout_config or {
            'enabled': True,
            'line_margin': 0.02,
            'court_type': 'auto',
            'check_only_bounces': True
        }

        # Create steps (order matters!)
        # 1. Detect hits first (affects entire rally)
        # 2. Detect bounces (need to know when ball contacts court)
        # 3. Check in/out (only for bounces)
        steps = [
            HitDetectionStep(hit_config),
            BounceDetectionStep(bounce_config),
            InOutDecisionStep(inout_config)
        ]

        # Initialize parent Pipeline
        super().__init__(name="EventPipeline", steps=steps)

    @classmethod
    def from_config(cls, config: dict):
        """
        Create EventPipeline from configuration dictionary

        Args:
            config: Configuration dictionary with keys:
                - hit: dict
                - bounce: dict
                - inout: dict

        Returns:
            EventPipeline instance
        """
        return cls(
            hit_config=config.get('hit', {}),
            bounce_config=config.get('bounce', {}),
            inout_config=config.get('inout', {})
        )
