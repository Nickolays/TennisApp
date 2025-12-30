#!/usr/bin/env python3
"""
Phase 4 Test - Event Detection Pipeline (Hit/Bounce/In-Out)

Tests the complete pipeline:
- Phase 1: Detection (Court + Ball)
- Phase 2: Temporal (Gap Filling + Smoothing + Windows)
- Phase 3: Geometry (Homography + Transform + Velocity)
- Phase 4: Events (Hit + Bounce + In/Out)

Usage:
    python test_phase4_events.py
"""
import sys
import time
import cv2

from app.core.context import ProcessingContext
from app.core.data_models import FrameDetection

# Phase 1: Detection
from app.steps.detection.court_detection import CourtDetectionStep
from app.steps.detection.ball_detection import BallDetectionStep

# Phase 2: Temporal
from app.pipelines.temporal_pipeline import TemporalPipeline

# Phase 3: Geometry
from app.pipelines.geometry_pipeline import GeometryPipeline

# Phase 4: Events
from app.pipelines.event_pipeline import EventPipeline


def load_video(video_path: str, max_frames: int = None):
    """Load video and create initial context"""
    print(f"\n{'='*60}")
    print("LOADING VIDEO")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames:
        frame_count = min(frame_count, max_frames)

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames to process: {frame_count}")

    # Read frames
    frames = []
    frame_ids = []

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        frame_ids.append(i)

    cap.release()

    print(f"Loaded {len(frames)} frames")

    # Create context
    context = ProcessingContext(
        video_path=video_path,
        fps=fps,
        total_frames=len(frames),
        width=width,
        height=height,
        frames=frames,
        frame_ids=frame_ids
    )

    return context


def run_phase1_detection(context: ProcessingContext):
    """Phase 1: Detection (Court + Ball)"""
    print(f"\n{'='*60}")
    print("PHASE 1: DETECTION")
    print(f"{'='*60}")

    # Court detection configuration
    court_config = {
        'enabled': True,
        'model_path': 'models/court_model_best.pt',
        'interval': 30,
        'confidence_threshold': 0.5,
        'input_size': [640, 360]
    }

    # Ball detection configuration
    ball_config = {
        'enabled': True,
        'model_path': 'models/ball_model_best.pt',
        'interval': 1,
        'confidence_threshold': 0.3,
        'input_size': [640, 360]
    }

    # Run detection steps
    court_step = CourtDetectionStep(court_config)
    ball_step = BallDetectionStep(ball_config)

    print("\n[1/2] Court Detection")
    start_time = time.time()
    context = court_step.process(context)
    court_time = time.time() - start_time
    print(f"  Time: {court_time:.2f}s")

    print("\n[2/2] Ball Detection")
    start_time = time.time()
    context = ball_step.process(context)
    ball_time = time.time() - start_time
    print(f"  Time: {ball_time:.2f}s")

    total_time = court_time + ball_time
    print(f"\nPhase 1 Total: {total_time:.2f}s")

    return context


def run_phase2_temporal(context: ProcessingContext):
    """Phase 2: Temporal Processing"""
    print(f"\n{'='*60}")
    print("PHASE 2: TEMPORAL PROCESSING")
    print(f"{'='*60}")

    # Temporal pipeline configuration
    temporal_config = {
        'gap_filling': {
            'enabled': True,
            'max_gap_linear': 5,
            'max_gap_poly': 15,
            'poly_order': 2
        },
        'smoothing': {
            'enabled': True,
            'process_noise': 0.1,
            'measurement_noise': 10.0,
            'smooth_interpolated': False
        },
        'window_extraction': {
            'enabled': True,
            'window_size': 5,
            'stride': 1,
            'only_with_ball': True
        }
    }

    # Run temporal pipeline
    temporal_pipeline = TemporalPipeline.from_config(temporal_config)

    start_time = time.time()
    context = temporal_pipeline.run(context)
    total_time = time.time() - start_time

    print(f"\nPhase 2 Total: {total_time:.2f}s")

    return context


def run_phase3_geometry(context: ProcessingContext):
    """Phase 3: Geometry Processing"""
    print(f"\n{'='*60}")
    print("PHASE 3: GEOMETRY PROCESSING")
    print(f"{'='*60}")

    # Geometry pipeline configuration
    geometry_config = {
        'homography': {
            'enabled': True,
            'interval': 30,
            'min_keypoints': 8,
            'ransac_threshold': 5.0,
            'min_inliers': 6
        },
        'transform': {
            'enabled': True,
            'transform_ball': True,
            'transform_players': False
        },
        'velocity': {
            'enabled': True,
            'use_kalman': True,
            'fallback_to_finite_diff': True,
            'acceleration_threshold': 20.0
        }
    }

    # Run geometry pipeline
    geometry_pipeline = GeometryPipeline.from_config(geometry_config)

    start_time = time.time()
    context = geometry_pipeline.run(context)
    total_time = time.time() - start_time

    print(f"\nPhase 3 Total: {total_time:.2f}s")

    return context


def run_phase4_events(context: ProcessingContext):
    """Phase 4: Event Detection"""
    print(f"\n{'='*60}")
    print("PHASE 4: EVENT DETECTION")
    print(f"{'='*60}")

    # Event pipeline configuration
    event_config = {
        'hit': {
            'enabled': True,
            'acceleration_threshold': 15.0,
            'min_speed_increase': 3.0,
            'min_frames_between': 8,
            'use_temporal_windows': False
        },
        'bounce': {
            'enabled': True,
            'min_vy_flip': 2.0,
            'max_height_threshold': 0.5,
            'speed_decrease_ratio': 0.3,
            'min_frames_between': 5
        },
        'inout': {
            'enabled': True,
            'line_margin': 0.02,
            'court_type': 'auto',
            'check_only_bounces': True
        }
    }

    # Run event pipeline
    event_pipeline = EventPipeline.from_config(event_config)

    start_time = time.time()
    context = event_pipeline.run(context)
    total_time = time.time() - start_time

    print(f"\nPhase 4 Total: {total_time:.2f}s")

    return context


def print_summary(context: ProcessingContext):
    """Print final summary"""
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Phase 1: Detection
    print("\n[Phase 1] Detection:")
    print(f"  Court detections: {sum(1 for d in context.detections if d.court_keypoints is not None)}")
    print(f"  Ball detections: {sum(1 for d in context.detections if d.ball_position_px is not None)}")

    # Phase 2: Temporal
    print("\n[Phase 2] Temporal:")
    print(f"  Temporal windows: {len(context.temporal_windows) if hasattr(context, 'temporal_windows') else 0}")

    # Phase 3: Geometry
    print("\n[Phase 3] Geometry:")
    print(f"  Homography matrices: {len(context.homography_cache)}")
    print(f"  Ball states (court coords): {len(context.ball_states) if hasattr(context, 'ball_states') else 0}")

    # Phase 4: Events
    print("\n[Phase 4] Events:")

    if hasattr(context, 'ball_states') and context.ball_states:
        hit_count = sum(1 for s in context.ball_states if s.is_hit)
        bounce_count = sum(1 for s in context.ball_states if s.is_bounce)
        print(f"  Hits detected: {hit_count}")
        print(f"  Bounces detected: {bounce_count}")

        # In/Out decisions
        if hasattr(context, 'inout_decisions'):
            in_count = sum(1 for is_in, _ in context.inout_decisions.values() if is_in)
            out_count = sum(1 for is_in, _ in context.inout_decisions.values() if not is_in)
            print(f"  In/Out decisions: {in_count} in, {out_count} out")

        # Show event timeline
        print("\n  Event Timeline:")
        events = []
        for state in context.ball_states:
            if state.is_hit:
                events.append((state.frame_id, "HIT"))
            if state.is_bounce:
                events.append((state.frame_id, "BOUNCE"))

        # Sort by frame
        events.sort(key=lambda x: x[0])

        # Show first 10 events
        for frame_id, event_type in events[:10]:
            print(f"    Frame {frame_id}: {event_type}")

        if len(events) > 10:
            print(f"    ... ({len(events) - 10} more events)")

    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print(f"{'='*60}\n")


def main():
    """Main test function"""
    video_path = "tests/video3.mp4"
    max_frames = 100

    # Load video
    context = load_video(video_path, max_frames)

    # Run pipelines
    context = run_phase1_detection(context)
    context = run_phase2_temporal(context)
    context = run_phase3_geometry(context)
    context = run_phase4_events(context)

    # Print summary
    print_summary(context)


if __name__ == '__main__':
    main()
