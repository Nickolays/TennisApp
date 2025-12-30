#!/usr/bin/env python3
"""
Phase 3 Test - Geometry Pipeline (Homography + Transform + Velocity)

Usage:
    cd ~/Projects/TennisAnalysis
    source .venv/bin/activate
    python TennisApp/test_phase3_geometry.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import torch

from app.core.context import ProcessingContext
from app.core.pipeline import Pipeline
from app.steps.detection.court_detection import CourtDetectionStep
from app.steps.detection.ball_detection import BallDetectionStep
from app.pipelines.temporal_pipeline import TemporalPipeline
from app.pipelines.geometry_pipeline import GeometryPipeline
from app.core.data_models import FrameDetection


def load_test_video(video_path: str, max_frames: int = 100):
    """Load test video frames"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo Info:")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Total frames: {total_frames}")
    print(f"  Loading: {min(max_frames, total_frames)} frames\n")

    frames = []
    frame_ids = []

    for i in range(min(max_frames, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_ids.append(i)

    cap.release()

    metadata = {'fps': fps, 'total_frames': total_frames, 'width': width, 'height': height}
    return frames, frame_ids, metadata


def test_geometry_pipeline():
    """Test Detection + Temporal + Geometry Pipeline"""
    print("="*80)
    print("PHASE 3 TEST: Detection + Temporal + Geometry Pipeline")
    print("="*80)

    base_path = Path(__file__).parent
    video_path = base_path / "tests/video3.mp4"

    if not video_path.exists():
        print(f"\n⚠️  Test video not found: {video_path}")
        return False

    # Load video
    frames, frame_ids, metadata = load_test_video(str(video_path), max_frames=100)

    # Create context
    context = ProcessingContext(
        video_path=str(video_path),
        fps=metadata['fps'],
        total_frames=metadata['total_frames'],
        width=metadata['width'],
        height=metadata['height'],
        duration_seconds=metadata['total_frames'] / metadata['fps']
    )

    context.frames = frames
    context.frame_ids = frame_ids
    context.chunk_start = 0
    context.chunk_end = len(frames)
    context.detections = [FrameDetection(frame_id=fid, timestamp=fid / metadata['fps']) for fid in frame_ids]

    # Check models
    court_model = base_path / 'models/court_model_best.pt'
    ball_model = base_path / 'models/ball_model_best.pt'

    if not court_model.exists() or not ball_model.exists():
        print(f"\n⚠️  Models not found")
        return False

    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}\n")
    else:
        print("⚠️  Using CPU\n")

    # ========== PHASE 1: DETECTION ==========
    print("="*80)
    print("PHASE 1: DETECTION")
    print("="*80 + "\n")

    detection_pipeline = Pipeline(
        name="DetectionPipeline",
        steps=[
            CourtDetectionStep({
                'enabled': True,
                'model_path': str(court_model),
                'model_type': 'tracknet',
                'interval': 30,
                'confidence_threshold': 0.5,
                'input_size': [640, 360]
            }),
            BallDetectionStep({
                'enabled': True,
                'model_path': str(ball_model),
                'model_type': 'tracknet',
                'interval': 1,
                'confidence_threshold': 0.3,
                'input_size': [640, 360]
            }),
        ]
    )

    context = detection_pipeline.run(context)

    court_detected = sum(1 for d in context.detections if d.court_keypoints is not None)
    ball_detected = sum(1 for d in context.detections if d.ball_position_px is not None)

    print(f"\nDetection Results:")
    print(f"  Court: {court_detected}/{len(context.detections)} frames")
    print(f"  Ball: {ball_detected}/{len(context.detections)} frames")

    # ========== PHASE 2: TEMPORAL ==========
    print(f"\n{'='*80}")
    print("PHASE 2: TEMPORAL PROCESSING")
    print("="*80 + "\n")

    temporal_pipeline = TemporalPipeline(
        gap_filling_config={'enabled': True, 'max_gap_linear': 5, 'max_gap_poly': 15},
        smoothing_config={'enabled': True, 'process_noise': 0.1, 'measurement_noise': 10.0},
        window_config={'enabled': True, 'window_size': 5, 'stride': 1}
    )

    context = temporal_pipeline.run(context)

    # ========== PHASE 3: GEOMETRY ==========
    print(f"\n{'='*80}")
    print("PHASE 3: GEOMETRY PROCESSING")
    print("="*80 + "\n")

    geometry_pipeline = GeometryPipeline(
        homography_config={'enabled': True, 'interval': 30, 'min_inliers': 4, 'ransac_threshold': 10.0},
        transform_config={'enabled': True, 'transform_ball': True},
        velocity_config={'enabled': True, 'use_kalman': True}
    )

    context = geometry_pipeline.run(context)

    # ========== RESULTS ==========
    print(f"\n{'='*80}")
    print("RESULTS")
    print("="*80)

    print(f"\nHomography:")
    print(f"  Cached frames: {len(context.homography_cache)}")

    print(f"\nBall States:")
    if hasattr(context, 'ball_states'):
        print(f"  Total: {len(context.ball_states)}")
        states_with_velocity = sum(1 for s in context.ball_states if s.velocity is not None)
        print(f"  With velocity: {states_with_velocity}/{len(context.ball_states)}")

        if context.ball_states:
            sample = context.ball_states[0]
            print(f"\n  Sample (Frame {sample.frame_id}):")
            print(f"    Pixel: ({sample.position_px[0]:.1f}, {sample.position_px[1]:.1f})")
            if sample.position_court:
                print(f"    Court: ({sample.position_court[0]:.2f}, {sample.position_court[1]:.2f}) meters")
            if sample.velocity:
                print(f"    Velocity: ({sample.velocity[0]:.2f}, {sample.velocity[1]:.2f}) m/s")
            if sample.speed:
                print(f"    Speed: {sample.speed:.2f} m/s ({sample.speed*3.6:.1f} km/h)")

    print(f"\nPerformance:")
    for step_name, duration in context.step_timings.items():
        print(f"  {step_name}: {duration:.2f}s")
    total = context.get_total_processing_time()
    print(f"  Total: {total:.2f}s ({len(context.frames)/total:.1f} FPS)")

    # ========== VALIDATION ==========
    print(f"\n{'='*80}")
    print("VALIDATION")
    print("="*80)

    success = True

    if len(context.homography_cache) == 0:
        print("❌ No homographies computed!")
        success = False
    else:
        print(f"✓ {len(context.homography_cache)} homographies computed")

    if not hasattr(context, 'ball_states') or len(context.ball_states) == 0:
        print("❌ No ball states created!")
        success = False
    else:
        print(f"✓ {len(context.ball_states)} ball states created")

    states_with_velocity = sum(1 for s in context.ball_states if s.velocity is not None)
    if states_with_velocity == 0:
        print("❌ No velocities computed!")
        success = False
    else:
        print(f"✓ {states_with_velocity} velocities computed")

    print()

    if success:
        print("🎉 Phase 3 Geometry Pipeline WORKS!\n")
        print("Next Steps:")
        print("  1. Add EventPipeline (hit/bounce detection)")
        print("  2. Add AnalyticsPipeline (rally stats)")
        print("  3. Add enhanced visualization")
        print()
    else:
        print("❌ Geometry pipeline needs debugging\n")

    return success


if __name__ == "__main__":
    success = test_geometry_pipeline()
    sys.exit(0 if success else 1)
