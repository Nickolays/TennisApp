"""
Phase 2 Test - Temporal Pipeline (Gap Filling, Smoothing, Windows)

Usage:
    cd ~/Projects/TennisAnalysis
    source .venv/bin/activate
    python TennisApp/test_phase2_temporal.py
"""
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import torch

from app.core.context import ProcessingContext
from app.core.pipeline import Pipeline
from app.steps.detection.court_detection import CourtDetectionStep
from app.steps.detection.ball_detection import BallDetectionStep
from app.pipelines.temporal_pipeline import TemporalPipeline


def load_test_video(video_path: str, max_frames: int = 100):
    """
    Load test video frames

    Args:
        video_path: Path to video file
        max_frames: Maximum frames to load (for testing)

    Returns:
        (frames, frame_ids, metadata)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo Info:")
    print(f"  Path: {video_path}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Loading: {min(max_frames, total_frames)} frames\n")

    # Load frames
    frames = []
    frame_ids = []

    for i in range(min(max_frames, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_ids.append(i)

    cap.release()

    metadata = {
        'fps': fps,
        'total_frames': total_frames,
        'width': width,
        'height': height,
    }

    return frames, frame_ids, metadata


def simulate_gaps_in_detections(context: ProcessingContext, gap_probability: float = 0.3):
    """
    Randomly remove some ball detections to simulate gaps for testing

    Args:
        context: Processing context
        gap_probability: Probability of removing a detection (0.0 to 1.0)
    """
    print(f"\n[TEST] Simulating gaps (removing {gap_probability*100:.0f}% of detections)")

    original_count = sum(1 for det in context.detections if det.ball_position_px is not None)

    for det in context.detections:
        if det.ball_position_px is not None and np.random.random() < gap_probability:
            det.ball_position_px = None
            det.ball_confidence = 0.0

    remaining_count = sum(1 for det in context.detections if det.ball_position_px is not None)
    removed_count = original_count - remaining_count

    print(f"  Removed: {removed_count}/{original_count} detections ({removed_count/original_count*100:.1f}%)")
    print(f"  Remaining: {remaining_count}/{len(context.detections)} frames ({remaining_count/len(context.detections)*100:.1f}%)")


def test_temporal_pipeline():
    """
    Test Detection + Temporal Pipeline
    """
    print("="*60)
    print("PHASE 2 TEST: Detection + Temporal Pipeline")
    print("="*60)

    # Check for test video
    base_path = Path(__file__).parent
    video_path = base_path / "tests/video3.mp4"

    if not video_path.exists():
        print(f"\n⚠️  Test video not found: {video_path}")
        print("Please ensure video3.mp4 is in tests/ directory")
        return False

    # Load video
    try:
        frames, frame_ids, metadata = load_test_video(str(video_path), max_frames=100)
    except Exception as e:
        print(f"\n❌ Failed to load video: {e}")
        return False

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

    # Configure detection steps
    court_config = {
        'enabled': True,
        'model_path': str(base_path / 'models/court_model_best.pt'),
        'model_type': 'tracknet',
        'interval': 10,
        'confidence_threshold': 0.5,
        'input_size': [640, 360]
    }

    ball_config = {
        'enabled': True,
        'model_path': str(base_path / 'models/ball_model_best.pt'),
        'model_type': 'tracknet',
        'interval': 1,
        'confidence_threshold': 0.3,
        'input_size': [640, 360]
    }

    # Check if models exist
    if not Path(court_config['model_path']).exists():
        print(f"\n⚠️  Court model not found: {court_config['model_path']}")
        return False

    if not Path(ball_config['model_path']).exists():
        print(f"\n⚠️  Ball model not found: {ball_config['model_path']}")
        return False

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}\n")
    else:
        print("⚠️  No GPU available, using CPU (will be slower)\n")

    # ========== PHASE 1: DETECTION ==========
    print("\n" + "="*60)
    print("PHASE 1: DETECTION")
    print("="*60)

    detection_pipeline = Pipeline(
        name="DetectionPipeline",
        steps=[
            CourtDetectionStep(court_config),
            BallDetectionStep(ball_config),
        ]
    )

    try:
        context = detection_pipeline.run(context)
    except Exception as e:
        print(f"\n❌ Detection pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Report detection results
    court_detected = sum(1 for det in context.detections if det.court_keypoints is not None)
    ball_detected_initial = sum(1 for det in context.detections if det.ball_position_px is not None)

    print(f"\nDetection Results:")
    print(f"  Court: {court_detected}/{len(context.detections)} frames ({court_detected/len(context.detections)*100:.1f}%)")
    print(f"  Ball: {ball_detected_initial}/{len(context.detections)} frames ({ball_detected_initial/len(context.detections)*100:.1f}%)")

    # Simulate gaps for testing (optional)
    if ball_detected_initial > 50:
        simulate_gaps_in_detections(context, gap_probability=0.3)

    # ========== PHASE 2: TEMPORAL PROCESSING ==========
    print("\n" + "="*60)
    print("PHASE 2: TEMPORAL PROCESSING")
    print("="*60)

    # Create temporal pipeline
    temporal_pipeline = TemporalPipeline(
        gap_filling_config={
            'enabled': True,
            'max_gap_linear': 5,
            'max_gap_poly': 15,
            'poly_order': 2
        },
        smoothing_config={
            'enabled': True,
            'process_noise': 0.1,
            'measurement_noise': 10.0,
            'smooth_interpolated': False
        },
        window_config={
            'enabled': True,
            'window_size': 5,
            'stride': 1,
            'only_with_ball': True,
            'min_ball_confidence': 0.0
        }
    )

    # Run temporal pipeline
    print("\nRunning Temporal Pipeline...")
    print("-" * 60)

    try:
        context = temporal_pipeline.run(context)
    except Exception as e:
        print(f"\n❌ Temporal pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ========== RESULTS ==========
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    # Ball detection after gap filling
    ball_detected_final = sum(1 for det in context.detections if det.ball_position_px is not None)

    print(f"\nBall Trajectory:")
    print(f"  Before gap filling: {ball_detected_initial}/{len(context.detections)} frames ({ball_detected_initial/len(context.detections)*100:.1f}%)")
    print(f"  After gap filling:  {ball_detected_final}/{len(context.detections)} frames ({ball_detected_final/len(context.detections)*100:.1f}%)")
    print(f"  Gap filling improvement: +{ball_detected_final - ball_detected_initial} frames")

    # Temporal windows
    print(f"\nTemporal Windows:")
    print(f"  Extracted: {len(context.temporal_windows)} windows")
    if context.temporal_windows:
        avg_window_size = np.mean([len(w.frame_ids) for w in context.temporal_windows])
        print(f"  Avg window size: {avg_window_size:.1f} frames")
        print(f"  Coverage: {len(context.temporal_windows)}/{len(context.detections)} frames")

        # Sample windows
        print(f"\n  Sample windows (first 3):")
        for i, window in enumerate(context.temporal_windows[:3]):
            ball_count = sum(1 for pos in window.ball_positions if pos is not None)
            print(f"    Window {i+1}: center={window.center_frame_id}, frames={len(window.frame_ids)}, ball_count={ball_count}/{len(window.ball_positions)}")

    # Performance
    print(f"\nPerformance:")
    for step_name, duration in context.step_timings.items():
        print(f"  {step_name}: {duration:.2f}s")
    print(f"  Total: {context.get_total_processing_time():.2f}s")

    fps_processed = len(context.frames) / context.get_total_processing_time()
    print(f"  Processing FPS: {fps_processed:.1f}")

    # ========== VALIDATION ==========
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)

    success = True

    # Check gap filling worked
    if ball_detected_final <= ball_detected_initial:
        print("⚠️  Gap filling did not add any frames (may be expected if no gaps)")
    else:
        print(f"✓ Gap filling added {ball_detected_final - ball_detected_initial} frames")

    # Check temporal windows
    if len(context.temporal_windows) == 0:
        print("❌ No temporal windows extracted!")
        success = False
    else:
        print(f"✓ {len(context.temporal_windows)} temporal windows extracted")

    print()

    if success:
        print("🎉 Phase 2 Temporal Pipeline WORKS!\n")
        print("Next Steps:")
        print("  1. Add PlayerDetectionStep (YOLO)")
        print("  2. Create GeometryPipeline (homography)")
        print("  3. Create EventPipeline (hit/bounce detection)")
        print()
    else:
        print("❌ Temporal pipeline needs debugging\n")

    return success


if __name__ == "__main__":
    success = test_temporal_pipeline()
    sys.exit(0 if success else 1)
