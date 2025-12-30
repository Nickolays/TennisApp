"""
Phase 1 Live Test - Test with real video

Usage:
    cd ~/Projects/TennisAnalysis
    source .venv/bin/activate
    python TennisApp/test_phase1_live.py
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


def load_test_video(video_path: str, max_frames: int = 100):
    """
    Load test video frames

    Args:
        video_path: Path to video file
        max_frames: Maximum frames to load (for testing)

    Returns:
        (frames, metadata)
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


def test_detection_pipeline():
    """
    Test CourtDetection + BallDetection pipeline
    """
    print("="*60)
    print("PHASE 1 LIVE TEST: Detection Pipeline")
    print("="*60)

    # Check for test videos (use absolute paths)
    base_path = Path(__file__).parent
    test_videos = [
        base_path / "tests/video1.mp4",
        base_path / "tests/video2.mp4",
        base_path / "tests/video3.mp4",
    ]

    video_path = None
    for path in test_videos:
        if path.exists():
            video_path = str(path)
            break

    if not video_path:
        print("\n⚠️  No test video found!")
        print("Looking for:")
        for path in test_videos:
            print(f"  - {path}")
        print("\nPlease provide a video path or place a test video in tests/")
        return False

    # Load video
    try:
        frames, frame_ids, metadata = load_test_video(video_path, max_frames=50)
    except Exception as e:
        print(f"\n❌ Failed to load video: {e}")
        return False

    # Create context
    context = ProcessingContext(
        video_path=video_path,
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

    # Configure detection steps (use absolute paths)
    base_path = Path(__file__).parent

    court_config = {
        'enabled': True,
        'model_path': str(base_path / 'models/court_model_best.pt'),
        'model_type': 'tracknet',
        'interval': 10,  # Detect every 10 frames (faster for testing)
        'confidence_threshold': 0.5,
        'input_size': [640, 360]
    }

    ball_config = {
        'enabled': True,
        'model_path': str(base_path / 'models/ball_model_best.pt'),
        'model_type': 'tracknet',
        'interval': 1,  # Every frame
        'confidence_threshold': 0.3,
        'input_size': [640, 360]
    }

    # Check if models exist
    if not Path(court_config['model_path']).exists():
        print(f"\n⚠️  Court model not found: {court_config['model_path']}")
        print("Please ensure models are downloaded to models/ directory")
        print(f"\nLooking in: {base_path / 'models/'}")
        available_models = list((base_path / 'models').glob('*.pt'))
        if available_models:
            print(f"Available models:")
            for model in available_models:
                print(f"  - {model.name}")
        return False

    if not Path(ball_config['model_path']).exists():
        print(f"\n⚠️  Ball model not found: {ball_config['model_path']}")
        print("Please ensure models are downloaded to models/ directory")
        print(f"\nLooking in: {base_path / 'models/'}")
        available_models = list((base_path / 'models').glob('*.pt'))
        if available_models:
            print(f"Available models:")
            for model in available_models:
                print(f"  - {model.name}")
        return False

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}\n")
    else:
        print("⚠️  No GPU available, using CPU (will be slower)\n")

    # Create pipeline
    detection_pipeline = Pipeline(
        name="DetectionPipeline",
        steps=[
            CourtDetectionStep(court_config),
            BallDetectionStep(ball_config),
        ]
    )

    # Run pipeline
    print("\nRunning Detection Pipeline...")
    print("-" * 60)

    try:
        result = detection_pipeline.run(context)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Analyze results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    # Court detection stats
    court_detected = sum(1 for det in result.detections if det.court_keypoints is not None)
    court_rate = court_detected / len(result.detections) * 100
    print(f"\nCourt Detection:")
    print(f"  Detected: {court_detected}/{len(result.detections)} frames ({court_rate:.1f}%)")

    if court_detected > 0:
        avg_conf = np.mean([det.court_confidence for det in result.detections if det.court_keypoints is not None])
        print(f"  Avg Confidence: {avg_conf:.3f}")

    # Ball detection stats
    ball_detected = sum(1 for det in result.detections if det.ball_position_px is not None)
    ball_rate = ball_detected / len(result.detections) * 100
    print(f"\nBall Detection:")
    print(f"  Detected: {ball_detected}/{len(result.detections)} frames ({ball_rate:.1f}%)")

    if ball_detected > 0:
        avg_conf = np.mean([det.ball_confidence for det in result.detections if det.ball_position_px is not None])
        print(f"  Avg Confidence: {avg_conf:.3f}")

        # Sample positions
        print(f"\nSample Ball Positions (first 5 detections):")
        count = 0
        for det in result.detections:
            if det.ball_position_px is not None:
                x, y = det.ball_position_px
                print(f"  Frame {det.frame_id}: ({x:.0f}, {y:.0f}) - conf={det.ball_confidence:.3f}")
                count += 1
                if count >= 5:
                    break

    # Performance
    print(f"\nPerformance:")
    for step_name, duration in result.step_timings.items():
        print(f"  {step_name}: {duration:.2f}s")
    print(f"  Total: {result.get_total_processing_time():.2f}s")

    fps_processed = len(result.frames) / result.get_total_processing_time()
    print(f"  Processing FPS: {fps_processed:.1f}")

    # Validate
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)

    success = True

    if court_detected == 0:
        print("❌ No courts detected!")
        success = False
    else:
        print(f"✓ Courts detected in {court_rate:.1f}% of frames")

    if ball_detected == 0:
        print("⚠️  No balls detected (this is OK if video doesn't show ball)")
    else:
        print(f"✓ Balls detected in {ball_rate:.1f}% of frames")

    print()

    if success:
        print("🎉 Phase 1 Detection Pipeline WORKS!\n")
        print("Next Steps:")
        print("  1. Add PlayerDetectionStep (YOLO)")
        print("  2. Create TemporalPipeline (gap filling)")
        print("  3. Create GeometryPipeline (homography)")
        print()
    else:
        print("❌ Detection pipeline needs debugging\n")

    return success


if __name__ == "__main__":
    success = test_detection_pipeline()
    sys.exit(0 if success else 1)
