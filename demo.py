#!/usr/bin/env python3
"""
Tennis Computer Vision - Demo Script
File: demo.py

Demonstrate the tennis analysis pipeline with video processing and detection models
"""
import sys
import os
import numpy as np
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from app.core.video_processor import VideoProcessor
from app.core.data_models import ProcessingConfig
from app.models.detection_models import DetectionPipeline, ModelType


def demo_detection_models():
    """Demonstrate the Detection Models architecture"""
    print("="*60)
    print("DETECTION MODELS DEMO")
    print("="*60)
    
    # Create configuration
    config = ProcessingConfig()
    
    # Create detection pipeline
    pipeline = DetectionPipeline(config)
    
    # Add different detection models
    print("Adding detection models to pipeline...")
    pipeline.add_model(ModelType.COURT_DETECTION, "models/model_tennis_court_det.pt")
    pipeline.add_model(ModelType.BALL_DETECTION, "models/ball_model_best.pt")
    # pipeline.add_model(ModelType.PLAYER_DETECTION, "models/model_best.pt")
    
    # Show pipeline info
    info = pipeline.get_pipeline_info()
    print(f"Available models: {info['available_models']}")
    
    # Test with synthetic frames
    print("\nTesting with synthetic frames...")
    frames = np.random.randint(0, 255, (3, 512, 512, 3), dtype=np.uint8)
    
    # Run detection
    results = pipeline.detect(frames)
    
    print(f"Detection results:")
    for model_type, model_results in results.items():
        print(f"  {model_type.value}: {len(model_results)} detections")
        if model_results:
            print(f"    Confidence: {model_results[0].confidence:.2f}")
    
    print("✓ Detection Models demo completed!")
    return True


def demo_video_processing():
    """Demo: Run detection overlays only; ignore pipeline analytics/game segments"""
    print("\n" + "="*60)
    print("VIDEO PROCESSING DEMO (Direct Detection)")
    print("="*60)
    
    test_video = "tests/video3.mp4"
    if not os.path.exists(test_video):
        print(f"⚠️ Test video not found: {test_video}")
        print("Skipping video processing demo...")
        return False
    
    # Config for demo
    config = ProcessingConfig()
    
    from app.core.base import CourtDetector, BallDetector, PlayerDetector, VideoRenderer
    court_detector = CourtDetector(config)
    ball_detector = BallDetector(config)
    player_detector = PlayerDetector(config, model_name_or_path="yolo11n.pt")  # Use YOLO v11 nano
    renderer = VideoRenderer(config)
    
    import cv2
    cap = cv2.VideoCapture(test_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = "results/demo_detection_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    print(f"Direct detection demo, input: {test_video}, output: {out_path}")
    
    max_frames = 100
    frame_id = 0
    import numpy as np
    from app.core.data_models import HomographyData, BallTrajectory
    
    while frame_id < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        batch = frame[np.newaxis, ...]
        # Court detection
        try:
            court_result = court_detector(batch)[0]
            court_keypoints, court_conf = court_result
            if court_keypoints is not None:
                homography = HomographyData(matrix=None, source_keypoints=court_keypoints, target_keypoints=None, valid_from_frame=frame_id, valid_to_frame=frame_id, confidence=court_conf)
            else:
                homography = None
        except Exception as e:
            print(f"Court detection error in demo: {e}")
            homography = None
        # Ball detection
        try:
            ball_result = ball_detector(batch)[0]
            ball_pos = ball_result['ball_pos']
            trajectory = None
            if ball_pos is not None:
                # Provide a fake trajectory with just the ball position
                trajectory = BallTrajectory(frame_ids=[frame_id], positions_px=np.array([ball_pos]))
        except Exception as e:
            print(f"Ball detection error in demo: {e}")
            trajectory = None
        
        # Player detection
        try:
            player_result = player_detector(batch)[0]
            players = {
                'player_boxes': player_result.get('player_boxes', []),
                'player_confs': player_result.get('player_confs', []),
                'player_class_ids': player_result.get('player_class_ids', [])
            }
        except Exception as e:
            print(f"Player detection error in demo: {e}")
            players = {'player_boxes': [], 'player_confs': [], 'player_class_ids': []}
        
        # Draw overlays, always call renderer
        render_data = {
            'frame_id': frame_id,
            'homography': homography,
            'trajectory': trajectory,
            'players': players
        }
        rendered = renderer(frame, render_data)
        writer.write(rendered)
        frame_id += 1
    cap.release()
    writer.release()
    print("✓ Detection overlays written to", out_path)
    return True


def demo_frame_filter():
    """Demonstrate frame filtering"""
    print("\n" + "="*60)
    print("FRAME FILTER DEMO")
    print("="*60)
    
    from app.core.base import FrameFilter
    
    config = ProcessingConfig(motion_threshold=5.0)
    frame_filter = FrameFilter(config)
    
    # Create synthetic frames with different motion patterns
    print("Testing motion detection with synthetic frames...")
    
    # High motion frames
    high_motion_frames = []
    for i in range(5):
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        high_motion_frames.append(frame)
    
    # Low motion frames (similar)
    base_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    low_motion_frames = [base_frame]
    for i in range(4):
        frame = base_frame + np.random.randint(-5, 5, base_frame.shape, dtype=np.int16)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        low_motion_frames.append(frame)
    
    # Test high motion
    high_motion_results = frame_filter(np.array(high_motion_frames))
    print(f"High motion frames: {[is_active for _, is_active in high_motion_results]}")
    
    # Test low motion
    low_motion_results = frame_filter(np.array(low_motion_frames))
    print(f"Low motion frames: {[is_active for _, is_active in low_motion_results]}")
    
    print("✓ Frame filter demo completed!")
    return True


def main():
    """Run all demos"""
    print("🎾 TENNIS COMPUTER VISION - DEMO")
    print("="*60)
    
    demos = [
        ("Detection Models", demo_detection_models),
        # ("Frame Filter", demo_frame_filter),
        ("Video Processing", demo_video_processing),
    ]
    
    results = []
    for name, demo_func in demos:
        try:
            result = demo_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} demo failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} demos passed")
    
    if passed == total:
        print("\n🎉 All demos completed successfully!")
        print("\nNext steps:")
        print("1. Process your own video: python process_video.py your_video.mp4")
        print("2. Adjust parameters in ProcessingConfig")
        print("3. Implement real detection models")
        print("4. Add more visualization features")
    else:
        print(f"\n⚠️ {total - passed} demos failed")
        print("Check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)


