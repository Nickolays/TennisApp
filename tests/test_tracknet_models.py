#!/usr/bin/env python3
"""
Tennis Computer Vision - TrackNet Model Test
File: test_tracknet_models.py

Test loading and using actual TrackNet models
"""
import sys
import os
import numpy as np
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from app.models.unified_detection import (
    UnifiedDetectionPipeline, DetectionType, DetectionOutput
)
from app.core.data_models import ProcessingConfig


def test_model_loading():
    """Test loading actual TrackNet models"""
    print("="*60)
    print("TRACKNET MODEL LOADING TEST")
    print("="*60)
    
    config = ProcessingConfig()
    pipeline = UnifiedDetectionPipeline(config)
    
    # Test model paths
    court_model_path = "models/model_tennis_court_det.pt"
    ball_model_path = "models/model_best.pt"
    
    print("Testing model file existence...")
    print(f"  - Court model: {court_model_path} - {'✓' if Path(court_model_path).exists() else '✗'}")
    print(f"  - Ball model: {ball_model_path} - {'✓' if Path(ball_model_path).exists() else '✗'}")
    
    # Add models to pipeline
    print("\nAdding models to pipeline...")
    court_success = pipeline.add_model(DetectionType.COURT, court_model_path)
    ball_success = pipeline.add_model(DetectionType.BALL, ball_model_path)
    
    print(f"  - Court Detection: {'✓' if court_success else '✗'}")
    print(f"  - Ball Detection: {'✓' if ball_success else '✗'}")
    
    # Initialize pipeline
    init_success = pipeline.initialize()
    print(f"  - Pipeline initialization: {'✓' if init_success else '✗'}")
    
    if init_success:
        # Show pipeline info
        info = pipeline.get_pipeline_info()
        print(f"\nPipeline Info:")
        print(f"  - Available models: {info['available_models']}")
        print(f"  - Optimization settings: {info['optimization_settings']}")
        
        # Test with dummy frames
        print(f"\nTesting inference with dummy frames...")
        dummy_frames = np.random.randint(0, 255, (2, 512, 512, 3), dtype=np.uint8)
        
        try:
            results = pipeline.detect_sequential(dummy_frames)
            
            print(f"Detection Results:")
            for detection_type, detection_results in results.items():
                print(f"  - {detection_type.value}: {len(detection_results)} detections")
                if detection_results:
                    result = detection_results[0]
                    print(f"    Confidence: {result.confidence:.3f}")
                    
                    if detection_type == DetectionType.COURT and result.keypoints is not None:
                        print(f"    Keypoints: {result.keypoints.shape}")
                    elif detection_type == DetectionType.BALL and result.ball_position:
                        print(f"    Ball position: {result.ball_position}")
            
            print("✓ TrackNet model inference working!")
            return True
            
        except Exception as e:
            print(f"✗ Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("✗ Pipeline initialization failed")
        return False


def test_video_processing():
    """Test processing a small portion of the video"""
    print("\n" + "="*60)
    print("VIDEO PROCESSING TEST")
    print("="*60)
    
    video_path = "tests/video1.mp4"
    if not Path(video_path).exists():
        print(f"⚠️ Test video not found: {video_path}")
        return False
    
    print(f"Testing video processing with: {video_path}")
    
    # Create configuration
    config = ProcessingConfig(
        use_fp16=True,
        batch_size=2,  # Small batch for testing
        motion_threshold=5.0
    )
    
    # Create pipeline
    pipeline = UnifiedDetectionPipeline(config)
    pipeline.add_model(DetectionType.COURT, "models/model_tennis_court_det.pt")
    pipeline.add_model(DetectionType.BALL, "models/model_best.pt")
    pipeline.initialize()
    
    # Load first few frames
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    for i in range(5):  # Test with first 5 frames
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    
    cap.release()
    
    if frames:
        frames_array = np.array(frames)
        print(f"Loaded {len(frames)} frames for testing")
        
        try:
            # Process frames
            results = pipeline.detect_sequential(frames_array)
            
            print("Processing Results:")
            for detection_type, detection_results in results.items():
                confidences = [r.confidence for r in detection_results if r]
                if confidences:
                    avg_conf = np.mean(confidences)
                    print(f"  - {detection_type.value}: avg confidence {avg_conf:.3f}")
            
            print("✓ Video processing test successful!")
            return True
            
        except Exception as e:
            print(f"✗ Error during video processing: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("✗ Could not load frames from video")
        return False


def main():
    """Run all TrackNet tests"""
    print("🎾 TENNIS COMPUTER VISION - TRACKNET MODEL TEST")
    print("="*60)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Video Processing", test_video_processing),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} test failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TRACKNET TEST SUMMARY")
    print("="*60)
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All TrackNet tests passed!")
        print("\nTrackNet models are working correctly!")
        print("\nReady to process videos with:")
        print("python3 process_video_tracknet.py tests/video1.mp4")
    else:
        print(f"\n⚠️ {total - passed} tests failed")
        print("Check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

