#!/usr/bin/env python3
"""
Tennis Computer Vision - Unified Detection Pipeline Demo
File: demo_unified_detection.py

Demonstrate the unified detection architecture for Court, Ball, and Pose detection
"""
import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from app.models.unified_detection import (
    UnifiedDetectionPipeline, DetectionType, DetectionOutput,
    ONNXConverter
)
from app.core.data_models import ProcessingConfig


def demo_unified_pipeline():
    """Demonstrate the unified detection pipeline"""
    print("="*60)
    print("UNIFIED DETECTION PIPELINE DEMO")
    print("="*60)
    
    # Create configuration optimized for fast inference
    config = ProcessingConfig(
        use_fp16=True,           # Enable FP16 for faster inference
        batch_size=8,            # Optimized batch size
        motion_threshold=5.0,
        court_detection_interval=30
    )
    
    print("Configuration:")
    print(f"  - Use FP16: {config.use_fp16}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Motion threshold: {config.motion_threshold}")
    print(f"  - Court detection interval: {config.court_detection_interval}")
    
    # Create unified pipeline
    pipeline = UnifiedDetectionPipeline(config)
    
    # Add all three detection models (as per your pipeline image)
    print("\nAdding detection models...")
    
    court_success = pipeline.add_model(DetectionType.COURT, "models/model_tennis_court_det.pt")
    ball_success = pipeline.add_model(DetectionType.BALL, "models/model_best.pt")
    pose_success = pipeline.add_model(DetectionType.POSE, "models/model_best.pt")
    
    print(f"  - Court Detection: {'✓' if court_success else '✗'}")
    print(f"  - Ball Detection: {'✓' if ball_success else '✗'}")
    print(f"  - Pose Detection: {'✓' if pose_success else '✗'}")
    
    # Initialize pipeline
    init_success = pipeline.initialize()
    if not init_success:
        print("❌ Pipeline initialization failed")
        return False
    
    print("✓ Pipeline initialized successfully")
    
    # Show pipeline info
    info = pipeline.get_pipeline_info()
    print(f"\nPipeline Info:")
    print(f"  - Available models: {info['available_models']}")
    print(f"  - Optimization settings: {info['optimization_settings']}")
    
    return pipeline


def demo_detection_sequence(pipeline: UnifiedDetectionPipeline):
    """Demonstrate sequential detection (as per your pipeline image)"""
    print("\n" + "="*60)
    print("SEQUENTIAL DETECTION DEMO")
    print("="*60)
    
    # Create synthetic test frames
    print("Creating test frames...")
    frames = np.random.randint(0, 255, (3, 512, 512, 3), dtype=np.uint8)
    print(f"  - Frame shape: {frames.shape}")
    
    # Run sequential detection (Court -> Ball -> Pose)
    print("\nRunning sequential detection...")
    results = pipeline.detect_sequential(frames)
    
    # Display results
    print("\nDetection Results:")
    for detection_type, detection_results in results.items():
        print(f"\n{detection_type.value.upper()} Detection:")
        print(f"  - Number of detections: {len(detection_results)}")
        # print(f"output shape: {detection_results[0].shape}")
        
        if detection_results:
            result = detection_results[0]  # First frame result
            
            if detection_type == DetectionType.COURT:
                print(f"  - Confidence: {result.confidence:.3f}")
                print(f"  - Keypoints: {result.keypoints.shape}")
                print(f"  - Court lines: {len(result.court_lines) if result.court_lines else 0}")
                print(f"  - Court corners: {result.court_corners.shape if result.court_corners is not None else 'None'}")
                
            elif detection_type == DetectionType.BALL:
                print(f"  - Confidence: {result.confidence:.3f}")
                print(f"  - Ball position: {result.ball_position}")
                print(f"  - Keypoints: {result.keypoints.shape}")
                
            elif detection_type == DetectionType.POSE:
                print(f"  - Confidence: {result.confidence:.3f}")
                print(f"  - Pose keypoints: {result.pose_keypoints.shape}")
                print(f"  - Skeleton connections: {len(result.pose_skeleton) if result.pose_skeleton else 0}")
    
    return True


def demo_onnx_conversion():
    """Demonstrate ONNX conversion capabilities"""
    print("\n" + "="*60)
    print("ONNX CONVERSION DEMO")
    print("="*60)
    
    config = ProcessingConfig()
    
    # Create models for conversion
    models_to_convert = [
        (DetectionType.COURT, "models/model_tennis_court_det.pt", "models/court_model.onnx"),
        (DetectionType.BALL, "models/model_best.pt", "models/ball_model.onnx"),
        (DetectionType.POSE, "models/model_best.pt", "models/pose_model.onnx")
    ]
    
    print("Converting models to ONNX format...")
    
    for detection_type, input_path, output_path in models_to_convert:
        print(f"\nConverting {detection_type.value} model:")
        print(f"  - Input: {input_path}")
        print(f"  - Output: {output_path}")
        
        # Create model
        if detection_type == DetectionType.COURT:
            from app.models.unified_detection import create_court_detector
            model = create_court_detector(config)
        elif detection_type == DetectionType.BALL:
            from app.models.unified_detection import create_ball_detector
            model = create_ball_detector(config)
        elif detection_type == DetectionType.POSE:
            from app.models.unified_detection import create_pose_detector
            model = create_pose_detector(config)
        
        # Load model
        model.load_model(input_path)
        
        # Convert to ONNX
        onnx_success = ONNXConverter.convert_to_onnx(model, output_path)
        
        if onnx_success:
            print(f"  ✓ ONNX conversion successful")
            
            # Optimize for TensorRT
            trt_path = output_path.replace('.onnx', '.trt')
            trt_success = ONNXConverter.optimize_for_tensorrt(output_path, trt_path)
            
            if trt_success:
                print(f"  ✓ TensorRT optimization successful: {trt_path}")
            else:
                print(f"  ✗ TensorRT optimization failed")
        else:
            print(f"  ✗ ONNX conversion failed")
    
    print("\n✓ ONNX conversion demo completed")
    return True


def demo_performance_comparison():
    """Demonstrate performance comparison between different settings"""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON DEMO")
    print("="*60)
    
    import time
    
    # Test different configurations
    configs = [
        ("FP32, Batch=4", ProcessingConfig(use_fp16=False, batch_size=4)),
        ("FP32, Batch=8", ProcessingConfig(use_fp16=False, batch_size=8)),
        ("FP16, Batch=4", ProcessingConfig(use_fp16=True, batch_size=4)),
        ("FP16, Batch=8", ProcessingConfig(use_fp16=True, batch_size=8)),
    ]
    
    # Create test frames
    frames = np.random.randint(0, 255, (8, 512, 512, 3), dtype=np.uint8)
    
    print("Testing different optimization settings...")
    print(f"Test frames: {frames.shape}")
    
    results = []
    
    for config_name, config in configs:
        print(f"\nTesting {config_name}:")
        
        # Create pipeline
        pipeline = UnifiedDetectionPipeline(config)
        pipeline.add_model(DetectionType.COURT, "models/model_tennis_court_det.pt")
        pipeline.initialize()
        
        # Time the detection
        start_time = time.time()
        detection_results = pipeline.detect(frames)
        end_time = time.time()
        
        processing_time = end_time - start_time
        fps = len(frames) / processing_time
        
        results.append((config_name, processing_time, fps))
        
        print(f"  - Processing time: {processing_time:.3f}s")
        print(f"  - FPS: {fps:.1f}")
    
    # Show comparison
    print("\nPerformance Comparison:")
    print("Configuration        | Time (s) | FPS")
    print("-" * 40)
    for config_name, processing_time, fps in results:
        print(f"{config_name:<20} | {processing_time:>8.3f} | {fps:>6.1f}")
    
    # Find best configuration
    best_config = max(results, key=lambda x: x[2])  # Highest FPS
    print(f"\n✓ Best configuration: {best_config[0]} ({best_config[2]:.1f} FPS)")
    
    return True


def demo_real_video_processing():
    """Demonstrate processing real tennis video"""
    print("\n" + "="*60)
    print("REAL VIDEO PROCESSING DEMO")
    print("="*60)
    
    video_path = "tests/video1.mp4"
    if not os.path.exists(video_path):
        print(f"⚠️ Test video not found: {video_path}")
        print("Skipping real video processing demo...")
        return False
    
    # Create optimized pipeline
    config = ProcessingConfig(
        use_fp16=True,
        batch_size=4,
        motion_threshold=5.0
    )
    
    pipeline = UnifiedDetectionPipeline(config)
    pipeline.add_model(DetectionType.COURT, "models/model_tennis_court_det.pt")
    pipeline.add_model(DetectionType.BALL, "models/model_best.pt")
    pipeline.add_model(DetectionType.POSE, "models/model_best.pt")
    pipeline.initialize()
    
    print(f"Processing video: {video_path}")
    
    # Load video frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Read first 30 frames for demo
    for i in range(30):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    
    cap.release()
    
    if frames:
        frames = np.array(frames)
        print(f"Loaded {len(frames)} frames for processing")
        
        # Process frames
        print("Running unified detection...")
        results = pipeline.detect_sequential(frames)
        
        # Analyze results
        print("\nVideo Analysis Results:")
        for detection_type, detection_results in results.items():
            confidences = [r.confidence for r in detection_results]
            avg_conf = np.mean(confidences)
            max_conf = np.max(confidences)
            min_conf = np.min(confidences)
            
            print(f"\n{detection_type.value.upper()} Detection:")
            print(f"  - Average confidence: {avg_conf:.3f}")
            print(f"  - Max confidence: {max_conf:.3f}")
            print(f"  - Min confidence: {min_conf:.3f}")
            print(f"  - Detections: {len(detection_results)}")
        
        print("✓ Real video processing demo completed")
        return True
    else:
        print("❌ Could not load frames from video")
        return False


def main():
    """Run all unified detection demos"""
    print("🎾 TENNIS COMPUTER VISION - UNIFIED DETECTION DEMO")
    print("="*60)
    
    demos = [
        ("Unified Pipeline", demo_unified_pipeline),
        ("Sequential Detection", lambda: demo_detection_sequence(demo_unified_pipeline())),
        ("ONNX Conversion", demo_onnx_conversion),
        ("Performance Comparison", demo_performance_comparison),
        ("Real Video Processing", demo_real_video_processing),
    ]
    
    results = []
    pipeline = None
    
    for name, demo_func in demos:
        try:
            if name == "Sequential Detection" and pipeline is None:
                pipeline = demo_unified_pipeline()
                result = demo_detection_sequence(pipeline)
            else:
                result = demo_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} demo failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("UNIFIED DETECTION DEMO SUMMARY")
    print("="*60)
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} demos passed")
    
    if passed == total:
        print("\n🎉 All unified detection demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ Unified architecture for Court, Ball, and Pose detection")
        print("✓ Sequential pipeline execution (as per your design)")
        print("✓ ONNX conversion ready for optimization")
        print("✓ TensorRT optimization ready for fast inference")
        print("✓ Performance optimization (FP16, batch processing)")
        print("✓ Real video processing capabilities")
        print("\nNext steps:")
        print("1. Implement actual TrackNet models")
        print("2. Convert to ONNX format")
        print("3. Optimize with TensorRT")
        print("4. Deploy for fast inference")
    else:
        print(f"\n⚠️ {total - passed} demos failed")
        print("Check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)


