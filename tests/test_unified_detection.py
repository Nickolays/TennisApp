#!/usr/bin/env python3
"""
Tennis Computer Vision - Unified Detection Pipeline Test
File: tests/test_unified_detection.py

Test the unified detection architecture for Court, Ball, and Pose detection
"""
import sys
import os
import numpy as np

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.models.unified_detection import (
    UnifiedDetectionPipeline, DetectionType, DetectionOutput,
    TrackNetUnifiedModel, ONNXConverter,
    create_court_detector, create_ball_detector, create_pose_detector
)
from app.core.data_models import ProcessingConfig


def test_unified_model_creation():
    """Test creation of unified detection models"""
    print("\n" + "="*60)
    print("TEST 1: Unified Model Creation")
    print("="*60)
    
    config = ProcessingConfig()
    
    # Test creating different model types
    court_detector = create_court_detector(config)
    ball_detector = create_ball_detector(config)
    pose_detector = create_pose_detector(config)
    
    assert court_detector.detection_type == DetectionType.COURT
    assert ball_detector.detection_type == DetectionType.BALL
    assert pose_detector.detection_type == DetectionType.POSE
    
    print("✓ All unified models created successfully")
    print("✓ Detection types correctly assigned")
    
    return True


def test_model_loading():
    """Test model loading functionality"""
    print("\n" + "="*60)
    print("TEST 2: Model Loading")
    print("="*60)
    
    config = ProcessingConfig()
    
    # Test loading each model type
    models = [
        (create_court_detector(config), "models/court_model.pt"),
        (create_ball_detector(config), "models/ball_model.pt"),
        (create_pose_detector(config), "models/pose_model.pt")
    ]
    
    for model, model_path in models:
        success = model.load_model(model_path)
        assert success == True
        assert model.is_loaded == True
        assert model.model_path == model_path
        
        # Test model info
        info = model.get_model_info()
        assert info['is_loaded'] == True
        assert info['detection_type'] == model.detection_type.value
        
        print(f"✓ {model.detection_type.value} model loaded successfully")
    
    return True


def test_model_inference():
    """Test model inference for all detection types"""
    print("\n" + "="*60)
    print("TEST 3: Model Inference")
    print("="*60)
    
    config = ProcessingConfig()
    
    # Create test frames
    frames = np.random.randint(0, 255, (3, 512, 512, 3), dtype=np.uint8)
    
    # Test court detection
    court_detector = create_court_detector(config)
    court_detector.load_model("models/court_model.pt")
    court_results = court_detector(frames)
    
    assert len(court_results) == 3
    assert all(isinstance(result, DetectionOutput) for result in court_results)
    assert all(result.detection_type == DetectionType.COURT for result in court_results)
    assert all(result.keypoints is not None for result in court_results)
    assert all(result.court_lines is not None for result in court_results)
    
    print("✓ Court detection inference working")
    
    # Test ball detection
    ball_detector = create_ball_detector(config)
    ball_detector.load_model("models/ball_model.pt")
    ball_results = ball_detector(frames)
    
    assert len(ball_results) == 3
    assert all(result.detection_type == DetectionType.BALL for result in ball_results)
    assert all(result.ball_position is not None for result in ball_results)
    
    print("✓ Ball detection inference working")
    
    # Test pose detection
    pose_detector = create_pose_detector(config)
    pose_detector.load_model("models/pose_model.pt")
    pose_results = pose_detector(frames)
    
    assert len(pose_results) == 3
    assert all(result.detection_type == DetectionType.POSE for result in pose_results)
    assert all(result.pose_keypoints is not None for result in pose_results)
    assert all(result.pose_skeleton is not None for result in pose_results)
    
    print("✓ Pose detection inference working")
    
    return True


def test_unified_pipeline():
    """Test UnifiedDetectionPipeline"""
    print("\n" + "="*60)
    print("TEST 4: Unified Detection Pipeline")
    print("="*60)
    
    config = ProcessingConfig()
    pipeline = UnifiedDetectionPipeline(config)
    
    # Add models to pipeline
    success1 = pipeline.add_model(DetectionType.COURT, "models/court_model.pt")
    success2 = pipeline.add_model(DetectionType.BALL, "models/ball_model.pt")
    success3 = pipeline.add_model(DetectionType.POSE, "models/pose_model.pt")
    
    assert success1 == True
    assert success2 == True
    assert success3 == True
    
    # Initialize pipeline
    init_success = pipeline.initialize()
    assert init_success == True
    
    # Test pipeline info
    info = pipeline.get_pipeline_info()
    assert info['is_initialized'] == True
    assert len(info['available_models']) == 3
    assert 'court' in info['available_models']
    assert 'ball' in info['available_models']
    assert 'pose' in info['available_models']
    
    print("✓ Pipeline created and initialized successfully")
    
    # Test detection
    frames = np.random.randint(0, 255, (2, 512, 512, 3), dtype=np.uint8)
    results = pipeline.detect(frames)
    
    assert len(results) == 3
    assert DetectionType.COURT in results
    assert DetectionType.BALL in results
    assert DetectionType.POSE in results
    
    print("✓ Pipeline detection working")
    
    # Test sequential detection (as per your pipeline image)
    sequential_results = pipeline.detect_sequential(frames)
    assert len(sequential_results) == 3
    
    print("✓ Sequential detection working")
    
    return True


def test_onnx_conversion():
    """Test ONNX conversion utilities"""
    print("\n" + "="*60)
    print("TEST 5: ONNX Conversion")
    print("="*60)
    
    config = ProcessingConfig()
    
    # Create a model for testing
    court_detector = create_court_detector(config)
    court_detector.load_model("models/court_model.pt")
    
    # Test ONNX conversion
    onnx_success = ONNXConverter.convert_to_onnx(court_detector, "models/court_model.onnx")
    assert onnx_success == True
    
    # Test TensorRT optimization
    trt_success = ONNXConverter.optimize_for_tensorrt("models/court_model.onnx", "models/court_model.trt")
    assert trt_success == True
    
    print("✓ ONNX conversion working")
    print("✓ TensorRT optimization working")
    
    return True


def test_pipeline_optimization():
    """Test pipeline optimization features"""
    print("\n" + "="*60)
    print("TEST 6: Pipeline Optimization")
    print("="*60)
    
    # Test with different optimization settings
    config_fp16 = ProcessingConfig(use_fp16=True, batch_size=16)
    config_fp32 = ProcessingConfig(use_fp16=False, batch_size=8)
    
    pipeline_fp16 = UnifiedDetectionPipeline(config_fp16)
    pipeline_fp32 = UnifiedDetectionPipeline(config_fp32)
    
    # Add models with different settings
    pipeline_fp16.add_model(DetectionType.COURT, "models/court_model.pt")
    pipeline_fp32.add_model(DetectionType.COURT, "models/court_model.pt")
    
    pipeline_fp16.initialize()
    pipeline_fp32.initialize()
    
    # Test info shows optimization settings
    info_fp16 = pipeline_fp16.get_pipeline_info()
    info_fp32 = pipeline_fp32.get_pipeline_info()
    
    assert info_fp16['optimization_settings']['use_fp16'] == True
    assert info_fp16['optimization_settings']['batch_size'] == 16
    assert info_fp32['optimization_settings']['use_fp16'] == False
    assert info_fp32['optimization_settings']['batch_size'] == 8
    
    print("✓ FP16 optimization settings working")
    print("✓ Batch size optimization working")
    
    return True


def test_coordinate_mapping():
    """Test coordinate mapping between different input sizes"""
    print("\n" + "="*60)
    print("TEST 7: Coordinate Mapping")
    print("="*60)
    
    config = ProcessingConfig()
    
    # Test with different input sizes
    test_sizes = [(480, 640), (512, 512), (720, 1280)]
    
    for height, width in test_sizes:
        frames = np.random.randint(0, 255, (1, height, width, 3), dtype=np.uint8)
        
        # Test court detection
        court_detector = create_court_detector(config)
        court_detector.load_model("models/court_model.pt")
        results = court_detector(frames)
        
        # Check that coordinates are mapped back to original frame size
        keypoints = results[0].keypoints
        assert keypoints.shape[0] == 8  # 8 court keypoints
        assert np.all(keypoints[:, 0] >= 0) and np.all(keypoints[:, 0] <= width)
        assert np.all(keypoints[:, 1] >= 0) and np.all(keypoints[:, 1] <= height)
        
        print(f"✓ Coordinate mapping working for {width}x{height}")
    
    return True


def run_all_tests():
    """Run all unified detection tests"""
    print("\n" + "#"*60)
    print("# TENNIS CV - UNIFIED DETECTION PIPELINE TEST SUITE")
    print("#"*60)
    
    tests = [
        test_unified_model_creation,
        test_model_loading,
        test_model_inference,
        test_unified_pipeline,
        test_onnx_conversion,
        test_pipeline_optimization,
        test_coordinate_mapping
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "#"*60)
    if passed == len(tests):
        print("# ✅ ALL UNIFIED DETECTION TESTS PASSED!")
        print("#"*60)
        print("\nUnified Detection Pipeline is working correctly!")
        print("\nFeatures validated:")
        print("✓ Unified model architecture")
        print("✓ Court detection (keypoints + lines)")
        print("✓ Ball detection (position + confidence)")
        print("✓ Pose detection (keypoints + skeleton)")
        print("✓ Sequential pipeline execution")
        print("✓ ONNX conversion ready")
        print("✓ TensorRT optimization ready")
        print("✓ Coordinate mapping")
        print("✓ Optimization settings (FP16, batch size)")
        print("\nReady for fast pipeline inference!")
    else:
        print(f"# ❌ {len(tests) - passed} TESTS FAILED!")
        print("#"*60)
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)


