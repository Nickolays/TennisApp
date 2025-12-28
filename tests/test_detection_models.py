#!/usr/bin/env python3
"""
Tennis Computer Vision - Detection Models Test
File: tests/test_detection_models.py

Test the Detection Models architecture
"""
import sys
import os
import numpy as np

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.models.detection_models import (
    DetectionModelFactory, ModelType, DetectionPipeline,
    BaseDetectionModel, TrackNetDetectionModel
)
from app.core.data_models import ProcessingConfig


def test_detection_model_factory():
    """Test DetectionModelFactory"""
    print("\n" + "="*60)
    print("TEST 1: Detection Model Factory")
    print("="*60)
    
    config = ProcessingConfig()
    
    # Test creating different model types
    court_detector = DetectionModelFactory.create_court_detector(config)
    ball_detector = DetectionModelFactory.create_ball_detector(config)
    player_detector = DetectionModelFactory.create_player_detector(config)
    
    assert isinstance(court_detector, BaseDetectionModel)
    assert isinstance(ball_detector, BaseDetectionModel)
    assert isinstance(player_detector, BaseDetectionModel)
    
    assert court_detector.model_type == ModelType.COURT_DETECTION
    assert ball_detector.model_type == ModelType.BALL_DETECTION
    assert player_detector.model_type == ModelType.PLAYER_DETECTION
    
    print("✓ DetectionModelFactory working correctly")
    print("✓ All model types can be created")
    
    return True


def test_model_loading():
    """Test model loading functionality"""
    print("\n" + "="*60)
    print("TEST 2: Model Loading")
    print("="*60)
    
    config = ProcessingConfig()
    model = DetectionModelFactory.create_court_detector(config)
    
    # Test loading with dummy path
    success = model.load_model("dummy_model.pt")
    assert success == True
    
    assert model.is_loaded == True
    assert model.model_path == "dummy_model.pt"
    
    # Test model info
    info = model.get_model_info()
    assert info['model_type'] == 'court_detection'
    assert info['is_loaded'] == True
    
    print("✓ Model loading working correctly")
    print("✓ Model info retrieval working")
    
    return True


def test_model_inference():
    """Test model inference"""
    print("\n" + "="*60)
    print("TEST 3: Model Inference")
    print("="*60)
    
    config = ProcessingConfig()
    
    # Test court detection
    court_detector = DetectionModelFactory.create_court_detector(config)
    court_detector.load_model("dummy_court.pt")
    
    frames = np.random.randint(0, 255, (2, 512, 512, 3), dtype=np.uint8)
    court_results = court_detector(frames)
    
    assert len(court_results) == 2
    assert all(isinstance(result, type(court_results[0])) for result in court_results)
    assert all(result.model_type == ModelType.COURT_DETECTION for result in court_results)
    
    print("✓ Court detection inference working")
    
    # Test ball detection
    ball_detector = DetectionModelFactory.create_ball_detector(config)
    ball_detector.load_model("dummy_ball.pt")
    
    ball_results = ball_detector(frames)
    
    assert len(ball_results) == 2
    assert all(result.model_type == ModelType.BALL_DETECTION for result in ball_results)
    
    print("✓ Ball detection inference working")
    
    # Test player detection
    player_detector = DetectionModelFactory.create_player_detector(config)
    player_detector.load_model("dummy_player.pt")
    
    player_results = player_detector(frames)
    
    assert len(player_results) == 2
    assert all(result.model_type == ModelType.PLAYER_DETECTION for result in player_results)
    
    print("✓ Player detection inference working")
    
    return True


def test_detection_pipeline():
    """Test DetectionPipeline"""
    print("\n" + "="*60)
    print("TEST 4: Detection Pipeline")
    print("="*60)
    
    config = ProcessingConfig()
    pipeline = DetectionPipeline(config)
    
    # Add models to pipeline
    success1 = pipeline.add_model(ModelType.COURT_DETECTION, "dummy_court.pt")
    success2 = pipeline.add_model(ModelType.BALL_DETECTION, "dummy_ball.pt")
    success3 = pipeline.add_model(ModelType.PLAYER_DETECTION, "dummy_player.pt")
    
    assert success1 == True
    assert success2 == True
    assert success3 == True
    
    # Test pipeline info
    info = pipeline.get_pipeline_info()
    assert len(info['available_models']) == 3
    assert 'court_detection' in info['available_models']
    assert 'ball_detection' in info['available_models']
    assert 'player_detection' in info['available_models']
    
    print("✓ DetectionPipeline created successfully")
    print("✓ All models added to pipeline")
    
    # Test detection
    frames = np.random.randint(0, 255, (2, 512, 512, 3), dtype=np.uint8)
    results = pipeline.detect(frames)
    
    assert len(results) == 3
    assert ModelType.COURT_DETECTION in results
    assert ModelType.BALL_DETECTION in results
    assert ModelType.PLAYER_DETECTION in results
    
    print("✓ Pipeline detection working")
    
    # Test selective detection
    selective_results = pipeline.detect(frames, [ModelType.COURT_DETECTION])
    assert len(selective_results) == 1
    assert ModelType.COURT_DETECTION in selective_results
    
    print("✓ Selective detection working")
    
    return True


def test_model_extensibility():
    """Test model extensibility"""
    print("\n" + "="*60)
    print("TEST 5: Model Extensibility")
    print("="*60)
    
    # Test that we can create custom model types
    class CustomModelType(ModelType):
        CUSTOM_DETECTION = "custom_detection"
    
    # Test that BaseDetectionModel can be extended
    class CustomDetectionModel(BaseDetectionModel):
        def __init__(self, config: ProcessingConfig):
            super().__init__(config, ModelType.MULTI_OBJECT)
        
        def load_model(self, model_path: str) -> bool:
            self.model_path = model_path
            self.is_loaded = True
            return True
        
        def preprocess(self, frames: np.ndarray) -> np.ndarray:
            return frames.astype(np.float32) / 255.0
        
        def inference(self, preprocessed_frames: np.ndarray) -> List[Dict[str, Any]]:
            return [{'confidence': 0.8} for _ in range(len(preprocessed_frames))]
        
        def postprocess(self, raw_results: List[Dict[str, Any]]) -> List[DetectionResult]:
            from app.models.detection_models import DetectionResult
            return [DetectionResult(
                model_type=self.model_type,
                confidence=raw_result['confidence']
            ) for raw_result in raw_results]
    
    config = ProcessingConfig()
    custom_model = CustomDetectionModel(config)
    
    # Test custom model
    custom_model.load_model("custom_model.pt")
    assert custom_model.is_loaded == True
    
    frames = np.random.randint(0, 255, (2, 512, 512, 3), dtype=np.uint8)
    results = custom_model(frames)
    
    assert len(results) == 2
    assert all(result.model_type == ModelType.MULTI_OBJECT for result in results)
    
    print("✓ Custom model creation working")
    print("✓ Model extensibility validated")
    
    return True


def run_all_tests():
    """Run all detection model tests"""
    print("\n" + "#"*60)
    print("# TENNIS CV - DETECTION MODELS TEST SUITE")
    print("#"*60)
    
    tests = [
        test_detection_model_factory,
        test_model_loading,
        test_model_inference,
        test_detection_pipeline,
        test_model_extensibility
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
        print("# ✅ ALL DETECTION MODEL TESTS PASSED!")
        print("#"*60)
        print("\nDetection Models architecture is working correctly!")
        print("\nFeatures validated:")
        print("✓ Model factory pattern")
        print("✓ Model loading and management")
        print("✓ Inference pipeline")
        print("✓ Detection pipeline")
        print("✓ Extensibility for custom models")
        print("\nReady for integration with tennis pipeline!")
    else:
        print(f"# ❌ {len(tests) - passed} TESTS FAILED!")
        print("#"*60)
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)


