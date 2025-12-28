#!/usr/bin/env python3
"""
Tennis Computer Vision - Simple Pipeline Test Runner
File: tests/run_simple_test.py

Simple test that can run without pytest to validate the pipeline
"""
import sys
import os

# Add the parent directory to Python path to find the app module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from app.core.data_models import ProcessingConfig, SegmentType
        print("✓ Data models imported")
    except ImportError as e:
        print(f"✗ Failed to import data models: {e}")
        return False
    
    try:
        from app.core.base import FrameFilter, GameSegmenter
        print("✓ Base classes imported")
    except ImportError as e:
        print(f"✗ Failed to import base classes: {e}")
        return False
    
    try:
        from app.core.video_processor import VideoProcessor
        print("✓ Video processor imported")
    except ImportError as e:
        print(f"✗ Failed to import video processor: {e}")
        return False
    
    return True

def test_frame_filter():
    """Test FrameFilter with synthetic data"""
    print("\nTesting FrameFilter...")
    
    try:
        import numpy as np
        from app.core.base import FrameFilter
        from app.core.data_models import ProcessingConfig
        
        # Create config
        config = ProcessingConfig(motion_threshold=5.0)
        frame_filter = FrameFilter(config)
        
        # Create synthetic frames
        frames = np.random.randint(0, 255, (5, 100, 100, 3), dtype=np.uint8)
        
        # Test frame filtering
        result = frame_filter(frames)
        
        # Validate result
        assert len(result) == 5
        assert result[0][1] == True  # First frame always active
        
        print("✓ FrameFilter working with synthetic data")
        return True
        
    except Exception as e:
        print(f"✗ FrameFilter test failed: {e}")
        return False

def test_game_segmenter():
    """Test GameSegmenter with synthetic data"""
    print("\nTesting GameSegmenter...")
    
    try:
        from app.core.base import GameSegmenter
        from app.core.data_models import ProcessingConfig, SegmentType
        
        # Create config
        config = ProcessingConfig()
        segmenter = GameSegmenter(config)
        
        # Create synthetic frame activity
        frame_activity = [
            (0, True), (1, True), (2, True),  # Rally
            (3, False), (4, False),           # Idle
            (5, True), (6, True),             # Rally
        ]
        
        # Test segmentation
        segments = segmenter(frame_activity)
        
        # Validate result
        assert len(segments) > 0
        assert all(seg.duration_frames() > 0 for seg in segments)
        
        print("✓ GameSegmenter working with synthetic data")
        return True
        
    except Exception as e:
        print(f"✗ GameSegmenter test failed: {e}")
        return False

def test_video_processor_init():
    """Test VideoProcessor initialization"""
    print("\nTesting VideoProcessor initialization...")
    
    try:
        from app.core.video_processor import VideoProcessor
        from app.core.data_models import ProcessingConfig
        
        # Create config
        config = ProcessingConfig(save_visualization=False)
        
        # Initialize processor
        processor = VideoProcessor(config)
        
        # Check components
        assert processor.frame_filter is not None
        assert processor.game_segmenter is not None
        assert processor.court_detector is not None
        
        print("✓ VideoProcessor initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ VideoProcessor test failed: {e}")
        return False

def main():
    """Run all simple tests"""
    print("="*60)
    print("TENNIS CV - SIMPLE PIPELINE TEST")
    print("="*60)
    
    tests = [
        test_imports,
        test_frame_filter,
        test_game_segmenter,
        test_video_processor_init
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("="*60)
    if passed == len(tests):
        print("✅ ALL SIMPLE TESTS PASSED!")
        print("Pipeline components are working correctly.")
        print("\nTo run full tests with real video:")
        print("1. Install dependencies: pip install numpy opencv-python scipy")
        print("2. Run: python tests/test_pipeline_integration.py")
    else:
        print(f"❌ {len(tests) - passed} TESTS FAILED!")
        print("Please check the errors above.")
    
    print("="*60)

if __name__ == "__main__":
    main()


