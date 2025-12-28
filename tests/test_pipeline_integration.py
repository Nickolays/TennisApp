"""
Tennis Computer Vision - Pipeline Integration Tests
File: tests/test_pipeline_integration.py

Test full pipeline end-to-end with real video
"""
import sys
import os
import numpy as np
import cv2
import pytest
from pathlib import Path

# Add the parent directory to Python path to find the app module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core.video_processor import VideoProcessor
from app.core.data_models import ProcessingConfig, SegmentType


class TestPipelineIntegration:
    """Test full pipeline integration"""
    
    def setup_method(self):
        """Setup test configuration"""
        self.config = ProcessingConfig(
            court_detection_interval=30,
            batch_size=8,
            motion_threshold=5.0,
            min_rally_frames=10,
            save_visualization=False,  # Don't save for testing
            results_path="results/"
        )
        self.video_path = Path("tests/video1.mp4")
    
    def test_pipeline_initialization(self):
        """Test that pipeline components initialize correctly"""
        processor = VideoProcessor(self.config)
        
        # Check all components are initialized
        assert processor.court_detector is not None
        assert processor.ball_player_detector is not None
        assert processor.frame_filter is not None
        assert processor.game_segmenter is not None
        assert processor.homography_calculator is not None
        assert processor.ball_trajectory_preprocessor is not None
        assert processor.ball_analyzer is not None
        assert processor.renderer is not None
        
        print("✓ All pipeline components initialized successfully")
    
    def test_pipeline_with_real_video(self):
        """Test full pipeline processing with real tennis video"""
        if not self.video_path.exists():
            pytest.skip(f"Test video not found: {self.video_path}")
        
        # Create processor
        processor = VideoProcessor(self.config)
        
        # Process video
        print(f"\nProcessing video: {self.video_path}")
        analytics = processor(str(self.video_path))
        
        # Validate analytics structure
        assert analytics is not None
        assert analytics.video_path == str(self.video_path)
        assert analytics.total_frames > 0
        assert analytics.fps > 0
        assert analytics.duration_seconds > 0
        
        print(f"✓ Video processed: {analytics.total_frames} frames, {analytics.fps} fps")
        print(f"✓ Duration: {analytics.duration_seconds:.2f} seconds")
        
        # Validate game segments
        assert len(analytics.game_segments) > 0
        print(f"✓ Created {len(analytics.game_segments)} game segments")
        
        # Check segment types
        segment_types = [seg.segment_type for seg in analytics.game_segments]
        print(f"✓ Segment types: {set(segment_types)}")
        
        # Should have both RALLY and IDLE segments
        rally_segments = analytics.get_rally_segments()
        idle_segments = [seg for seg in analytics.game_segments if seg.segment_type == SegmentType.IDLE]
        
        print(f"✓ Rally segments: {len(rally_segments)}")
        print(f"✓ Idle segments: {len(idle_segments)}")
        
        # Validate segment properties
        for i, segment in enumerate(analytics.game_segments):
            assert segment.start_frame >= 0
            assert segment.end_frame >= segment.start_frame
            assert segment.duration_frames() > 0
            
            if i > 0:
                # Check segments don't overlap
                prev_segment = analytics.game_segments[i-1]
                assert segment.start_frame > prev_segment.end_frame
        
        print("✓ All segments have valid properties")
    
    def test_frame_filtering_efficiency(self):
        """Test that frame filtering reduces processing load"""
        if not self.video_path.exists():
            pytest.skip(f"Test video not found: {self.video_path}")
        
        # Load video metadata
        cap = cv2.VideoCapture(str(self.video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Process with frame filtering
        processor = VideoProcessor(self.config)
        analytics = processor(str(self.video_path))
        
        # Calculate filtering efficiency
        rally_segments = analytics.get_rally_segments()
        total_rally_frames = sum(seg.duration_frames() for seg in rally_segments)
        filtering_efficiency = total_rally_frames / total_frames
        
        print(f"✓ Total frames: {total_frames}")
        print(f"✓ Rally frames: {total_rally_frames}")
        print(f"✓ Filtering efficiency: {filtering_efficiency:.2%}")
        
        # Should filter out significant portion of frames
        assert filtering_efficiency < 0.8, "Should filter out at least 20% of frames"
        assert filtering_efficiency > 0.1, "Should keep at least 10% of frames"
    
    def test_pipeline_with_different_configs(self):
        """Test pipeline with different configuration settings"""
        if not self.video_path.exists():
            pytest.skip(f"Test video not found: {self.video_path}")
        
        # Test with different motion thresholds
        configs = [
            ProcessingConfig(motion_threshold=2.0, save_visualization=False),
            ProcessingConfig(motion_threshold=10.0, save_visualization=False),
            ProcessingConfig(motion_threshold=5.0, court_detection_interval=60, save_visualization=False),
        ]
        
        results = []
        
        for i, config in enumerate(configs):
            print(f"\nTesting config {i+1}: motion_threshold={config.motion_threshold}")
            
            processor = VideoProcessor(config)
            analytics = processor(str(self.video_path))
            
            rally_segments = analytics.get_rally_segments()
            total_rally_frames = sum(seg.duration_frames() for seg in rally_segments)
            
            results.append({
                'config': config,
                'rally_segments': len(rally_segments),
                'rally_frames': total_rally_frames,
                'efficiency': total_rally_frames / analytics.total_frames
            })
            
            print(f"  Rally segments: {len(rally_segments)}")
            print(f"  Rally frames: {total_rally_frames}")
            print(f"  Efficiency: {total_rally_frames / analytics.total_frames:.2%}")
        
        # Validate that different thresholds produce different results
        assert results[0]['efficiency'] != results[1]['efficiency'], "Different thresholds should produce different results"
        
        # Lower threshold should keep more frames
        assert results[0]['efficiency'] > results[1]['efficiency'], "Lower threshold should keep more frames"
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling with invalid inputs"""
        processor = VideoProcessor(self.config)
        
        # Test with non-existent file
        with pytest.raises(Exception):
            processor("non_existent_video.mp4")
        
        # Test with invalid video file
        invalid_path = "tests/invalid_video.txt"
        Path(invalid_path).write_text("This is not a video file")
        
        try:
            with pytest.raises(Exception):
                processor(invalid_path)
        finally:
            # Clean up
            Path(invalid_path).unlink(missing_ok=True)
    
    def test_pipeline_memory_efficiency(self):
        """Test that pipeline processes video in chunks without memory issues"""
        if not self.video_path.exists():
            pytest.skip(f"Test video not found: {self.video_path}")
        
        # Use small batch size to force chunking
        small_config = ProcessingConfig(
            batch_size=4,
            chunk_size=50,
            motion_threshold=5.0,
            save_visualization=False
        )
        
        processor = VideoProcessor(small_config)
        
        # Process video (should not cause memory issues)
        analytics = processor(str(self.video_path))
        
        assert analytics is not None
        assert len(analytics.game_segments) > 0
        
        print("✓ Pipeline processed video in chunks without memory issues")


class TestPipelinePerformance:
    """Test pipeline performance characteristics"""
    
    def setup_method(self):
        """Setup test configuration"""
        self.config = ProcessingConfig(
            court_detection_interval=30,
            batch_size=16,
            motion_threshold=5.0,
            save_visualization=False
        )
        self.video_path = Path("tests/video1.mp4")
    
    def test_processing_speed(self):
        """Test pipeline processing speed"""
        if not self.video_path.exists():
            pytest.skip(f"Test video not found: {self.video_path}")
        
        import time
        
        processor = VideoProcessor(self.config)
        
        start_time = time.time()
        analytics = processor(str(self.video_path))
        end_time = time.time()
        
        processing_time = end_time - start_time
        fps_processed = analytics.total_frames / processing_time
        
        print(f"✓ Processing time: {processing_time:.2f} seconds")
        print(f"✓ Processing speed: {fps_processed:.2f} frames/second")
        
        # Should process faster than real-time for short videos
        if analytics.duration_seconds < 30:  # Only for short videos
            assert fps_processed > analytics.fps, "Should process faster than real-time"
    
    def test_batch_size_impact(self):
        """Test impact of different batch sizes on performance"""
        if not self.video_path.exists():
            pytest.skip(f"Test video not found: {self.video_path}")
        
        batch_sizes = [4, 8, 16, 32]
        results = []
        
        for batch_size in batch_sizes:
            config = ProcessingConfig(
                batch_size=batch_size,
                motion_threshold=5.0,
                save_visualization=False
            )
            
            processor = VideoProcessor(config)
            
            import time
            start_time = time.time()
            analytics = processor(str(self.video_path))
            end_time = time.time()
            
            processing_time = end_time - start_time
            fps_processed = analytics.total_frames / processing_time
            
            results.append({
                'batch_size': batch_size,
                'processing_time': processing_time,
                'fps_processed': fps_processed
            })
            
            print(f"Batch size {batch_size}: {processing_time:.2f}s, {fps_processed:.2f} fps")
        
        # Validate that batch size affects performance
        assert len(set(r['fps_processed'] for r in results)) > 1, "Different batch sizes should affect performance"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

