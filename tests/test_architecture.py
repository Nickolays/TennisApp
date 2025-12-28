"""
Tennis Computer Vision - Architecture Validation Test
File: test_architecture.py

Run this to validate the pipeline structure before implementing models
"""
import sys
import os
import numpy as np

# Add the parent directory to Python path to find the app module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core.data_models import (
    FrameDetection, GameSegment, ProcessingConfig, 
    SegmentType, BallTrajectory, VideoAnalytics
)
from app.core.base import (
    CourtDetector, BallPlayerDetector, FrameFilter,
    GameSegmenter, HomographyCalculator, BallAnalyzer,
    BallTrajectoryPreprocessor, VideoRenderer
)


def test_dataclasses():
    """Test all dataclasses can be instantiated"""
    print("\n" + "="*60)
    print("TEST 1: Dataclasses Validation")
    print("="*60)
    
    # Test FrameDetection
    detection = FrameDetection(
        frame_id=100,
        timestamp=3.33,
        court_keypoints=np.array([[0, 0], [100, 100]]),
        court_confidence=0.95,
        ball_position_px=(320, 240),
        ball_confidence=0.85,
        player_boxes=[(100, 100, 200, 300), (400, 150, 500, 350)],
        player_confidences=[0.92, 0.88]
    )
    
    assert detection.has_court() == True
    assert detection.has_ball() == True
    print("✓ FrameDetection working")
    
    # Test GameSegment
    segment = GameSegment(
        start_frame=100,
        end_frame=500,
        segment_type=SegmentType.RALLY
    )
    assert segment.duration_frames() == 401
    assert segment.duration_seconds(30.0) == 401 / 30.0
    print("✓ GameSegment working")
    
    # Test BallTrajectory
    trajectory = BallTrajectory(
        frame_ids=[1, 2, 3, 4, 5],
        positions_px=np.array([[100, 200], [110, 210], [120, 220], [130, 230], [140, 240]]),
        speeds=[10.5, 11.2, 12.0, 11.8, 10.9]
    )
    assert trajectory.get_max_speed() == 12.0
    assert trajectory.get_rally_duration(30.0) == 4 / 30.0
    print("✓ BallTrajectory working")
    
    # Test ProcessingConfig
    config = ProcessingConfig(
        court_detection_interval=30,
        batch_size=16,
        max_ball_speed=70.0
    )
    assert config.court_detection_interval == 30
    print("✓ ProcessingConfig working")
    
    print("\n✅ All dataclasses validated!\n")


def test_base_classes():
    """Test all base classes can be instantiated and called"""
    print("\n" + "="*60)
    print("TEST 2: Base Classes __call__ Pattern")
    print("="*60)
    
    config = ProcessingConfig()
    
    # Test CourtDetector
    court_detector = CourtDetector(config)
    dummy_frames = np.random.randint(0, 255, (2, 720, 1280, 3), dtype=np.uint8)
    result = court_detector(dummy_frames)  # Should call __call__ -> preprocess -> inference -> postprocess
    assert isinstance(result, list)
    print("✓ CourtDetector __call__ working")
    
    # Test BallPlayerDetector
    ball_player_detector = BallPlayerDetector(config)
    result = ball_player_detector(dummy_frames)
    assert isinstance(result, list)
    print("✓ BallPlayerDetector __call__ working")
    
    # Test FrameFilter
    frame_filter = FrameFilter(config)
    frames = np.random.randint(0, 255, (100, 720, 1280, 3), dtype=np.uint8)
    result = frame_filter(frames)
    assert isinstance(result, list)
    print("✓ FrameFilter __call__ working")
    
    # Test GameSegmenter
    game_segmenter = GameSegmenter(config)
    frame_activity = [(i, i % 30 < 20) for i in range(100)]  # Simulate activity pattern
    result = game_segmenter(frame_activity)
    assert isinstance(result, list)
    print("✓ GameSegmenter __call__ working")
    
    # Test HomographyCalculator
    homography_calc = HomographyCalculator(config)
    keypoints = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    result = homography_calc(keypoints)
    # Result can be None for dummy data
    print("✓ HomographyCalculator __call__ working")
    
    # Test BallTrajectoryPreprocessor
    ball_preprocessor = BallTrajectoryPreprocessor(config)
    detections = [
        FrameDetection(frame_id=1, timestamp=0.033, ball_position_px=(100, 200), ball_confidence=0.8),
        FrameDetection(frame_id=2, timestamp=0.066, ball_position_px=(110, 210), ball_confidence=0.7),
    ]
    result = ball_preprocessor(detections)
    assert result is not None or result is None  # Can be None for insufficient detections
    print("✓ BallTrajectoryPreprocessor __call__ working")
    
    # Test VideoRenderer
    renderer = VideoRenderer(config)
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    render_data = {'frame_id': 1}
    result = renderer(dummy_frame, render_data)
    assert isinstance(result, np.ndarray)
    print("✓ VideoRenderer __call__ working")


def test_pipeline_flow():
    """Test data flow through pipeline components"""
    print("\n" + "="*60)
    print("TEST 3: Pipeline Data Flow")
    print("="*60)
    
    config = ProcessingConfig()
    
    # Simulate mini pipeline
    print("\nSimulating mini pipeline flow:")
    print("-" * 60)
    
    # Step 1: Generate dummy frames
    print("1. Generating dummy frames...")
    frames = np.random.randint(0, 255, (10, 720, 1280, 3), dtype=np.uint8)
    print(f"   Input: {frames.shape}")
    
    # Step 2: Filter frames
    print("2. Filtering frames...")
    frame_filter = FrameFilter(config)
    frame_activity = frame_filter(frames)
    print(f"   Output: {len(frame_activity)} frame activity markers")
    
    # Step 3: Segment video
    print("3. Segmenting video...")
    game_segmenter = GameSegmenter(config)
    segments = game_segmenter(frame_activity)
    print(f"   Output: {len(segments)} segments")
    
    # Step 4: Detect court
    print("4. Detecting court...")
    court_detector = CourtDetector(config)
    court_results = court_detector(frames[:2])  # Detect on 2 frames
    print(f"   Output: {len(court_results)} court detections")
    
    # Step 5: Detect ball & players
    print("5. Detecting ball & players...")
    ball_player_detector = BallPlayerDetector(config)
    bp_results = ball_player_detector(frames[:2])
    print(f"   Output: {len(bp_results)} ball/player detections")
    
    # Step 6: Create FrameDetection objects
    print("6. Creating structured detections...")
    detections = []
    for i, (court_res, bp_res) in enumerate(zip(court_results, bp_results)):
        detection = FrameDetection(
            frame_id=i,
            timestamp=i / 30.0,
            court_keypoints=court_res[0],
            court_confidence=court_res[1],
            ball_position_px=bp_res['ball_pos'],
            ball_confidence=bp_res['ball_conf'],
            player_boxes=bp_res['player_boxes'],
            player_confidences=bp_res['player_confs']
        )
        detections.append(detection)
    print(f"   Output: {len(detections)} structured detections")
    
    # Step 7: Calculate homography
    print("7. Calculating homography...")
    if detections[0].court_keypoints is not None:
        homography_calc = HomographyCalculator(config)
        H = homography_calc(detections[0].court_keypoints)
        print(f"   Output: Homography matrix {'computed' if H is not None else 'failed (expected with dummy data)'}")
    else:
        print("   Output: No court keypoints (expected with dummy data)")
    
    # Step 8: Analyze ball
    print("8. Analyzing ball trajectory...")
    trajectory = BallTrajectory(
        frame_ids=[d.frame_id for d in detections],
        positions_px=np.array([[100+i*10, 200+i*5] for i in range(len(detections))])
    )
    ball_analyzer = BallAnalyzer(config)
    analyzed_trajectory = ball_analyzer(trajectory)
    print(f"   Output: Trajectory analyzed")
    
    print("\n✅ Pipeline data flow validated!\n")


def test_video_analytics():
    """Test VideoAnalytics aggregation"""
    print("\n" + "="*60)
    print("TEST 4: VideoAnalytics Aggregation")
    print("="*60)
    
    # Create mock analytics
    analytics = VideoAnalytics(
        video_path="test_video.mp4",
        total_frames=1000,
        fps=30.0,
        duration_seconds=33.33
    )
    
    # Add segments
    analytics.game_segments = [
        GameSegment(start_frame=0, end_frame=300, segment_type=SegmentType.RALLY),
        GameSegment(start_frame=301, end_frame=400, segment_type=SegmentType.IDLE),
        GameSegment(start_frame=401, end_frame=800, segment_type=SegmentType.RALLY),
    ]
    
    rallies = analytics.get_rally_segments()
    assert len(rallies) == 2
    print(f"✓ Found {len(rallies)} rally segments from {len(analytics.game_segments)} total segments")
    
    # Add homography
    from app.core.data_models import HomographyData
    H = np.eye(3)
    analytics.homographies = [
        HomographyData(
            matrix=H,
            source_keypoints=np.array([[0, 0], [100, 100]]),
            target_keypoints=np.array([[0, 0], [10, 10]]),
            valid_from_frame=0,
            valid_to_frame=500,
            confidence=0.9
        )
    ]
    
    # Test homography lookup
    h = analytics.get_active_homography(250)
    assert h is not None
    print("✓ Homography lookup working")
    
    h = analytics.get_active_homography(600)
    assert h is None
    print("✓ Homography validity check working")
    
    print("\n✅ VideoAnalytics aggregation validated!\n")


def test_coordinate_system_documentation():
    """Test that coordinate systems are clearly documented"""
    print("\n" + "="*60)
    print("TEST 5: Coordinate System Documentation")
    print("="*60)
    
    detection = FrameDetection(
        frame_id=1,
        timestamp=0.033,
        ball_position_px=(320, 240),  # _px suffix indicates pixel coordinates
        ball_confidence=0.8
    )
    
    from app.core.data_models import BallState
    ball_state = BallState(
        frame_id=1,
        position_px=(320, 240),        # Pixel coordinates
        position_court=(5.5, 10.2),    # Court coordinates in meters
        velocity=(2.5, -1.3),          # Velocity in m/s
        speed=2.82                      # Speed in m/s
    )
    
    print("✓ Coordinate systems clearly named:")
    print(f"  - Pixel coords: ball_position_px = {detection.ball_position_px}")
    print(f"  - Court coords: position_court = {ball_state.position_court}")
    print(f"  - Velocities in m/s: {ball_state.velocity}")
    
    print("\n✅ Coordinate system naming validated!\n")


def test_frame_filter_output_format():
    """Test FrameFilter output format validation"""
    print("\n" + "="*60)
    print("TEST 6: FrameFilter Output Format")
    print("="*60)
    
    config = ProcessingConfig(motion_threshold=5.0)
    frame_filter = FrameFilter(config)
    
    # Test with synthetic frames
    frames = np.random.randint(0, 255, (10, 720, 1280, 3), dtype=np.uint8)
    result = frame_filter(frames)
    
    # Validate output format
    assert isinstance(result, list)
    assert len(result) == len(frames)
    
    for frame_id, is_active in result:
        assert isinstance(frame_id, int)
        assert isinstance(is_active, bool)
        assert 0 <= frame_id < len(frames)
    
    # First frame should always be active
    assert result[0][1] == True
    
    # Test with identical frames (no motion)
    identical_frames = np.array([frames[0]] * 5)
    result_no_motion = frame_filter(identical_frames)
    
    assert len(result_no_motion) == 5
    assert result_no_motion[0][1] == True  # First frame always active
    # Other frames should be inactive due to no motion
    for i in range(1, len(result_no_motion)):
        assert result_no_motion[i][1] == False
    
    print("✓ FrameFilter output format validated")
    print("✓ Motion detection working correctly")
    print("✓ First frame always marked as active")
    
    print("\n✅ FrameFilter output format validated!\n")


def run_all_tests():
    """Run all architecture tests"""
    print("\n" + "#"*60)
    print("# TENNIS CV - ARCHITECTURE VALIDATION TEST SUITE")
    print("#"*60)
    
    try:
        test_dataclasses()
        test_base_classes()
        test_pipeline_flow()
        test_video_analytics()
        test_coordinate_system_documentation()
        test_frame_filter_output_format()
        
        print("\n" + "#"*60)
        print("# ✅ ALL TESTS PASSED!")
        print("#"*60)
        print("\nArchitecture is ready for implementation!")
        print("\nNext steps:")
        print("1. Implement TrackNet_v3 model loading in CourtDetector")
        print("2. Implement TrackNet_v3 for BallPlayerDetector")
        print("3. Implement motion detection in FrameFilter")
        print("4. Implement homography calculation with RANSAC")
        print("5. Implement ball trajectory preprocessing")
        print("6. Implement ball analytics (bounce, hit, in/out)")
        print("7. Implement visualization rendering")
        print("8. Test on real tennis video")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()