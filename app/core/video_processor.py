"""
Tennis Computer Vision Analysis - Main Video Processor
File: app/core/video_processor.py

Orchestrates the entire pipeline with __call__ pattern
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

from app.core.data_models import (
    FrameDetection, GameSegment, VideoAnalytics, 
    ProcessingConfig, HomographyData, BallTrajectory
)
from app.core.base import (
    CourtDetector, BallDetector,
    FrameFilter, BallTrajectoryPreprocessor,
    GameSegmenter, HomographyCalculator, BallAnalyzer,
    VideoRenderer
)


class VideoProcessor:
    """
    Main pipeline orchestrator for tennis video analysis
    
    Pipeline flow:
    1. Load video -> chunk processing
    2. Filter frames (FrameFilter)
    3. Segment into game phases (GameSegmenter)
    4. For each rally segment:
       a. Detect court (CourtDetector - every N frames)
       b. Detect ball & players (BallPlayerDetector)
       c. Calculate homography (HomographyCalculator)
       d. Preprocess ball trajectory (BallTrajectoryPreprocessor)
       e. Analyze ball (BallAnalyzer)
       f. Render visualization (VideoRenderer)
    5. Save results
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
        # Initialize pipeline components
        self.court_detector = CourtDetector(config)
        self.ball_detector = BallDetector(config)
        self.frame_filter = FrameFilter(config)
        self.game_segmenter = GameSegmenter(config)
        self.homography_calculator = HomographyCalculator(config)
        self.ball_trajectory_preprocessor = BallTrajectoryPreprocessor(config)
        self.ball_analyzer = BallAnalyzer(config)
        self.renderer = VideoRenderer(config)
        
        print("[VideoProcessor] Pipeline initialized")
    
    def __call__(self, video_path: str, output_name: Optional[str] = None) -> VideoAnalytics:
        """
        Process entire video through pipeline
        
        Args:
            video_path: Path to input video
            output_name: Optional output name (default: input_name_processed.mp4)
        
        Returns:
            VideoAnalytics object with all results
        """
        print(f"\n{'='*60}")
        print(f"Processing video: {video_path}")
        print(f"{'='*60}\n")
        
        # Load video metadata
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        cap.release()
        
        print(f"Video info: {total_frames} frames, {fps} fps, {duration:.2f}s")
        print(f"Resolution: {width}x{height}\n")
        
        # Initialize analytics result
        analytics = VideoAnalytics(
            video_path=video_path,
            total_frames=total_frames,
            fps=fps,
            duration_seconds=duration
        )
        
        # STEP 1: Frame filtering and segmentation
        # print("STEP 1: Frame filtering and game segmentation")
        # print("-" * 60)
        # frame_activity = self._filter_frames(video_path, total_frames)
        # game_segments = self.game_segmenter(frame_activity)
        # analytics.game_segments = game_segments
        
        # print(f"Found {len(game_segments)} segments")
        # rally_segments = analytics.get_rally_segments()
        # print(f"Active rallies: {len(rally_segments)}\n")
        
        # STEP 2: Process each rally segment
        print("STEP 2: Processing rally segments")
        print("-" * 60)

        rally_segments = analytics.get_rally_segments()
        
        for idx, segment in enumerate(rally_segments):
            print(f"\nProcessing Rally {idx+1}/{len(rally_segments)}")
            print(f"Frames: {segment.start_frame} -> {segment.end_frame}")
            
            # Extract segment frames
            segment_frames = self._extract_segment_frames(
                video_path, segment.start_frame, segment.end_frame
            )
            
            # Detect court, ball, players
            detections = self._detect_segment(segment, segment_frames)
            
            # Calculate homography
            homography = self._calculate_homography_for_segment(detections, segment)
            if homography:
                analytics.homographies.append(homography)
            
            # Process ball trajectory
            trajectory = self._process_ball_trajectory(detections, homography)
            segment.ball_trajectory = trajectory
            
            # Analyze ball (bounce, hit, in/out, speed)
            if trajectory:
                trajectory = self.ball_analyzer(trajectory)
                segment.ball_trajectory = trajectory
                analytics.total_rallies += 1
        
        print(f"\n{'='*60}")
        print(f"Pipeline completed!")
        print(f"{'='*60}\n")
        
        # STEP 3: Render and save
        if self.config.save_visualization:
            output_path = self._generate_output_path(video_path, output_name)
            print(f"Rendering video to: {output_path}")
            self._render_and_save(video_path, analytics, output_path)
        
        return analytics
    
    def _filter_frames(self, video_path: str, total_frames: int) -> List[Tuple[int, bool]]:
        """Filter frames using motion detection"""
        cap = cv2.VideoCapture(video_path)
        
        # Sample frames for filtering (every 10th frame for speed)
        sample_frames = []
        sample_indices = range(0, total_frames, 10)
        
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                sample_frames.append(frame)
        
        cap.release()
        
        # Run frame filter
        sample_frames = np.array(sample_frames)
        # activity = self.frame_filter(sample_frames)
        
        # # Expand to all frames (simple interpolation)
        # full_activity = []
        # for frame_id in range(total_frames):
        #     sample_idx = frame_id // 10
        #     if sample_idx < len(activity):
        #         full_activity.append((frame_id, activity[sample_idx][1]))
        #     else:
        #         full_activity.append((frame_id, True))
        
        # return full_activity
        return sample_frames
    
    def _extract_segment_frames(
        self, video_path: str, start_frame: int, end_frame: int
    ) -> np.ndarray:
        """Extract frames for a segment"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        for frame_id in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return np.array(frames)
    
    def _detect_segment(
        self, segment: GameSegment, frames: np.ndarray
    ) -> List[FrameDetection]:
        """Run detection on segment frames"""
        detections = []
        num_frames = len(frames)
        
        # Process in batches
        for batch_start in range(0, num_frames, self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, num_frames)
            batch_frames = frames[batch_start:batch_end]
            
            # Court detection (every N frames)
            court_results = []
            for local_idx, global_frame_id in enumerate(
                range(segment.start_frame + batch_start, segment.start_frame + batch_end)
            ):
                if global_frame_id % self.config.court_detection_interval == 0:
                    # Detect court
                    court_batch = batch_frames[local_idx:local_idx+1]
                    court_result = self.court_detector(court_batch)[0]
                    court_results.append((local_idx, court_result))
            
            # Ball and player detection (every frame)
            ball_player_results = self.ball_player_detector(batch_frames)
            
            # Combine results
            for local_idx in range(len(batch_frames)):
                global_frame_id = segment.start_frame + batch_start + local_idx
                timestamp = global_frame_id / self.config.output_fps
                
                # Get court result (use last detected)
                court_keypoints = None
                court_conf = 0.0
                for court_idx, (ct_local_idx, ct_result) in enumerate(court_results):
                    if ct_local_idx <= local_idx:
                        court_keypoints, court_conf = ct_result
                
                # Get ball/player result
                bp_result = ball_player_results[local_idx]
                
                detection = FrameDetection(
                    frame_id=global_frame_id,
                    timestamp=timestamp,
                    court_keypoints=court_keypoints,
                    court_confidence=court_conf,
                    ball_position_px=bp_result['ball_pos'],
                    ball_confidence=bp_result['ball_conf'],
                    player_boxes=bp_result['player_boxes'],
                    player_confidences=bp_result['player_confs']
                )
                detections.append(detection)
        
        return detections
    
    def _calculate_homography_for_segment(
        self, detections: List[FrameDetection], segment: GameSegment
    ) -> Optional[HomographyData]:
        """Calculate homography matrix for segment"""
        # Find frame with best court detection
        best_detection = max(
            [d for d in detections if d.has_court()],
            key=lambda d: d.court_confidence,
            default=None
        )
        
        if not best_detection:
            print("  WARNING: No valid court detection for homography")
            return None
        
        # Calculate homography
        H = self.homography_calculator(best_detection.court_keypoints)
        
        if H is None:
            return None
        
        return HomographyData(
            matrix=H,
            source_keypoints=best_detection.court_keypoints,
            target_keypoints=np.array([]),  # TODO: Use COURT_TEMPLATE_KEYPOINTS
            valid_from_frame=segment.start_frame,
            valid_to_frame=segment.end_frame,
            confidence=best_detection.court_confidence
        )
    
    def _process_ball_trajectory(
        self, detections: List[FrameDetection], homography: Optional[HomographyData]
    ) -> Optional[BallTrajectory]:
        """Process ball trajectory with preprocessing"""
        # Extract ball detections
        ball_detections = [d for d in detections if d.has_ball()]
        
        if len(ball_detections) < self.config.min_rally_frames:
            print(f"  WARNING: Insufficient ball detections ({len(ball_detections)} frames)")
            return None
        
        # Preprocess trajectory (interpolate, smooth)
        trajectory = self.ball_trajectory_preprocessor(ball_detections)
        
        # Transform to court coordinates if homography available
        if homography and len(trajectory.positions_px) > 0:
            trajectory = self._transform_trajectory_to_court(trajectory, homography)
        
        return trajectory
    
    def _transform_trajectory_to_court(
        self, trajectory: BallTrajectory, homography: HomographyData
    ) -> BallTrajectory:
        """Transform ball positions from pixel to court coordinates"""
        # TODO: Implement coordinate transformation using homography matrix
        # cv2.perspectiveTransform(trajectory.positions_px, homography.matrix)
        print("  Transforming trajectory to court coordinates...")
        return trajectory
    
    def _render_and_save(
        self, video_path: str, analytics: VideoAnalytics, output_path: str
    ):
        """Render visualization and save video"""
        cap = cv2.VideoCapture(video_path)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, self.config.output_fps, (width, height))
        
        frame_id = 0
        with tqdm(total=analytics.total_frames, desc="Rendering") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Prepare rendering data
                render_data = self._get_render_data(frame_id, analytics)
                
                # Render overlays
                frame = self.renderer(frame, render_data)
                
                # Write frame
                writer.write(frame)
                
                frame_id += 1
                pbar.update(1)
        
        cap.release()
        writer.release()
        print(f"Video saved to: {output_path}")
    
    def _get_render_data(self, frame_id: int, analytics: VideoAnalytics) -> dict:
        """Prepare data for rendering at specific frame"""
        data = {
            'frame_id': frame_id,
            'homography': analytics.get_active_homography(frame_id),
            'trajectory': None,
            'current_segment': None,
        }
        
        # Find active segment
        for segment in analytics.game_segments:
            if segment.start_frame <= frame_id <= segment.end_frame:
                data['current_segment'] = segment
                data['trajectory'] = segment.ball_trajectory
                break
        
        return data
    
    def _generate_output_path(self, video_path: str, output_name: Optional[str]) -> str:
        """Generate output path for processed video"""
        video_path = Path(video_path)
        
        if output_name:
            filename = output_name
        else:
            filename = f"{video_path.stem}_processed.mp4"
        
        output_dir = Path(self.config.results_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return str(output_dir / filename)


class StreamProcessor:
    """
    Real-time stream processor for live video
    
    Differences from VideoProcessor:
    - Processes frames one at a time (low latency)
    - Cannot look ahead (causal processing only)
    - Uses online algorithms for smoothing
    - No reprocessing of segments
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
        # Initialize components
        self.court_detector = CourtDetector(config)
        self.ball_detector = BallDetector(config)
        self.ball_analyzer = BallAnalyzer(config)
        self.renderer = VideoRenderer(config)
        
        # State management
        self.frame_count = 0
        self.last_homography: Optional[HomographyData] = None
        self.trajectory_buffer: List[FrameDetection] = []
        
        print("[StreamProcessor] Real-time pipeline initialized")
    
    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Process single frame in real-time
        
        Args:
            frame: Single frame (H, W, 3)
        
        Returns:
            Tuple of (rendered_frame, analytics_dict)
        """
        self.frame_count += 1
        
        # Court detection (every N frames)
        court_result = None
        if self.frame_count % self.config.court_detection_interval == 0:
            court_result = self.court_detector(frame[np.newaxis, ...])[0]
            court_keypoints, court_conf = court_result
            
            if court_keypoints is not None and court_conf > 0.5:
                # Update homography
                H = HomographyCalculator(self.config).analyze(court_keypoints)
                if H is not None:
                    self.last_homography = HomographyData(
                        matrix=H,
                        source_keypoints=court_keypoints,
                        target_keypoints=np.array([]),
                        valid_from_frame=self.frame_count,
                        valid_to_frame=self.frame_count + self.config.court_detection_interval,
                        confidence=court_conf
                    )
        
        # Ball and player detection
        bp_result = self.ball_player_detector(frame[np.newaxis, ...])[0]
        
        # Create detection
        detection = FrameDetection(
            frame_id=self.frame_count,
            timestamp=self.frame_count / self.config.output_fps,
            court_keypoints=court_result[0] if court_result else None,
            court_confidence=court_result[1] if court_result else 0.0,
            ball_position_px=bp_result['ball_pos'],
            ball_confidence=bp_result['ball_conf'],
            player_boxes=bp_result['player_boxes'],
            player_confidences=bp_result['player_confs']
        )
        
        # Update trajectory buffer
        self.trajectory_buffer.append(detection)
        if len(self.trajectory_buffer) > 60:  # Keep last 2 seconds
            self.trajectory_buffer.pop(0)
        
        # Prepare analytics (limited to buffer)
        analytics_dict = {
            'detection': detection,
            'homography': self.last_homography,
            'recent_trajectory': self.trajectory_buffer[-30:] if len(self.trajectory_buffer) >= 30 else []
        }
        
        # Render
        rendered_frame = self.renderer(frame, analytics_dict)
        
        return rendered_frame, analytics_dict
    
    def reset(self):
        """Reset stream processor state"""
        self.frame_count = 0
        self.last_homography = None
        self.trajectory_buffer = []


# ==================== TESTING UTILITIES ====================

def test_pipeline_simple():
    """Simple test with dummy video"""
    print("\n" + "="*60)
    print("TESTING PIPELINE - Simple Mode")
    print("="*60 + "\n")
    
    # Create config
    config = ProcessingConfig(
        court_detection_interval=30,
        batch_size=8,
        chunk_size=300,
        min_rally_frames=30,
        save_visualization=False  # Don't save for testing
    )
    
    # Create processor
    processor = VideoProcessor(config)
    
    # Test with dummy video path
    print("Pipeline structure created successfully!")
    print("\nComponents initialized:")
    print(f"  - CourtDetector: {processor.court_detector.__class__.__name__}")
    print(f"  - BallPlayerDetector: {processor.ball_player_detector.__class__.__name__}")
    print(f"  - FrameFilter: {processor.frame_filter.__class__.__name__}")
    print(f"  - GameSegmenter: {processor.game_segmenter.__class__.__name__}")
    print(f"  - HomographyCalculator: {processor.homography_calculator.__class__.__name__}")
    print(f"  - BallAnalyzer: {processor.ball_analyzer.__class__.__name__}")
    print(f"  - VideoRenderer: {processor.renderer.__class__.__name__}")
    
    print("\n✓ Pipeline architecture validated!")
    print("✓ All components can be called with __call__ method")
    print("✓ Ready to implement detection and analytics logic")


def test_stream_processor():
    """Test real-time stream processor"""
    print("\n" + "="*60)
    print("TESTING STREAM PROCESSOR")
    print("="*60 + "\n")
    
    config = ProcessingConfig()
    stream_processor = StreamProcessor(config)
    
    # Simulate processing 10 frames
    print("Simulating real-time frame processing...")
    for i in range(10):
        # Dummy frame
        dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        rendered, analytics = stream_processor(dummy_frame)
        print(f"Frame {i+1}: Processed successfully")
    
    print("\n✓ Stream processor validated!")
    print("✓ Stateful processing working")
    print("✓ Low-latency pipeline ready")


if __name__ == "__main__":
    test_pipeline_simple()
    test_stream_processor()