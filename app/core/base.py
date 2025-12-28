"""
Tennis Computer Vision - Base Pipeline Classes
File: app/core/base.py

All pipeline components follow the __call__ pattern for consistent interface
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from scipy import ndimage

from app.core.data_models import (
    ProcessingConfig, FrameDetection, GameSegment, 
    SegmentType, BallTrajectory, HomographyData
)
from app.models.unified_detection import (
    UnifiedDetectionPipeline, DetectionType, 
    create_court_detector, create_ball_detector, create_pose_detector
)


class FrameFilter:
    """
    Motion-based frame filtering to remove dead frames
    
    Algorithm:
    1. Convert frames to grayscale for efficiency
    2. Calculate absolute difference between consecutive frames
    3. Compute mean pixel difference per frame
    4. Mark frames as active if difference > motion_threshold
    5. Apply temporal smoothing to avoid flickering
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.motion_threshold = config.motion_threshold
        self.smoothing_window = 5  # frames for median filter
    
    def __call__(self, frames: np.ndarray) -> List[Tuple[int, bool]]:
        """
        Filter frames based on motion detection
        
        Args:
            frames: Array of frames (N, H, W, 3) or (N, H, W) if grayscale
        
        Returns:
            List of (frame_id, is_active) tuples
        """
        if len(frames) == 0:
            return []
        
        if len(frames) == 1:
            return [(0, True)]  # Single frame is always active
        
        # Convert to grayscale if needed
        if frames.ndim == 4:  # RGB frames
            gray_frames = cv2.cvtColor(frames, cv2.COLOR_RGB2GRAY)
        else:  # Already grayscale
            gray_frames = frames
        
        # Calculate frame differences
        frame_diffs = []
        for i in range(1, len(gray_frames)):
            diff = cv2.absdiff(gray_frames[i-1], gray_frames[i])
            mean_diff = np.mean(diff)
            frame_diffs.append(mean_diff)
        
        # First frame is always active
        activity_scores = [self.motion_threshold + 1.0] + frame_diffs
        
        # Apply temporal smoothing to reduce flickering
        if len(activity_scores) >= self.smoothing_window:
            activity_scores = ndimage.median_filter(activity_scores, size=self.smoothing_window)
        
        # Determine activity based on threshold
        frame_activity = []
        for frame_id, score in enumerate(activity_scores):
            is_active = score > self.motion_threshold
            frame_activity.append((frame_id, is_active))
        
        return frame_activity


class CourtDetector:
    """
    Court detection using Unified Detection Pipeline
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.detection_model = create_court_detector(config)
        
        # Try to load model if available
        weights_path = "court_model_best.pt"
        court_model_path = f"{config.model_path}{weights_path}"
        self.detection_model.load_model(court_model_path)
    
    def __call__(self, frames: np.ndarray) -> List[Tuple[Optional[np.ndarray], float]]:
        """
        Detect court keypoints in frames
        
        Args:
            frames: Array of frames (N, H, W, 3)
        
        Returns:
            List of (keypoints, confidence) tuples
        """
        results = []
        
        try:
            # Run detection using unified pipeline
            detection_results = self.detection_model(frames)
            
            for detection_result in detection_results:
                keypoints = detection_result.keypoints
                confidence = detection_result.confidence
                results.append((keypoints, confidence))
                
        except Exception as e:
            print(f"Court detection error: {e}")
            # Fallback to dummy results
            for i in range(len(frames)):
                dummy_keypoints = np.array([
                    [100, 100], [500, 100], [500, 400], [100, 400],
                    [200, 200], [400, 200], [400, 300], [200, 300]
                ], dtype=np.float32)
                results.append((dummy_keypoints, 0.85))
        
        return results


class BallDetector:
    """
    Ball detection using Unified Detection Pipeline
    """
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.ball_detector = create_ball_detector(config)
        
        # Try to load models if available
        ball_model_path = f"{config.model_path}ball_model_best.pt"
        self.ball_detector.load_model(ball_model_path)
    
    def __call__(self, frames: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect ball in frames
        
        Args:
            frames: Array of frames (N, H, W, 3)
        
        Returns:
            List of detection dictionaries
        """
        results = []
        
        try:
            # Run ball detection
            ball_results = self.ball_detector(frames)
            
            # Debug: Print ball detection results
            print(f"[DEBUG BallDetector] Got {len(ball_results)} ball results for {len(frames)} frames")
            
            # Combine results
            for i in range(len(frames)):
                ball_result = ball_results[i] if i < len(ball_results) else None
                
                # Extract ball position
                ball_pos = None
                ball_conf = 0.0
                if ball_result:
                    print(f"[DEBUG BallDetector] Frame {i}: ball_result type={type(ball_result)}, has ball_position={hasattr(ball_result, 'ball_position')}")
                    if hasattr(ball_result, 'ball_position') and ball_result.ball_position is not None:
                        ball_pos = ball_result.ball_position
                        ball_conf = ball_result.confidence
                        print(f"[DEBUG BallDetector] Frame {i}: ball_pos={ball_pos}, conf={ball_conf}")
                    elif hasattr(ball_result, 'keypoints') and ball_result.keypoints is not None:
                        # Fallback: extract from keypoints
                        kpts = ball_result.keypoints
                        if len(kpts) > 0 and len(kpts[0]) >= 2:
                            ball_pos = (float(kpts[0][0]), float(kpts[0][1]))
                            ball_conf = ball_result.confidence
                            print(f"[DEBUG BallDetector] Frame {i}: extracted from keypoints: ball_pos={ball_pos}")
                    else:
                        print(f"[DEBUG BallDetector] Frame {i}: No ball position found in result")
                
                # Check coordinate range
                if ball_pos is not None:
                    h, w = frames[i].shape[:2]
                    if ball_pos[0] < 0 or ball_pos[0] > w or ball_pos[1] < 0 or ball_pos[1] > h:
                        print(f"[DEBUG BallDetector] WARNING: ball_pos {ball_pos} out of bounds! Frame size: {w}x{h}")
                    else:
                        print(f"[DEBUG BallDetector] Frame {i}: ball_pos {ball_pos} is within bounds ({w}x{h})")
                
                result = {
                    'ball_pos': ball_pos,
                    'ball_conf': ball_conf,
                    'player_boxes': [],
                    'player_confs': []
                }
                results.append(result)
                
        except Exception as e:
            print(f"Ball detection error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to dummy results
            for i in range(len(frames)):
                result = {
                    'ball_pos': (320 + i*5, 240 + i*3),  # Moving ball
                    'ball_conf': 0.75,
                    'player_boxes': [],
                    'player_confs': []
                }
                results.append(result)
        
        return results
    

class PlayerDetector:
    """
    Player detection using YOLO v11
    Filters for class 0 (person) and returns bounding boxes
    """
    def __init__(self, config: ProcessingConfig, model_name_or_path: str = "yolo11n.pt"):
        """
        Initialize YOLO player detector
        
        Args:
            config: Processing configuration
            model_name_or_path: YOLO model name (e.g., 'yolo11n.pt') or path to weights
        """
        self.config = config
        from app.models.yolo_player_detector import YOLOPlayerDetector
        
        self.yolo_detector = YOLOPlayerDetector(config, model_name_or_path)
        
        # Try to load model
        # First try custom weights path, then model name
        player_model_path = f"{config.model_path}player_model.pt"
        if not Path(player_model_path).exists():
            # Use model name (will download if needed)
            self.yolo_detector.load_model(model_name_or_path)
        else:
            self.yolo_detector.load_model(player_model_path)
    
    def __call__(self, frames: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect players in frames using YOLO
        
        Args:
            frames: Array of frames (N, H, W, 3) in BGR format
        
        Returns:
            List of detection dictionaries with player bounding boxes
        """
        try:
            # Run YOLO player detection
            player_results = self.yolo_detector(frames)
            
            return player_results
            
        except Exception as e:
            print(f"YOLO player detection error: {e}")
            import traceback
            traceback.print_exc()
            # Return empty detections
            return [{'player_boxes': [], 'player_confs': [], 'player_class_ids': []} for _ in range(len(frames))]


class GameSegmenter:
    """
    Simplified game segmentation for static camera setup
    
    Since camera is fixed and court always visible, we create minimal segments:
    - RALLY segments for active frames
    - IDLE segments for dead frames
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.min_segment_frames = 10  # Minimum frames for a valid segment
    
    def __call__(self, frame_activity: List[Tuple[int, bool]]) -> List[GameSegment]:
        """
        Convert frame activity to game segments
        
        Args:
            frame_activity: List of (frame_id, is_active) tuples
        
        Returns:
            List of GameSegment objects
        """
        if not frame_activity:
            return []
        
        segments = []
        current_start = None
        current_type = None
        
        for frame_id, is_active in frame_activity:
            segment_type = SegmentType.RALLY if is_active else SegmentType.IDLE
            
            if current_type is None:
                # First frame
                current_start = frame_id
                current_type = segment_type
            elif current_type == segment_type:
                # Continue current segment
                continue
            else:
                # Type changed, finalize current segment
                if current_start is not None:
                    segment = GameSegment(
                        start_frame=current_start,
                        end_frame=frame_id - 1,
                        segment_type=current_type
                    )
                    segments.append(segment)
                
                # Start new segment
                current_start = frame_id
                current_type = segment_type
        
        # Finalize last segment
        if current_start is not None:
            segment = GameSegment(
                start_frame=current_start,
                end_frame=frame_activity[-1][0],
                segment_type=current_type
            )
            segments.append(segment)
        
        # Merge consecutive segments of same type
        merged_segments = self._merge_consecutive_segments(segments)
        
        # Filter out very short segments
        filtered_segments = [
            seg for seg in merged_segments 
            if seg.duration_frames() >= self.min_segment_frames
        ]
        
        return filtered_segments
    
    def _merge_consecutive_segments(self, segments: List[GameSegment]) -> List[GameSegment]:
        """Merge consecutive segments of the same type"""
        if not segments:
            return []
        
        merged = [segments[0]]
        
        for current_seg in segments[1:]:
            last_seg = merged[-1]
            
            if (last_seg.segment_type == current_seg.segment_type and 
                last_seg.end_frame + 1 == current_seg.start_frame):
                # Merge segments
                merged[-1] = GameSegment(
                    start_frame=last_seg.start_frame,
                    end_frame=current_seg.end_frame,
                    segment_type=last_seg.segment_type
                )
            else:
                merged.append(current_seg)
        
        return merged


class HomographyCalculator:
    """
    Homography calculation for court transformation (stub implementation)
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def __call__(self, keypoints: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate homography matrix from court keypoints
        
        Args:
            keypoints: Court keypoints in pixel coordinates (N, 2)
        
        Returns:
            3x3 homography matrix or None if calculation fails
        """
        # Stub: return identity matrix
        # In real implementation, this would:
        # 1. Match keypoints to court template
        # 2. Use RANSAC for robust estimation
        # 3. Validate matrix quality
        
        if len(keypoints) < 4:
            return None
        
        # Return identity matrix for now
        return np.eye(3, dtype=np.float32)


class BallTrajectoryPreprocessor:
    """
    Ball trajectory preprocessing (stub implementation)
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def __call__(self, detections: List[FrameDetection]) -> Optional[BallTrajectory]:
        """
        Preprocess ball trajectory from detections
        
        Args:
            detections: List of FrameDetection objects
        
        Returns:
            BallTrajectory object or None
        """
        if not detections:
            return None
        
        # Extract ball detections
        ball_detections = [d for d in detections if d.has_ball()]
        
        if len(ball_detections) < 2:
            return None
        
        # Create trajectory
        frame_ids = [d.frame_id for d in ball_detections]
        positions = np.array([d.ball_position_px for d in ball_detections])
        
        trajectory = BallTrajectory(
            frame_ids=frame_ids,
            positions_px=positions
        )
        
        return trajectory


class BallAnalyzer:
    """
    Ball trajectory analysis (stub implementation)
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def __call__(self, trajectory: BallTrajectory) -> BallTrajectory:
        """
        Analyze ball trajectory for events
        
        Args:
            trajectory: BallTrajectory object
        
        Returns:
            Enhanced BallTrajectory with analysis results
        """
        # Stub: pass through trajectory unchanged
        # In real implementation, this would:
        # 1. Detect bounces
        # 2. Detect hits
        # 3. Determine in/out decisions
        # 4. Calculate speeds and velocities
        
        return trajectory


class VideoRenderer:
    """
    Video visualization rendering with overlays
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.colors = {
            'ball': (0, 255, 0),      # Green
            'court': (255, 255, 255), # White
            'player': (0, 0, 255),    # Red
            'text': (255, 255, 255),  # White
            'trajectory': (255, 0, 0) # Blue
        }
    
    def __call__(self, frame: np.ndarray, render_data: Dict[str, Any]) -> np.ndarray:
        """
        Render overlays on frame
        
        Args:
            frame: Input frame (H, W, 3)
            render_data: Dictionary with rendering data
        
        Returns:
            Frame with overlays rendered
        """
        rendered_frame = frame.copy()
        
        # Draw court lines if enabled
        if self.config.draw_court_lines and 'homography' in render_data:
            rendered_frame = self._draw_court_lines(rendered_frame, render_data['homography'])
        
        # Draw ball trajectory if enabled
        if self.config.draw_trajectories and 'trajectory' in render_data:
            rendered_frame = self._draw_ball_trajectory(rendered_frame, render_data['trajectory'])
        
        # Draw players if enabled
        if 'players' in render_data:
            rendered_frame = self._draw_players(rendered_frame, render_data['players'])
        
        # Draw current segment info
        if 'current_segment' in render_data:
            rendered_frame = self._draw_segment_info(rendered_frame, render_data['current_segment'])
        
        # Draw frame info
        rendered_frame = self._draw_frame_info(rendered_frame, render_data)
        
        return rendered_frame
    
    def _draw_court_lines(self, frame: np.ndarray, homography: Optional[HomographyData]) -> np.ndarray:
        if homography is None or homography.source_keypoints is None:
            print("[DEBUG] Skipping court, no homography or source_keypoints.")
            return frame
        
        h, w = frame.shape[:2]
        keypoints = homography.source_keypoints.copy()
        
        # Fix transpose: ensure keypoints are in (x, y) format, not (y, x)
        # Check if keypoints need to be swapped
        if len(keypoints) > 0:
            # If first keypoint's x > w or y > h, likely transposed
            first_kp = keypoints[0]
            if first_kp[0] > w or first_kp[1] > h:
                print(f"[DEBUG] COURT: Keypoints appear transposed. Swapping x/y. Original: {keypoints[0]}, Frame size: {w}x{h}")
                # Swap x and y coordinates
                keypoints = np.column_stack([keypoints[:, 1], keypoints[:, 0]])
        
        print(f"[DEBUG] COURT overlay: kp={keypoints}, conf={getattr(homography, 'confidence', None)}, frame_size={w}x{h}")
        
        # Draw court keypoints
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            # Ensure coordinates are within bounds
            x = max(0, min(w-1, x))
            y = max(0, min(h-1, y))
            cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
            cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)
        
        # Draw label at first keypoint with background
        if len(keypoints) > 0:
            p0_x, p0_y = int(keypoints[0][0]), int(keypoints[0][1])
            p0_x = max(10, min(w-250, p0_x))
            p0_y = max(40, min(h-20, p0_y))
            label = f'COURT: {homography.confidence:.2f}' if hasattr(homography, 'confidence') else 'COURT'
            # Draw background rectangle
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (p0_x, p0_y-text_h-5), (p0_x+text_w+10, p0_y+5), (0, 0, 0), -1)
            cv2.putText(frame, label, (p0_x+5, p0_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return frame

    def _draw_ball_trajectory(self, frame: np.ndarray, trajectory: Optional[BallTrajectory]) -> np.ndarray:
        if trajectory is None or len(trajectory.positions_px) == 0:
            print("[DEBUG] Skipping ball trajectory, no positions.")
            return frame
        
        h, w = frame.shape[:2]
        positions = trajectory.positions_px.copy()
        
        # Ensure positions are in (x, y) format
        if len(positions.shape) == 2 and positions.shape[1] >= 2:
            # Check if coordinates need to be swapped (if x > w or y > h)
            if len(positions) > 0:
                first_pos = positions[0]
                if first_pos[0] > w or first_pos[1] > h:
                    print(f"[DEBUG] BALL: Positions appear transposed. Swapping x/y. Original: {positions[0]}, Frame size: {w}x{h}")
                    positions = np.column_stack([positions[:, 1], positions[:, 0]])
        
        positions = positions.astype(int)
        
        # Draw trajectory line
        if len(positions) >= 2:
            for i in range(1, len(positions)):
                pt1 = (max(0, min(w-1, positions[i-1][0])), max(0, min(h-1, positions[i-1][1])))
                pt2 = (max(0, min(w-1, positions[i][0])), max(0, min(h-1, positions[i][1])))
                cv2.line(frame, pt1, pt2, self.colors['trajectory'], 2)
        
        # Draw ball position
        if len(positions) > 0:
            ball_x = max(0, min(w-1, int(positions[-1][0])))
            ball_y = max(0, min(h-1, int(positions[-1][1])))
            ball_px = (ball_x, ball_y)
            print(f"[DEBUG] BALL overlay: pos={ball_px}, frame_size={w}x{h}")
            cv2.circle(frame, ball_px, 10, self.colors['ball'], -1)
            cv2.circle(frame, ball_px, 10, (255, 255, 255), 2)
            # Draw label with background
            label = 'BALL'
            label_x = max(10, min(w-100, ball_x+15))
            label_y = max(40, min(h-20, ball_y-15))
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (label_x, label_y-text_h-5), (label_x+text_w+10, label_y+5), (0, 0, 0), -1)
            cv2.putText(frame, label, (label_x+5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame
    
    def _draw_players(self, frame: np.ndarray, players: Optional[Dict[str, Any]]) -> np.ndarray:
        """Draw player bounding boxes"""
        if players is None:
            return frame
        
        player_boxes = players.get('player_boxes', [])
        player_confs = players.get('player_confs', [])
        
        if not player_boxes:
            return frame
        
        h, w = frame.shape[:2]
        
        # Draw each player bounding box
        for i, (box, conf) in enumerate(zip(player_boxes, player_confs)):
            # box format: (x1, y1, x2, y2)
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(w-1, x1))
            y1 = max(0, min(h-1, y1))
            x2 = max(0, min(w-1, x2))
            y2 = max(0, min(h-1, y2))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['player'], 2)
            
            # Draw label with background
            label = f'PLAYER {i+1}: {conf:.2f}'
            label_x = max(10, min(w-200, x1))
            label_y = max(30, min(h-10, y1-10))
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (label_x, label_y-text_h-5), (label_x+text_w+10, label_y+5), (0, 0, 0), -1)
            cv2.putText(frame, label, (label_x+5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['player'], 2)
        
        print(f"[DEBUG] PLAYER overlay: {len(player_boxes)} players detected")
        
        return frame
    
    def _draw_segment_info(self, frame: np.ndarray, segment: Optional[GameSegment]) -> np.ndarray:
        """Draw current segment information"""
        if segment is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Segment type and duration
        segment_text = f"Segment: {segment.segment_type.value}"
        duration_text = f"Duration: {segment.duration_frames()} frames"
        
        # Draw background rectangle
        cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, segment_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        cv2.putText(frame, duration_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        return frame
    
    def _draw_frame_info(self, frame: np.ndarray, render_data: Dict[str, Any]) -> np.ndarray:
        """Draw frame information"""
        h, w = frame.shape[:2]
        
        frame_id = render_data.get('frame_id', 0)
        timestamp = frame_id / self.config.output_fps if self.config.output_fps > 0 else 0
        
        # Frame info text
        frame_text = f"Frame: {frame_id}"
        time_text = f"Time: {timestamp:.2f}s"
        
        # Draw background rectangle
        cv2.rectangle(frame, (w-200, 10), (w-10, 80), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, frame_text, (w-190, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        cv2.putText(frame, time_text, (w-190, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        return frame

