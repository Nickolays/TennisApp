"""
Tennis Computer Vision Analysis - Core Data Structures
File: app/core/data_models.py
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np


class SegmentType(Enum):
    """Types of video segments"""
    RALLY = "rally"
    IDLE = "idle"
    PREPARATION = "preparation"
    UNKNOWN = "unknown"


class CourtRegion(Enum):
    """Court regions for ball position"""
    IN_BOUNDS = "in_bounds"
    OUT_BOUNDS = "out_bounds"
    NET = "net"
    UNKNOWN = "unknown"


@dataclass
class FrameDetection:
    """Detection results for a single frame"""
    frame_id: int
    timestamp: float  # seconds
    
    # Court detection (keypoints in pixel coordinates)
    court_keypoints: Optional[np.ndarray] = None  # Shape: (N, 2) - court line intersections
    court_confidence: float = 0.0
    
    # Ball detection
    ball_position_px: Optional[Tuple[float, float]] = None  # (x, y) in pixels
    ball_confidence: float = 0.0
    
    # Player detections
    player_boxes: List[Tuple[float, float, float, float]] = field(default_factory=list)  # [(x1,y1,x2,y2), ...]
    player_confidences: List[float] = field(default_factory=list)
    player_track_ids: List[int] = field(default_factory=list)  # Track IDs from ByteTrack (matches player_boxes order)
    
    def has_court(self) -> bool:
        return self.court_keypoints is not None and self.court_confidence > 0.5

    def has_ball(self) -> bool:
        return self.ball_position_px is not None and self.ball_confidence > 0.3

    def has_players(self) -> bool:
        return len(self.player_boxes) > 0


@dataclass
class TemporalWindow:
    """Temporal window of frames for event detection"""
    center_frame_id: int
    frame_ids: List[int]  # All frame IDs in window (e.g., ±5 frames = 11 frames)
    frames: Optional[List[np.ndarray]] = None  # Actual frame data (if available)
    ball_positions: List[Optional[Tuple[float, float]]] = field(default_factory=list)
    center_ball_position: Optional[Tuple[float, float]] = None
    center_ball_confidence: float = 0.0

    def window_size(self) -> int:
        """Get window size (half-width)"""
        return (len(self.frame_ids) - 1) // 2

    def is_complete(self, expected_size: int) -> bool:
        """Check if window has expected number of frames"""
        return len(self.frame_ids) == 2 * expected_size + 1


@dataclass
class BallState:
    """Ball state at a specific frame (after homography transformation)"""
    frame_id: int
    position_px: Tuple[float, float]  # Pixel coordinates
    position_court: Optional[Tuple[float, float]] = None  # Court coordinates (meters)
    velocity: Optional[Tuple[float, float]] = None  # m/s
    speed: Optional[float] = None  # m/s
    is_bounce: bool = False
    is_hit: bool = False
    court_region: CourtRegion = CourtRegion.UNKNOWN


@dataclass
class BallTrajectory:
    """Complete ball trajectory for a rally"""
    frame_ids: List[int] = field(default_factory=list)
    positions_px: np.ndarray = field(default_factory=lambda: np.array([]))  # Shape: (N, 2)
    positions_court: np.ndarray = field(default_factory=lambda: np.array([]))  # Shape: (N, 2)
    velocities: np.ndarray = field(default_factory=lambda: np.array([]))  # Shape: (N, 2)
    speeds: List[float] = field(default_factory=list)
    
    bounces: List[int] = field(default_factory=list)  # Frame IDs where ball bounced
    hits: List[int] = field(default_factory=list)  # Frame IDs where ball was hit
    in_out_decisions: List[Tuple[int, CourtRegion]] = field(default_factory=list)  # (frame_id, region)
    
    def get_max_speed(self) -> Optional[float]:
        return max(self.speeds) if self.speeds else None
    
    def get_rally_duration(self, fps: float) -> float:
        """Duration in seconds"""
        if len(self.frame_ids) < 2:
            return 0.0
        return (self.frame_ids[-1] - self.frame_ids[0]) / fps


@dataclass
class GameSegment:
    """Video segment representing a game phase"""
    start_frame: int
    end_frame: int
    segment_type: SegmentType
    
    # Analytics results for this segment
    ball_trajectory: Optional[BallTrajectory] = None
    score: Optional[Tuple[int, int]] = None  # (player1, player2)
    winner: Optional[int] = None  # Player index who won the point
    
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1
    
    def duration_seconds(self, fps: float) -> float:
        return self.duration_frames() / fps


@dataclass
class HomographyData:
    """Homography transformation data"""
    matrix: np.ndarray  # 3x3 transformation matrix
    source_keypoints: np.ndarray  # Court keypoints in pixel space
    target_keypoints: np.ndarray  # Court keypoints in world space (meters)
    valid_from_frame: int
    valid_to_frame: int
    confidence: float = 1.0
    
    def is_valid_for_frame(self, frame_id: int) -> bool:
        return self.valid_from_frame <= frame_id <= self.valid_to_frame


@dataclass
class PlayerStats:
    """Statistics for a player"""
    player_id: int
    total_hits: int = 0
    avg_ball_speed: float = 0.0
    max_ball_speed: float = 0.0
    points_won: int = 0
    points_lost: int = 0
    distance_covered: float = 0.0  # meters
    

@dataclass
class VideoAnalytics:
    """Complete analytics for entire video"""
    video_path: str
    total_frames: int
    fps: float
    duration_seconds: float
    
    # Segmentation
    game_segments: List[GameSegment] = field(default_factory=list)
    
    # Homography (can have multiple if court changes)
    homographies: List[HomographyData] = field(default_factory=list)
    
    # Player statistics
    player_stats: Dict[int, PlayerStats] = field(default_factory=dict)
    
    # Overall match info
    final_score: Optional[Tuple[int, int]] = None
    total_rallies: int = 0
    
    def get_active_homography(self, frame_id: int) -> Optional[HomographyData]:
        """Get valid homography for given frame"""
        for h in self.homographies:
            if h.is_valid_for_frame(frame_id):
                return h
        return None
    
    def get_rally_segments(self) -> List[GameSegment]:
        """Filter only rally segments"""
        return [seg for seg in self.game_segments if seg.segment_type == SegmentType.RALLY]


@dataclass
class ProcessingConfig:
    """Configuration for video processing pipeline"""
    # Detection settings
    court_detection_interval: int = 30  # Detect court every N frames
    ball_confidence_threshold: float = 0.3
    player_confidence_threshold: float = 0.5
    
    # Processing settings
    batch_size: int = 1
    chunk_size: int = 300  # Process video in chunks
    use_fp16: bool = True
    
    # Analytics settings
    motion_threshold: float = 5.0  # For frame filtering
    min_rally_frames: int = 30  # Minimum frames to consider as rally
    max_ball_speed: float = 70.0  # m/s (252 km/h - realistic max)
    interpolate_max_gap: int = 5  # Max frames to interpolate missing ball
    
    # Output settings
    save_visualization: bool = True
    output_fps: int = 25
    draw_court_lines: bool = True
    draw_trajectories: bool = True
    
    # Paths
    model_path: str = "models/"
    results_path: str = "results/"


# Standard tennis court dimensions (in meters)
COURT_DIMENSIONS = {
    'length': 23.77,  # Full court length
    'width': 10.97,   # Full court width (doubles)
    'singles_width': 8.23,  # Singles court width
    'service_line': 6.40,  # Distance from net to service line
    'net_height': 0.914,  # Net height at center
}

# Standard court keypoints (template for homography)
# These represent the real-world court coordinates (14 keypoints)
# Based on standard tennis court layout
COURT_TEMPLATE_KEYPOINTS = np.array([
    # Baseline (back)
    [0.0, 0.0],                                      # 0: Back-left baseline
    [COURT_DIMENSIONS['width'], 0.0],                # 1: Back-right baseline

    # Service line (back)
    [0.0, COURT_DIMENSIONS['service_line']],         # 2: Back-left service line
    [COURT_DIMENSIONS['width'], COURT_DIMENSIONS['service_line']],  # 3: Back-right service line

    # Net line (center)
    [0.0, COURT_DIMENSIONS['length'] / 2],           # 4: Left net post
    [COURT_DIMENSIONS['width'], COURT_DIMENSIONS['length'] / 2],  # 5: Right net post

    # Service line (front)
    [0.0, COURT_DIMENSIONS['length'] - COURT_DIMENSIONS['service_line']],  # 6: Front-left service line
    [COURT_DIMENSIONS['width'], COURT_DIMENSIONS['length'] - COURT_DIMENSIONS['service_line']],  # 7: Front-right service line

    # Baseline (front)
    [0.0, COURT_DIMENSIONS['length']],               # 8: Front-left baseline
    [COURT_DIMENSIONS['width'], COURT_DIMENSIONS['length']],  # 9: Front-right baseline

    # Center service line
    [COURT_DIMENSIONS['width'] / 2, COURT_DIMENSIONS['service_line']],  # 10: Back center service
    [COURT_DIMENSIONS['width'] / 2, COURT_DIMENSIONS['length'] / 2],  # 11: Net center
    [COURT_DIMENSIONS['width'] / 2, COURT_DIMENSIONS['length'] - COURT_DIMENSIONS['service_line']],  # 12: Front center service

    # Singles sidelines (optional, use doubles width for now)
    [COURT_DIMENSIONS['width'] / 2, 0.0],            # 13: Back center baseline
], dtype=np.float32)


# Singles court template (narrower than doubles)
# Singles sidelines are 1.37m inward from doubles sidelines on each side
SINGLES_SIDELINE_OFFSET = (COURT_DIMENSIONS['width'] - COURT_DIMENSIONS['singles_width']) / 2

COURT_TEMPLATE_KEYPOINTS_SINGLES = np.array([
    # Baseline (back)
    [SINGLES_SIDELINE_OFFSET, 0.0],                                      # 0: Back-left baseline
    [SINGLES_SIDELINE_OFFSET + COURT_DIMENSIONS['singles_width'], 0.0],  # 1: Back-right baseline

    # Service line (back)
    [SINGLES_SIDELINE_OFFSET, COURT_DIMENSIONS['service_line']],         # 2: Back-left service line
    [SINGLES_SIDELINE_OFFSET + COURT_DIMENSIONS['singles_width'], COURT_DIMENSIONS['service_line']],  # 3: Back-right service line

    # Net line (center)
    [SINGLES_SIDELINE_OFFSET, COURT_DIMENSIONS['length'] / 2],           # 4: Left net post
    [SINGLES_SIDELINE_OFFSET + COURT_DIMENSIONS['singles_width'], COURT_DIMENSIONS['length'] / 2],  # 5: Right net post

    # Service line (front)
    [SINGLES_SIDELINE_OFFSET, COURT_DIMENSIONS['length'] - COURT_DIMENSIONS['service_line']],  # 6: Front-left service line
    [SINGLES_SIDELINE_OFFSET + COURT_DIMENSIONS['singles_width'], COURT_DIMENSIONS['length'] - COURT_DIMENSIONS['service_line']],  # 7: Front-right service line

    # Baseline (front)
    [SINGLES_SIDELINE_OFFSET, COURT_DIMENSIONS['length']],               # 8: Front-left baseline
    [SINGLES_SIDELINE_OFFSET + COURT_DIMENSIONS['singles_width'], COURT_DIMENSIONS['length']],  # 9: Front-right baseline

    # Center service line
    [COURT_DIMENSIONS['width'] / 2, COURT_DIMENSIONS['service_line']],  # 10: Back center service
    [COURT_DIMENSIONS['width'] / 2, COURT_DIMENSIONS['length'] / 2],  # 11: Net center
    [COURT_DIMENSIONS['width'] / 2, COURT_DIMENSIONS['length'] - COURT_DIMENSIONS['service_line']],  # 12: Front center service

    # Center baseline
    [COURT_DIMENSIONS['width'] / 2, 0.0],            # 13: Back center baseline
], dtype=np.float32)


def get_court_template(court_type: str = 'doubles') -> np.ndarray:
    """
    Get appropriate court template based on court type

    Args:
        court_type: 'singles' or 'doubles'

    Returns:
        Court template keypoints (14, 2) array in meters
    """
    if court_type == 'singles':
        return COURT_TEMPLATE_KEYPOINTS_SINGLES
    else:
        return COURT_TEMPLATE_KEYPOINTS