"""
Processing Context - Shared state for pipeline

File: app/core/context.py
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

# Import existing data models
from app.core.data_models import (
    FrameDetection, GameSegment, BallTrajectory,
    HomographyData, VideoAnalytics, TemporalWindow, BallState
)


@dataclass
class ProcessingContext:
    """
    Shared state that flows through the entire pipeline.

    Think of this as a "blackboard" where steps read from and write to.
    All pipeline state is stored here, not in individual steps.

    Design Principles:
    - Steps modify this context in-place
    - All data is accessible to all steps (no hidden state)
    - Clear separation between input, intermediate, and output data
    """

    # ========== VIDEO METADATA (Input) ==========
    video_path: str
    fps: float
    total_frames: int
    width: int
    height: int
    duration_seconds: float = 0.0

    # ========== CURRENT PROCESSING CHUNK ==========
    chunk_start: int = 0
    chunk_end: int = 0
    chunk_size: int = 500

    # ========== FRAME DATA (Populated by PreprocessingPipeline) ==========
    frames: List[np.ndarray] = field(default_factory=list)  # Raw frames (N, H, W, 3)
    frame_ids: List[int] = field(default_factory=list)  # Frame indices
    active_frame_mask: List[bool] = field(default_factory=list)  # Motion filter results

    # ========== DETECTION RESULTS (Populated by DetectionPipeline) ==========
    # Using existing FrameDetection dataclass
    detections: List[FrameDetection] = field(default_factory=list)

    # ========== HOMOGRAPHY MATRICES (Cached by GeometryPipeline) ==========
    # Key: frame_id, Value: 3x3 homography matrix (pixel → court coordinates)
    homography_cache: Dict[int, np.ndarray] = field(default_factory=dict)

    # ========== TEMPORAL DATA (Populated by TemporalPipeline) ==========
    ball_trajectories: List[BallTrajectory] = field(default_factory=list)

    # Temporal windows for multi-frame models (e.g., ball hit detection)
    # List of windows, each containing ±N frames around a center frame
    temporal_windows: List[TemporalWindow] = field(default_factory=list)

    # ========== GEOMETRY DATA (Populated by GeometryPipeline) ==========
    # Ball states with court coordinates and velocities
    ball_states: List[BallState] = field(default_factory=list)

    # ========== EVENTS (Populated by EventPipeline) ==========
    hit_events: List[int] = field(default_factory=list)  # Frame IDs where ball was hit
    bounce_events: List[int] = field(default_factory=list)  # Frame IDs where ball bounced
    in_out_events: List[Tuple[int, str]] = field(default_factory=list)  # (frame_id, 'in'/'out')

    # ========== ANALYTICS (Populated by AnalyticsPipeline) ==========
    game_segments: List[GameSegment] = field(default_factory=list)  # Rally/idle/prep segments
    player_statistics: Dict[str, Any] = field(default_factory=dict)  # Aggregated stats

    # ========== RENDERING (Populated by RenderingPipeline) ==========
    annotated_frames: List[np.ndarray] = field(default_factory=list)  # Frames with overlays
    output_video_path: Optional[str] = None
    output_json_path: Optional[str] = None

    # ========== PERFORMANCE TRACKING ==========
    step_timings: Dict[str, float] = field(default_factory=dict)  # step_name → duration (seconds)

    # ========== FINAL OUTPUT ==========
    video_analytics: Optional[VideoAnalytics] = None  # Complete results

    def get_homography_for_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """
        Get homography matrix for a frame.

        Uses cached value from nearest computed frame if exact frame not available.
        This allows us to compute homography every N frames (e.g., 30) but use it
        for all intermediate frames.

        Args:
            frame_id: Frame index

        Returns:
            3x3 homography matrix (pixel → court coords) or None if not available
        """
        # Exact match
        if frame_id in self.homography_cache:
            return self.homography_cache[frame_id]

        # Find closest cached frame
        cached_frames = sorted(self.homography_cache.keys())
        if not cached_frames:
            return None

        # Use nearest neighbor
        closest = min(cached_frames, key=lambda x: abs(x - frame_id))
        return self.homography_cache[closest]

    def get_detection(self, frame_id: int) -> Optional[FrameDetection]:
        """
        Get detection results for a specific frame.

        Args:
            frame_id: Frame index

        Returns:
            FrameDetection object or None if frame not in current chunk
        """
        # Find detection by frame_id
        for detection in self.detections:
            if detection.frame_id == frame_id:
                return detection
        return None

    def clear_frames(self):
        """
        Clear frame data to free memory after processing chunk.

        Call this after processing each chunk to avoid memory issues
        with long videos.
        """
        self.frames = []
        self.annotated_frames = []

    def get_total_processing_time(self) -> float:
        """Get total processing time across all steps"""
        return sum(self.step_timings.values())

    def get_progress(self) -> float:
        """
        Get processing progress (0.0 to 1.0)

        Based on chunk progress within total video.
        """
        if self.total_frames == 0:
            return 0.0
        return min(self.chunk_end / self.total_frames, 1.0)

    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"<ProcessingContext "
            f"video={self.video_path}, "
            f"frames={len(self.frames)}, "
            f"detections={len(self.detections)}, "
            f"chunk=[{self.chunk_start}:{self.chunk_end}]>"
        )
