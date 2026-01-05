"""
Player Tracking Step - Track players across frames using ByteTrack

File: app/steps/tracking/player_tracking.py

ByteTrack: Simple and effective multi-object tracking algorithm
- Uses Kalman filter for motion prediction
- Handles low-confidence detections for robust tracking
- Maintains player IDs across frames even when detection fails
- Fast: ~30 FPS on GPU

Integration:
- Input: Player bounding boxes from YOLO detection (List[Tuple[float, float, float, float]])
- Input: Player confidences (List[float])
- Output: Track IDs for each detected player (List[int])
- ByteTrack available in ultralytics (no extra dependencies except lap>=0.5.12)
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
import cv2

from app.steps.base import PipelineStep
from app.core.context import ProcessingContext
from app.core.data_models import FrameDetection

try:
    from ultralytics.trackers.byte_tracker import BYTETracker
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False
    print("Warning: ByteTrack not available. Install ultralytics>=8.0.0 and lap>=0.5.12")


class PlayerTrackingStep(PipelineStep):
    """
    Track tennis players across frames using ByteTrack algorithm.

    Features:
    - Maintains player IDs across frames (even when detection fails)
    - Uses Kalman filter for motion prediction
    - Handles occlusions and temporary detection failures
    - Fast and simple (no appearance features needed)

    Data Flow:
    1. Input from PlayerDetectionStep:
       - det.player_boxes: List[Tuple[float, float, float, float]]  # [(x1, y1, x2, y2), ...]
       - det.player_confidences: List[float]  # [0.85, 0.92, ...]

    2. Convert to ByteTrack format:
       - numpy array shape (N, 5): [x1, y1, x2, y2, confidence]
       - dtype: float32

    3. ByteTrack processing:
       - Input: detections (N, 5) numpy array
       - Output: List[STrack] objects with .track_id, .tlbr (bounding box), .score

    4. Store results in FrameDetection:
       - det.player_track_ids: List[int]  # [1, 2, 1, 2, ...]
       - Matches 1-to-1 with det.player_boxes

    Configuration:
        tracking:
          players:
            enabled: true
            tracker: 'bytetrack'        # Tracker type
            track_thresh: 0.5            # High confidence detection threshold
            track_buffer: 30             # Frames to keep lost tracks
            match_thresh: 0.8            # IOU threshold for matching
            min_box_area: 10             # Minimum box area (filter tiny boxes)
            mot20: false                 # MOT20 protocol (use False for tennis)

    ByteTrack Parameters Explained:
    - track_thresh: Detections above this are "high confidence" tracks (0.5 for tennis)
    - track_buffer: Keep lost tracks for N frames (30 frames = 1 second at 30fps)
    - match_thresh: IOU threshold for matching detections to tracks (0.8 = tight matching)
    - min_box_area: Filter out very small detections (10 pixels² minimum)
    - mot20: MOT20 protocol uses different matching strategy (False for tennis)
    """

    def __init__(self, config: Dict):
        """
        Initialize player tracking step

        Args:
            config: Configuration dict with tracking parameters
        """
        super().__init__(config)

        if not BYTETRACK_AVAILABLE:
            raise ImportError(
                "ByteTrack not available. Please install:\n"
                "  pip install ultralytics>=8.0.0 lap>=0.5.12"
            )

        # Get tracking config
        tracking_config = config.get('tracking', {}).get('players', {})
        self.enabled = tracking_config.get('enabled', True)
        self.tracker_type = tracking_config.get('tracker', 'bytetrack')

        # ByteTrack parameters
        self.track_thresh = tracking_config.get('track_thresh', 0.5)
        self.track_buffer = tracking_config.get('track_buffer', 30)
        self.match_thresh = tracking_config.get('match_thresh', 0.8)
        self.min_box_area = tracking_config.get('min_box_area', 10)
        self.mot20 = tracking_config.get('mot20', False)

        # Frame rate (needed for ByteTrack)
        self.frame_rate = tracking_config.get('frame_rate', 30)

        # Initialize ByteTrack tracker
        # BYTETracker expects args object with these attributes
        from types import SimpleNamespace
        args = SimpleNamespace(
            track_thresh=self.track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            min_box_area=self.min_box_area,
            mot20=self.mot20
        )

        self.tracker = BYTETracker(args, frame_rate=self.frame_rate)

        # Statistics
        self.total_frames_tracked = 0
        self.total_tracks = 0
        self.active_track_ids = set()

        print(f"PlayerTrackingStep initialized:")
        print(f"  Tracker: {self.tracker_type}")
        print(f"  Track threshold: {self.track_thresh}")
        print(f"  Track buffer: {self.track_buffer} frames")
        print(f"  Match threshold: {self.match_thresh}")
        print(f"  Min box area: {self.min_box_area} px²")

    def process(self, context: ProcessingContext):
        """
        Track players across all frames

        Args:
            context: Processing context with detections
        """
        if not self.enabled:
            print("Player tracking disabled")
            return

        print(f"\n{'='*60}")
        print("STEP: Player Tracking (ByteTrack)")
        print(f"{'='*60}")
        print(f"Total frames to track: {len(context.detections)}")
        print()

        # Reset tracker for new video
        from types import SimpleNamespace
        args = SimpleNamespace(
            track_thresh=self.track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            min_box_area=self.min_box_area,
            mot20=self.mot20
        )
        self.tracker = BYTETracker(args, frame_rate=self.frame_rate)
        self.active_track_ids = set()

        # Track players frame by frame
        frames_with_tracks = 0
        total_track_assignments = 0

        for det in context.detections.values():
            # Skip frames without player detections
            if not det.has_players():
                continue

            # Convert detection data to ByteTrack format
            # Input: det.player_boxes (List[Tuple[float, float, float, float]])
            # Input: det.player_confidences (List[float])
            # Output: numpy array shape (N, 5) [x1, y1, x2, y2, confidence]
            detections = self._convert_to_bytetrack_format(
                det.player_boxes,
                det.player_confidences
            )

            # Run ByteTrack
            # Input: detections (N, 5) numpy array
            # Output: List[STrack] with .track_id, .tlbr, .score
            tracks = self.tracker.update(detections, [det.frame_id], [det.frame_id])

            # Extract track IDs and store in detection
            # Output: List[int] matching order of det.player_boxes
            track_ids = self._extract_track_ids(tracks, det.player_boxes)

            # Store track IDs in detection
            # Add new field to FrameDetection: player_track_ids
            det.player_track_ids = track_ids

            # Update statistics
            if track_ids:
                frames_with_tracks += 1
                total_track_assignments += len(track_ids)
                self.active_track_ids.update(track_ids)

        # Statistics
        self.total_frames_tracked = frames_with_tracks
        self.total_tracks = len(self.active_track_ids)

        print(f"\n✓ Player tracking complete")
        print(f"  Frames tracked: {frames_with_tracks}")
        print(f"  Total track assignments: {total_track_assignments}")
        print(f"  Unique player IDs: {self.total_tracks}")
        print(f"  Active track IDs: {sorted(self.active_track_ids)}")
        print(f"  Avg tracks/frame: {total_track_assignments / max(frames_with_tracks, 1):.2f}")
        print()

    def _convert_to_bytetrack_format(
        self,
        boxes: List[Tuple[float, float, float, float]],
        confidences: List[float]
    ) -> np.ndarray:
        """
        Convert detection data to ByteTrack format

        Args:
            boxes: List of bounding boxes [(x1, y1, x2, y2), ...]
            confidences: List of confidence scores [0.85, 0.92, ...]

        Returns:
            numpy array shape (N, 5): [x1, y1, x2, y2, confidence]
            dtype: float32

        Example:
            Input:
                boxes = [(100, 200, 150, 300), (400, 500, 450, 600)]
                confidences = [0.85, 0.92]

            Output:
                array([[100., 200., 150., 300., 0.85],
                       [400., 500., 450., 600., 0.92]], dtype=float32)
        """
        if not boxes:
            # Return empty array with correct shape (0, 5)
            return np.zeros((0, 5), dtype=np.float32)

        # Convert to numpy arrays
        boxes_array = np.array(boxes, dtype=np.float32)  # Shape: (N, 4)
        conf_array = np.array(confidences, dtype=np.float32).reshape(-1, 1)  # Shape: (N, 1)

        # Concatenate: [x1, y1, x2, y2, confidence]
        detections = np.hstack([boxes_array, conf_array])  # Shape: (N, 5)

        return detections

    def _extract_track_ids(
        self,
        tracks: List,
        original_boxes: List[Tuple[float, float, float, float]]
    ) -> List[int]:
        """
        Extract track IDs from ByteTrack output

        ByteTrack returns STrack objects, we need to match them back to original detections
        and extract track IDs in the same order.

        Args:
            tracks: List[STrack] from ByteTrack with .track_id, .tlbr, .score
            original_boxes: Original detection boxes for ordering

        Returns:
            List of track IDs matching order of original_boxes

        Note:
            ByteTrack may return tracks in different order than input detections.
            We match by IOU between track bounding box and original detection box.
        """
        if not tracks or not original_boxes:
            return []

        # Extract track data
        # STrack.tlbr returns [x1, y1, x2, y2] as list
        track_boxes = np.array([track.tlbr for track in tracks], dtype=np.float32)
        track_ids = [int(track.track_id) for track in tracks]

        # Convert original boxes to numpy
        original_boxes_array = np.array(original_boxes, dtype=np.float32)

        # Match tracks to original detections by IOU
        # This ensures track IDs are in same order as det.player_boxes
        matched_track_ids = []

        for orig_box in original_boxes_array:
            # Find best matching track by IOU
            best_iou = 0.0
            best_track_id = -1

            for track_box, track_id in zip(track_boxes, track_ids):
                iou = self._compute_iou(orig_box, track_box)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            # Only assign track ID if IOU is high enough (avoid mismatches)
            if best_iou > 0.3:  # Loose threshold since these are same-frame matches
                matched_track_ids.append(best_track_id)
            else:
                # No good match - assign new temporary ID (-1 means untracked)
                matched_track_ids.append(-1)

        return matched_track_ids

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Compute Intersection over Union (IOU) between two boxes

        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]

        Returns:
            IOU score (0.0 to 1.0)
        """
        # Intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        # IOU
        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def get_statistics(self) -> Dict:
        """Get tracking statistics"""
        return {
            'total_frames_tracked': self.total_frames_tracked,
            'total_unique_tracks': self.total_tracks,
            'active_track_ids': sorted(self.active_track_ids),
            'tracker_type': self.tracker_type,
            'track_thresh': self.track_thresh,
            'track_buffer': self.track_buffer,
        }
