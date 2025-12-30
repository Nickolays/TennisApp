"""
Player Detection Step - Detect players using YOLO v11

File: app/steps/detection/player_detection.py
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict

from app.steps.base import PipelineStep
from app.core.context import ProcessingContext
from app.core.data_models import FrameDetection
from app.models.model_registry import ModelRegistry


class PlayerDetectionStep(PipelineStep):
    """
    Detect tennis players using YOLO v11 with intelligent filtering.

    Features:
    - Detects all people in frame using YOLO v11
    - Filters to keep only active players on court
    - Excludes: judges, ballboys, spectators, photographers
    - Uses court boundaries and activity metrics for filtering
    - Tracks player positions across frames

    Filtering Strategy:
    1. Court Boundary Filter: Keep only people inside court boundaries (+margin)
    2. Size Filter: Players are larger than ballboys/judges (closer to camera)
    3. Activity Filter: Players move more than static judges
    4. Position Filter: Players typically in specific court regions
    5. Count Filter: Limit to 2-4 players (singles/doubles)

    Configuration:
        detection:
          players:
            enabled: true
            model_path: 'models/yolo11n.pt'  # YOLO v11 nano (fast)
            confidence_threshold: 0.5
            interval: 1                       # Detect every frame
            input_size: [640, 640]            # YOLO input size

            # Filtering parameters
            court_margin: 100                 # Pixels outside court bounds
            min_box_area: 5000                # Minimum player box area (pixels²)
            max_players: 4                    # Maximum players (doubles)
            activity_window: 10               # Frames to check for movement
            min_movement: 10.0                # Minimum movement (pixels)
    """

    def __init__(self, config: dict):
        """
        Initialize player detection step

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        self.model_path = config.get('model_path', 'models/yolo11n.pt')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.interval = config.get('interval', 1)
        self.input_size = tuple(config.get('input_size', [640, 640]))

        # Filtering parameters
        self.court_margin = config.get('court_margin', 100)
        self.min_box_area = config.get('min_box_area', 5000)
        self.max_players = config.get('max_players', 4)
        self.activity_window = config.get('activity_window', 10)
        self.min_movement = config.get('min_movement', 10.0)

        # Batch inference configuration
        self.batch_size = config.get('batch_size', 8)  # Process 8 frames at once

        # Model
        self.model = None

        # Player tracking across frames
        self.player_history: Dict[int, List[Tuple[float, float, float, float]]] = {}

    def _load_model(self):
        """Load YOLO model from registry"""
        if self.model is None:
            self.model = ModelRegistry.load_model(
                model_type='yolo',
                model_path=self.model_path,
                conf_threshold=self.confidence_threshold
            )

    def _get_court_bounds(
        self,
        court_keypoints: Optional[np.ndarray],
        frame_shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Get court boundaries from keypoints

        Args:
            court_keypoints: Court keypoints (N, 2) or None
            frame_shape: Frame shape (height, width)

        Returns:
            (x_min, y_min, x_max, y_max) - Court bounds with margin
        """
        if court_keypoints is None or len(court_keypoints) == 0:
            # No court detected - use full frame
            height, width = frame_shape
            return (0, 0, width, height)

        # Get bounding box of all court keypoints
        x_coords = court_keypoints[:, 0]
        y_coords = court_keypoints[:, 1]

        x_min = int(np.min(x_coords)) - self.court_margin
        x_max = int(np.max(x_coords)) + self.court_margin
        y_min = int(np.min(y_coords)) - self.court_margin
        y_max = int(np.max(y_coords)) + self.court_margin

        # Clamp to frame bounds
        height, width = frame_shape
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)

        return (x_min, y_min, x_max, y_max)

    def _is_inside_court(
        self,
        box: Tuple[float, float, float, float],
        court_bounds: Tuple[int, int, int, int]
    ) -> bool:
        """
        Check if player bounding box is inside court boundaries

        Args:
            box: Player box (x1, y1, x2, y2)
            court_bounds: Court bounds (x_min, y_min, x_max, y_max)

        Returns:
            True if box center is inside court bounds
        """
        x1, y1, x2, y2 = box
        court_x_min, court_y_min, court_x_max, court_y_max = court_bounds

        # Use box center for checking
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Check if center is inside court bounds
        if (court_x_min <= center_x <= court_x_max and
            court_y_min <= center_y <= court_y_max):
            return True

        return False

    def _compute_box_area(self, box: Tuple[float, float, float, float]) -> float:
        """
        Compute bounding box area

        Args:
            box: Box (x1, y1, x2, y2)

        Returns:
            Area in pixels²
        """
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        return width * height

    def _compute_movement(
        self,
        frame_id: int,
        box: Tuple[float, float, float, float]
    ) -> float:
        """
        Compute movement of player over recent frames

        Args:
            frame_id: Current frame ID
            box: Current box (x1, y1, x2, y2)

        Returns:
            Average movement in pixels
        """
        if frame_id not in self.player_history:
            return 0.0

        # Get box center
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2

        # Look back at recent frames
        movements = []
        for past_frame_id in range(max(0, frame_id - self.activity_window), frame_id):
            if past_frame_id not in self.player_history:
                continue

            past_boxes = self.player_history[past_frame_id]

            # Find closest box in past frame
            min_dist = float('inf')
            for past_box in past_boxes:
                past_center_x = (past_box[0] + past_box[2]) / 2
                past_center_y = (past_box[1] + past_box[3]) / 2

                dist = np.sqrt((center_x - past_center_x)**2 + (center_y - past_center_y)**2)
                min_dist = min(min_dist, dist)

            if min_dist < float('inf'):
                movements.append(min_dist)

        # Return average movement
        if movements:
            return np.mean(movements)
        return 0.0

    def _filter_players(
        self,
        boxes: List[Tuple[float, float, float, float]],
        confidences: List[float],
        frame_id: int,
        court_bounds: Tuple[int, int, int, int]
    ) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
        """
        Filter detected people to keep only active players

        Args:
            boxes: All detected person boxes
            confidences: Detection confidences
            frame_id: Current frame ID
            court_bounds: Court boundaries

        Returns:
            (filtered_boxes, filtered_confidences)
        """
        filtered_boxes = []
        filtered_confidences = []

        candidates = []

        for box, conf in zip(boxes, confidences):
            # Filter 1: Court boundary (must be inside court + margin)
            if not self._is_inside_court(box, court_bounds):
                continue

            # Filter 2: Size (players are large enough)
            area = self._compute_box_area(box)
            if area < self.min_box_area:
                continue

            # Filter 3: Activity (players move, judges/ballboys less so)
            movement = self._compute_movement(frame_id, box)

            # Store candidate with score
            # Score = confidence + movement_bonus
            # Players with more movement get higher score
            movement_bonus = min(movement / 50.0, 1.0)  # Cap at 1.0
            score = conf + movement_bonus

            candidates.append((box, conf, score))

        # Sort by score (descending)
        candidates.sort(key=lambda x: x[2], reverse=True)

        # Keep top N players
        for box, conf, score in candidates[:self.max_players]:
            filtered_boxes.append(box)
            filtered_confidences.append(conf)

        return filtered_boxes, filtered_confidences

    def _detect_players_in_frame(
        self,
        frame: np.ndarray,
        court_keypoints: Optional[np.ndarray],
        frame_id: int
    ) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
        """
        Detect players in a single frame

        Args:
            frame: Input frame (H, W, 3)
            court_keypoints: Court keypoints for filtering
            frame_id: Frame ID for tracking

        Returns:
            (boxes, confidences) - Filtered player detections
        """
        # Run YOLO detection
        # YOLOModelWrapper already filters for person class (class_id=0)
        # conf_threshold is already set in the wrapper initialization
        results = self.model(frame)

        # Extract person detections from YOLO results
        # results is a list with one Results object for single image
        all_boxes = []
        all_confidences = []

        if len(results) > 0:
            result = results[0]  # First result (single image)

            # Get boxes and confidences
            # result.boxes contains all detections
            if result.boxes is not None and len(result.boxes) > 0:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4) - x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()  # (N,) - confidence scores

                for box, conf in zip(boxes_xyxy, confidences):
                    x1, y1, x2, y2 = box
                    all_boxes.append((float(x1), float(y1), float(x2), float(y2)))
                    all_confidences.append(float(conf))

        # Get court bounds for filtering
        court_bounds = self._get_court_bounds(court_keypoints, frame.shape[:2])

        # Filter to keep only active players
        filtered_boxes, filtered_confidences = self._filter_players(
            all_boxes,
            all_confidences,
            frame_id,
            court_bounds
        )

        # Update player history for tracking
        self.player_history[frame_id] = filtered_boxes

        return filtered_boxes, filtered_confidences

    def _detect_players_batch(
        self,
        frames: List[np.ndarray],
        frame_indices: List[int],
        court_keypoints_list: List[Optional[np.ndarray]]
    ) -> List[Tuple[List[Tuple[float, float, float, float]], List[float]]]:
        """
        Detect players in a batch of frames (faster)

        Args:
            frames: List of frames to process
            frame_indices: Corresponding frame indices
            court_keypoints_list: Court keypoints for each frame

        Returns:
            List of (boxes, confidences) for each frame
        """
        # Run YOLO on batch
        results = self.model(frames, verbose=False)

        # Process results for each frame
        batch_results = []

        for idx, (result, frame_id, court_keypoints, frame) in enumerate(
            zip(results, frame_indices, court_keypoints_list, frames)
        ):
            # Extract detections
            all_boxes = []
            all_confidences = []

            if result.boxes is not None and len(result.boxes) > 0:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                for box, conf in zip(boxes_xyxy, confidences):
                    x1, y1, x2, y2 = box
                    all_boxes.append((float(x1), float(y1), float(x2), float(y2)))
                    all_confidences.append(float(conf))

            # Get court bounds for filtering
            court_bounds = self._get_court_bounds(court_keypoints, frame.shape[:2])

            # Filter to keep only active players
            filtered_boxes, filtered_confidences = self._filter_players(
                all_boxes,
                all_confidences,
                frame_id,
                court_bounds
            )

            # Update player history
            self.player_history[frame_id] = filtered_boxes

            batch_results.append((filtered_boxes, filtered_confidences))

        return batch_results

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Detect players in video frames using batch inference

        Updates context.detections with player boxes and confidences

        Args:
            context: Processing context with frames and detections

        Returns:
            Updated context
        """
        self._load_model()

        if not context.frames:
            print("  No frames to process")
            return context

        # Process frames at intervals
        frames_to_process = list(range(0, len(context.frames), self.interval))
        total_players = 0

        # Process in batches for efficiency
        for batch_start in range(0, len(frames_to_process), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(frames_to_process))
            batch_indices = frames_to_process[batch_start:batch_end]

            # Prepare batch data
            batch_frames = []
            batch_frame_ids = []
            batch_court_keypoints = []

            for i in batch_indices:
                batch_frames.append(context.frames[i])
                batch_frame_ids.append(context.frame_ids[i])

                # Get court keypoints for filtering
                court_keypoints = None
                if i < len(context.detections):
                    court_keypoints = context.detections[i].court_keypoints
                batch_court_keypoints.append(court_keypoints)

            # Detect players in batch
            batch_results = self._detect_players_batch(
                batch_frames,
                batch_frame_ids,
                batch_court_keypoints
            )

            # Store results
            for i, (boxes, confidences) in zip(batch_indices, batch_results):
                if i < len(context.detections):
                    context.detections[i].player_boxes = boxes
                    context.detections[i].player_confidences = confidences
                    total_players += len(boxes)

        # Fill intermediate frames (nearest neighbor)
        if self.interval > 1:
            for i in range(len(context.detections)):
                if i % self.interval == 0:
                    continue

                # Find nearest processed frame
                nearest_idx = (i // self.interval) * self.interval
                if nearest_idx < len(context.detections):
                    context.detections[i].player_boxes = context.detections[nearest_idx].player_boxes
                    context.detections[i].player_confidences = context.detections[nearest_idx].player_confidences

        # Summary
        avg_players = total_players / len(frames_to_process) if frames_to_process else 0
        frames_with_players = sum(1 for d in context.detections if len(d.player_boxes) > 0)

        print(f"  Players detected in {frames_with_players}/{len(context.detections)} frames")
        print(f"  Average players per frame: {avg_players:.1f}")

        return context
