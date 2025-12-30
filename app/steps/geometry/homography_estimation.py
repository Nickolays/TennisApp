"""
Homography Estimation Step - Transform pixel coordinates to court coordinates

File: app/steps/geometry/homography_estimation.py
"""
import numpy as np
import cv2
from typing import Optional, Tuple

from app.steps.base import PipelineStep
from app.core.context import ProcessingContext
from app.core.data_models import COURT_TEMPLATE_KEYPOINTS, get_court_template


class HomographyEstimationStep(PipelineStep):
    """
    Estimate homography matrices from court keypoints.

    Features:
    - Computes homography every N frames (court is static)
    - Caches matrices for intermediate frames
    - Uses RANSAC for robust estimation
    - Validates homography quality
    - Falls back to previous homography if estimation fails

    Configuration:
        geometry:
          homography:
            enabled: true
            interval: 30              # Compute every 30 frames
            min_keypoints: 8          # Minimum points needed (of 14)
            ransac_threshold: 5.0     # RANSAC inlier threshold (pixels)
            min_inliers: 6            # Minimum inliers to accept
    """

    def __init__(self, config: dict):
        """
        Initialize homography estimation step

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        self.interval = config.get('interval', 30)
        self.min_keypoints = config.get('min_keypoints', 8)
        self.ransac_threshold = config.get('ransac_threshold', 5.0)
        self.min_inliers = config.get('min_inliers', 6)
        self.court_type = config.get('court_type', 'auto')  # 'singles', 'doubles', or 'auto'

        # Standard court template (real-world coordinates in meters)
        # Will be updated dynamically if court_type='auto'
        self.court_template = COURT_TEMPLATE_KEYPOINTS

    def _estimate_homography(
        self,
        detected_keypoints: np.ndarray,
        template_keypoints: np.ndarray
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Estimate homography matrix from detected court keypoints

        Args:
            detected_keypoints: Detected keypoints in pixel coordinates (N, 2)
            template_keypoints: Template keypoints in court coordinates (N, 2)

        Returns:
            (homography_matrix, num_inliers) or (None, 0) if estimation fails
        """
        # Need at least 4 points for homography
        if len(detected_keypoints) < 4:
            return None, 0

        try:
            # Estimate homography with RANSAC
            # Maps: detected_pixels → template_court_coords
            H, mask = cv2.findHomography(
                detected_keypoints,
                template_keypoints,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold
            )

            if H is None:
                return None, 0

            # Count inliers
            num_inliers = np.sum(mask)

            # Check if we have enough inliers
            if num_inliers < self.min_inliers:
                return None, 0

            return H, int(num_inliers)

        except Exception as e:
            print(f"    Homography estimation failed: {e}")
            return None, 0

    def _validate_homography(self, H: np.ndarray) -> bool:
        """
        Validate homography matrix for reasonableness

        Args:
            H: 3x3 homography matrix

        Returns:
            True if homography is valid
        """
        if H is None:
            return False

        # Check matrix is not degenerate
        if np.linalg.det(H) < 1e-6:
            return False

        # Check matrix values are reasonable (not too extreme)
        if np.any(np.abs(H) > 1e6):
            return False

        return True

    def _detect_court_type(
        self,
        detected_keypoints: np.ndarray
    ) -> str:
        """
        Auto-detect court type (singles vs doubles) from detected keypoints.

        Simple heuristic:
        - Measure width of detected court (distance between left and right keypoints)
        - Compare to expected singles/doubles ratio
        - If width ratio < 0.85, likely singles

        Args:
            detected_keypoints: Detected keypoints (N, 2)

        Returns:
            'singles' or 'doubles'
        """
        if len(detected_keypoints) < 2:
            return 'doubles'  # Default

        # Get leftmost and rightmost points
        x_coords = detected_keypoints[:, 0]
        x_min = np.min(x_coords)
        x_max = np.max(x_coords)
        detected_width = x_max - x_min

        # Expected width ratio (singles/doubles = 8.23/10.97 ≈ 0.75)
        # In pixel space, ratio should be similar
        # This is a simplified heuristic - in reality needs more sophisticated detection

        # For now, just return doubles (since we can't reliably detect from pixels alone)
        # TODO: Improve with actual court line detection or ML classification
        return 'doubles'

    def _match_keypoints(
        self,
        detected_keypoints: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match detected keypoints to template keypoints.

        Since we have 14 detected keypoints and 14 template keypoints,
        we use all available points for better homography estimation.

        Args:
            detected_keypoints: All 14 detected keypoints (14, 2)

        Returns:
            (matched_detected, matched_template) arrays
        """
        # Use all available keypoints
        num_template = len(self.court_template)

        if len(detected_keypoints) < num_template:
            # If we have fewer detected points than template, use what we have
            matched_detected = detected_keypoints
            matched_template = self.court_template[:len(detected_keypoints)]
        else:
            # Use first N points that match template
            matched_detected = detected_keypoints[:num_template]
            matched_template = self.court_template

        return matched_detected, matched_template

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Estimate homography matrices for detected court keypoints

        Updates context.homography_cache with computed matrices

        Args:
            context: Processing context with detections

        Returns:
            Updated context
        """
        if not context.detections:
            print("  No detections to process")
            return context

        # Auto-detect court type if configured
        if self.court_type == 'auto':
            # Use first valid detection to detect court type
            for det in context.detections:
                if det.court_keypoints is not None and len(det.court_keypoints) >= 4:
                    detected_type = self._detect_court_type(det.court_keypoints)
                    self.court_template = get_court_template(detected_type)
                    print(f"  Auto-detected court type: {detected_type}")
                    break
        elif self.court_type in ['singles', 'doubles']:
            self.court_template = get_court_template(self.court_type)

        # Estimate homography at intervals
        computed_count = 0
        failed_count = 0
        last_valid_H = None

        for i in range(0, len(context.detections), self.interval):
            det = context.detections[i]

            # Check if we have court keypoints
            if det.court_keypoints is None:
                failed_count += 1
                print(f"    Frame {det.frame_id}: No court keypoints")
                if last_valid_H is not None:
                    context.homography_cache[det.frame_id] = last_valid_H
                continue

            if len(det.court_keypoints) < 4:
                failed_count += 1
                print(f"    Frame {det.frame_id}: Too few keypoints ({len(det.court_keypoints)} < 4)")
                if last_valid_H is not None:
                    context.homography_cache[det.frame_id] = last_valid_H
                continue

            # Match keypoints to template
            matched_detected, matched_template = self._match_keypoints(det.court_keypoints)

            # Debug output
            # print(f"    Frame {det.frame_id}: Matched {len(matched_detected)} keypoints")

            # Estimate homography
            H, num_inliers = self._estimate_homography(matched_detected, matched_template)

            # Validate
            if H is not None and self._validate_homography(H):
                context.homography_cache[det.frame_id] = H
                last_valid_H = H
                computed_count += 1

                print(f"    Frame {det.frame_id}: Homography computed ({num_inliers} inliers)")
            else:
                failed_count += 1
                # Use previous valid homography
                if last_valid_H is not None:
                    context.homography_cache[det.frame_id] = last_valid_H
                    print(f"    Frame {det.frame_id}: Failed (H={H is not None}, inliers={num_inliers}), using previous")
                else:
                    print(f"    Frame {det.frame_id}: Failed (H={H is not None}, inliers={num_inliers}), no previous homography")

        # Summary
        total_frames = len(context.detections)
        frames_with_homography = len(context.homography_cache)

        print(f"  Computed {computed_count} homographies ({failed_count} failed)")
        print(f"  Coverage: {frames_with_homography}/{total_frames} frames have homography")

        return context
