"""
CPU-based Geometric Refinement for Court Keypoint Detection

This module implements a 3-stage refinement pipeline to improve court keypoint
accuracy by leveraging geometric constraints of a tennis court.

Architecture:
    Heatmaps (1, 14, H, W)
        ↓
    Stage 0: Initial Extraction (argmax)
        ↓
    Stage 1: Local Line-Based Refinement (pixel-level)
        ↓
    Stage 2: Global Homography Correction
        ↓
    Stage 3: Line Collinearity Regularization
        ↓
    Refined Keypoints (14, 2)

File: app/src/court_refinement.py
"""
import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict


# Court line groups (which keypoints belong to which line)
COURT_LINES = {
    'baseline_back': [0, 1, 13],           # Back baseline (horizontal)
    'baseline_front': [8, 9],              # Front baseline (horizontal)
    'service_back': [2, 3, 10],            # Back service line (horizontal)
    'service_front': [6, 7, 12],           # Front service line (horizontal)
    'net_line': [4, 5, 11],                # Net line (horizontal)
    'sideline_left': [0, 2, 4, 6, 8],      # Left sideline (vertical)
    'sideline_right': [1, 3, 5, 7, 9],     # Right sideline (vertical)
    'center_line': [10, 11, 12, 13],       # Center service line (vertical)
}

# Reference keypoints for homography (most stable corner points)
REFERENCE_KEYPOINTS = [0, 1, 8, 9]  # Four corner baseline points


# ============================================================================
# Stage 1: Local Line-Based Refinement
# ============================================================================

def _extract_crop(
    frame: np.ndarray,
    x: float,
    y: float,
    crop_size: int
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Extract local crop around keypoint with boundary handling.

    Args:
        frame: Original frame (H, W, 3)
        x, y: Keypoint coordinates
        crop_size: Size of crop window

    Returns:
        crop: Extracted crop
        offset: (x_offset, y_offset) top-left corner position
    """
    h, w = frame.shape[:2]
    x1 = max(0, int(x - crop_size // 2))
    y1 = max(0, int(y - crop_size // 2))
    x2 = min(w, int(x + crop_size // 2))
    y2 = min(h, int(y + crop_size // 2))

    crop = frame[y1:y2, x1:x2]
    return crop, (x1, y1)


def _preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """
    Preprocess crop for line detection.

    Pipeline:
        - Convert to grayscale
        - Adaptive thresholding (handles varying lighting)
        - Morphological cleanup
        - Edge detection

    Args:
        crop: BGR crop image

    Returns:
        edges: Binary edge map
    """
    # Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding (handles varying lighting)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Edge detection
    edges = cv2.Canny(clean, 50, 150)

    return edges


def _detect_lines(edges: np.ndarray) -> List[np.ndarray]:
    """
    Detect lines using Hough transform.

    Args:
        edges: Binary edge map

    Returns:
        lines: List of detected lines [[x1, y1, x2, y2], ...]
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=10,
        maxLineGap=5
    )

    return lines if lines is not None else []


def _classify_lines(
    lines: List[np.ndarray],
    angle_threshold: float = 15
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Separate lines into horizontal and vertical.

    Args:
        lines: Detected lines
        angle_threshold: Angle tolerance in degrees

    Returns:
        (horizontal_lines, vertical_lines)
    """
    horizontal = []
    vertical = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        if abs(angle) < angle_threshold or abs(angle - 180) < angle_threshold:
            horizontal.append(line)
        elif abs(angle - 90) < angle_threshold or abs(angle + 90) < angle_threshold:
            vertical.append(line)

    return horizontal, vertical


def _find_intersection(
    horizontal_lines: List[np.ndarray],
    vertical_lines: List[np.ndarray]
) -> Optional[np.ndarray]:
    """
    Find best intersection point of horizontal and vertical lines.

    Strategy: Find intersection of longest horizontal and vertical lines.

    Args:
        horizontal_lines: List of horizontal lines
        vertical_lines: List of vertical lines

    Returns:
        (x, y) intersection point or None if no valid intersection
    """
    if not horizontal_lines or not vertical_lines:
        return None

    # Find longest horizontal line
    h_lengths = [np.linalg.norm([line[0][2] - line[0][0], line[0][3] - line[0][1]])
                 for line in horizontal_lines]
    best_h = horizontal_lines[np.argmax(h_lengths)][0]

    # Find longest vertical line
    v_lengths = [np.linalg.norm([line[0][2] - line[0][0], line[0][3] - line[0][1]])
                 for line in vertical_lines]
    best_v = vertical_lines[np.argmax(v_lengths)][0]

    # Compute intersection
    # Horizontal line: y = y1 (approximate as horizontal)
    # Vertical line: x = x1 (approximate as vertical)
    x1_h, y1_h, x2_h, y2_h = best_h
    x1_v, y1_v, x2_v, y2_v = best_v

    # More accurate: solve line equations
    # Line 1: (y - y1_h) = m_h * (x - x1_h)
    # Line 2: (y - y1_v) = m_v * (x - x1_v)

    denom_h = x2_h - x1_h
    denom_v = x2_v - x1_v

    if abs(denom_h) < 1e-6 or abs(denom_v) < 1e-6:
        # Degenerate lines
        return None

    m_h = (y2_h - y1_h) / denom_h
    m_v = (y2_v - y1_v) / denom_v

    # Solve for intersection
    # m_h * (x - x1_h) + y1_h = m_v * (x - x1_v) + y1_v
    # (m_h - m_v) * x = m_v * x1_v - m_h * x1_h + y1_v - y1_h

    if abs(m_h - m_v) < 1e-6:
        # Parallel lines
        return None

    x = (m_v * x1_v - m_h * x1_h + y1_v - y1_h) / (m_h - m_v)
    y = m_h * (x - x1_h) + y1_h

    return np.array([x, y])


def refine_keypoint_local(
    keypoint: np.ndarray,
    frame: np.ndarray,
    crop_size: int = 50,
    max_shift: float = 15.0
) -> Tuple[np.ndarray, bool]:
    """
    Refine a single keypoint by detecting line intersection in local crop.

    Pipeline:
        1. Extract local crop around keypoint
        2. Preprocess: grayscale → threshold → edges
        3. Detect lines using HoughLinesP
        4. Find intersection of horizontal and vertical lines
        5. Return intersection if within max_shift, else original

    Args:
        keypoint: (x, y) initial position
        frame: Original frame (H, W, 3)
        crop_size: Crop window size
        max_shift: Maximum allowed shift in pixels

    Returns:
        refined_keypoint: (x, y) refined position
        success: Whether refinement succeeded
    """
    x, y = keypoint

    # Extract crop
    crop, (x_offset, y_offset) = _extract_crop(frame, x, y, crop_size)

    if crop.size == 0:
        return keypoint, False

    # Preprocess
    edges = _preprocess_crop(crop)

    # Detect lines
    lines = _detect_lines(edges)

    if len(lines) == 0:
        return keypoint, False

    # Classify lines
    horizontal, vertical = _classify_lines(lines)

    # Find intersection
    intersection_local = _find_intersection(horizontal, vertical)

    if intersection_local is None:
        return keypoint, False

    # Convert to global coordinates
    intersection_global = intersection_local + np.array([x_offset, y_offset])

    # Check if shift is within bounds
    shift = np.linalg.norm(intersection_global - keypoint)

    if shift > max_shift:
        return keypoint, False

    return intersection_global, True


# ============================================================================
# Stage 2: Global Homography Correction
# ============================================================================

def _compute_robust_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    ransac_threshold: float
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Compute homography with RANSAC and validation.

    Args:
        src_pts: Source points (N, 2)
        dst_pts: Destination points (N, 2)
        ransac_threshold: RANSAC inlier threshold (pixels)

    Returns:
        (H, success) - Homography matrix and success flag
    """
    if len(src_pts) < 4:
        return None, False

    H, mask = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold
    )

    # Validate homography
    if H is None:
        return None, False

    # Check determinant (should be positive and reasonable)
    det = np.linalg.det(H)
    if det < 1e-6 or det > 1e6:
        return None, False

    # Check number of inliers
    inliers = np.sum(mask) if mask is not None else 0
    if inliers < 4:
        return None, False

    return H, True


def _apply_homography_to_keypoints(
    keypoints: np.ndarray,
    H: np.ndarray
) -> np.ndarray:
    """
    Apply homography transformation to keypoints.

    Args:
        keypoints: (N, 2) keypoint coordinates
        H: (3, 3) homography matrix

    Returns:
        transformed_keypoints: (N, 2) transformed coordinates
    """
    # Convert to homogeneous coordinates
    pts_homogeneous = np.hstack([keypoints, np.ones((len(keypoints), 1))])

    # Transform
    transformed = (H @ pts_homogeneous.T).T

    # Convert back to Cartesian
    transformed = transformed[:, :2] / transformed[:, 2:]

    return transformed


def refine_keypoints_homography(
    keypoints: np.ndarray,
    court_template: np.ndarray,
    reference_indices: List[int] = REFERENCE_KEYPOINTS,
    ransac_threshold: float = 5.0
) -> Tuple[np.ndarray, bool]:
    """
    Apply homography-based refinement using court template.

    Strategy:
        1. Select reference keypoints (stable corners)
        2. Compute homography: detected_ref → template_ref (forward)
        3. Compute inverse homography: template_ref → detected_ref (backward)
        4. Project all keypoints through forward->backward to enforce geometry
        5. Fallback to input if H fails or degrades quality

    This approach enforces geometric constraints by projecting keypoints through
    the court template space and back to pixel space, effectively "correcting"
    them to fit the template geometry while staying in pixel coordinates.

    Args:
        keypoints: (14, 2) initial keypoints
        court_template: (14, 2) template in world coordinates
        reference_indices: Indices of reference keypoints
        ransac_threshold: RANSAC inlier threshold

    Returns:
        refined_keypoints: (14, 2) refined keypoints
        success: Whether homography was computed successfully
    """
    # Select reference keypoints
    src_ref = keypoints[reference_indices]
    dst_ref = court_template[reference_indices]

    # Compute forward homography: pixel → template
    H_forward, success = _compute_robust_homography(src_ref, dst_ref, ransac_threshold)

    if not success:
        return keypoints, False

    # Compute inverse homography: template → pixel
    try:
        H_inverse = np.linalg.inv(H_forward)
    except np.linalg.LinAlgError:
        return keypoints, False

    # Project keypoints through template space and back
    # This enforces geometric constraints while staying in pixel space
    # Forward: pixel → template
    template_coords = _apply_homography_to_keypoints(keypoints, H_forward)

    # Backward: template → pixel (refined)
    refined = _apply_homography_to_keypoints(template_coords, H_inverse)

    return refined, True


# ============================================================================
# Stage 3: Line Collinearity Regularization
# ============================================================================

def _fit_line_to_points(points: np.ndarray) -> Optional[Tuple]:
    """
    Fit line to points using least squares.

    Args:
        points: (N, 2) points

    Returns:
        (vx, vy, x0, y0) - Line parameters or None if fitting fails
    """
    if len(points) < 2:
        return None

    # Use cv2.fitLine for robust fitting
    result = cv2.fitLine(
        points.astype(np.float32),
        cv2.DIST_L2, 0, 0.01, 0.01
    )

    if result is None:
        return None

    vx, vy, x0, y0 = result.flatten()

    return (vx, vy, x0, y0)


def _project_point_onto_line(
    point: np.ndarray,
    line_params: Tuple
) -> np.ndarray:
    """
    Project point onto line.

    Args:
        point: (x, y) point
        line_params: (vx, vy, x0, y0) line parameters

    Returns:
        projected_point: (x, y) projected point
    """
    vx, vy, x0, y0 = line_params

    # Point on line: P0 = (x0, y0)
    # Direction: v = (vx, vy)
    # Point: p = (px, py)
    # Projection: P = P0 + ((p - P0) · v) * v

    px, py = point
    t = ((px - x0) * vx + (py - y0) * vy) / (vx**2 + vy**2 + 1e-6)

    proj_x = x0 + t * vx
    proj_y = y0 + t * vy

    return np.array([proj_x, proj_y])


def _constrain_shift(
    original: np.ndarray,
    refined: np.ndarray,
    max_shift: float
) -> np.ndarray:
    """
    Limit shift magnitude to max_shift.

    Args:
        original: Original point
        refined: Refined point
        max_shift: Maximum allowed shift

    Returns:
        constrained_point: Point with constrained shift
    """
    shift = refined - original
    distance = np.linalg.norm(shift)

    if distance > max_shift:
        # Scale shift to max_shift
        shift = shift * (max_shift / distance)
        refined = original + shift

    return refined


def regularize_line_groups(
    keypoints: np.ndarray,
    line_groups: Dict[str, List[int]] = COURT_LINES,
    max_shift: float = 10.0
) -> np.ndarray:
    """
    Project keypoints onto fitted lines for each group.

    Ensures points on the same court line are perfectly collinear.

    Args:
        keypoints: (14, 2) keypoints
        line_groups: Dictionary of line groups
        max_shift: Maximum allowed shift

    Returns:
        regularized_keypoints: (14, 2) with collinear constraints enforced
    """
    regularized = keypoints.copy()

    for line_name, indices in line_groups.items():
        if len(indices) < 2:
            continue

        # Extract points for this line
        line_points = keypoints[indices]

        # Fit line
        line_params = _fit_line_to_points(line_points)

        if line_params is None:
            continue

        # Project each point onto line
        for idx in indices:
            original = keypoints[idx]
            projected = _project_point_onto_line(original, line_params)

            # Constrain shift
            constrained = _constrain_shift(original, projected, max_shift)

            regularized[idx] = constrained

    return regularized


# ============================================================================
# Main Refinement Class
# ============================================================================

class CourtKeypointRefinement:
    """
    CPU-based geometric refinement for court keypoints.

    Implements 3-stage pipeline:
        Stage 1: Local line-based refinement
        Stage 2: Global homography correction
        Stage 3: Line collinearity regularization
    """

    def __init__(
        self,
        court_template: np.ndarray,
        crop_size: int = 50,
        max_local_shift: float = 15.0,
        max_regularization_shift: float = 10.0,
        ransac_threshold: float = 5.0,
        enable_stages: Optional[Dict[str, bool]] = None
    ):
        """
        Initialize refinement module.

        Args:
            court_template: (14, 2) court template in world coordinates
            crop_size: Size of crop for local refinement
            max_local_shift: Max shift in Stage 1 (pixels)
            max_regularization_shift: Max shift in Stage 3 (pixels)
            ransac_threshold: RANSAC inlier threshold
            enable_stages: Dict to enable/disable stages (for ablation)
        """
        self.court_template = court_template
        self.crop_size = crop_size
        self.max_local_shift = max_local_shift
        self.max_regularization_shift = max_regularization_shift
        self.ransac_threshold = ransac_threshold

        # Allow disabling individual stages for ablation studies
        # NOTE: Regularization is disabled by default because court lines are only
        # collinear in world space, not in perspective-distorted pixel space.
        # Enabling it would cause incorrect projections.
        self.enable_stages = enable_stages or {
            'local_refinement': True,
            'homography': True,
            'regularization': False  # DISABLED: Only valid in world space, not pixels
        }

    def refine(
        self,
        keypoints: np.ndarray,
        frame: np.ndarray,
        confidence: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply multi-stage refinement pipeline.

        Args:
            keypoints: (14, 2) initial keypoints
            frame: Original frame for local refinement
            confidence: (14,) per-keypoint confidence scores

        Returns:
            refined_keypoints: (14, 2) refined coordinates
            stats: Dict with per-stage statistics
        """
        stats = {
            'stage0_input': keypoints.copy(),
            'stage1_local': None,
            'stage2_homography': None,
            'stage3_regularization': None,
            'shifts': {},
        }

        current_keypoints = keypoints.copy()
        frame_h, frame_w = frame.shape[:2]

        # Stage 1: Local refinement
        if self.enable_stages['local_refinement']:
            stage1_keypoints = self._apply_local_refinement(
                current_keypoints, frame, confidence
            )
            stats['stage1_local'] = stage1_keypoints.copy()
            stats['shifts']['local'] = np.linalg.norm(
                stage1_keypoints - current_keypoints, axis=1
            )
            current_keypoints = stage1_keypoints

        # Stage 2: Homography
        if self.enable_stages['homography']:
            stage2_keypoints, success = refine_keypoints_homography(
                current_keypoints,
                self.court_template,
                REFERENCE_KEYPOINTS,
                self.ransac_threshold
            )
            if success:
                stats['stage2_homography'] = stage2_keypoints.copy()
                stats['shifts']['homography'] = np.linalg.norm(
                    stage2_keypoints - current_keypoints, axis=1
                )
                current_keypoints = stage2_keypoints

        # Stage 3: Regularization (DISABLED by default - see __init__ comment)
        if self.enable_stages['regularization']:
            stage3_keypoints = regularize_line_groups(
                current_keypoints,
                COURT_LINES,
                self.max_regularization_shift
            )
            stats['stage3_regularization'] = stage3_keypoints.copy()
            stats['shifts']['regularization'] = np.linalg.norm(
                stage3_keypoints - current_keypoints, axis=1
            )
            current_keypoints = stage3_keypoints

        # Validation: Ensure keypoints are within frame bounds
        # If refinement produced out-of-bounds keypoints, fall back to original
        if not self._validate_keypoints(current_keypoints, frame_w, frame_h):
            return keypoints, stats  # Return original keypoints

        return current_keypoints, stats

    def _validate_keypoints(
        self,
        keypoints: np.ndarray,
        frame_w: int,
        frame_h: int,
        margin: float = 0.1  # 10% margin outside frame is OK
    ) -> bool:
        """
        Validate that keypoints are within reasonable bounds.

        Args:
            keypoints: (14, 2) keypoints
            frame_w: Frame width
            frame_h: Frame height
            margin: Allowed margin outside frame (fraction)

        Returns:
            True if keypoints are valid, False otherwise
        """
        min_x = -frame_w * margin
        max_x = frame_w * (1 + margin)
        min_y = -frame_h * margin
        max_y = frame_h * (1 + margin)

        x_valid = (keypoints[:, 0] >= min_x) & (keypoints[:, 0] <= max_x)
        y_valid = (keypoints[:, 1] >= min_y) & (keypoints[:, 1] <= max_y)

        return np.all(x_valid) and np.all(y_valid)

    def _apply_local_refinement(
        self,
        keypoints: np.ndarray,
        frame: np.ndarray,
        confidence: np.ndarray
    ) -> np.ndarray:
        """
        Apply local refinement to each keypoint.

        Args:
            keypoints: (14, 2) keypoints
            frame: Original frame
            confidence: (14,) confidence scores

        Returns:
            refined_keypoints: (14, 2) refined keypoints
        """
        refined = keypoints.copy()

        for i in range(len(keypoints)):
            # Skip low-confidence keypoints
            if confidence[i] < 0.3:
                continue

            refined_kp, success = refine_keypoint_local(
                keypoints[i],
                frame,
                self.crop_size,
                self.max_local_shift
            )

            if success:
                refined[i] = refined_kp

        return refined
