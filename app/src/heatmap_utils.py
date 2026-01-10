"""
Heatmap generation utilities for keypoint detection
"""
import numpy as np
import torch


def generate_gaussian_heatmap(heatmap, center, radius=7):
    """
    Draw a 2D Gaussian blob on a heatmap at the specified center.

    Args:
        heatmap: numpy array of shape (H, W)
        center: tuple (x, y) in pixel coordinates
        radius: Gaussian radius (controls blob size)
    """
    height, width = heatmap.shape
    x, y = int(center[0]), int(center[1])

    # Create a square region around the center
    x_min = max(0, x - 3 * radius)
    x_max = min(width, x + 3 * radius + 1)
    y_min = max(0, y - 3 * radius)
    y_max = min(height, y + 3 * radius + 1)

    # Generate Gaussian values for this region
    for dy in range(y_min, y_max):
        for dx in range(x_min, x_max):
            dist_sq = (dx - x) ** 2 + (dy - y) ** 2
            gaussian_val = np.exp(-dist_sq / (2 * radius ** 2))
            heatmap[dy, dx] = max(heatmap[dy, dx], gaussian_val)

    return heatmap


def generate_heatmaps(keypoints, output_h, output_w, radius=7):
    """
    Generate heatmaps for all keypoints.

    Args:
        keypoints: torch.Tensor of shape (num_keypoints, 2) in pixel coordinates
        output_h: output heatmap height
        output_w: output heatmap width
        radius: Gaussian radius

    Returns:
        heatmaps: numpy array of shape (num_keypoints, output_h, output_w)
    """
    num_keypoints = keypoints.shape[0]
    heatmaps = np.zeros((num_keypoints, output_h, output_w), dtype=np.float32)

    for i in range(num_keypoints):
        x, y = keypoints[i, 0].item(), keypoints[i, 1].item()

        # Skip invalid keypoints (outside image bounds)
        if x < 0 or y < 0 or x >= output_w or y >= output_h:
            continue

        generate_gaussian_heatmap(heatmaps[i], (x, y), radius=radius)

    return heatmaps
