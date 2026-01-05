#!/usr/bin/env python3
"""
Court Detection Video Visualization
Visualize court keypoint predictions on tennis videos with advanced features
"""
import os
import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from collections import deque

from app.models.tracknet import TrackNet
from app.src.postprocess import court_postprocess

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Court keypoint connections (skeleton)
COURT_SKELETON = [
    # Outer boundaries
    [0, 4], [4, 8], [8, 12], [12, 6], [6, 1],  # Back side (left to right)
    [2, 5], [5, 10], [10, 13], [13, 7], [7, 3],  # Front side (left to right)

    # Sidelines
    [0, 2],  # Left sideline (back to front)
    [1, 3],  # Right sideline (back to front)

    # Service lines
    [4, 5],  # Left service line
    [6, 7],  # Right service line

    # Center lines
    [8, 9],   # Back center service
    [12, 13], # Front center service
    [9, 11],  # Center service line (back half)
    [11, 13], # Center service line (front half)

    # Net
    [10, 11], # Net center line
]

# Keypoint names
KEYPOINT_NAMES = [
    "01-BL", "02-BR", "03-FL", "04-FR",  # Baselines (Back/Front Left/Right)
    "05-BSL", "06-FSL", "07-BSR", "08-FSR",  # Service Lines (Back/Front Service Left/Right)
    "09-BCS", "10-FCS",  # Center Service (Back/Front)
    "11-NL", "12-NR",    # Net (Left/Right)
    "13-BC", "14-FC"     # Center (Back/Front)
]

# Colors
COLOR_KEYPOINT = (0, 255, 0)      # Green
COLOR_SKELETON = (0, 255, 255)    # Yellow
COLOR_TEXT = (255, 255, 255)      # White
COLOR_BG = (0, 0, 0)              # Black
COLOR_ERROR = (0, 0, 255)         # Red
COLOR_HEATMAP_COLD = (255, 0, 0)  # Blue
COLOR_HEATMAP_HOT = (0, 0, 255)   # Red


class CourtVisualizer:
    """Court detection visualizer with advanced features"""

    def __init__(self, model, target_size=(512, 512), device='cuda'):
        """
        Initialize visualizer

        Args:
            model: Trained TrackNet model
            target_size: Model input size (H, W)
            device: cuda or cpu
        """
        self.model = model
        self.target_size = target_size
        self.device = device

        # Statistics
        self.keypoint_history = []  # Store keypoints for temporal smoothing
        self.smoothing_window = 5   # Frames for temporal smoothing

        # Heatmap accumulation
        self.heatmap_accumulator = None
        self.heatmap_alpha = 0.3

    def preprocess_frame(self, frame):
        """
        Preprocess frame for model input

        Args:
            frame: BGR frame (H, W, 3) uint8

        Returns:
            tensor: (1, 3, H, W) normalized tensor
            scale: (scale_x, scale_y) for converting predictions back
        """
        orig_h, orig_w = frame.shape[:2]
        target_h, target_w = self.target_size

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize
        resized = cv2.resize(frame_rgb, (target_w, target_h))

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Convert to tensor (C, H, W)
        tensor = torch.from_numpy(normalized).permute(2, 0, 1)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        # Calculate scale factors
        scale_x = orig_w / target_w
        scale_y = orig_h / target_h

        return tensor, (scale_x, scale_y)

    def predict_keypoints(self, frame):
        """
        Predict court keypoints for a frame

        Args:
            frame: BGR frame (H, W, 3) uint8

        Returns:
            keypoints: (14, 2) numpy array [x, y] in original frame coordinates
        """
        # Preprocess
        tensor, (scale_x, scale_y) = self.preprocess_frame(frame)
        tensor = tensor.to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(tensor)  # (1, 14, H, W)
            preds = court_postprocess(outputs)  # (1, 14, 2)

        # Convert to numpy
        keypoints = preds[0].cpu().numpy()  # (14, 2)

        # Scale back to original frame size
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        return keypoints

    def smooth_keypoints(self, keypoints):
        """
        Temporal smoothing of keypoints using sliding window average

        Args:
            keypoints: (14, 2) numpy array

        Returns:
            smoothed: (14, 2) numpy array
        """
        # Add to history
        self.keypoint_history.append(keypoints.copy())

        # Keep only last N frames
        if len(self.keypoint_history) > self.smoothing_window:
            self.keypoint_history.pop(0)

        # Average over window
        if len(self.keypoint_history) >= 2:
            smoothed = np.mean(self.keypoint_history, axis=0)
        else:
            smoothed = keypoints

        return smoothed

    def compute_keypoint_stability(self):
        """
        Compute stability metric (how much keypoints move between frames)

        Returns:
            stability: Average pixel movement between consecutive frames
        """
        if len(self.keypoint_history) < 2:
            return 0.0

        # Calculate movement between last two frames
        prev = self.keypoint_history[-2]
        curr = self.keypoint_history[-1]

        movement = np.sqrt(np.sum((curr - prev) ** 2, axis=1))  # (14,)
        avg_movement = np.mean(movement)

        return avg_movement

    def draw_keypoints(self, frame, keypoints, draw_skeleton=True, draw_labels=True,
                       keypoint_radius=6, line_thickness=2, alpha=1.0):
        """
        Draw court keypoints and skeleton on frame

        Args:
            frame: BGR frame to draw on
            keypoints: (14, 2) numpy array [x, y]
            draw_skeleton: Whether to draw skeleton lines
            draw_labels: Whether to draw keypoint labels
            keypoint_radius: Radius of keypoint circles
            line_thickness: Thickness of skeleton lines
            alpha: Transparency (1.0 = opaque, 0.0 = transparent)

        Returns:
            frame: Frame with drawings
        """
        if alpha < 1.0:
            overlay = frame.copy()
        else:
            overlay = frame

        # Draw skeleton first (so keypoints appear on top)
        if draw_skeleton:
            for i, j in COURT_SKELETON:
                pt1 = tuple(keypoints[i].astype(int))
                pt2 = tuple(keypoints[j].astype(int))
                cv2.line(overlay, pt1, pt2, COLOR_SKELETON, line_thickness)

        # Draw keypoints
        for idx, (x, y) in enumerate(keypoints):
            x, y = int(x), int(y)

            # Draw filled circle
            cv2.circle(overlay, (x, y), keypoint_radius, COLOR_KEYPOINT, -1)

            # Draw border
            cv2.circle(overlay, (x, y), keypoint_radius + 1, (0, 0, 0), 1)

            # Draw label
            if draw_labels:
                label = KEYPOINT_NAMES[idx]

                # Calculate text size
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
                )

                # Draw background
                cv2.rectangle(
                    overlay,
                    (x - text_w // 2 - 2, y - keypoint_radius - text_h - 5),
                    (x + text_w // 2 + 2, y - keypoint_radius - 2),
                    COLOR_BG,
                    -1
                )

                # Draw text
                cv2.putText(
                    overlay,
                    label,
                    (x - text_w // 2, y - keypoint_radius - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    COLOR_TEXT,
                    1,
                    cv2.LINE_AA
                )

        # Blend with alpha
        if alpha < 1.0:
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame

    def draw_info_panel(self, frame, frame_idx, total_frames, fps, stability=None,
                       position='top'):
        """
        Draw information panel

        Args:
            frame: BGR frame
            frame_idx: Current frame index
            total_frames: Total number of frames
            fps: Video FPS
            stability: Keypoint stability metric
            position: 'top' or 'bottom'

        Returns:
            frame: Frame with info panel
        """
        h, w = frame.shape[:2]

        # Build info text
        info_lines = [
            f"Frame: {frame_idx + 1}/{total_frames}",
            f"Time: {frame_idx / fps:.2f}s",
            f"FPS: {fps} | Resolution: {w}x{h}",
        ]

        if stability is not None:
            info_lines.append(f"Stability: {stability:.2f}px")

        # Panel dimensions
        panel_height = 30 + len(info_lines) * 22
        panel_y = 0 if position == 'top' else h - panel_height

        # Draw semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_y), (w, panel_y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw text
        y_offset = panel_y + 20
        for text in info_lines:
            cv2.putText(
                frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_TEXT,
                1,
                cv2.LINE_AA
            )
            y_offset += 22

        return frame

    def draw_minimap(self, frame, keypoints, position='bottom_right', size=(200, 150)):
        """
        Draw top-down court minimap

        Args:
            frame: BGR frame
            keypoints: (14, 2) numpy array
            position: 'top_left', 'top_right', 'bottom_left', 'bottom_right'
            size: (width, height) of minimap

        Returns:
            frame: Frame with minimap
        """
        h, w = frame.shape[:2]
        map_w, map_h = size

        # Create minimap canvas
        minimap = np.zeros((map_h, map_w, 3), dtype=np.uint8)
        minimap[:] = (34, 139, 34)  # Tennis court green

        # Normalize keypoints to minimap size
        kp_normalized = keypoints.copy()

        # Find bounding box of court
        min_x, min_y = kp_normalized.min(axis=0)
        max_x, max_y = kp_normalized.max(axis=0)

        # Add margin
        margin = 10
        kp_normalized[:, 0] = (kp_normalized[:, 0] - min_x) / (max_x - min_x) * (map_w - 2 * margin) + margin
        kp_normalized[:, 1] = (kp_normalized[:, 1] - min_y) / (max_y - min_y) * (map_h - 2 * margin) + margin

        # Draw skeleton
        for i, j in COURT_SKELETON:
            pt1 = tuple(kp_normalized[i].astype(int))
            pt2 = tuple(kp_normalized[j].astype(int))
            cv2.line(minimap, pt1, pt2, (255, 255, 255), 1)

        # Draw keypoints
        for x, y in kp_normalized:
            cv2.circle(minimap, (int(x), int(y)), 3, (255, 255, 0), -1)

        # Position on main frame
        if position == 'top_left':
            x_offset, y_offset = 10, 10
        elif position == 'top_right':
            x_offset, y_offset = w - map_w - 10, 10
        elif position == 'bottom_left':
            x_offset, y_offset = 10, h - map_h - 10
        else:  # bottom_right
            x_offset, y_offset = w - map_w - 10, h - map_h - 10

        # Overlay minimap
        frame[y_offset:y_offset + map_h, x_offset:x_offset + map_w] = minimap

        # Draw border
        cv2.rectangle(
            frame,
            (x_offset, y_offset),
            (x_offset + map_w, y_offset + map_h),
            (255, 255, 255),
            2
        )

        return frame

    def draw_trajectory_trace(self, frame, keypoint_idx=0, trace_length=30):
        """
        Draw trajectory trace for a specific keypoint

        Args:
            frame: BGR frame
            keypoint_idx: Index of keypoint to trace (0-13)
            trace_length: Number of past frames to show

        Returns:
            frame: Frame with trajectory trace
        """
        if len(self.keypoint_history) < 2:
            return frame

        # Get trajectory for this keypoint
        trajectory = [kp[keypoint_idx] for kp in self.keypoint_history[-trace_length:]]

        # Draw line through trajectory
        for i in range(len(trajectory) - 1):
            pt1 = tuple(trajectory[i].astype(int))
            pt2 = tuple(trajectory[i + 1].astype(int))

            # Color fades from old to new
            alpha = (i + 1) / len(trajectory)
            color = (int(255 * (1 - alpha)), int(255 * alpha), 0)

            cv2.line(frame, pt1, pt2, color, 2)

        return frame


def visualize_video(model_path, video_path, output_path=None,
                   target_size=(512, 512), device='cuda',
                   max_frames=None, skip_frames=1,
                   draw_skeleton=True, draw_labels=True,
                   draw_minimap=True, smooth_keypoints=True,
                   save_json=False, json_path=None):
    """
    Visualize court detection on video with advanced features

    Args:
        model_path: Path to trained model checkpoint
        video_path: Path to input video
        output_path: Path to save output video
        target_size: Model input size
        device: cuda or cpu
        max_frames: Maximum frames to process
        skip_frames: Process every N frames
        draw_skeleton: Draw court skeleton lines
        draw_labels: Draw keypoint labels
        draw_minimap: Draw top-down minimap
        smooth_keypoints: Apply temporal smoothing
        save_json: Save keypoints to JSON
        json_path: Path to save JSON

    Returns:
        stats: Processing statistics
    """
    print(f"{'='*60}")
    print("COURT DETECTION VIDEO VISUALIZATION")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Target size: {target_size}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    model = TrackNet(out_channels=14).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        if "val_loss" in checkpoint:
            print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print(f"  Loaded raw state dict")

    model.eval()

    # Initialize visualizer
    visualizer = CourtVisualizer(model, target_size, device)

    # Open video
    print(f"\nOpening video...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames / fps:.2f}s")

    # Output video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"\nOutput: {output_path}")

    # JSON output
    keypoints_data = {
        "video": str(video_path),
        "fps": fps,
        "resolution": [width, height],
        "total_frames": 0,
        "frames": []
    }

    # Process frames
    print(f"\nProcessing frames...")
    frame_count = 0
    processed_count = 0

    pbar = tqdm(total=min(total_frames, max_frames) if max_frames else total_frames,
                desc="Processing")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check max frames
        if max_frames and processed_count >= max_frames:
            break

        # Skip frames
        if frame_count % skip_frames != 0:
            frame_count += 1
            continue

        # Predict keypoints
        keypoints = visualizer.predict_keypoints(frame)

        # Apply temporal smoothing
        if smooth_keypoints:
            keypoints = visualizer.smooth_keypoints(keypoints)

        # Compute stability
        stability = visualizer.compute_keypoint_stability()

        # Draw visualization
        result = visualizer.draw_keypoints(
            frame.copy(),
            keypoints,
            draw_skeleton=draw_skeleton,
            draw_labels=draw_labels
        )

        # Draw info panel
        result = visualizer.draw_info_panel(
            result,
            processed_count,
            total_frames,
            fps,
            stability=stability
        )

        # Draw minimap
        if draw_minimap:
            result = visualizer.draw_minimap(
                result,
                keypoints,
                position='bottom_right'
            )

        # Save frame
        if writer:
            writer.write(result)

        # Save keypoints data
        if save_json:
            keypoints_data["frames"].append({
                "frame": processed_count,
                "timestamp": processed_count / fps,
                "keypoints": keypoints.tolist(),
                "stability": float(stability)
            })

        processed_count += 1
        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer:
        writer.release()

    # Save JSON
    if save_json and json_path:
        keypoints_data["total_frames"] = processed_count
        with open(json_path, 'w') as f:
            json.dump(keypoints_data, f, indent=2)
        print(f"\n✓ Keypoints saved: {json_path}")

    # Statistics
    stats = {
        "total_frames": total_frames,
        "processed_frames": processed_count,
        "fps": fps,
        "duration": total_frames / fps,
        "output": output_path,
    }

    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed: {processed_count} frames")
    print(f"Duration: {processed_count / fps:.2f}s")
    if output_path:
        print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Visualize court detection on tennis videos"
    )

    # Required arguments
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--model", required=True, help="Trained model checkpoint")

    # Output options
    parser.add_argument("--output", help="Output video path")
    parser.add_argument("--save-json", action='store_true',
                       help="Save keypoints to JSON")
    parser.add_argument("--json-output", help="JSON output path (default: same as output)")

    # Processing options
    parser.add_argument("--target-size", type=int, nargs=2, default=[512, 512],
                       help="Model input size (H W)")
    parser.add_argument("--max-frames", type=int,
                       help="Maximum frames to process")
    parser.add_argument("--skip-frames", type=int, default=1,
                       help="Process every N frames")
    parser.add_argument("--device", default='cuda', choices=['cuda', 'cpu'],
                       help="Device to use")

    # Visualization options
    parser.add_argument("--no-skeleton", action='store_true',
                       help="Don't draw court skeleton")
    parser.add_argument("--no-labels", action='store_true',
                       help="Don't draw keypoint labels")
    parser.add_argument("--no-minimap", action='store_true',
                       help="Don't draw minimap")
    parser.add_argument("--no-smoothing", action='store_true',
                       help="Don't apply temporal smoothing")

    args = parser.parse_args()

    # Check input
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return

    # JSON output path
    json_path = None
    if args.save_json:
        if args.json_output:
            json_path = args.json_output
        elif args.output:
            json_path = str(Path(args.output).with_suffix('.json'))
        else:
            json_path = str(video_path.with_suffix('.json'))

    # Run visualization
    stats = visualize_video(
        model_path=args.model,
        video_path=video_path,
        output_path=args.output,
        target_size=tuple(args.target_size),
        device=args.device,
        max_frames=args.max_frames,
        skip_frames=args.skip_frames,
        draw_skeleton=not args.no_skeleton,
        draw_labels=not args.no_labels,
        draw_minimap=not args.no_minimap,
        smooth_keypoints=not args.no_smoothing,
        save_json=args.save_json,
        json_path=json_path,
    )


if __name__ == "__main__":
    main()
