#!/usr/bin/env python3
"""
Court Detection Visualization Script
Test trained model on sample images and videos
"""
import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from app.models.tracknet import TrackNet
from app.src.postprocess import court_postprocess

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Court keypoint connections
COURT_SKELETON = [
    [0, 4], [4, 8], [8, 12], [12, 6], [6, 1],  # Back side
    [2, 5], [5, 10], [10, 13], [13, 7], [7, 3],  # Front side
    [0, 2], [1, 3],  # Sidelines
    [4, 5], [6, 7],  # Service lines
    [8, 9], [12, 13], [9, 11], [11, 13],  # Center lines
    [10, 11],  # Net
]

# Keypoint names
KEYPOINT_NAMES = [
    "BL", "BR", "FL", "FR",  # Baselines
    "BSL", "FSL", "BSR", "FSR",  # Service Lines
    "BCS", "FCS",  # Center Service
    "NL", "NR",  # Net
    "BC", "FC"  # Center
]


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = TrackNet(out_channels=14).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A')}")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        print(f"  Mean Error: {metrics['mean_error_px']:.2f} px")
        print(f"  PCK@10px: {metrics['pck@10px']:.1f}%")

    return model


def preprocess_image(image, target_size=(640, 640)):
    """
    Preprocess image for model input

    Args:
        image: BGR image (H, W, 3) uint8
        target_size: (H, W) model input size

    Returns:
        tensor: (1, 3, H, W) normalized tensor
        scale: (scale_x, scale_y) for converting predictions back
    """
    orig_h, orig_w = image.shape[:2]
    target_h, target_w = target_size

    # Resize
    resized = cv2.resize(image, (target_w, target_h))

    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0

    # Transpose to (C, H, W)
    transposed = normalized.transpose(2, 0, 1)

    # Add batch dimension and convert to tensor
    tensor = torch.from_numpy(transposed).unsqueeze(0)

    # Scale factors for converting back to original size
    scale_x = orig_w / target_w
    scale_y = orig_h / target_h

    return tensor, (scale_x, scale_y)


def postprocess_predictions(heatmaps, scale):
    """
    Convert heatmaps to keypoint coordinates

    Args:
        heatmaps: (1, K, H, W) heatmap tensor
        scale: (scale_x, scale_y) for converting to original size

    Returns:
        keypoints: (K, 2) numpy array of (x, y) coordinates
        confidence: (K,) numpy array of confidence scores
    """
    # Use postprocess to get keypoints in heatmap coordinates
    kps = court_postprocess(heatmaps)  # (1, K, 2)
    kps = kps[0].cpu().numpy()  # (K, 2)

    # Scale to original image size
    scale_x, scale_y = scale
    kps[:, 0] *= scale_x
    kps[:, 1] *= scale_y

    # Get confidence scores (max value in each heatmap)
    confidence = heatmaps[0].max(dim=-1)[0].max(dim=-1)[0].cpu().numpy()  # (K,)

    return kps, confidence


def draw_court_keypoints(image, keypoints, confidence, threshold=0.5):
    """
    Draw court keypoints and skeleton on image

    Args:
        image: BGR image
        keypoints: (K, 2) array of (x, y) coordinates
        confidence: (K,) array of confidence scores
        threshold: minimum confidence to draw

    Returns:
        annotated image
    """
    result = image.copy()

    # Draw skeleton (connections)
    for i, j in COURT_SKELETON:
        if i < len(keypoints) and j < len(keypoints):
            if confidence[i] > threshold and confidence[j] > threshold:
                pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
                pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
                cv2.line(result, pt1, pt2, (0, 255, 255), 2)

    # Draw keypoints
    for i, (kp, conf) in enumerate(zip(keypoints, confidence)):
        if conf > threshold:
            x, y = int(kp[0]), int(kp[1])

            # Draw circle
            cv2.circle(result, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(result, (x, y), 7, (255, 255, 255), 1)

            # Draw label
            label = f"{i+1:02d}"
            cv2.putText(result, label, (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return result


def visualize_heatmaps(heatmaps, num_cols=7):
    """
    Create visualization grid of all heatmaps

    Args:
        heatmaps: (1, K, H, W) tensor
        num_cols: number of columns in grid

    Returns:
        grid image
    """
    heatmaps = heatmaps[0].cpu().numpy()  # (K, H, W)
    K, H, W = heatmaps.shape

    num_rows = (K + num_cols - 1) // num_cols

    # Create grid
    grid_h = num_rows * H
    grid_w = num_cols * W
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for idx in range(K):
        row = idx // num_cols
        col = idx % num_cols

        # Normalize heatmap to [0, 255]
        hm = heatmaps[idx]
        hm_norm = (hm * 255).astype(np.uint8)

        # Apply colormap
        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)

        # Add to grid
        y1 = row * H
        y2 = (row + 1) * H
        x1 = col * W
        x2 = (col + 1) * W
        grid[y1:y2, x1:x2] = hm_color

        # Add label
        label = f"KP{idx+1:02d}"
        cv2.putText(grid, label, (x1 + 5, y1 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return grid


def visualize_image(model, image_path, output_path=None, show_heatmaps=False):
    """
    Visualize court detection on a single image

    Args:
        model: trained model
        image_path: path to input image
        output_path: path to save output (optional)
        show_heatmaps: whether to show heatmap grid
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    print(f"\nProcessing: {image_path}")
    print(f"  Image size: {image.shape[1]}x{image.shape[0]}")

    # Preprocess
    tensor, scale = preprocess_image(image, target_size=(640, 640))
    tensor = tensor.to(device)

    # Inference
    with torch.no_grad():
        heatmaps = model(tensor)  # (1, 14, 640, 640)

    print(f"  Heatmaps shape: {heatmaps.shape}")
    print(f"  Heatmaps range: [{heatmaps.min():.4f}, {heatmaps.max():.4f}]")

    # Postprocess
    keypoints, confidence = postprocess_predictions(heatmaps, scale)

    print(f"  Keypoints detected: {(confidence > 0.5).sum()}/14")
    print(f"  Confidence range: [{confidence.min():.4f}, {confidence.max():.4f}]")

    # Draw results
    result = draw_court_keypoints(image, keypoints, confidence, threshold=0.3)

    # Show or save
    if output_path:
        cv2.imwrite(str(output_path), result)
        print(f"  Saved to: {output_path}")

    if show_heatmaps:
        heatmap_grid = visualize_heatmaps(heatmaps)
        if output_path:
            heatmap_path = Path(output_path).parent / (Path(output_path).stem + "_heatmaps.jpg")
            cv2.imwrite(str(heatmap_path), heatmap_grid)
            print(f"  Heatmaps saved to: {heatmap_path}")

    # Display
    cv2.imshow("Court Detection", result)
    if show_heatmaps:
        cv2.imshow("Heatmaps", heatmap_grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_video(model, video_path, output_path=None, max_frames=None):
    """
    Visualize court detection on a video

    Args:
        model: trained model
        video_path: path to input video
        output_path: path to save output video (optional)
        max_frames: maximum number of frames to process
    """
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"\nProcessing video: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")

    # Setup video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Process frames
    pbar = tqdm(total=total_frames, desc="Processing")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames and frame_idx >= max_frames:
            break

        # Preprocess
        tensor, scale = preprocess_image(frame, target_size=(640, 640))
        tensor = tensor.to(device)

        # Inference
        with torch.no_grad():
            heatmaps = model(tensor)

        # Postprocess
        keypoints, confidence = postprocess_predictions(heatmaps, scale)

        # Draw results
        result = draw_court_keypoints(frame, keypoints, confidence, threshold=0.3)

        # Add frame info
        cv2.putText(result, f"Frame: {frame_idx+1}/{total_frames}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Write frame
        if writer:
            writer.write(result)

        # Display
        cv2.imshow("Court Detection", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer:
        writer.release()
        print(f"\nSaved to: {output_path}")

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Visualize court detection")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input image or video")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save output")
    parser.add_argument("--heatmaps", action="store_true",
                       help="Show heatmap visualizations")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames to process (for videos)")

    args = parser.parse_args()

    # Load model
    model = load_model(args.checkpoint, device=device)

    # Determine input type
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    # Process based on file type
    if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        visualize_image(model, input_path, args.output, args.heatmaps)
    elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
        visualize_video(model, input_path, args.output, args.max_frames)
    else:
        print(f"Error: Unsupported file type: {input_path.suffix}")


if __name__ == "__main__":
    # Example usage:
    # python visualize_court_test.py --checkpoint checkpoints/court_detection_fixed/best_model.pth --input data/test_image.jpg --output results/test_result.jpg --heatmaps
    main()
