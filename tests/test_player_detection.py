#!/usr/bin/env python3
"""
Player Detection Test - Test YOLO v11 player detection with filtering

Usage:
    python test_player_detection.py
"""
import sys
import time
import cv2

from app.core.context import ProcessingContext
from app.core.data_models import FrameDetection

# Detection steps
from app.steps.detection.court_detection import CourtDetectionStep
from app.steps.detection.ball_detection import BallDetectionStep
from app.steps.detection.player_detection import PlayerDetectionStep


def load_video(video_path: str, max_frames: int = None):
    """Load video and create initial context"""
    print(f"\n{'='*60}")
    print("LOADING VIDEO")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames:
        frame_count = min(frame_count, max_frames)

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames to process: {frame_count}")

    # Read frames
    frames = []
    frame_ids = []

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        frame_ids.append(i)

    cap.release()

    print(f"Loaded {len(frames)} frames")

    # Create context
    context = ProcessingContext(
        video_path=video_path,
        fps=fps,
        total_frames=len(frames),
        width=width,
        height=height,
        frames=frames,
        frame_ids=frame_ids
    )

    return context


def run_detection(context: ProcessingContext):
    """Run detection pipeline"""
    print(f"\n{'='*60}")
    print("DETECTION PIPELINE")
    print(f"{'='*60}")

    # Court detection configuration
    court_config = {
        'enabled': True,
        'model_path': 'models/court_model_best.pt',
        'interval': 30,
        'confidence_threshold': 0.5,
        'input_size': [640, 360]
    }

    # Ball detection configuration
    ball_config = {
        'enabled': True,
        'model_path': 'models/ball_model_best.pt',
        'interval': 1,
        'confidence_threshold': 0.3,
        'input_size': [640, 360]
    }

    # Player detection configuration
    player_config = {
        'enabled': True,
        'model_path': 'models/yolo11n.pt',  # YOLO v11 nano
        'confidence_threshold': 0.5,
        'interval': 1,
        'input_size': [640, 640],

        # Filtering parameters
        'court_margin': 100,
        'min_box_area': 5000,
        'max_players': 4,
        'activity_window': 10,
        'min_movement': 10.0
    }

    # Run detection steps
    court_step = CourtDetectionStep(court_config)
    ball_step = BallDetectionStep(ball_config)
    player_step = PlayerDetectionStep(player_config)

    print("\n[1/3] Court Detection")
    start_time = time.time()
    context = court_step.process(context)
    court_time = time.time() - start_time
    print(f"  Time: {court_time:.2f}s")

    print("\n[2/3] Ball Detection")
    start_time = time.time()
    context = ball_step.process(context)
    ball_time = time.time() - start_time
    print(f"  Time: {ball_time:.2f}s")

    print("\n[3/3] Player Detection")
    start_time = time.time()
    context = player_step.process(context)
    player_time = time.time() - start_time
    print(f"  Time: {player_time:.2f}s")

    total_time = court_time + ball_time + player_time
    print(f"\nTotal Detection Time: {total_time:.2f}s")

    return context


def print_summary(context: ProcessingContext):
    """Print detection summary"""
    print(f"\n{'='*60}")
    print("DETECTION SUMMARY")
    print(f"{'='*60}")

    # Count detections
    court_count = sum(1 for d in context.detections if d.has_court())
    ball_count = sum(1 for d in context.detections if d.has_ball())
    player_count = sum(1 for d in context.detections if d.has_players())

    print(f"\nDetection Coverage:")
    print(f"  Court: {court_count}/{len(context.detections)} frames")
    print(f"  Ball:  {ball_count}/{len(context.detections)} frames")
    print(f"  Players: {player_count}/{len(context.detections)} frames")

    # Player statistics
    total_players = 0
    max_players_frame = 0
    max_players_count = 0

    for det in context.detections:
        num_players = len(det.player_boxes)
        total_players += num_players

        if num_players > max_players_count:
            max_players_count = num_players
            max_players_frame = det.frame_id

    avg_players = total_players / len(context.detections) if context.detections else 0

    print(f"\nPlayer Statistics:")
    print(f"  Total player detections: {total_players}")
    print(f"  Average players per frame: {avg_players:.2f}")
    print(f"  Max players in single frame: {max_players_count} (frame {max_players_frame})")

    # Show sample detections
    print(f"\nSample Detections (first 5 frames with players):")
    count = 0
    for det in context.detections:
        if det.has_players() and count < 5:
            print(f"  Frame {det.frame_id}: {len(det.player_boxes)} players")
            for i, (box, conf) in enumerate(zip(det.player_boxes, det.player_confidences)):
                x1, y1, x2, y2 = box
                print(f"    Player {i+1}: box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}), conf={conf:.2f}")
            count += 1

    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print(f"{'='*60}\n")


def visualize_frame(context: ProcessingContext, frame_idx: int = 0, output_path: str = "results/player_detection_sample.jpg"):
    """Visualize a frame with all detections"""
    import os

    if frame_idx >= len(context.frames):
        return

    frame = context.frames[frame_idx].copy()
    det = context.detections[frame_idx]

    # Draw court keypoints (green)
    if det.court_keypoints is not None:
        for kp in det.court_keypoints:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # Draw ball (yellow)
    if det.ball_position_px is not None:
        x, y = det.ball_position_px
        x, y = int(x), int(y)
        cv2.circle(frame, (x, y), 8, (0, 255, 255), 2)
        cv2.putText(frame, f"Ball {det.ball_confidence:.2f}", (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Draw players (blue boxes)
    for i, (box, conf) in enumerate(zip(det.player_boxes, det.player_confidences)):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"P{i+1} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save frame
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame)
    print(f"\nVisualization saved to: {output_path}")


def main():
    """Main test function"""
    video_path = "tests/video3.mp4"
    max_frames = 100

    # Load video
    context = load_video(video_path, max_frames)

    # Run detection
    context = run_detection(context)

    # Print summary
    print_summary(context)

    # Visualize sample frame
    # Find first frame with players
    for i, det in enumerate(context.detections):
        if det.has_players():
            visualize_frame(context, frame_idx=i)
            break


if __name__ == '__main__':
    main()
