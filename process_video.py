#!/usr/bin/env python3
"""
Process Tennis Video - Simple Pipeline Script

Usage:
    python process_video.py tests/video3.mp4
    python process_video.py tests/video3.mp4 --max-frames 500
"""
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import json
import torch
from datetime import datetime

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.context import ProcessingContext
from app.core.pipeline import Pipeline
from app.steps.detection.court_detection import CourtDetectionStep
from app.steps.detection.ball_detection import BallDetectionStep
from app.steps.detection.player_detection import PlayerDetectionStep
from app.pipelines.temporal_pipeline import TemporalPipeline
from app.pipelines.geometry_pipeline import GeometryPipeline
from app.pipelines.event_pipeline import EventPipeline
from app.core.data_models import FrameDetection


def process_video_simple(video_path: str, max_frames: int = None):
    """
    Process video with Detection + Temporal pipeline

    Args:
        video_path: Path to video
        max_frames: Maximum frames to process (None = all)
    """
    print("="*80)
    print("TENNIS VIDEO PROCESSING")
    print("="*80)

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"\n❌ Video not found: {video_path}")
        return None

    # Load video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"\n❌ Could not open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo Info:")
    print(f"  Path: {video_path}")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Total frames: {total_frames}")

    # Determine how many frames to process
    frames_to_process = min(max_frames, total_frames) if max_frames else total_frames
    print(f"  Processing: {frames_to_process} frames")

    # Check models
    base_path = Path(__file__).parent
    court_model = base_path / 'models/court_model_best.pt'
    ball_model = base_path / 'models/ball_model_best.pt'

    if not court_model.exists():
        print(f"\n❌ Court model not found: {court_model}")
        cap.release()
        return None

    if not ball_model.exists():
        print(f"\n❌ Ball model not found: {ball_model}")
        cap.release()
        return None

    # GPU check
    if torch.cuda.is_available():
        print(f"\n✓ Using GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print("\n⚠️  Using CPU (slower)\n")

    # Load frames
    print("Loading frames...")
    frames = []
    frame_ids = []

    for i in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_ids.append(i)

        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{frames_to_process} frames...")

    cap.release()
    print(f"✓ Loaded {len(frames)} frames\n")

    # Create context
    context = ProcessingContext(
        video_path=str(video_path),
        fps=fps,
        total_frames=total_frames,
        width=width,
        height=height,
        duration_seconds=total_frames / fps
    )

    context.frames = frames
    context.frame_ids = frame_ids
    context.chunk_start = 0
    context.chunk_end = len(frames)

    # Initialize detections
    context.detections = [
        FrameDetection(frame_id=fid, timestamp=fid / fps)
        for fid in frame_ids
    ]

    # Create pipelines
    print("="*80)
    print("RUNNING PIPELINE")
    print("="*80 + "\n")

    detection_pipeline = Pipeline(
        name="DetectionPipeline",
        steps=[
            CourtDetectionStep({
                'enabled': True,
                'model_path': str(court_model),
                'model_type': 'tracknet',
                'interval': 30,
                'confidence_threshold': 0.5,
                'input_size': [640, 360]
            }),
            BallDetectionStep({
                'enabled': True,
                'model_path': str(ball_model),
                'model_type': 'tracknet',
                'interval': 1,
                'confidence_threshold': 0.3,
                'input_size': [640, 360]
            }),
            PlayerDetectionStep({
                'enabled': True,
                'model_path': 'models/yolo11n.pt',
                'confidence_threshold': 0.35,  # Lower to catch far players
                'interval': 1,
                'input_size': [640, 640],
                'batch_size': 16,  # Process 16 frames at once for speed
                'court_margin': 100,
                'min_box_area': 1500,  # Reduced from 5000 to catch smaller/far players
                'max_players': 4,
                'activity_window': 10,
                'min_movement': 5.0  # Reduced from 10.0 for less active far players
            }),
        ]
    )

    temporal_pipeline = TemporalPipeline(
        gap_filling_config={'enabled': True, 'max_gap_linear': 5, 'max_gap_poly': 15},
        smoothing_config={'enabled': True, 'process_noise': 0.1, 'measurement_noise': 10.0},
        window_config={'enabled': True, 'window_size': 5, 'stride': 1}
    )

    geometry_pipeline = GeometryPipeline(
        homography_config={'enabled': True, 'interval': 30, 'min_keypoints': 8},
        transform_config={'enabled': True, 'transform_ball': True},
        velocity_config={'enabled': True, 'use_kalman': True}
    )

    event_pipeline = EventPipeline(
        hit_config={'enabled': True, 'acceleration_threshold': 15.0, 'min_frames_between': 8},
        bounce_config={'enabled': True, 'min_vy_flip': 2.0, 'min_frames_between': 5},
        inout_config={'enabled': True, 'court_type': 'auto'}
    )

    # Run detection
    print("Step 1/4: Detection Pipeline")
    print("-" * 80)
    context = detection_pipeline.run(context)

    # Run temporal
    print("\nStep 2/4: Temporal Pipeline")
    print("-" * 80)
    context = temporal_pipeline.run(context)

    # Run geometry
    print("\nStep 3/4: Geometry Pipeline")
    print("-" * 80)
    context = geometry_pipeline.run(context)

    # Run events
    print("\nStep 4/4: Event Pipeline")
    print("-" * 80)
    context = event_pipeline.run(context)

    return context


def save_results(context: ProcessingContext, output_dir: Path):
    """Save results to JSON and visualization video"""
    output_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(context.video_path).stem

    # 1. Save JSON results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print("="*80)

    json_path = output_dir / f"{video_name}_results.json"

    results = {
        'video': {
            'path': context.video_path,
            'fps': context.fps,
            'frames': len(context.detections),
        },
        'detections': {
            'court': sum(1 for d in context.detections if d.court_keypoints is not None),
            'ball': sum(1 for d in context.detections if d.ball_position_px is not None),
            'players': sum(1 for d in context.detections if d.has_players()),
        },
        'temporal': {
            'windows': len(context.temporal_windows)
        },
        'events': {
            'bounces': len(context.bounce_events),
            'hits': len(context.hit_events),
            'bounce_frames': context.bounce_events,
            'hit_frames': context.hit_events
        },
        'frames': []
    }

    for det in context.detections:
        frame_data = {
            'frame_id': det.frame_id,
            'timestamp': det.timestamp,
            'ball': None,
            'court': None,
            'players': []
        }

        if det.ball_position_px:
            frame_data['ball'] = {
                'x': float(det.ball_position_px[0]),
                'y': float(det.ball_position_px[1]),
                'confidence': float(det.ball_confidence)
            }

        if det.court_keypoints is not None:
            frame_data['court'] = {
                'keypoints': det.court_keypoints.tolist(),
                'confidence': float(det.court_confidence)
            }

        if det.has_players():
            for box, conf in zip(det.player_boxes, det.player_confidences):
                frame_data['players'].append({
                    'box': [float(x) for x in box],
                    'confidence': float(conf)
                })

        results['frames'].append(frame_data)

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ JSON results: {json_path}")

    # 2. Save visualization video
    video_path = output_dir / f"{video_name}_visualized.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, context.fps, (context.width, context.height))

    print(f"\nCreating visualization video...")

    for i, (frame, det) in enumerate(zip(context.frames, context.detections)):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()

        # Check if this frame has events
        is_bounce = det.frame_id in context.bounce_events
        is_hit = det.frame_id in context.hit_events

        # Draw court keypoints (green dots)
        if det.court_keypoints is not None:
            for kp in det.court_keypoints:
                cv2.circle(frame_bgr, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)

        # Draw players (blue boxes with labels)
        if det.has_players():
            for idx, (box, conf) in enumerate(zip(det.player_boxes, det.player_confidences)):
                x1, y1, x2, y2 = map(int, box)

                # Draw bounding box
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 100, 0), 3)

                # Draw label background
                label = f"Player {idx+1}: {conf:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame_bgr, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), (255, 100, 0), -1)

                # Draw label text
                cv2.putText(frame_bgr, label, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw ball (yellow circle with glow effect)
        if det.ball_position_px:
            x, y = int(det.ball_position_px[0]), int(det.ball_position_px[1])

            # Change ball color based on events
            if is_bounce:
                # Bounce - Red/Orange glow
                cv2.circle(frame_bgr, (x, y), 25, (0, 100, 255), 3)
                cv2.circle(frame_bgr, (x, y), 18, (0, 150, 255), 3)
                cv2.circle(frame_bgr, (x, y), 12, (0, 200, 255), -1)
            elif is_hit:
                # Hit - Bright green glow
                cv2.circle(frame_bgr, (x, y), 25, (0, 255, 100), 3)
                cv2.circle(frame_bgr, (x, y), 18, (0, 255, 150), 3)
                cv2.circle(frame_bgr, (x, y), 12, (0, 255, 0), -1)
            else:
                # Normal - Yellow glow
                cv2.circle(frame_bgr, (x, y), 15, (0, 200, 255), 2)
                cv2.circle(frame_bgr, (x, y), 10, (0, 230, 255), 2)
                cv2.circle(frame_bgr, (x, y), 6, (0, 255, 255), -1)

            # Ball label with event info
            if is_bounce:
                ball_label = "BOUNCE!"
                label_color = (0, 100, 255)  # Red/Orange
            elif is_hit:
                ball_label = "HIT!"
                label_color = (0, 255, 0)  # Green
            else:
                ball_label = f"Ball: {det.ball_confidence:.2f}"
                label_color = (0, 255, 255)  # Yellow

            cv2.putText(frame_bgr, ball_label, (x + 20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

        # Info panel (top-left corner with semi-transparent background)
        info_height = 120
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (0, 0), (350, info_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0, frame_bgr)

        # Frame info
        cv2.putText(frame_bgr, f"Frame: {det.frame_id}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_bgr, f"Time: {det.timestamp:.2f}s", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Detection counts
        cv2.putText(frame_bgr, f"Court: {'YES' if det.has_court() else 'NO'}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if det.has_court() else (0, 0, 255), 2)
        cv2.putText(frame_bgr, f"Ball: {'YES' if det.has_ball() else 'NO'}", (150, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if det.has_ball() else (0, 0, 255), 2)
        cv2.putText(frame_bgr, f"Players: {len(det.player_boxes)}", (250, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0) if det.has_players() else (100, 100, 100), 2)

        writer.write(frame_bgr)

        if (i + 1) % 50 == 0:
            print(f"  Rendered {i + 1}/{len(context.frames)} frames...")

    writer.release()
    print(f"✓ Visualization video: {video_path}")

    return json_path, video_path


def main():
    parser = argparse.ArgumentParser(description="Process tennis video")
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--max-frames', type=int, default=None, help='Max frames to process')
    parser.add_argument('--output', '-o', default='results/', help='Output directory')

    args = parser.parse_args()

    # Process video
    context = process_video_simple(args.video, args.max_frames)

    if context is None:
        print("\n❌ Processing failed!")
        return 1

    # Save results
    output_dir = Path(args.output)
    json_path, video_path = save_results(context, output_dir)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)

    court_count = sum(1 for d in context.detections if d.court_keypoints is not None)
    ball_count = sum(1 for d in context.detections if d.ball_position_px is not None)
    player_count = sum(1 for d in context.detections if d.has_players())
    total_player_detections = sum(len(d.player_boxes) for d in context.detections)

    print(f"\nProcessed: {len(context.detections)} frames")
    print(f"\nDetection Coverage:")
    print(f"  Court: {court_count}/{len(context.detections)} ({court_count/len(context.detections)*100:.1f}%)")
    print(f"  Ball: {ball_count}/{len(context.detections)} ({ball_count/len(context.detections)*100:.1f}%)")
    print(f"  Players: {player_count}/{len(context.detections)} ({player_count/len(context.detections)*100:.1f}%)")
    print(f"  Total player detections: {total_player_detections}")
    print(f"  Avg players/frame: {total_player_detections/len(context.detections):.2f}")
    print(f"\nTemporal Windows: {len(context.temporal_windows)}")

    print(f"\nEvents Detected:")
    print(f"  Ball Bounces: {len(context.bounce_events)}")
    print(f"  Ball Hits: {len(context.hit_events)}")
    if context.bounce_events:
        print(f"  Bounce frames: {context.bounce_events}")
    if context.hit_events:
        print(f"  Hit frames: {context.hit_events}")

    print(f"\nPerformance:")
    total_time = sum(context.step_timings.values())
    print(f"  Total: {total_time:.2f}s")
    print(f"  FPS: {len(context.detections)/total_time:.1f}")

    print(f"\nOutput:")
    print(f"  {json_path}")
    print(f"  {video_path}")

    print("\n🎉 Done!\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
