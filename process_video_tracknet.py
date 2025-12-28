#!/usr/bin/env python3
"""
Tennis Computer Vision - Real TrackNet Video Processing
File: process_video_tracknet.py

Process tennis video using actual TrackNet models and save output with visualizations
"""
import sys
import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import Dict

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from app.models.unified_detection import (
    UnifiedDetectionPipeline, DetectionType, DetectionOutput
)
from app.core.data_models import ProcessingConfig


def create_output_directory(video_path: str) -> Path:
    """Create output directory structure"""
    video_name = Path(video_path).stem
    output_dir = Path("results") / "line" / video_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def draw_court_detection(frame: np.ndarray, detection: DetectionOutput, color=(0, 255, 0)) -> np.ndarray:
    """Draw court detection results on frame"""
    if detection.keypoints is not None and len(detection.keypoints) > 0:
        # Draw court keypoints
        for i, (x, y) in enumerate(detection.keypoints):
            cv2.circle(frame, (int(x), int(y)), 5, color, -1)
            cv2.putText(frame, str(i), (int(x)+10, int(y)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw court lines if available
        if detection.court_lines:
            for line in detection.court_lines:
                pt1, pt2 = line
                cv2.line(frame, (int(pt1[0]), int(pt1[1])), 
                        (int(pt2[0]), int(pt2[1])), color, 2)
    
    return frame


def draw_ball_detection(frame: np.ndarray, detection: DetectionOutput, color=(0, 0, 255)) -> np.ndarray:
    """Draw ball detection results on frame"""
    if detection.ball_position:
        x, y = detection.ball_position
        cv2.circle(frame, (int(x), int(y)), 8, color, -1)
        cv2.putText(frame, f"Ball: {detection.confidence:.2f}", 
                   (int(x)+15, int(y)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame


def draw_pose_detection(frame: np.ndarray, detection: DetectionOutput, color=(255, 0, 0)) -> np.ndarray:
    """Draw pose detection results on frame"""
    if detection.pose_keypoints is not None and len(detection.pose_keypoints) > 0:
        pose_kpts = detection.pose_keypoints[0]  # First person
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(pose_kpts):
            if conf > 0.3:  # Only draw confident keypoints
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        
        # Draw skeleton connections
        if detection.pose_skeleton:
            for connection in detection.pose_skeleton:
                start_idx, end_idx = connection
                if (start_idx < len(pose_kpts) and end_idx < len(pose_kpts) and
                    pose_kpts[start_idx][2] > 0.3 and pose_kpts[end_idx][2] > 0.3):
                    
                    start_point = (int(pose_kpts[start_idx][0]), int(pose_kpts[start_idx][1]))
                    end_point = (int(pose_kpts[end_idx][0]), int(pose_kpts[end_idx][1]))
                    cv2.line(frame, start_point, end_point, color, 2)
    
    return frame


def draw_detection_info(frame: np.ndarray, detections: Dict[DetectionType, DetectionOutput], 
                       frame_id: int, fps: float) -> np.ndarray:
    """Draw detection information overlay"""
    h, w = frame.shape[:2]
    
    # Create info panel
    info_height = 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, info_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Frame info
    timestamp = frame_id / fps
    cv2.putText(frame, f"Frame: {frame_id} | Time: {timestamp:.2f}s", 
               (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Detection info
    y_offset = 50
    for detection_type, detection in detections.items():
        if detection:
            info_text = f"{detection_type.value.upper()}: {detection.confidence:.3f}"
            cv2.putText(frame, info_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
    
    return frame


def process_video_with_tracknet(input_path: str, output_path: str = None, 
                              max_frames: int = None) -> bool:
    """
    Process video using actual TrackNet models
    
    Args:
        input_path: Path to input video
        output_path: Path for output video
        max_frames: Maximum frames to process (None = all)
    """
    print("="*60)
    print("TENNIS COMPUTER VISION - TRACKNET PROCESSING")
    print("="*60)
    
    # Validate input file
    if not os.path.exists(input_path):
        print(f"❌ Error: Input video not found: {input_path}")
        return False
    
    print(f"📹 Input video: {input_path}")
    
    # Create output directory and path
    output_dir = create_output_directory(input_path)
    if output_path is None:
        output_path = output_dir / f"{Path(input_path).stem}_tracknet_processed.mp4"
    
    print(f"📁 Output video: {output_path}")
    
    # Create configuration optimized for TrackNet
    config = ProcessingConfig(
        use_fp16=True,           # Enable FP16 for faster inference
        batch_size=4,            # Smaller batch for memory efficiency
        motion_threshold=5.0,
        court_detection_interval=30,
        save_visualization=True,
        draw_court_lines=True,
        draw_trajectories=True,
        results_path=str(output_dir)
    )
    
    print(f"⚙️ Configuration:")
    print(f"   - Use FP16: {config.use_fp16}")
    print(f"   - Batch size: {config.batch_size}")
    print(f"   - Court detection interval: {config.court_detection_interval}")
    
    try:
        # Create unified pipeline
        print("\n🔧 Initializing TrackNet pipeline...")
        pipeline = UnifiedDetectionPipeline(config)
        
        # Add TrackNet models
        court_model_path = "models/model_tennis_court_det.pt"
        ball_model_path = "models/model_best.pt"
        
        print(f"Adding TrackNet models...")
        court_success = pipeline.add_model(DetectionType.COURT, court_model_path)
        ball_success = pipeline.add_model(DetectionType.BALL, ball_model_path)
        
        print(f"  - Court Detection: {'✓' if court_success else '✗'}")
        print(f"  - Ball Detection: {'✓' if ball_success else '✗'}")
        
        # Initialize pipeline
        init_success = pipeline.initialize()
        if not init_success:
            print("❌ Pipeline initialization failed")
            return False
        
        print("✓ TrackNet pipeline initialized successfully")
        
        # Load video
        print(f"\n📹 Loading video: {input_path}")
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print("❌ Error: Could not open video file")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"Video properties:")
        print(f"  - Resolution: {width}x{height}")
        print(f"  - FPS: {fps}")
        print(f"  - Total frames: {total_frames}")
        print(f"  - Duration: {total_frames/fps:.2f}s")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not writer.isOpened():
            print("❌ Error: Could not create output video writer")
            cap.release()
            return False
        
        # Process video
        print(f"\n🎾 Processing video with TrackNet...")
        
        frame_count = 0
        batch_size = config.batch_size
        batch_frames = []
        batch_frame_ids = []
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            batch_frames.append(frame)
            batch_frame_ids.append(frame_count)
            
            # Process batch when full or at end
            if len(batch_frames) == batch_size or frame_count == total_frames - 1:
                if batch_frames:
                    print(f"Processing frames {batch_frame_ids[0]}-{batch_frame_ids[-1]}...")
                    
                    # Convert to numpy array
                    frames_array = np.array(batch_frames)
                    
                    # Run TrackNet detection
                    try:
                        results = pipeline.detect_sequential(frames_array)
                        
                        # Draw results on frames
                        for i, frame in enumerate(batch_frames):
                            frame_id = batch_frame_ids[i]
                            
                            # Get detections for this frame
                            frame_detections = {}
                            for detection_type, detection_results in results.items():
                                if i < len(detection_results):
                                    frame_detections[detection_type] = detection_results[i]
                                else:
                                    frame_detections[detection_type] = None
                            
                            # Draw detections
                            if DetectionType.COURT in frame_detections and frame_detections[DetectionType.COURT]:
                                frame = draw_court_detection(frame, frame_detections[DetectionType.COURT])
                            
                            if DetectionType.BALL in frame_detections and frame_detections[DetectionType.BALL]:
                                frame = draw_ball_detection(frame, frame_detections[DetectionType.BALL])
                            
                            # Draw info overlay
                            frame = draw_detection_info(frame, frame_detections, frame_id, fps)
                            
                            # Write frame
                            writer.write(frame)
                        
                        print(f"  ✓ Processed {len(batch_frames)} frames")
                        
                    except Exception as e:
                        print(f"  ✗ Error processing batch: {e}")
                        # Write original frames if processing fails
                        for frame in batch_frames:
                            writer.write(frame)
                    
                    # Clear batch
                    batch_frames = []
                    batch_frame_ids = []
            
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        # Cleanup
        cap.release()
        writer.release()
        
        print(f"\n✅ Video processing completed!")
        print(f"📁 Output saved to: {output_path}")
        print(f"📊 Processed {frame_count} frames")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Process tennis video with TrackNet models")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument("-m", "--max-frames", type=int, help="Maximum frames to process")
    
    args = parser.parse_args()
    
    # Process video
    success = process_video_with_tracknet(
        args.input_video, 
        args.output, 
        args.max_frames
    )
    
    if success:
        print("\n🎉 TrackNet video processing completed successfully!")
        print("\nNext steps:")
        print("1. View the processed video to see TrackNet detections")
        print("2. Check the court detection accuracy")
        print("3. Verify ball tracking performance")
        print("4. Adjust parameters if needed")
    else:
        print("\n💥 TrackNet video processing failed!")
        print("Check the error messages above and try again.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
