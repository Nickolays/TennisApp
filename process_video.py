#!/usr/bin/env python3
"""
Tennis Computer Vision - Video Processing Script
File: process_video.py

Process a tennis video and generate visualization output
"""
import sys
import os
import argparse
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from app.core.video_processor import VideoProcessor
from app.core.data_models import ProcessingConfig


def process_video(input_path: str, output_path: str = None, config: ProcessingConfig = None):
    """
    Process a tennis video and generate output
    
    Args:
        input_path: Path to input video file
        output_path: Path for output video (optional)
        config: Processing configuration (optional)
    """
    print("="*60)
    print("TENNIS COMPUTER VISION - VIDEO PROCESSING")
    print("="*60)
    
    # Validate input file
    if not os.path.exists(input_path):
        print(f"❌ Error: Input video not found: {input_path}")
        return False
    
    print(f"📹 Input video: {input_path}")
    
    # Create default config if not provided
    if config is None:
        config = ProcessingConfig(
            motion_threshold=5.0,
            court_detection_interval=30,
            batch_size=8,
            save_visualization=True,
            draw_court_lines=True,
            draw_trajectories=True,
            results_path="results/"
        )
    
    # Set output path
    if output_path is None:
        input_file = Path(input_path)
        output_path = f"results/{input_file.stem}_processed.mp4"
    
    print(f"📁 Output video: {output_path}")
    print(f"⚙️ Configuration:")
    print(f"   - Motion threshold: {config.motion_threshold}")
    print(f"   - Court detection interval: {config.court_detection_interval}")
    print(f"   - Batch size: {config.batch_size}")
    print(f"   - Save visualization: {config.save_visualization}")
    print(f"   - Draw court lines: {config.draw_court_lines}")
    print(f"   - Draw trajectories: {config.draw_trajectories}")
    
    try:
        # Create processor
        print("\n🔧 Initializing pipeline...")
        processor = VideoProcessor(config)
        
        # Process video
        print("\n🎾 Processing video...")
        analytics = processor(input_path, output_path)
        
        # Display results
        print("\n📊 Processing Results:")
        print(f"   - Total frames: {analytics.total_frames}")
        print(f"   - FPS: {analytics.fps}")
        print(f"   - Duration: {analytics.duration_seconds:.2f} seconds")
        print(f"   - Game segments: {len(analytics.game_segments)}")
        
        rally_segments = analytics.get_rally_segments()
        print(f"   - Rally segments: {len(rally_segments)}")
        print(f"   - Total rallies: {analytics.total_rallies}")
        
        # Show segment details
        if rally_segments:
            print("\n🏆 Rally Segments:")
            for i, segment in enumerate(rally_segments[:5]):  # Show first 5
                duration = segment.duration_seconds(analytics.fps)
                print(f"   {i+1}. Frames {segment.start_frame}-{segment.end_frame} ({duration:.2f}s)")
            
            if len(rally_segments) > 5:
                print(f"   ... and {len(rally_segments) - 5} more segments")
        
        print(f"\n✅ Video processing completed!")
        print(f"📁 Output saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Process tennis video with computer vision")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument("-t", "--threshold", type=float, default=5.0, 
                       help="Motion detection threshold (default: 5.0)")
    parser.add_argument("-c", "--court-interval", type=int, default=30,
                       help="Court detection interval in frames (default: 30)")
    parser.add_argument("-b", "--batch-size", type=int, default=8,
                       help="Batch size for processing (default: 8)")
    parser.add_argument("--no-court-lines", action="store_true",
                       help="Disable court line drawing")
    parser.add_argument("--no-trajectories", action="store_true",
                       help="Disable trajectory drawing")
    parser.add_argument("--no-visualization", action="store_true",
                       help="Disable video output (analysis only)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = ProcessingConfig(
        motion_threshold=args.threshold,
        court_detection_interval=args.court_interval,
        batch_size=args.batch_size,
        save_visualization=not args.no_visualization,
        draw_court_lines=not args.no_court_lines,
        draw_trajectories=not args.no_trajectories,
        results_path="results/"
    )
    
    # Process video
    success = process_video(args.input_video, args.output, config)
    
    if success:
        print("\n🎉 Processing completed successfully!")
        print("\nNext steps:")
        print("1. View the processed video to see visualizations")
        print("2. Check the analytics results")
        print("3. Adjust parameters if needed")
    else:
        print("\n💥 Processing failed!")
        print("Check the error messages above and try again.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())


