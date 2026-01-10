#!/usr/bin/env python3
"""
Test script for court keypoint refinement.

This script tests the geometric refinement pipeline on a single frame
and visualizes the before/after results.

Usage:
    python test_court_refinement.py

Requirements:
    - Test frame: tests/test_court_frame.jpg (or any video frame)
    - Initial keypoints: tests/test_keypoints_initial.npy
    - Confidences: tests/test_confidences.npy

If test data doesn't exist, the script will extract it from a video.

File: test_court_refinement.py
"""
import cv2
import numpy as np
from pathlib import Path
import argparse

from app.src.court_refinement import CourtKeypointRefinement
from app.core.data_models import COURT_TEMPLATE_KEYPOINTS


def extract_test_data_from_video(video_path: str, output_dir: str = "tests"):
    """
    Extract test frame and keypoints from a video for testing refinement.

    Args:
        video_path: Path to test video
        output_dir: Directory to save test data
    """
    print(f"\n📹 Extracting test data from {video_path}...")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return False

    # Read first frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Cannot read frame from video")
        return False

    # Save frame
    frame_path = output_dir / "test_court_frame.jpg"
    cv2.imwrite(str(frame_path), frame)
    print(f"✓ Saved test frame: {frame_path}")

    # Generate dummy keypoints for testing
    # (In real use, these would come from model predictions)
    h, w = frame.shape[:2]

    # Simulate 14 keypoints roughly in court positions
    # These are dummy positions - in real use, run court detection model
    keypoints = np.array([
        [w * 0.2, h * 0.8],   # 0: Back-left baseline
        [w * 0.8, h * 0.8],   # 1: Back-right baseline
        [w * 0.25, h * 0.65], # 2: Back-left service
        [w * 0.75, h * 0.65], # 3: Back-right service
        [w * 0.25, h * 0.5],  # 4: Left net post
        [w * 0.75, h * 0.5],  # 5: Right net post
        [w * 0.25, h * 0.35], # 6: Front-left service
        [w * 0.75, h * 0.35], # 7: Front-right service
        [w * 0.2, h * 0.2],   # 8: Front-left baseline
        [w * 0.8, h * 0.2],   # 9: Front-right baseline
        [w * 0.5, h * 0.65],  # 10: Back center service
        [w * 0.5, h * 0.5],   # 11: Net center
        [w * 0.5, h * 0.35],  # 12: Front center service
        [w * 0.5, h * 0.8],   # 13: Back center baseline
    ], dtype=np.float32)

    # Add some noise to simulate imperfect detection
    keypoints += np.random.randn(14, 2) * 5.0

    # Confidences (all high for test)
    confidences = np.random.rand(14) * 0.3 + 0.7  # 0.7 to 1.0

    # Save
    np.save(str(output_dir / "test_keypoints_initial.npy"), keypoints)
    np.save(str(output_dir / "test_confidences.npy"), confidences)

    print(f"✓ Saved test keypoints: {output_dir / 'test_keypoints_initial.npy'}")
    print(f"✓ Saved test confidences: {output_dir / 'test_confidences.npy'}")

    return True


def test_refinement_on_frame(
    frame_path: str = "tests/test_court_frame.jpg",
    keypoints_path: str = "tests/test_keypoints_initial.npy",
    confidences_path: str = "tests/test_confidences.npy",
    output_path: str = "results/refinement_test.jpg"
):
    """
    Test refinement on a single frame.

    Args:
        frame_path: Path to test frame
        keypoints_path: Path to initial keypoints (.npy)
        confidences_path: Path to confidence scores (.npy)
        output_path: Path to save visualization
    """
    print("\n" + "=" * 60)
    print("COURT KEYPOINT REFINEMENT TEST")
    print("=" * 60)

    # Check files exist
    if not Path(frame_path).exists():
        print(f"❌ Test frame not found: {frame_path}")
        print("   Use --extract-from-video to generate test data")
        return False

    if not Path(keypoints_path).exists():
        print(f"❌ Keypoints not found: {keypoints_path}")
        print("   Use --extract-from-video to generate test data")
        return False

    if not Path(confidences_path).exists():
        print(f"❌ Confidences not found: {confidences_path}")
        print("   Use --extract-from-video to generate test data")
        return False

    # Load data
    frame = cv2.imread(frame_path)
    initial_keypoints = np.load(keypoints_path)
    confidences = np.load(confidences_path)

    print(f"\n📂 Loaded test data:")
    print(f"   Frame: {frame.shape}")
    print(f"   Keypoints: {initial_keypoints.shape}")
    print(f"   Confidences: {confidences.shape}")

    # Initialize refinement
    print(f"\n🔧 Initializing refinement module...")
    refinement = CourtKeypointRefinement(
        court_template=COURT_TEMPLATE_KEYPOINTS,
        crop_size=50,
        max_local_shift=15.0,
        max_regularization_shift=10.0,
        ransac_threshold=5.0
    )

    # Apply refinement
    print(f"\n🎯 Applying 3-stage refinement...")
    refined_keypoints, stats = refinement.refine(
        initial_keypoints, frame, confidences
    )

    # Print statistics
    print("\n" + "=" * 60)
    print("REFINEMENT STATISTICS")
    print("=" * 60)

    for stage, shifts in stats['shifts'].items():
        if shifts is not None and len(shifts) > 0:
            print(f"\n{stage.upper()} stage:")
            print(f"  Mean shift: {shifts.mean():.2f} px")
            print(f"  Max shift: {shifts.max():.2f} px")
            print(f"  Median shift: {np.median(shifts):.2f} px")
            print(f"  Keypoints moved: {np.sum(shifts > 0.5)}/14")

    # Compute overall improvement
    total_shift = np.linalg.norm(refined_keypoints - initial_keypoints, axis=1)
    print(f"\nOVERALL REFINEMENT:")
    print(f"  Mean shift: {total_shift.mean():.2f} px")
    print(f"  Max shift: {total_shift.max():.2f} px")
    print(f"  Keypoints refined: {np.sum(total_shift > 0.5)}/14")

    # Visualize results
    print(f"\n🎨 Creating visualization...")
    vis_frame = frame.copy()

    # Draw initial keypoints (red circles)
    for i, kp in enumerate(initial_keypoints):
        cv2.circle(vis_frame, tuple(kp.astype(int)), 8, (0, 0, 255), -1)
        cv2.putText(
            vis_frame, str(i),
            tuple((kp + [10, 10]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
        )

    # Draw refined keypoints (green circles with outline)
    for i, kp in enumerate(refined_keypoints):
        cv2.circle(vis_frame, tuple(kp.astype(int)), 6, (0, 255, 0), 2)

    # Draw shift vectors
    for i in range(14):
        pt1 = tuple(initial_keypoints[i].astype(int))
        pt2 = tuple(refined_keypoints[i].astype(int))
        cv2.arrowedLine(vis_frame, pt1, pt2, (255, 255, 0), 1, tipLength=0.3)

    # Add legend
    legend_y = 30
    cv2.putText(vis_frame, "Red = Initial keypoints", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(vis_frame, "Green = Refined keypoints", (10, legend_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(vis_frame, "Yellow = Shift vectors", (10, legend_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Save visualization
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), vis_frame)

    print(f"\n✅ Test completed successfully!")
    print(f"   Visualization saved: {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test court keypoint geometric refinement"
    )
    parser.add_argument(
        '--extract-from-video',
        type=str,
        help='Extract test data from video (e.g., tests/video3.mp4)'
    )
    parser.add_argument(
        '--frame',
        type=str,
        default='tests/test_court_frame.jpg',
        help='Path to test frame'
    )
    parser.add_argument(
        '--keypoints',
        type=str,
        default='tests/test_keypoints_initial.npy',
        help='Path to initial keypoints'
    )
    parser.add_argument(
        '--confidences',
        type=str,
        default='tests/test_confidences.npy',
        help='Path to confidence scores'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/refinement_test.jpg',
        help='Path to save visualization'
    )

    args = parser.parse_args()

    # Extract test data if requested
    if args.extract_from_video:
        success = extract_test_data_from_video(args.extract_from_video)
        if not success:
            return 1

    # Run test
    success = test_refinement_on_frame(
        frame_path=args.frame,
        keypoints_path=args.keypoints,
        confidences_path=args.confidences,
        output_path=args.output
    )

    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
