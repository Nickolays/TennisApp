"""
Test new pipeline architecture (Phase 1)

File: tests/test_new_pipeline.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2

from app.core.context import ProcessingContext
from app.core.pipeline import Pipeline
from app.steps.detection.court_detection import CourtDetectionStep
from app.steps.detection.ball_detection import BallDetectionStep


def test_pipeline_step_base():
    """Test that PipelineStep base class works"""
    from app.steps.base import PipelineStep

    class DummyStep(PipelineStep):
        def process(self, context):
            print("  DummyStep processing...")
            context.test_value = 42
            return context

    # Create step
    step = DummyStep(config={'enabled': True})

    # Create context
    context = ProcessingContext(
        video_path="test.mp4",
        fps=30,
        total_frames=100,
        width=1920,
        height=1080
    )

    # Execute
    result = step(context)

    # Verify
    assert result.test_value == 42
    assert 'DummyStep' in result.step_timings
    print("✓ PipelineStep base class works!")


def test_processing_context():
    """Test ProcessingContext dataclass"""
    from app.core.context import ProcessingContext

    context = ProcessingContext(
        video_path="test.mp4",
        fps=30,
        total_frames=1000,
        width=1920,
        height=1080
    )

    # Test basic attributes
    assert context.fps == 30
    assert context.total_frames == 1000

    # Test homography cache
    H = np.eye(3)
    context.homography_cache[0] = H
    context.homography_cache[30] = H

    # Get exact frame
    assert np.array_equal(context.get_homography_for_frame(0), H)

    # Get nearest frame (should use frame 30)
    assert np.array_equal(context.get_homography_for_frame(25), H)

    print("✓ ProcessingContext works!")


def test_pipeline_composition():
    """Test Pipeline composition (nested pipelines)"""
    from app.core.pipeline import Pipeline
    from app.steps.base import PipelineStep

    class Step1(PipelineStep):
        def process(self, context):
            context.values = [1]
            return context

    class Step2(PipelineStep):
        def process(self, context):
            context.values.append(2)
            return context

    class Step3(PipelineStep):
        def process(self, context):
            context.values.append(3)
            return context

    # Create sub-pipeline
    sub_pipeline = Pipeline(
        name="SubPipeline",
        steps=[Step1({'enabled': True}), Step2({'enabled': True})]
    )

    # Create main pipeline with sub-pipeline
    main_pipeline = Pipeline(
        name="MainPipeline",
        steps=[sub_pipeline, Step3({'enabled': True})]
    )

    # Execute
    context = ProcessingContext(
        video_path="test.mp4",
        fps=30,
        total_frames=100,
        width=1920,
        height=1080
    )

    result = main_pipeline.run(context)

    # Verify
    assert result.values == [1, 2, 3]
    print("✓ Pipeline composition (nested) works!")


def test_detection_pipeline_mock():
    """Test detection pipeline with mock data (no actual model)"""
    from app.core.pipeline import Pipeline
    from app.steps.base import PipelineStep

    class MockCourtDetectionStep(PipelineStep):
        """Mock court detection for testing"""
        def process(self, context):
            from app.core.data_models import FrameDetection

            # Initialize detections
            context.detections = [
                FrameDetection(frame_id=i, timestamp=i/context.fps)
                for i in range(len(context.frames))
            ]

            # Add mock court keypoints
            for det in context.detections:
                det.court_keypoints = np.random.rand(14, 2) * [context.width, context.height]
                det.court_confidence = 0.95

            return context

    class MockBallDetectionStep(PipelineStep):
        """Mock ball detection for testing"""
        def process(self, context):
            # Add mock ball positions
            for det in context.detections:
                if np.random.rand() > 0.2:  # 80% detection rate
                    det.ball_position_px = (
                        np.random.rand() * context.width,
                        np.random.rand() * context.height
                    )
                    det.ball_confidence = 0.85
                else:
                    det.ball_position_px = None
                    det.ball_confidence = 0.0

            return context

    # Create pipeline
    detection_pipeline = Pipeline(
        name="MockDetectionPipeline",
        steps=[
            MockCourtDetectionStep({'enabled': True}),
            MockBallDetectionStep({'enabled': True}),
        ]
    )

    # Create context with dummy frames
    context = ProcessingContext(
        video_path="test.mp4",
        fps=30,
        total_frames=10,
        width=1920,
        height=1080
    )

    # Add dummy frames
    context.frames = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(10)]
    context.frame_ids = list(range(10))

    # Execute
    result = detection_pipeline.run(context)

    # Verify
    assert len(result.detections) == 10
    assert all(det.court_keypoints is not None for det in result.detections)

    ball_detected = sum(1 for det in result.detections if det.ball_position_px is not None)
    print(f"  Ball detected in {ball_detected}/10 frames")
    print("✓ Detection pipeline (mock) works!")


def test_model_registry():
    """Test ModelRegistry (without actual models)"""
    from app.models.model_registry import ModelRegistry

    # Check that model types are registered
    assert "tracknet" in ModelRegistry._model_classes
    assert "yolo" in ModelRegistry._model_classes

    print("✓ ModelRegistry is set up correctly!")


def run_all_tests():
    """Run all Phase 1 tests"""
    print("\n" + "="*60)
    print("TESTING PHASE 1: New Pipeline Architecture")
    print("="*60 + "\n")

    tests = [
        ("PipelineStep Base Class", test_pipeline_step_base),
        ("ProcessingContext", test_processing_context),
        ("Pipeline Composition (Nested)", test_pipeline_composition),
        ("Detection Pipeline (Mock)", test_detection_pipeline_mock),
        ("ModelRegistry", test_model_registry),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        print("-" * 60)
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)

    if failed == 0:
        print("\n🎉 All tests passed! Phase 1 infrastructure is ready.\n")
    else:
        print("\n❌ Some tests failed. Please fix before proceeding.\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
