"""
Test new pipeline structure (no dependencies)

File: tests/test_structure_only.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all new modules can be imported"""
    print("[TEST] Testing imports...")

    # Core modules
    from app.core.context import ProcessingContext
    print("  ✓ ProcessingContext imported")

    from app.core.pipeline import Pipeline, AsyncPipeline
    print("  ✓ Pipeline, AsyncPipeline imported")

    from app.steps.base import PipelineStep
    print("  ✓ PipelineStep imported")

    from app.models.model_registry import ModelRegistry
    print("  ✓ ModelRegistry imported")

    # Detection steps
    from app.steps.detection.court_detection import CourtDetectionStep
    print("  ✓ CourtDetectionStep imported")

    from app.steps.detection.ball_detection import BallDetectionStep
    print("  ✓ BallDetectionStep imported")

    print("✓ All imports successful!")
    return True


def test_pipeline_step_instantiation():
    """Test that PipelineStep can be instantiated"""
    print("\n[TEST] Testing PipelineStep instantiation...")

    from app.steps.base import PipelineStep

    class DummyStep(PipelineStep):
        def process(self, context):
            return context

    step = DummyStep(config={'enabled': True})
    print(f"  Step name: {step.name}")
    print(f"  Step enabled: {step.enabled}")
    print(f"  Step repr: {step}")

    print("✓ PipelineStep can be instantiated!")
    return True


def test_pipeline_instantiation():
    """Test that Pipeline can be instantiated"""
    print("\n[TEST] Testing Pipeline instantiation...")

    from app.core.pipeline import Pipeline
    from app.steps.base import PipelineStep

    class Step1(PipelineStep):
        def process(self, context):
            return context

    class Step2(PipelineStep):
        def process(self, context):
            return context

    pipeline = Pipeline(
        name="TestPipeline",
        steps=[
            Step1({'enabled': True}),
            Step2({'enabled': True}),
        ]
    )

    print(f"  Pipeline name: {pipeline.name}")
    print(f"  Pipeline steps: {len(pipeline.steps)}")
    print(f"  Pipeline repr: {pipeline}")

    print("✓ Pipeline can be instantiated!")
    return True


def test_model_registry_setup():
    """Test that ModelRegistry is set up"""
    print("\n[TEST] Testing ModelRegistry setup...")

    from app.models.model_registry import ModelRegistry

    print(f"  Registered model types: {list(ModelRegistry._model_classes.keys())}")

    assert "tracknet" in ModelRegistry._model_classes
    assert "yolo" in ModelRegistry._model_classes

    print("✓ ModelRegistry is set up!")
    return True


def test_detection_steps_instantiation():
    """Test that detection steps can be instantiated"""
    print("\n[TEST] Testing detection steps instantiation...")

    from app.steps.detection.court_detection import CourtDetectionStep
    from app.steps.detection.ball_detection import BallDetectionStep

    court_config = {
        'enabled': True,
        'model_path': 'models/court_model_best.pt',
        'model_type': 'tracknet',
        'interval': 30
    }

    ball_config = {
        'enabled': True,
        'model_path': 'models/ball_model_best.pt',
        'model_type': 'tracknet',
        'interval': 1
    }

    court_step = CourtDetectionStep(court_config)
    print(f"  CourtDetectionStep: {court_step}")

    ball_step = BallDetectionStep(ball_config)
    print(f"  BallDetectionStep: {ball_step}")

    print("✓ Detection steps can be instantiated!")
    return True


def test_directory_structure():
    """Test that directory structure is correct"""
    print("\n[TEST] Testing directory structure...")

    project_root = Path(__file__).parent.parent

    required_dirs = [
        "app/core",
        "app/steps",
        "app/steps/preprocessing",
        "app/steps/detection",
        "app/steps/temporal",
        "app/steps/geometry",
        "app/steps/events",
        "app/steps/analytics",
        "app/steps/rendering",
        "app/pipelines",
        "app/services",
        "configs",
    ]

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (missing)")
            return False

    print("✓ Directory structure is correct!")
    return True


def run_all_tests():
    """Run all structure tests"""
    print("\n" + "="*60)
    print("TESTING PHASE 1: New Pipeline Architecture (Structure Only)")
    print("="*60)

    tests = [
        ("Imports", test_imports),
        ("PipelineStep Instantiation", test_pipeline_step_instantiation),
        ("Pipeline Instantiation", test_pipeline_instantiation),
        ("ModelRegistry Setup", test_model_registry_setup),
        ("Detection Steps Instantiation", test_detection_steps_instantiation),
        ("Directory Structure", test_directory_structure),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)

    if failed == 0:
        print("\n🎉 All structure tests passed!")
        print("\nPhase 1 Complete:")
        print("✓ Directory structure created")
        print("✓ PipelineStep base class implemented")
        print("✓ ProcessingContext dataclass created")
        print("✓ Pipeline & AsyncPipeline implemented")
        print("✓ ModelRegistry created")
        print("✓ CourtDetectionStep migrated")
        print("✓ BallDetectionStep migrated")
        print("\nNext Steps:")
        print("- Install dependencies: pip install numpy opencv-python torch")
        print("- Run full tests: python tests/test_new_pipeline.py")
        print("- Migrate more steps (Temporal, Geometry, Events, etc.)")
        print()
    else:
        print("\n❌ Some tests failed. Please fix before proceeding.\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
