"""
Pipeline - Composable execution of steps

File: app/core/pipeline.py
"""
from typing import List, Optional
from app.core.context import ProcessingContext
from app.steps.base import PipelineStep
import time


class Pipeline:
    """
    Composable pipeline of steps.

    Can contain individual steps OR other pipelines (nested composition).

    Key Features:
    - Sequential execution of steps
    - Automatic timing and logging
    - Error propagation
    - Nested pipeline support (Composite Pattern)

    Usage:
        # Simple pipeline
        pipeline = Pipeline(
            name="DetectionPipeline",
            steps=[
                CourtDetectionStep(config),
                BallDetectionStep(config),
                PlayerDetectionStep(config),
            ]
        )

        # Nested pipeline
        main_pipeline = Pipeline(
            name="MainPipeline",
            steps=[
                PreprocessingPipeline(config),  # ← Sub-pipeline!
                DetectionPipeline(config),       # ← Sub-pipeline!
                TemporalPipeline(config),        # ← Sub-pipeline!
            ]
        )

        # Execute
        result = pipeline.run(context)
    """

    def __init__(self, name: str, steps: List[PipelineStep]):
        """
        Initialize pipeline

        Args:
            name: Pipeline name (for logging)
            steps: List of PipelineStep objects (or other Pipelines!)
        """
        self.name = name
        self.steps = steps

    def run(self, context: ProcessingContext) -> ProcessingContext:
        """
        Execute all steps sequentially

        Args:
            context: ProcessingContext to process

        Returns:
            Updated ProcessingContext

        Raises:
            Exception: If any step fails
        """
        print(f"\n{'='*60}")
        print(f"Starting Pipeline: {self.name}")
        print(f"{'='*60}")

        pipeline_start = time.time()

        for i, step in enumerate(self.steps, 1):
            print(f"\n[{self.name}] Step {i}/{len(self.steps)}: {step.name if hasattr(step, 'name') else step.__class__.__name__}")

            # Execute step (or nested pipeline)
            context = step(context)

        pipeline_duration = time.time() - pipeline_start

        print(f"\n{'='*60}")
        print(f"[{self.name}] ✓ Complete ({pipeline_duration:.2f}s)")
        print(f"{'='*60}\n")

        return context

    def __call__(self, context: ProcessingContext) -> ProcessingContext:
        """Make pipeline callable like a step"""
        return self.run(context)

    def add_step(self, step: PipelineStep, position: Optional[int] = None):
        """
        Add a step to the pipeline

        Args:
            step: PipelineStep to add
            position: Insert position (None = append at end)
        """
        if position is None:
            self.steps.append(step)
        else:
            self.steps.insert(position, step)

    def remove_step(self, step_name: str):
        """
        Remove a step by name

        Args:
            step_name: Name of step to remove (class name)
        """
        self.steps = [
            s for s in self.steps
            if (hasattr(s, 'name') and s.name != step_name)
        ]

    def get_step(self, step_name: str) -> Optional[PipelineStep]:
        """
        Get a step by name

        Args:
            step_name: Name of step to retrieve

        Returns:
            PipelineStep or None if not found
        """
        for step in self.steps:
            if hasattr(step, 'name') and step.name == step_name:
                return step
        return None

    def __repr__(self) -> str:
        """String representation"""
        step_names = [
            s.name if hasattr(s, 'name') else s.__class__.__name__
            for s in self.steps
        ]
        return f"<Pipeline '{self.name}' with {len(self.steps)} steps: {', '.join(step_names)}>"


class AsyncPipeline(Pipeline):
    """
    Asynchronous pipeline for parallel execution.

    NOT IMPLEMENTED YET - Placeholder for future parallel execution.

    For Phase 1, use sequential Pipeline. We'll add parallel execution
    in Phase 2 using the ParallelExecutor pattern.

    Usage:
        # Future:
        pipeline = AsyncPipeline(name="ParallelPipeline", steps=[...])
        result = await pipeline.run_async(context)
    """

    async def run_async(self, context: ProcessingContext) -> ProcessingContext:
        """
        Execute steps asynchronously (NOT IMPLEMENTED YET)

        For now, fall back to synchronous execution.
        """
        print(f"[WARNING] AsyncPipeline not implemented yet, using sequential execution")
        return self.run(context)

    def __repr__(self) -> str:
        """String representation"""
        return f"<AsyncPipeline '{self.name}' (not implemented, uses sequential)>"
