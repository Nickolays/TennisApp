"""
Base class for all pipeline steps

File: app/steps/base.py
"""
from abc import ABC, abstractmethod
from typing import Any, Dict
import time


class PipelineStep(ABC):
    """
    Base class for all pipeline steps.

    Design Principles:
    - Single Responsibility: Each step does ONE thing
    - Independently Testable: Can test without full pipeline
    - Chainable: Input/output via ProcessingContext
    - Stateless: All state stored in context, not step instance

    Usage:
        class MyCustomStep(PipelineStep):
            def __init__(self, config: dict):
                super().__init__(config)
                # Initialize your step-specific resources

            def process(self, context: ProcessingContext) -> ProcessingContext:
                # Do your processing
                # Modify context in-place or return new context
                return context
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline step

        Args:
            config: Step-specific configuration dictionary
        """
        self.config = config
        self.name = self.__class__.__name__
        self.enabled = config.get('enabled', True)

    @abstractmethod
    def process(self, context: 'ProcessingContext') -> 'ProcessingContext':
        """
        Process data from context and return updated context.

        This is the main method that each step must implement.

        Args:
            context: ProcessingContext containing all pipeline state

        Returns:
            Updated ProcessingContext (can be same object modified in-place)

        Raises:
            Exception: If processing fails (will be caught by pipeline)
        """
        pass

    def __call__(self, context: 'ProcessingContext') -> 'ProcessingContext':
        """
        Callable interface for pipeline chaining.

        Automatically handles:
        - Timing
        - Logging
        - Error handling (basic)
        - Skipping if disabled

        Args:
            context: ProcessingContext

        Returns:
            Updated ProcessingContext
        """
        if not self.enabled:
            print(f"[{self.name}] ⊘ Skipped (disabled in config)")
            return context

        print(f"[{self.name}] Starting...")
        start_time = time.time()

        try:
            result_context = self.process(context)
            duration = time.time() - start_time

            # Store timing in context
            if hasattr(result_context, 'step_timings'):
                result_context.step_timings[self.name] = duration

            print(f"[{self.name}] ✓ Complete ({duration:.2f}s)")
            return result_context

        except Exception as e:
            duration = time.time() - start_time
            print(f"[{self.name}] ✗ Failed after {duration:.2f}s: {e}")
            raise

    def __repr__(self) -> str:
        """String representation"""
        status = "enabled" if self.enabled else "disabled"
        return f"<{self.name} ({status})>"
