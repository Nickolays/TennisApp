"""
In/Out Decision Step - Determine if ball is in or out using geometry

File: app/steps/events/inout_decision.py
"""
import numpy as np
from typing import List, Tuple, Optional

from app.steps.base import PipelineStep
from app.core.context import ProcessingContext
from app.core.data_models import BallState, COURT_DIMENSIONS


class InOutDecisionStep(PipelineStep):
    """
    Determine if ball bounces are inside or outside court boundaries.

    Geometry-Based Method (Simple, No ML):
    1. Get ball position in court coordinates (x, y in meters)
    2. Check if position is within court boundaries
    3. Apply margin for line thickness (ball can touch line and be "in")
    4. Mark bounce as in/out

    Tennis Rules:
    - Ball is "in" if any part touches the line
    - Need to check at bounce frame (when ball contacts court)
    - Different boundaries for singles vs doubles

    Configuration:
        events:
          inout:
            enabled: true
            line_margin: 0.02          # Line thickness margin (meters)
            court_type: 'auto'         # 'singles', 'doubles', or 'auto'
            check_only_bounces: true   # Only check at bounce frames
    """

    def __init__(self, config: dict):
        """
        Initialize in/out decision step

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        self.line_margin = config.get('line_margin', 0.02)
        self.court_type = config.get('court_type', 'auto')
        self.check_only_bounces = config.get('check_only_bounces', True)

        # Court dimensions (from COURT_DIMENSIONS)
        self.court_width_doubles = COURT_DIMENSIONS['width']  # 10.97m
        self.court_width_singles = COURT_DIMENSIONS['singles_width']  # 8.23m
        self.court_length = COURT_DIMENSIONS['length']  # 23.77m

    def _is_inside_court(
        self,
        position_court: Tuple[float, float],
        court_type: str = 'doubles'
    ) -> Tuple[bool, str]:
        """
        Check if position is inside court boundaries

        Args:
            position_court: Position in court coordinates (x, y) meters
            court_type: 'singles' or 'doubles'

        Returns:
            (is_inside, reason) - True if inside, reason string
        """
        x, y = position_court

        # Determine court width
        if court_type == 'singles':
            court_width = self.court_width_singles
            # Singles court is centered on doubles court
            # Singles sidelines are at x = 1.37m and x = 9.60m
            # (doubles goes from 0 to 10.97m)
            x_min = (self.court_width_doubles - self.court_width_singles) / 2
            x_max = x_min + self.court_width_singles
        else:  # doubles
            court_width = self.court_width_doubles
            x_min = 0.0
            x_max = court_width

        # Apply line margin (ball can touch line and be "in")
        x_min -= self.line_margin
        x_max += self.line_margin
        y_min = -self.line_margin
        y_max = self.court_length + self.line_margin

        # Check boundaries
        if x < x_min:
            return False, f"out_left (x={x:.2f} < {x_min:.2f})"
        if x > x_max:
            return False, f"out_right (x={x:.2f} > {x_max:.2f})"
        if y < y_min:
            return False, f"out_back (y={y:.2f} < {y_min:.2f})"
        if y > y_max:
            return False, f"out_front (y={y:.2f} > {y_max:.2f})"

        return True, "in"

    def _detect_court_type(
        self,
        ball_states: List[BallState]
    ) -> str:
        """
        Auto-detect court type (singles vs doubles) from ball positions

        Simple heuristic:
        - If most ball positions are within singles width → singles
        - Otherwise → doubles

        Args:
            ball_states: List of ball states

        Returns:
            'singles' or 'doubles'
        """
        if not ball_states:
            return 'doubles'  # Default

        singles_count = 0
        total_count = 0

        # Singles sidelines on doubles court
        singles_x_min = (self.court_width_doubles - self.court_width_singles) / 2
        singles_x_max = singles_x_min + self.court_width_singles

        for state in ball_states:
            if state.position_court is None:
                continue

            x = state.position_court[0]
            total_count += 1

            # Check if within singles boundaries
            if singles_x_min <= x <= singles_x_max:
                singles_count += 1

        if total_count == 0:
            return 'doubles'

        # If >80% of balls are within singles width, likely singles match
        singles_ratio = singles_count / total_count
        return 'singles' if singles_ratio > 0.8 else 'doubles'

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Determine if ball bounces are in or out of court

        Updates context with in/out decisions (stores in context.inout_decisions)

        Args:
            context: Processing context with ball_states

        Returns:
            Updated context
        """
        if not hasattr(context, 'ball_states') or not context.ball_states:
            print("  No ball states to process")
            return context

        ball_states = context.ball_states

        # Determine court type
        if self.court_type == 'auto':
            detected_court_type = self._detect_court_type(ball_states)
            print(f"  Auto-detected court type: {detected_court_type}")
        else:
            detected_court_type = self.court_type

        # Check each ball state
        in_count = 0
        out_count = 0
        inout_decisions = {}  # frame_id -> (is_in, reason)

        for state in ball_states:
            # Only check bounces if configured
            if self.check_only_bounces and not state.is_bounce:
                continue

            # Skip if no court position
            if state.position_court is None:
                continue

            # Check in/out
            is_inside, reason = self._is_inside_court(
                state.position_court,
                detected_court_type
            )

            # Store decision
            inout_decisions[state.frame_id] = (is_inside, reason)

            if is_inside:
                in_count += 1
            else:
                out_count += 1

        # Store in context
        context.inout_decisions = inout_decisions

        # Summary
        print(f"  In/Out decisions: {in_count} in, {out_count} out")

        if out_count > 0:
            # Show first few out balls
            out_frames = [fid for fid, (is_in, _) in inout_decisions.items() if not is_in]
            print(f"  Out frames: {out_frames[:5]}" + (" ..." if len(out_frames) > 5 else ""))

        return context
