"""Policy switching predicates with hysteresis.

Decides when to hand control from PPO to the LLM planner (dialog or puzzle)
and when to hand it back. Uses frame-count hysteresis to avoid thrashing.

Rules (from new/POLICY_SWITCHING.md):
  Switch to planner when:
    - Dialog detected for >= 3 consecutive frames, OR
    - Puzzle flags active for >= 3 consecutive frames.
  Return to PPO on:
    - Dialog clear, puzzle complete, timeout, HP drop, or fallback.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class ControlMode(Enum):
    PPO = auto()
    DIALOG = auto()
    PUZZLE = auto()


@dataclass
class PolicySwitch:
    """Hysteresis-based policy switch between PPO and LLM planner."""

    # Hysteresis thresholds
    enter_frames: int = 3  # consecutive frames before switching
    confidence_threshold: float = 0.5
    dialog_timeout_s: float = 2.0
    puzzle_timeout_s: float = 6.0

    # Internal state
    mode: ControlMode = ControlMode.PPO
    _dialog_counter: int = 0
    _puzzle_counter: int = 0
    _mode_start_time: float = 0.0
    _entry_hp: int = 0
    _prev_dialog: bool = False
    _prev_puzzle: bool = False

    def update(
        self,
        dialog_active: bool,
        puzzle_active: bool,
        hp: int,
        confidence: float = 1.0,
    ) -> ControlMode:
        """Update mode based on current frame observations.

        Args:
            dialog_active: Whether dialog state is nonzero.
            puzzle_active: Whether puzzle flags are nonzero.
            hp: Current player HP (hearts).
            confidence: Planner confidence (0..1).

        Returns:
            Current control mode after update.
        """
        now = time.monotonic()

        if self.mode == ControlMode.PPO:
            return self._update_ppo(dialog_active, puzzle_active, hp, confidence, now)
        elif self.mode == ControlMode.DIALOG:
            return self._update_dialog(dialog_active, hp, now)
        elif self.mode == ControlMode.PUZZLE:
            return self._update_puzzle(puzzle_active, hp, now)
        return self.mode

    def _update_ppo(
        self,
        dialog: bool,
        puzzle: bool,
        hp: int,
        confidence: float,
        now: float,
    ) -> ControlMode:
        # Dialog hysteresis
        if dialog:
            self._dialog_counter += 1
        else:
            self._dialog_counter = 0

        # Puzzle hysteresis
        if puzzle:
            self._puzzle_counter += 1
        else:
            self._puzzle_counter = 0

        # Check for mode switch
        if self._dialog_counter >= self.enter_frames and confidence >= self.confidence_threshold:
            self.mode = ControlMode.DIALOG
            self._mode_start_time = now
            self._entry_hp = hp
            logger.info("Switch PPO -> DIALOG (counter=%d)", self._dialog_counter)
        elif self._puzzle_counter >= self.enter_frames and confidence >= self.confidence_threshold:
            self.mode = ControlMode.PUZZLE
            self._mode_start_time = now
            self._entry_hp = hp
            logger.info("Switch PPO -> PUZZLE (counter=%d)", self._puzzle_counter)

        return self.mode

    def _update_dialog(self, dialog: bool, hp: int, now: float) -> ControlMode:
        elapsed = now - self._mode_start_time

        # Give back to PPO if:
        if not dialog:
            self.mode = ControlMode.PPO
            self._dialog_counter = 0
            logger.info("DIALOG -> PPO (dialog cleared)")
        elif elapsed > self.dialog_timeout_s:
            self.mode = ControlMode.PPO
            self._dialog_counter = 0
            logger.info("DIALOG -> PPO (timeout %.1fs)", elapsed)
        elif hp < self._entry_hp:
            self.mode = ControlMode.PPO
            self._dialog_counter = 0
            logger.info("DIALOG -> PPO (HP drop %d->%d)", self._entry_hp, hp)

        return self.mode

    def _update_puzzle(self, puzzle: bool, hp: int, now: float) -> ControlMode:
        elapsed = now - self._mode_start_time

        if not puzzle:
            self.mode = ControlMode.PPO
            self._puzzle_counter = 0
            logger.info("PUZZLE -> PPO (puzzle complete)")
        elif elapsed > self.puzzle_timeout_s:
            self.mode = ControlMode.PPO
            self._puzzle_counter = 0
            logger.info("PUZZLE -> PPO (timeout %.1fs)", elapsed)
        elif hp < self._entry_hp:
            self.mode = ControlMode.PPO
            self._puzzle_counter = 0
            logger.info("PUZZLE -> PPO (HP drop %d->%d)", self._entry_hp, hp)

        return self.mode

    def reset(self) -> None:
        self.mode = ControlMode.PPO
        self._dialog_counter = 0
        self._puzzle_counter = 0
        self._mode_start_time = 0.0
        self._entry_hp = 0


def is_dialog_state(dialog_flag: int) -> bool:
    """Simple predicate: dialog state RAM byte is nonzero."""
    return dialog_flag != 0


def is_puzzle_state(puzzle_flags: int) -> bool:
    """Simple predicate: puzzle flags bitfield is nonzero."""
    return puzzle_flags != 0
