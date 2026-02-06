"""Macro executor: translates high-level macros into frame-level button presses.

Supported macros (from new/SCHEMAS.md):
  MOVE_TO(x, y)       — navigate toward target pixel coords
  PUSH_BLOCK(dir, n)  — push a block N steps in a direction
  USE_ITEM(item)       — equip and use an item (A or B button)

Each macro has a timeout to avoid getting stuck.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


class MacroStatus(IntEnum):
    RUNNING = 0
    COMPLETED = 1
    TIMEOUT = 2
    FAILED = 3


@dataclass
class MacroState:
    """Tracks execution of a single macro."""

    name: str
    args: dict[str, Any]
    timeout_s: float = 1.5
    start_time: float = field(default_factory=time.monotonic)
    steps_executed: int = 0
    status: MacroStatus = MacroStatus.RUNNING

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    @property
    def timed_out(self) -> bool:
        return self.elapsed > self.timeout_s


# Direction to action mapping
_DIR_TO_ACTION = {
    "up": 1,
    "down": 2,
    "left": 3,
    "right": 4,
    "UP": 1,
    "DOWN": 2,
    "LEFT": 3,
    "RIGHT": 4,
}


class MacroExecutor:
    """Executes a sequence of macros, yielding one action per step call."""

    def __init__(self):
        self._queue: list[MacroState] = []
        self._current: MacroState | None = None

    def enqueue(self, macros: list[dict[str, Any]]) -> None:
        """Enqueue a list of macro dicts from the planner.

        Each macro: {"name": "MOVE_TO", "args": {...}, "timeout_s": 1.5}
        """
        for m in macros:
            self._queue.append(
                MacroState(
                    name=m["name"],
                    args=m.get("args", {}),
                    timeout_s=m.get("timeout_s", 1.5),
                )
            )

    def step(self, pixel_x: int, pixel_y: int) -> tuple[int, MacroStatus]:
        """Get next action for this frame.

        Args:
            pixel_x: Current player X pixel position.
            pixel_y: Current player Y pixel position.

        Returns:
            (action_int, status) where status indicates macro progress.
        """
        # Advance to next macro if needed
        if self._current is None or self._current.status != MacroStatus.RUNNING:
            if self._queue:
                self._current = self._queue.pop(0)
            else:
                return 0, MacroStatus.COMPLETED  # NOP, nothing to do

        macro = self._current

        # Timeout check
        if macro.timed_out:
            macro.status = MacroStatus.TIMEOUT
            logger.warning("Macro %s timed out after %.1fs", macro.name, macro.elapsed)
            return 0, MacroStatus.TIMEOUT

        macro.steps_executed += 1
        action = self._execute_step(macro, pixel_x, pixel_y)
        return action, macro.status

    def _execute_step(self, macro: MacroState, px: int, py: int) -> int:
        """Dispatch to the appropriate macro handler."""
        if macro.name == "MOVE_TO":
            return self._move_to(macro, px, py)
        elif macro.name == "PUSH_BLOCK":
            return self._push_block(macro, px, py)
        elif macro.name == "USE_ITEM":
            return self._use_item(macro)
        else:
            logger.warning("Unknown macro: %s", macro.name)
            macro.status = MacroStatus.FAILED
            return 0

    def _move_to(self, macro: MacroState, px: int, py: int) -> int:
        """Navigate toward target (x, y) pixel coords."""
        tx = macro.args.get("x", px)
        ty = macro.args.get("y", py)
        dx = tx - px
        dy = ty - py
        threshold = 4  # pixels

        if abs(dx) <= threshold and abs(dy) <= threshold:
            macro.status = MacroStatus.COMPLETED
            return 0  # arrived

        # Move along the axis with larger delta
        if abs(dx) > abs(dy):
            return 4 if dx > 0 else 3  # RIGHT or LEFT
        else:
            return 2 if dy > 0 else 1  # DOWN or UP

    def _push_block(self, macro: MacroState, px: int, py: int) -> int:
        """Push a block: face direction, then A button alternate."""
        direction = macro.args.get("dir", "RIGHT")
        steps = macro.args.get("steps", 1)
        total_needed = steps * 2  # face + push per step

        if macro.steps_executed > total_needed:
            macro.status = MacroStatus.COMPLETED
            return 0

        # Alternate: move in direction, then press A
        if macro.steps_executed % 2 == 1:
            return _DIR_TO_ACTION.get(direction, 4)
        else:
            return _DIR_TO_ACTION.get(direction, 4)  # keep pressing direction to push

    def _use_item(self, macro: MacroState) -> int:
        """Use currently equipped item (A or B button)."""
        button = macro.args.get("button", "A")
        if macro.steps_executed >= 2:
            macro.status = MacroStatus.COMPLETED
            return 0
        return 5 if button.upper() == "A" else 6  # A or B action

    @property
    def is_active(self) -> bool:
        return bool(self._queue) or (
            self._current is not None and self._current.status == MacroStatus.RUNNING
        )

    def clear(self) -> None:
        self._queue.clear()
        self._current = None
