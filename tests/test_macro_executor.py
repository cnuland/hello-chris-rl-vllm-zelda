"""Tests for macro executor."""

import pytest

from agent.planner.macro_executor import MacroExecutor, MacroStatus


class TestMacroExecutor:
    def test_move_to_reaches_target(self):
        ex = MacroExecutor()
        ex.enqueue([{"name": "MOVE_TO", "args": {"x": 100, "y": 72}, "timeout_s": 5.0}])

        # At (80, 72), target (100, 72) — should move right
        action, status = ex.step(pixel_x=80, pixel_y=72)
        assert action == 4  # RIGHT
        assert status == MacroStatus.RUNNING

    def test_move_to_completes_at_target(self):
        ex = MacroExecutor()
        ex.enqueue([{"name": "MOVE_TO", "args": {"x": 80, "y": 72}, "timeout_s": 5.0}])

        # Already at target
        action, status = ex.step(pixel_x=80, pixel_y=72)
        assert action == 0  # NOP (arrived)
        assert status == MacroStatus.COMPLETED

    def test_use_item_a(self):
        ex = MacroExecutor()
        ex.enqueue([{"name": "USE_ITEM", "args": {"button": "A"}, "timeout_s": 2.0}])

        action, status = ex.step(0, 0)
        assert action == 5  # A button

    def test_use_item_b(self):
        ex = MacroExecutor()
        ex.enqueue([{"name": "USE_ITEM", "args": {"button": "B"}, "timeout_s": 2.0}])

        action, status = ex.step(0, 0)
        assert action == 6  # B button

    def test_multiple_macros_sequential(self):
        ex = MacroExecutor()
        ex.enqueue([
            {"name": "MOVE_TO", "args": {"x": 80, "y": 72}, "timeout_s": 5.0},
            {"name": "USE_ITEM", "args": {"button": "A"}, "timeout_s": 2.0},
        ])

        # First macro completes immediately (at target)
        action1, status1 = ex.step(80, 72)
        assert status1 == MacroStatus.COMPLETED

        # Second macro starts
        action2, status2 = ex.step(80, 72)
        assert action2 == 5  # A button

    def test_empty_executor_returns_nop(self):
        ex = MacroExecutor()
        action, status = ex.step(0, 0)
        assert action == 0
        assert status == MacroStatus.COMPLETED

    def test_is_active(self):
        ex = MacroExecutor()
        assert not ex.is_active
        ex.enqueue([{"name": "MOVE_TO", "args": {"x": 100, "y": 100}, "timeout_s": 5.0}])
        assert ex.is_active

    def test_clear(self):
        ex = MacroExecutor()
        ex.enqueue([{"name": "MOVE_TO", "args": {"x": 100, "y": 100}, "timeout_s": 5.0}])
        ex.step(0, 0)
        ex.clear()
        assert not ex.is_active
