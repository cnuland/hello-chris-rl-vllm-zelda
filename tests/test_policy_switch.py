"""Tests for policy switch predicates and hysteresis."""

import time

import pytest

from agent.planner.policy_switch import (
    ControlMode,
    PolicySwitch,
    is_dialog_state,
    is_puzzle_state,
)


class TestPredicates:
    def test_is_dialog_state(self):
        assert is_dialog_state(0) is False
        assert is_dialog_state(1) is True
        assert is_dialog_state(255) is True

    def test_is_puzzle_state(self):
        assert is_puzzle_state(0) is False
        assert is_puzzle_state(1) is True
        assert is_puzzle_state(0xFF) is True


class TestPolicySwitch:
    def test_starts_in_ppo_mode(self):
        ps = PolicySwitch()
        assert ps.mode == ControlMode.PPO

    def test_dialog_hysteresis_requires_consecutive_frames(self):
        ps = PolicySwitch(enter_frames=3)
        # 2 frames not enough
        ps.update(dialog_active=True, puzzle_active=False, hp=3)
        ps.update(dialog_active=True, puzzle_active=False, hp=3)
        assert ps.mode == ControlMode.PPO

        # 3rd frame triggers switch
        ps.update(dialog_active=True, puzzle_active=False, hp=3)
        assert ps.mode == ControlMode.DIALOG

    def test_dialog_clears_returns_to_ppo(self):
        ps = PolicySwitch(enter_frames=1)
        ps.update(dialog_active=True, puzzle_active=False, hp=3)
        assert ps.mode == ControlMode.DIALOG

        ps.update(dialog_active=False, puzzle_active=False, hp=3)
        assert ps.mode == ControlMode.PPO

    def test_puzzle_hysteresis(self):
        ps = PolicySwitch(enter_frames=2)
        ps.update(dialog_active=False, puzzle_active=True, hp=3)
        assert ps.mode == ControlMode.PPO
        ps.update(dialog_active=False, puzzle_active=True, hp=3)
        assert ps.mode == ControlMode.PUZZLE

    def test_hp_drop_returns_to_ppo(self):
        ps = PolicySwitch(enter_frames=1)
        ps.update(dialog_active=True, puzzle_active=False, hp=5)
        assert ps.mode == ControlMode.DIALOG

        ps.update(dialog_active=True, puzzle_active=False, hp=3)
        assert ps.mode == ControlMode.PPO

    def test_confidence_threshold(self):
        ps = PolicySwitch(enter_frames=1, confidence_threshold=0.8)
        ps.update(dialog_active=True, puzzle_active=False, hp=3, confidence=0.3)
        assert ps.mode == ControlMode.PPO  # below threshold

        ps.update(dialog_active=True, puzzle_active=False, hp=3, confidence=0.9)
        assert ps.mode == ControlMode.DIALOG

    def test_reset(self):
        ps = PolicySwitch(enter_frames=1)
        ps.update(dialog_active=True, puzzle_active=False, hp=3)
        assert ps.mode == ControlMode.DIALOG
        ps.reset()
        assert ps.mode == ControlMode.PPO
