"""Tests for config loading."""

import pytest

from agent.utils.config import AgentConfig, load_config


class TestConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.emulator.headless is True
        assert cfg.observation.vector_size == 128
        assert cfg.training.lr == 3e-4
        assert cfg.rewards.death == -50.0

    def test_load_nonexistent(self):
        cfg = load_config("/nonexistent/path.yaml")
        assert cfg.emulator.headless is True

    def test_load_none(self):
        cfg = load_config(None)
        assert isinstance(cfg, AgentConfig)
