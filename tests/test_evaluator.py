"""Tests for evaluator ingest and reward model."""

import numpy as np
import pytest

from agent.evaluator.ingest import EvaluatorIngest, RUBRIC_WEIGHTS
from agent.evaluator.reward_model import (
    RewardModel,
    SelfImitationBuffer,
    build_preferences,
)


class TestEvaluatorIngest:
    def test_evaluate_segment_without_llm(self):
        """Without an LLM client, judge returns default 0.5 scores."""
        evaluator = EvaluatorIngest(llm_client=None, s3_client=None)
        segment = {"segment_id": "test-001", "total_reward": 10.0}
        result = evaluator.evaluate_segment(segment)

        assert result["segment_id"] == "test-001"
        assert "scores" in result
        assert "weighted_score" in result
        assert "rationale" in result
        # Default scores are all 0.5
        for key in ["progress", "dialog", "puzzle", "novelty", "efficiency"]:
            assert result["scores"][key] == 0.5

    def test_weighted_score_calculation(self):
        evaluator = EvaluatorIngest(llm_client=None, consistency_m=1)
        segment = {"segment_id": "test-002"}
        result = evaluator.evaluate_segment(segment)
        # All scores 0.5 → weighted = 0.5
        assert abs(result["weighted_score"] - 0.5) < 0.01

    def test_self_consistency_m3(self):
        evaluator = EvaluatorIngest(llm_client=None, consistency_m=3)
        segment = {"segment_id": "test-003"}
        result = evaluator.evaluate_segment(segment)
        # With no LLM, all 3 trials produce identical scores
        assert len(result["raw_trials"]) == 3


class TestRewardModel:
    def test_predict_returns_scalar(self):
        rm = RewardModel(obs_dim=128)
        obs = np.random.randn(128).astype(np.float32)
        val = rm.predict(obs)
        assert isinstance(val, float)

    def test_train_on_preferences(self):
        rm = RewardModel(obs_dim=128)
        prefs = []
        for _ in range(10):
            a = np.random.randn(128).astype(np.float32)
            b = np.random.randn(128).astype(np.float32)
            prefs.append((a, b, 1.0))
        loss = rm.train_on_preferences(prefs)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_build_preferences(self):
        scores = [
            {"weighted_score": 0.8, "scores": {"progress": 0.9, "dialog": 0.7, "puzzle": 0.8, "novelty": 0.6, "efficiency": 0.7}},
            {"weighted_score": 0.3, "scores": {"progress": 0.2, "dialog": 0.4, "puzzle": 0.3, "novelty": 0.3, "efficiency": 0.3}},
            {"weighted_score": 0.5, "scores": {"progress": 0.5, "dialog": 0.5, "puzzle": 0.5, "novelty": 0.5, "efficiency": 0.5}},
        ]
        prefs = build_preferences(scores)
        assert len(prefs) > 0
        # Each preference is (obs_a, obs_b, label)
        for obs_a, obs_b, label in prefs:
            assert obs_a.shape == (128,)
            assert obs_b.shape == (128,)
            assert label in (0.0, 1.0)


class TestSelfImitationBuffer:
    def test_add_and_sample(self):
        buf = SelfImitationBuffer(capacity=5)
        for i in range(10):
            buf.add({"segment_id": f"s{i}", "weighted_score": float(i)})
        assert len(buf) == 5
        samples = buf.sample(3)
        assert len(samples) == 3
        # Should have highest scores
        assert samples[0]["weighted_score"] == 9.0

    def test_min_score(self):
        buf = SelfImitationBuffer(capacity=3)
        buf.add({"weighted_score": 1.0})
        buf.add({"weighted_score": 5.0})
        buf.add({"weighted_score": 3.0})
        assert buf.min_score == 1.0
