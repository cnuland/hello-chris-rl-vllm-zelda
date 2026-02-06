"""Tests for metrics helpers."""

import pytest

from agent.utils.metrics import ExplorationMetrics, LLMCallMetrics, Timer


class TestExplorationMetrics:
    def test_record_new_tile(self):
        m = ExplorationMetrics()
        is_new = m.record_position(1, 5, 3)
        assert is_new is True
        assert m.num_unique_rooms == 1
        assert m.total_unique_tiles == 1

    def test_record_duplicate_tile(self):
        m = ExplorationMetrics()
        m.record_position(1, 5, 3)
        is_new = m.record_position(1, 5, 3)
        assert is_new is False
        assert m.total_unique_tiles == 1

    def test_multiple_rooms(self):
        m = ExplorationMetrics()
        m.record_position(1, 0, 0)
        m.record_position(2, 0, 0)
        m.record_position(3, 0, 0)
        assert m.num_unique_rooms == 3

    def test_doorway_pingpong(self):
        m = ExplorationMetrics()
        m.record_position(1, 0, 0)
        m.record_position(2, 0, 0)
        m.record_position(1, 0, 0)
        m.record_position(2, 0, 0)
        assert m.doorway_pingpong == 1


class TestLLMCallMetrics:
    def test_record_call(self):
        m = LLMCallMetrics()
        m.record_call(100.0, tokens=50, cache_hit=True)
        assert m.call_count == 1
        assert m.avg_latency_ms == 100.0
        assert m.cache_hit_rate == 1.0

    def test_avg_latency(self):
        m = LLMCallMetrics()
        m.record_call(100.0)
        m.record_call(200.0)
        assert m.avg_latency_ms == 150.0

    def test_empty_metrics(self):
        m = LLMCallMetrics()
        assert m.avg_latency_ms == 0.0
        assert m.cache_hit_rate == 0.0


class TestTimer:
    def test_timer_measures(self):
        with Timer() as t:
            pass  # nearly instant
        assert t.elapsed_ms >= 0.0
