"""Tests for coverage reward with decay and potential shaping."""

import numpy as np
import pytest

from agent.rl.rewards import CoverageReward, PotentialShaping


class TestCoverageReward:
    def test_new_room_tracked(self):
        cov = CoverageReward()
        cov.step(room_id=1, pixel_x=80, pixel_y=72)
        assert 1 in cov._visited_rooms
        assert cov.unique_rooms == 1

    def test_new_tile_tracked(self):
        cov = CoverageReward()
        cov.step(room_id=1, pixel_x=80, pixel_y=72)
        assert cov.total_tiles == 1
        # Same tile — no new entry
        cov.step(room_id=1, pixel_x=80, pixel_y=72)
        assert cov.total_tiles == 1
        # New tile
        cov.step(room_id=1, pixel_x=40, pixel_y=36)
        assert cov.total_tiles == 2

    def test_total_value_increases_with_new_tiles(self):
        cov = CoverageReward(exploration_inc=1.0, room_inc=1.0)
        assert cov.total_tile_value() == 0.0
        cov.step(1, 0, 0)
        assert cov.total_tile_value() == 1.0
        cov.step(1, 40, 0)
        assert cov.total_tile_value() == 2.0

    def test_room_value(self):
        cov = CoverageReward(room_inc=1.0)
        cov.step(1, 0, 0)
        assert cov.total_room_value() == 1.0
        cov.step(2, 0, 0)
        assert cov.total_room_value() == 2.0

    def test_decay_reduces_values(self):
        cov = CoverageReward(
            exploration_inc=1.0, room_inc=1.0,
            decay_factor=0.5, decay_frequency=1, decay_floor=0.1,
        )
        cov.step(1, 0, 0)
        initial_tile = cov.total_tile_value()
        # Next step triggers decay (freq=1)
        cov.step(1, 0, 0)  # revisit, triggers decay
        assert cov.total_tile_value() < initial_tile

    def test_decay_floor(self):
        cov = CoverageReward(
            exploration_inc=1.0, decay_factor=0.01,
            decay_frequency=1, decay_floor=0.15,
        )
        cov.step(1, 0, 0)
        # Run many steps to fully decay
        for _ in range(100):
            cov.step(1, 0, 0)
        # Value should hit floor, not zero
        assert cov.total_tile_value() >= 0.15

    def test_reset(self):
        cov = CoverageReward()
        cov.step(1, 80, 72)
        cov.reset()
        assert cov.unique_rooms == 0
        assert cov.total_tiles == 0
        assert cov.total_tile_value() == 0.0

    def test_visited_rooms_property(self):
        cov = CoverageReward()
        cov.step(1, 0, 0)
        cov.step(2, 0, 0)
        assert cov._visited_rooms == {1, 2}


class TestPotentialShaping:
    def test_shaping_preserves_zero(self):
        ps = PotentialShaping(gamma=0.99, lam=0.15)
        shaped = ps.shape(1.0, 0.0)
        assert abs(shaped - 1.0) < 0.01

    def test_positive_potential_increase(self):
        ps = PotentialShaping(gamma=0.99, lam=0.15)
        r1 = ps.shape(0.0, 1.0)
        assert r1 > 0

    def test_reset(self):
        ps = PotentialShaping()
        ps.shape(1.0, 2.0)
        ps.reset()
        assert ps._prev_potential == 0.0
