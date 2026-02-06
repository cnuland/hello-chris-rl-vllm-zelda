"""Tests for coverage reward and RND curiosity."""

import numpy as np
import pytest

from agent.rl.rewards import CoverageReward, PotentialShaping


class TestCoverageReward:
    def test_new_room_bonus(self):
        cov = CoverageReward()
        r = cov.step(room_id=1, pixel_x=80, pixel_y=72)
        assert r > 0  # new room + new tile
        assert 1 in cov._visited_rooms

    def test_new_tile_bonus(self):
        cov = CoverageReward()
        # First visit
        r1 = cov.step(room_id=1, pixel_x=80, pixel_y=72)
        # Same tile, same room
        r2 = cov.step(room_id=1, pixel_x=80, pixel_y=72)
        # New tile, same room
        r3 = cov.step(room_id=1, pixel_x=40, pixel_y=36)
        assert r1 > r3  # first tile + new room > just new tile
        assert r2 < 0  # revisit penalty
        assert r3 > 0  # new tile bonus

    def test_coverage_increments(self):
        cov = CoverageReward()
        cov.step(1, 0, 0)
        cov.step(1, 40, 0)
        cov.step(2, 0, 0)
        assert cov.unique_rooms == 2
        assert cov.total_tiles >= 2  # at least 2 unique tiles

    def test_reset(self):
        cov = CoverageReward()
        cov.step(1, 80, 72)
        cov.reset()
        assert cov.unique_rooms == 0
        assert cov.total_tiles == 0


class TestPotentialShaping:
    def test_shaping_preserves_zero(self):
        ps = PotentialShaping(gamma=0.99, lam=0.15)
        # With zero potential, shaped reward should be close to extrinsic
        shaped = ps.shape(1.0, 0.0)
        assert abs(shaped - 1.0) < 0.01

    def test_positive_potential_increase(self):
        ps = PotentialShaping(gamma=0.99, lam=0.15)
        # First step with positive potential
        r1 = ps.shape(0.0, 1.0)
        # Potential should add to reward
        assert r1 > 0

    def test_reset(self):
        ps = PotentialShaping()
        ps.shape(1.0, 2.0)
        ps.reset()
        assert ps._prev_potential == 0.0
