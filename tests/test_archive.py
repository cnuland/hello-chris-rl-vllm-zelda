"""Tests for Go-Explore-lite archive."""

import pytest

from agent.rl.archive import Archive


class TestArchive:
    def test_new_cell(self):
        arch = Archive(max_size=100)
        added = arch.update(
            room_id=1,
            pixel_x=80,
            pixel_y=72,
            save_state=b"state1",
            episode_reward=10.0,
            coverage=5,
        )
        assert added is True
        assert len(arch) == 1

    def test_update_existing_cell(self):
        arch = Archive()
        arch.update(1, 80, 72, b"state1", 10.0, 5)
        # Better reward should update
        improved = arch.update(1, 80, 72, b"state2", 20.0, 5)
        assert improved is True
        # Same reward, same coverage should not improve
        improved = arch.update(1, 80, 72, b"state3", 15.0, 5)
        assert improved is False

    def test_sample_frontier(self):
        arch = Archive()
        arch.update(1, 0, 0, b"s1", 1.0, 10)
        arch.update(1, 40, 0, b"s2", 1.0, 20)
        arch.update(2, 0, 0, b"s3", 1.0, 30)

        frontier = arch.sample_frontier(k=2)
        assert len(frontier) == 2
        # Highest novelty should be first (most coverage / least visits)
        assert frontier[0].total_coverage >= frontier[1].total_coverage

    def test_get_restart_state(self):
        arch = Archive()
        arch.update(1, 0, 0, b"state_data", 5.0, 10)
        state = arch.get_restart_state()
        assert state == b"state_data"

    def test_empty_frontier(self):
        arch = Archive()
        assert arch.get_restart_state() is None
        assert arch.sample_frontier() == []

    def test_eviction(self):
        arch = Archive(max_size=2)
        arch.update(1, 0, 0, b"s1", 1.0, 1)
        arch.update(1, 40, 0, b"s2", 1.0, 10)
        arch.update(2, 0, 0, b"s3", 1.0, 100)
        assert len(arch) == 2  # evicted the least novel

    def test_unique_rooms(self):
        arch = Archive()
        arch.update(1, 0, 0, b"s1", 1.0, 1)
        arch.update(1, 40, 0, b"s2", 1.0, 1)
        arch.update(2, 0, 0, b"s3", 1.0, 1)
        assert arch.unique_rooms == 2

    def test_reset(self):
        arch = Archive()
        arch.update(1, 0, 0, b"s1", 1.0, 1)
        arch.reset()
        assert len(arch) == 0
