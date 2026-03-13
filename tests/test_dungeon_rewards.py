"""Tests for the dungeon reward tracker (agent/env/dungeon_rewards.py)."""

import pytest

from agent.env.dungeon_rewards import DungeonRewardTracker
from agent.env.ram_addresses import (
    BOSS_KEYS,
    DUNGEON_COMPASS,
    DUNGEON_KEYS,
    DUNGEON_MAP,
    ENEMIES_COUNT,
)


class FakeRAM:
    """Simulates RAM reads for testing."""

    def __init__(self):
        self._data: dict[int, int] = {}

    def write(self, addr: int, val: int) -> None:
        self._data[addr] = val

    def read(self, addr: int) -> int:
        return self._data.get(addr, 0)


@pytest.fixture
def ram():
    return FakeRAM()


@pytest.fixture
def tracker():
    return DungeonRewardTracker(dungeon_index=1)


class TestDungeonRewardTracker:
    def test_initial_state_value_is_zero(self, tracker):
        vals = tracker.get_state_value()
        assert all(v == 0.0 for v in vals.values())
        assert "d_rooms" in vals
        assert "d_keys" in vals
        assert "d_boss_key" in vals

    def test_no_update_outside_dungeon(self, tracker, ram):
        milestones = tracker.update(
            read_fn=ram.read,
            room_id=0xD9,
            active_group=0,  # overworld
            dungeon_index=0xFF,
            essences_before=0,
            essences_now=0,
        )
        assert milestones == []
        assert tracker.unique_rooms == 0

    def test_room_tracking_in_dungeon(self, tracker, ram):
        # Enter dungeon room 0x1C
        tracker.update(
            read_fn=ram.read,
            room_id=0x1C,
            active_group=4,
            dungeon_index=1,
            essences_before=0,
            essences_now=0,
        )
        assert tracker.unique_rooms == 1

        # Same room again — no new room
        tracker.update(
            read_fn=ram.read,
            room_id=0x1C,
            active_group=4,
            dungeon_index=1,
            essences_before=0,
            essences_now=0,
        )
        assert tracker.unique_rooms == 1

        # New room
        tracker.update(
            read_fn=ram.read,
            room_id=0x2C,
            active_group=4,
            dungeon_index=1,
            essences_before=0,
            essences_now=0,
        )
        assert tracker.unique_rooms == 2

        vals = tracker.get_state_value()
        assert vals["d_rooms"] == 5.0 * 2  # 2 rooms * 5.0 weight

    def test_small_key_pickup(self, tracker, ram):
        # Enter dungeon
        tracker.update(
            read_fn=ram.read,
            room_id=0x1C,
            active_group=4,
            dungeon_index=1,
            essences_before=0,
            essences_now=0,
        )
        assert not tracker.milestone_first_key

        # Pick up a key (D1 = index 1, so DUNGEON_KEYS + 1)
        ram.write(DUNGEON_KEYS + 1, 1)
        milestones = tracker.update(
            read_fn=ram.read,
            room_id=0x1C,
            active_group=4,
            dungeon_index=1,
            essences_before=0,
            essences_now=0,
        )
        assert "got_first_dungeon_key" in milestones
        assert tracker.milestone_first_key
        vals = tracker.get_state_value()
        assert vals["d_keys"] == 15.0

        # Second key — no duplicate milestone
        ram.write(DUNGEON_KEYS + 1, 2)
        milestones = tracker.update(
            read_fn=ram.read,
            room_id=0x1C,
            active_group=4,
            dungeon_index=1,
            essences_before=0,
            essences_now=0,
        )
        assert "got_first_dungeon_key" not in milestones
        vals = tracker.get_state_value()
        assert vals["d_keys"] == 15.0 * 2

    def test_boss_key_pickup(self, tracker, ram):
        # Set boss key bit for dungeon 1 (bit 1)
        ram.write(BOSS_KEYS, 0b00000010)
        milestones = tracker.update(
            read_fn=ram.read,
            room_id=0x1C,
            active_group=4,
            dungeon_index=1,
            essences_before=0,
            essences_now=0,
        )
        assert "got_boss_key" in milestones
        assert tracker.milestone_boss_key
        vals = tracker.get_state_value()
        assert vals["d_boss_key"] == 25.0

    def test_map_and_compass(self, tracker, ram):
        # Map for D1 (bit 1)
        ram.write(DUNGEON_MAP, 0b00000010)
        tracker.update(
            read_fn=ram.read,
            room_id=0x1C,
            active_group=4,
            dungeon_index=1,
            essences_before=0,
            essences_now=0,
        )
        vals = tracker.get_state_value()
        assert vals["d_map"] == 5.0

        # Compass for D1 (bit 1)
        ram.write(DUNGEON_COMPASS, 0b00000010)
        tracker.update(
            read_fn=ram.read,
            room_id=0x2C,
            active_group=4,
            dungeon_index=1,
            essences_before=0,
            essences_now=0,
        )
        vals = tracker.get_state_value()
        assert vals["d_compass"] == 5.0

    def test_room_clear(self, tracker, ram):
        # Enter room with enemies
        ram.write(ENEMIES_COUNT, 3)
        tracker.update(
            read_fn=ram.read,
            room_id=0x1C,
            active_group=4,
            dungeon_index=1,
            essences_before=0,
            essences_now=0,
        )

        # Kill all enemies
        ram.write(ENEMIES_COUNT, 0)
        tracker.update(
            read_fn=ram.read,
            room_id=0x1C,
            active_group=4,
            dungeon_index=1,
            essences_before=0,
            essences_now=0,
        )
        vals = tracker.get_state_value()
        assert vals["d_clears"] == 3.0  # 1 clear * 3.0 weight

    def test_boss_defeat_via_essence(self, tracker, ram):
        milestones = tracker.update(
            read_fn=ram.read,
            room_id=0x1C,
            active_group=4,
            dungeon_index=1,
            essences_before=0,
            essences_now=1,  # Essence collected!
        )
        assert "defeated_boss" in milestones
        assert tracker.milestone_boss_defeated
        vals = tracker.get_state_value()
        assert vals["d_boss"] == 50.0

    def test_wrong_dungeon_index_ignored(self, tracker, ram):
        tracker.update(
            read_fn=ram.read,
            room_id=0x1C,
            active_group=4,
            dungeon_index=2,  # Not D1
            essences_before=0,
            essences_now=0,
        )
        assert tracker.unique_rooms == 0

    def test_reset_clears_all(self, tracker, ram):
        # Accumulate some state
        tracker.update(
            read_fn=ram.read,
            room_id=0x1C,
            active_group=4,
            dungeon_index=1,
            essences_before=0,
            essences_now=0,
        )
        assert tracker.unique_rooms == 1

        tracker.reset()
        assert tracker.unique_rooms == 0
        assert not tracker.milestone_first_key
        assert not tracker.milestone_boss_key
        assert not tracker.milestone_boss_defeated
        vals = tracker.get_state_value()
        assert all(v == 0.0 for v in vals.values())

    def test_custom_reward_weights(self, ram):
        tracker = DungeonRewardTracker(
            reward_config={"d_room_visit": 10.0, "d_small_key": 30.0},
            dungeon_index=1,
        )
        tracker.update(
            read_fn=ram.read,
            room_id=0x1C,
            active_group=4,
            dungeon_index=1,
            essences_before=0,
            essences_now=0,
        )
        vals = tracker.get_state_value()
        assert vals["d_rooms"] == 10.0  # custom weight

        ram.write(DUNGEON_KEYS + 1, 1)
        tracker.update(
            read_fn=ram.read,
            room_id=0x1C,
            active_group=4,
            dungeon_index=1,
            essences_before=0,
            essences_now=0,
        )
        vals = tracker.get_state_value()
        assert vals["d_keys"] == 30.0  # custom weight
