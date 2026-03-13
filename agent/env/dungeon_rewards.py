"""Dungeon-specific reward tracking — isolated from overworld rewards.

Tracks D1 (Gnarled Root Dungeon) progression using dungeon-specific RAM:
  - Room exploration (unique dungeon rooms visited)
  - Small key collection
  - Dungeon items (map, compass, boss key)
  - Room clears (all enemies defeated in a room)
  - Boss defeat (via essence collection while in dungeon)

Design: RewardWrapper delegates to DungeonRewardTracker when
active_group is 4 or 5. Overworld coverage tracking is skipped
during dungeon play, preventing cross-pollination of exploration
signals between the two very different action spaces.

The tracker produces reward components prefixed with ``d_`` that are
merged into the state value dict. The delta between consecutive
calls produces the step reward — same pattern as the overworld system.
"""

from __future__ import annotations

import logging
from typing import Callable

from agent.env.ram_addresses import (
    BOSS_KEYS,
    DUNGEON_COMPASS,
    DUNGEON_G4_ROOM_FLAGS,
    DUNGEON_KEYS,
    DUNGEON_MAP,
    DUNGEON_ROOM_PROPERTIES,
    ENEMIES_COUNT,
    ROOMFLAG_VISITED,
)

logger = logging.getLogger(__name__)


# D1 room property bit masks (from wDungeonRoomProperties / 0xCC58)
ROOM_PROP_KEY = 0x01       # Room contains a small key
ROOM_PROP_CHEST = 0x02     # Room contains a chest
ROOM_PROP_BOSS = 0x04      # Boss room
ROOM_PROP_DARK = 0x08      # Dark room (needs lamp)
ROOM_PROP_MINIBOSS = 0x10  # Mini-boss room


class DungeonRewardTracker:
    """Track dungeon-specific rewards, completely separate from overworld.

    Produces a dict of ``d_``-prefixed reward components via
    ``get_state_value()``.  The caller (RewardWrapper) merges these
    into the global state value dict and computes deltas as usual.

    Args:
        reward_config: Dict of reward weights.  Keys:
            d_room_visit, d_small_key, d_boss_key, d_map, d_compass,
            d_room_clear, d_boss_defeated.
        dungeon_index: Which dungeon to track (1 = D1 Gnarled Root).
    """

    def __init__(
        self,
        reward_config: dict[str, float] | None = None,
        dungeon_index: int = 1,
    ):
        cfg = reward_config or {}
        self._dungeon_index = dungeon_index

        # Reward weights — configurable via env vars
        self._w_room_visit = float(cfg.get("d_room_visit", 5.0))
        self._w_small_key = float(cfg.get("d_small_key", 15.0))
        self._w_boss_key = float(cfg.get("d_boss_key", 25.0))
        self._w_map = float(cfg.get("d_map", 5.0))
        self._w_compass = float(cfg.get("d_compass", 5.0))
        self._w_room_clear = float(cfg.get("d_room_clear", 3.0))
        self._w_boss_defeated = float(cfg.get("d_boss_defeated", 50.0))

        self.reset()

    def reset(self) -> None:
        """Reset all dungeon tracking for a new episode."""
        self._visited_rooms: set[int] = set()
        self._small_keys: int = 0
        self._has_boss_key: bool = False
        self._has_map: bool = False
        self._has_compass: bool = False
        self._rooms_cleared: int = 0
        self._boss_defeated: bool = False

        # Per-step tracking
        self._prev_room: int = -1
        self._prev_enemies: int = -1
        self._in_room_with_enemies: bool = False

        # Milestone flags for advancing checkpoint system
        self.milestone_first_key: bool = False
        self.milestone_boss_key: bool = False
        self.milestone_boss_defeated: bool = False

    def get_state_value(self) -> dict[str, float]:
        """Compute dungeon reward components.

        Returns dict of ``d_``-prefixed components.  The delta between
        consecutive calls is the step reward contribution from the dungeon.
        Components only change when the agent is inside the dungeon.
        """
        return {
            "d_rooms": self._w_room_visit * len(self._visited_rooms),
            "d_keys": self._w_small_key * self._small_keys,
            "d_boss_key": self._w_boss_key * float(self._has_boss_key),
            "d_map": self._w_map * float(self._has_map),
            "d_compass": self._w_compass * float(self._has_compass),
            "d_clears": self._w_room_clear * self._rooms_cleared,
            "d_boss": self._w_boss_defeated * float(self._boss_defeated),
        }

    def update(
        self,
        *,
        read_fn: Callable[[int], int],
        room_id: int,
        active_group: int,
        dungeon_index: int,
        essences_before: int,
        essences_now: int,
    ) -> list[str]:
        """Update dungeon state from current RAM.  Call every step.

        Only processes updates when inside a real dungeon
        (active_group 4/5 and dungeon_index matches).

        Args:
            read_fn: Function to read a single RAM byte (env._read).
            room_id: Current PLAYER_ROOM value.
            active_group: Current ACTIVE_GROUP value.
            dungeon_index: Current DUNGEON_INDEX value.
            essences_before: Essence count at start of step.
            essences_now: Essence count after step (for boss defeat detection).

        Returns:
            List of milestone names achieved this step (for logging/checkpoint).
        """
        if active_group not in (4, 5):
            self._prev_room = -1
            self._prev_enemies = -1
            self._in_room_with_enemies = False
            return []

        if dungeon_index != self._dungeon_index:
            return []

        milestones: list[str] = []

        # --- Room exploration ---
        if room_id not in self._visited_rooms:
            self._visited_rooms.add(room_id)
            logger.info(
                "DUNGEON ROOM: room=0x%02X | total=%d",
                room_id, len(self._visited_rooms),
            )

        # --- Room clear detection ---
        enemies = read_fn(ENEMIES_COUNT)
        if room_id != self._prev_room:
            # Entered a new room — reset enemy tracking
            self._in_room_with_enemies = enemies > 0
        elif self._in_room_with_enemies and enemies == 0:
            # All enemies defeated in this room
            self._rooms_cleared += 1
            self._in_room_with_enemies = False
            logger.info(
                "DUNGEON ROOM CLEAR: room=0x%02X | total_clears=%d",
                room_id, self._rooms_cleared,
            )
        self._prev_room = room_id
        self._prev_enemies = enemies

        # --- Small keys ---
        # DUNGEON_KEYS is a 12-byte array; index by dungeon number
        keys = read_fn(DUNGEON_KEYS + self._dungeon_index)
        if keys > self._small_keys:
            old_keys = self._small_keys
            self._small_keys = keys
            if not self.milestone_first_key:
                self.milestone_first_key = True
                milestones.append("got_first_dungeon_key")
            logger.info(
                "DUNGEON KEY: %d -> %d (dungeon %d)",
                old_keys, keys, self._dungeon_index,
            )

        # --- Boss key ---
        # BOSS_KEYS is a 2-byte bitset; check bit for this dungeon
        boss_keys = read_fn(BOSS_KEYS)
        if not self._has_boss_key and (boss_keys & (1 << self._dungeon_index)):
            self._has_boss_key = True
            self.milestone_boss_key = True
            milestones.append("got_boss_key")
            logger.info("DUNGEON BOSS KEY obtained (dungeon %d)", self._dungeon_index)

        # --- Map ---
        dungeon_map = read_fn(DUNGEON_MAP)
        if not self._has_map and (dungeon_map & (1 << self._dungeon_index)):
            self._has_map = True
            logger.info("DUNGEON MAP obtained (dungeon %d)", self._dungeon_index)

        # --- Compass ---
        compass = read_fn(DUNGEON_COMPASS)
        if not self._has_compass and (compass & (1 << self._dungeon_index)):
            self._has_compass = True
            logger.info("DUNGEON COMPASS obtained (dungeon %d)", self._dungeon_index)

        # --- Boss defeat (essence collected while in dungeon) ---
        if not self._boss_defeated and essences_now > essences_before:
            self._boss_defeated = True
            self.milestone_boss_defeated = True
            milestones.append("defeated_boss")
            logger.info("DUNGEON BOSS DEFEATED (dungeon %d)", self._dungeon_index)

        return milestones

    @property
    def unique_rooms(self) -> int:
        """Number of unique dungeon rooms visited this episode."""
        return len(self._visited_rooms)
