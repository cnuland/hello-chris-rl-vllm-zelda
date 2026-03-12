"""v1.1 Reward wrapper: delta-based rewards with exploration decay.

Inspired by pokemonred_puffer (drubinstein), this wrapper computes a total
game state value each step and returns the DELTA as the step reward.
This eliminates one-shot reward spikes and produces smooth gradients.

Key design principles:
  1. Delta-based: reward = new_total - old_total (no spikes, no UNACCOUNTED)
  2. Exploration decay: tiles lose value over time, driving continuous movement
  3. No penalties: purely additive rewards (no loiter, no death penalty)
  4. No phases: single flat reward structure (game progression gates naturally)
  5. Many small signals: events ~5-20, exploration ~0.02/tile, dialog ~0.5
"""

from __future__ import annotations

import io
import json
import logging
import os
from typing import Any

import gymnasium as gym
import numpy as np

from agent.env.zelda_env import ButtonAction
from agent.env.ram_addresses import (
    ACTIVE_GROUP,
    DIALOG_STATE,
    DUNGEON_FLOOR,
    DUNGEON_INDEX,
    DUNGEON_KEYS,
    ESSENCES_COLLECTED,
    GNARLED_KEY_GIVEN_FLAG,
    GNARLED_KEY_GIVEN_MASK,
    GNARLED_KEY_OBTAINED,
    GNARLED_KEY_OBTAINED_MASK,
    LINK_PUSHING_DIRECTION,
    MAKU_ROOM_FLAGS,
    MAKU_SEED_FLAG,
    MAKU_SEED_MASK,
    MAKU_TREE_STAGE,
    OVERWORLD_ROOM_FLAGS,
    ROOMFLAG_GATE_HIT,
    SWORD_LEVEL,
)

from agent.rl.rewards import CoverageReward, PotentialShaping

logger = logging.getLogger(__name__)

# The Maku Tree gate is on the OVERWORLD (group 0) at room 0xD9 (row=13, col=9).
MAKU_GATE_ROOM = 0xD9


class RewardWrapper(gym.Wrapper):
    """Delta-based reward wrapper with exploration decay.

    Computes total game state value each step, returns the change as reward.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_config: dict[str, float] | None = None,
        enable_shaping: bool = False,
        reward_model_path: str | None = None,
        enable_export: bool = True,
        s3_config: dict[str, str] | None = None,
        epoch: int = 0,
    ):
        super().__init__(env)
        cfg = reward_config or {}

        # --- Event flag weights (permanent, delta fires once) ---
        self._w_gate_slash = float(cfg.get("gate_slash", 10.0))
        self._w_maku_visit = float(cfg.get("maku_tree_visit", 5.0))
        self._w_maku_dialog = float(cfg.get("maku_dialog", 8.0))
        self._w_gnarled_key = float(cfg.get("gnarled_key", 15.0))
        self._w_maku_seed = float(cfg.get("maku_seed", 25.0))
        self._w_got_sword = float(cfg.get("sword", 5.0))
        self._w_snow_region = float(cfg.get("snow_region", 10.0))
        self._w_d1_entrance = float(cfg.get("d1_entrance_bonus", 12.0))
        self._w_dungeon_entry = float(cfg.get("dungeon_entry", 20.0))
        self._w_essence = float(cfg.get("essence", 30.0))
        self._w_dungeon_key = float(cfg.get("dungeon_key_pickup", 5.0))
        self._w_dungeon_floor = float(cfg.get("dungeon_floor", 5.0))

        # --- Exploration weights ---
        self._w_exploration = float(cfg.get("grid_exploration", 0.02))
        self._w_rooms = float(cfg.get("new_room", 0.5))

        # --- Dialog weight ---
        self._w_dialog = float(cfg.get("dialog_advance", 0.5))
        self._dialog_cap = int(cfg.get("dialog_cap", 20))

        # --- Directional weight (sub-room granular) ---
        self._w_directional = float(cfg.get("directional_bonus", 0.0))
        self._directional_target_row = int(cfg.get("directional_target_row",
                                                     os.getenv("DIRECTIONAL_TARGET_ROW", "9")))
        self._directional_target_col = int(cfg.get("directional_target_col",
                                                     os.getenv("DIRECTIONAL_TARGET_COL", "6")))
        # Pre-compute target in absolute tile coords (10 tiles/room wide, 8 tall)
        self._target_tile_x = self._directional_target_col * 10 + 5  # center
        self._target_tile_y = self._directional_target_row * 8 + 4   # center

        # --- Idle penalty weight ---
        self._w_idle = float(cfg.get("idle_penalty", 0.002))
        # Rooms where idle penalty is exempt (milestone interaction locations)
        self._idle_exempt_rooms: set[int] = {
            0xD9,  # Maku Gate (217) — gate slashing
            0x96,  # D1 entrance (150) — dungeon entry
        }

        # --- Exploration decay parameters ---
        decay_factor = float(cfg.get("exploration_decay", 0.9995))
        decay_freq = int(cfg.get("exploration_decay_freq", 10))
        decay_floor = float(cfg.get("exploration_decay_floor", 0.15))

        # --- Snow region bounds ---
        self._snow_region_min_row = int(cfg.get("snow_region_min_row",
                                                 os.getenv("SNOW_REGION_MIN_ROW", "9")))
        self._snow_region_max_row = int(cfg.get("snow_region_max_row",
                                                 os.getenv("SNOW_REGION_MAX_ROW", "11")))
        self._snow_region_min_col = int(cfg.get("snow_region_min_col",
                                                 os.getenv("SNOW_REGION_MIN_COL", "6")))
        self._snow_region_max_col = int(cfg.get("snow_region_max_col",
                                                 os.getenv("SNOW_REGION_MAX_COL", "7")))

        # --- D1 entrance room ---
        self._d1_entrance_room = int(cfg.get("d1_entrance_room",
                                              os.getenv("D1_ENTRANCE_ROOM", "0x96")), 0)

        # Sub-modules
        self._coverage = CoverageReward(
            exploration_inc=1.0,
            room_inc=1.0,
            decay_factor=decay_factor,
            decay_frequency=decay_freq,
            decay_floor=decay_floor,
        )

        # Potential-based shaping (optional)
        self._shaping = PotentialShaping(lam=0.01, epoch=epoch) if enable_shaping else None
        self._reward_model = None
        if enable_shaping and reward_model_path:
            self._load_reward_model(reward_model_path)

        # Delta tracking
        self._total_value = 0.0
        self._episode_total_reward = 0.0

        # --- Event flags (set once per episode, contribute to state value) ---
        self._gate_slashed = False
        self._visited_maku = False
        self._maku_dialog_given = False
        self._has_gnarled_key = False
        self._has_maku_seed = False
        self._got_sword = False
        self._entered_snow_region = False
        self._reached_d1_entrance = False
        self._entered_dungeon = False

        # Baselines (from save state — don't award events already achieved)
        self._baseline_gate_slashed = False
        self._baseline_gnarled_key = False
        self._baseline_maku_seed = False
        self._baseline_sword = 0
        self._baseline_group = 0
        self._baseline_maku_stage = 0

        # Cumulative counters
        self._dialog_advance_count = 0
        self._essences_collected = 0
        self._dungeon_keys = 0
        self._max_dungeon_floor = 0
        self._maku_stage = 0

        # Directional tracking
        self._min_target_distance = 999
        self._last_overworld_room = None
        self._last_pixel_x = 0
        self._last_pixel_y = 0

        # Idle tracking
        self._consecutive_idle_steps = 0

        # Previous frame state
        self._prev_pixel_x = 0
        self._prev_pixel_y = 0
        self._prev_group = 0
        self._prev_dialog_active = False
        self._prev_dialog_value = 0
        self._current_button = 0

        # Milestone tracking (for info dict compatibility)
        self._milestone_got_sword = False
        self._milestone_entered_dungeon = False
        self._milestone_visited_maku_tree = False
        self._milestone_entered_snow_region = False
        self._milestone_reached_d1_entrance = False
        self._milestone_essences = 0
        self._milestone_dungeon_keys = 0
        self._milestone_max_rooms = 0

        # Epoch tracking
        self._epoch = epoch

        # Episode export
        self._exporter = None
        self._enable_export = enable_export
        self._milestone_export = os.getenv("MILESTONE_EXPORT", "").lower() in ("1", "true")
        self._episode_worthy = False
        self._episode_id = ""
        if enable_export:
            if self._milestone_export:
                self._init_exporter(s3_config, deferred=True)
            else:
                import random
                export_prob_str = os.getenv("EXPORT_PROB", "")
                if export_prob_str:
                    export_prob = float(export_prob_str)
                else:
                    export_prob = max(0.10, 1.0 / max(int(os.getenv("NUM_ENVS", "1")), 1))
                if random.random() < export_prob:
                    self._init_exporter(s3_config)

        # Milestone state capture
        self._milestone_state_dir = os.getenv("MILESTONE_STATE_DIR", "")
        self._captured_milestones: set[str] = set()

        # Breakdown logging
        self._logged_breakdown = False
        self._prev_state_dict: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Game state value computation (the heart of delta-based rewards)
    # ------------------------------------------------------------------

    def get_game_state_value(self, info: dict) -> dict[str, float]:
        """Compute total game state value as a dict of named components.

        The step reward is sum(new_values) - sum(old_values).
        Event flags contribute permanently once set.
        Exploration values decay over time.
        """
        active_group = info.get("active_group", 0)

        # Directional progress toward target — sub-room granular
        # Uses absolute tile coords for smooth within-room gradients
        directional_val = 0.0
        if self._w_directional > 0 and self._last_overworld_room is not None:
            room = self._last_overworld_room
            row, col = room // 16, room % 16
            abs_tile_x = col * 10 + (self._last_pixel_x // 16)
            abs_tile_y = row * 8 + (self._last_pixel_y // 16)
            tile_dist = abs(abs_tile_x - self._target_tile_x) + abs(abs_tile_y - self._target_tile_y)
            max_tile_dist = 200
            directional_val = self._w_directional * max(0.0, max_tile_dist - tile_dist) / 10.0

        return {
            # Permanent event flags
            "gate_slashed": self._w_gate_slash * float(self._gate_slashed and not self._baseline_gate_slashed),
            "maku_visited": self._w_maku_visit * float(self._visited_maku),
            "maku_dialog": self._w_maku_dialog * float(self._maku_dialog_given),
            "gnarled_key": self._w_gnarled_key * float(self._has_gnarled_key and not self._baseline_gnarled_key),
            "maku_seed": self._w_maku_seed * float(self._has_maku_seed and not self._baseline_maku_seed),
            "got_sword": self._w_got_sword * float(self._got_sword),
            "snow_region": self._w_snow_region * float(self._entered_snow_region),
            "d1_entrance": self._w_d1_entrance * float(self._reached_d1_entrance),
            "dungeon_entry": self._w_dungeon_entry * float(self._entered_dungeon),
            "essences": self._w_essence * self._essences_collected,
            "dungeon_keys": self._w_dungeon_key * self._dungeon_keys,
            "dungeon_floors": self._w_dungeon_floor * self._max_dungeon_floor,

            # Decaying exploration
            "exploration": self._w_exploration * self._coverage.total_tile_value(),
            "rooms": self._w_rooms * self._coverage.total_room_value(),

            # Cumulative counters
            "dialog": self._w_dialog * min(self._dialog_advance_count, self._dialog_cap),

            # Spatial progress
            "direction": directional_val,

            # Idle penalty (negative component — grows each idle step)
            "idle": -self._w_idle * self._consecutive_idle_steps,
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)

        # Snapshot baseline from save state RAM
        self._prev_group = info.get("active_group", 0)
        self._baseline_group = self._prev_group
        self._prev_pixel_x = info.get("pixel_x", 0)
        self._prev_pixel_y = info.get("pixel_y", 0)
        self._prev_dialog_active = False
        self._prev_dialog_value = 0

        if hasattr(self.env, "_read"):
            self._baseline_sword = self.env._read(SWORD_LEVEL)
            self._baseline_gnarled_key = bool(
                self.env._read(GNARLED_KEY_OBTAINED) & GNARLED_KEY_OBTAINED_MASK
            )
            self._baseline_maku_seed = bool(
                self.env._read(MAKU_SEED_FLAG) & MAKU_SEED_MASK
            )
            gate_flags = self.env._read(OVERWORLD_ROOM_FLAGS + MAKU_GATE_ROOM)
            self._baseline_gate_slashed = bool(gate_flags & ROOMFLAG_GATE_HIT)
            self._baseline_maku_stage = self.env._read(MAKU_TREE_STAGE)
        else:
            self._baseline_sword = 0
            self._baseline_gnarled_key = False
            self._baseline_maku_seed = False
            self._baseline_gate_slashed = False
            self._baseline_maku_stage = 0

        # Reset event flags
        self._gate_slashed = self._baseline_gate_slashed
        self._visited_maku = False
        self._maku_dialog_given = False
        self._has_gnarled_key = self._baseline_gnarled_key
        self._has_maku_seed = self._baseline_maku_seed
        self._got_sword = False
        self._entered_snow_region = False
        self._reached_d1_entrance = False
        self._entered_dungeon = False
        self._dialog_advance_count = 0
        self._essences_collected = 0
        self._dungeon_keys = 0
        self._max_dungeon_floor = 0
        self._maku_stage = self._baseline_maku_stage

        # Directional tracking + idle reset
        start_room = info.get("room_id", 0)
        self._last_overworld_room = None
        self._last_pixel_x = info.get("pixel_x", 0)
        self._last_pixel_y = info.get("pixel_y", 0)
        self._consecutive_idle_steps = 0
        if self._prev_group == 0:
            self._last_overworld_room = start_room
            row, col = start_room // 16, start_room % 16
            self._min_target_distance = (
                abs(row - self._directional_target_row) +
                abs(col - self._directional_target_col)
            )
        else:
            self._min_target_distance = -1

        # Reset exploration
        self._coverage.reset()

        # Compute initial state value
        self._total_value = sum(self.get_game_state_value(info).values())
        self._episode_total_reward = 0.0
        self._logged_breakdown = False
        self._prev_state_dict = {}

        # Reset milestones
        self._milestone_got_sword = False
        self._milestone_entered_dungeon = False
        self._milestone_visited_maku_tree = False
        self._milestone_entered_snow_region = False
        self._milestone_reached_d1_entrance = False
        self._milestone_essences = 0
        self._milestone_dungeon_keys = 0
        self._milestone_max_rooms = 0
        self._captured_milestones.clear()

        # Potential shaping
        if self._shaping:
            self._shaping.reset()

        # Expose visited rooms to base env
        self.env._visited_rooms_set = self._coverage._visited_rooms

        # Episode export
        self._episode_worthy = False
        if self._exporter:
            self._episode_id = self._exporter.begin_episode()
            info["episode_id"] = self._episode_id

        # Log start once per epoch
        if not hasattr(self, "_logged_start"):
            logger.info(
                "Episode start: room_id=%d (row=%d, col=%d), group=%d, sword=%d",
                start_room, start_room // 16, start_room % 16,
                self._prev_group, self._baseline_sword,
            )
            self._logged_start = True

        info["epoch"] = self._epoch
        return obs, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_arr = np.asarray(action).flatten()
        self._current_button = int(action_arr[1]) if len(action_arr) > 1 else 0

        obs, _, terminated, truncated, info = self.env.step(action)

        active_group = info.get("active_group", 0)
        room_id = info.get("room_id", 0)

        # --- Update game event flags ---
        self._update_flags(info)

        # --- Update exploration (only when moved) ---
        is_transitioning = info.get("transitioning", False)
        dialog_active = info.get("dialog_active", False)
        menu_active = info.get("menu_active", False)
        busy = is_transitioning or dialog_active or menu_active

        if not is_transitioning:
            cur_px = info.get("pixel_x", 0)
            cur_py = info.get("pixel_y", 0)
            moved = cur_px != self._prev_pixel_x or cur_py != self._prev_pixel_y
            if moved:
                # Qualify room for multi-group tracking
                if active_group == 0:
                    qualified_room = room_id
                else:
                    qualified_room = active_group * 256 + room_id
                self._coverage.step(qualified_room, cur_px, cur_py)

                # Track overworld room + pixel for sub-room directional
                if active_group == 0:
                    self._last_overworld_room = room_id
                    self._last_pixel_x = cur_px
                    self._last_pixel_y = cur_py

                # Reset idle counter on movement
                self._consecutive_idle_steps = 0
            elif busy or active_group != 0 or room_id in self._idle_exempt_rooms:
                # Don't penalize idle during dialog, menus, transitions,
                # non-overworld areas (Maku Tree interior, dungeons),
                # or milestone overworld rooms (gate, D1 entrance)
                pass
            else:
                self._consecutive_idle_steps += 1

            self._prev_pixel_x = cur_px
            self._prev_pixel_y = cur_py

        self._prev_group = active_group

        # --- Compute delta reward ---
        new_state = self.get_game_state_value(info)
        new_total = sum(new_state.values())
        reward = new_total - self._total_value
        self._total_value = new_total
        self._prev_state_dict = new_state

        # Potential-based shaping (optional)
        if self._shaping is not None and self._reward_model is not None:
            phi = self._reward_model.predict(obs)
            reward = self._shaping.shape(reward, phi)

        self._episode_total_reward += reward

        # --- Log room discoveries ---
        new_rooms = self._coverage.unique_rooms
        if new_rooms > getattr(self, "_prev_logged_rooms", 0):
            logger.info(
                "NEW ROOM: group=%d room=%d (row=%d, col=%d) | total=%d",
                active_group, room_id, room_id // 16, room_id % 16, new_rooms,
            )
        self._prev_logged_rooms = new_rooms

        # --- Record frame for export ---
        if self._exporter:
            screen = self.env.render() if hasattr(self.env, "render") else None
            self._exporter.record_frame(
                step=info.get("step", 0),
                state=info,
                action=action,
                reward=reward,
                screen_array=screen,
            )

        # --- Episode end ---
        if (terminated or truncated) and self._exporter:
            self._exporter.end_episode()
            if self._milestone_export:
                if self._episode_worthy:
                    self._exporter.commit_episode()
                else:
                    self._exporter.discard_episode()

        if (terminated or truncated) and not self._logged_breakdown:
            self._logged_breakdown = True
            self._log_breakdown(info)

        # --- Update milestone tracking for info dict ---
        self._milestone_max_rooms = max(self._milestone_max_rooms, new_rooms)
        if hasattr(self.env, "_read"):
            keys = self.env._read(DUNGEON_KEYS)
            self._milestone_dungeon_keys = max(self._milestone_dungeon_keys, keys)
            essences = bin(self.env._read(ESSENCES_COLLECTED)).count("1")
            self._milestone_essences = max(self._milestone_essences, essences)
        dungeon_idx = info.get("dungeon_index", 0xFF)
        if (active_group in (4, 5) and self._baseline_group not in (4, 5)
                and 1 <= dungeon_idx < 0xFF):
            self._milestone_entered_dungeon = True
        if active_group == 2 and self._baseline_group != 2:
            self._milestone_visited_maku_tree = True

        # --- Populate info dict ---
        info["epoch"] = self._epoch
        info["unique_rooms"] = new_rooms
        info["unique_tiles"] = self._coverage.total_tiles
        info["seen_coords"] = self._coverage.total_tiles
        info["episode_reward"] = self._episode_total_reward

        # Milestone flags
        info["milestone_got_sword"] = float(self._milestone_got_sword)
        info["milestone_entered_dungeon"] = float(self._milestone_entered_dungeon)
        info["milestone_visited_maku_tree"] = float(self._milestone_visited_maku_tree)
        info["milestone_essences"] = float(self._milestone_essences)
        info["milestone_dungeon_keys"] = float(self._milestone_dungeon_keys)
        info["milestone_max_rooms"] = float(self._milestone_max_rooms)
        info["milestone_gnarled_key"] = float(
            self._has_gnarled_key and not self._baseline_gnarled_key
        )
        info["milestone_gate_slashed"] = float(
            self._gate_slashed and not self._baseline_gate_slashed
        )
        info["milestone_entered_snow_region"] = float(self._milestone_entered_snow_region)
        info["milestone_reached_d1_entrance"] = float(self._milestone_reached_d1_entrance)
        info["milestone_maku_dialog"] = float(self._maku_dialog_given)
        info["milestone_maku_seed"] = float(
            self._has_maku_seed and not self._baseline_maku_seed
        )
        info["milestone_maku_rooms"] = float(self._coverage.unique_rooms)
        info["milestone_maku_stage"] = float(self._maku_stage > self._baseline_maku_stage)

        # Baselines
        info["baseline_has_sword"] = float(self._baseline_sword > 0)
        info["baseline_has_gnarled_key"] = float(self._baseline_gnarled_key)
        info["baseline_has_maku_seed"] = float(self._baseline_maku_seed)
        info["baseline_has_maku_dialog"] = float(False)
        info["baseline_in_maku"] = float(self._baseline_group == 2)
        info["baseline_gate_slashed"] = float(self._baseline_gate_slashed)
        info["baseline_maku_stage"] = float(self._baseline_maku_stage)
        start_room = getattr(self, "_start_room_id", info.get("room_id", 0))
        start_row = start_room // 16
        start_col = start_room % 16
        info["baseline_in_snow_region"] = float(
            self._baseline_group == 0
            and self._snow_region_min_row <= start_row <= self._snow_region_max_row
            and self._snow_region_min_col <= start_col <= self._snow_region_max_col
        )
        info["baseline_in_dungeon"] = float(self._baseline_group in (4, 5))

        # Phase compatibility (flat — always report current progression)
        info["current_phase"] = self._detect_phase()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Flag updates (detect game events from RAM)
    # ------------------------------------------------------------------

    def _update_flags(self, info: dict) -> None:
        """Read RAM and update event flags."""
        active_group = info.get("active_group", 0)
        room_id = info.get("room_id", 0)
        dungeon_idx = info.get("dungeon_index", 0xFF)

        if not hasattr(self.env, "_read"):
            return

        read = self.env._read

        # Sword upgrade
        sword = read(SWORD_LEVEL)
        if sword > self._baseline_sword and not self._got_sword:
            self._got_sword = True
            self._milestone_got_sword = True
            self._capture_milestone_state("got_sword", self._episode_total_reward)
            logger.info("MILESTONE: Got sword (level %d)", sword)

        # Gate slash
        if not self._gate_slashed:
            gate_flags = read(OVERWORLD_ROOM_FLAGS + MAKU_GATE_ROOM)
            if gate_flags & ROOMFLAG_GATE_HIT:
                self._gate_slashed = True
                if not self._baseline_gate_slashed:
                    self._capture_milestone_state("gate_slashed", self._episode_total_reward)
                    logger.info("MILESTONE: Gate slashed!")

        # Maku Tree visit
        if active_group == 2 and self._prev_group != 2 and not self._visited_maku:
            self._visited_maku = True
            logger.info("MILESTONE: Visited Maku Tree")

        # Maku dialog (Gnarled Key quest given)
        maku_dialog = bool(read(GNARLED_KEY_GIVEN_FLAG) & GNARLED_KEY_GIVEN_MASK)
        if maku_dialog and not self._maku_dialog_given:
            self._maku_dialog_given = True
            self._capture_milestone_state("maku_dialog", self._episode_total_reward)
            logger.info("MILESTONE: Maku Tree gave Gnarled Key quest!")

        # Gnarled Key obtained
        gnarled_key = bool(read(GNARLED_KEY_OBTAINED) & GNARLED_KEY_OBTAINED_MASK)
        if gnarled_key and not self._has_gnarled_key:
            self._has_gnarled_key = True
            if not self._baseline_gnarled_key:
                self._capture_milestone_state("gnarled_key", self._episode_total_reward)
                logger.info("MILESTONE: Gnarled Key obtained!")

        # Maku Seed
        maku_seed = bool(read(MAKU_SEED_FLAG) & MAKU_SEED_MASK)
        if maku_seed and not self._has_maku_seed:
            self._has_maku_seed = True
            if not self._baseline_maku_seed:
                self._capture_milestone_state("maku_seed", self._episode_total_reward)
                logger.info("MILESTONE: Maku Seed obtained!")

        # Maku stage
        self._maku_stage = read(MAKU_TREE_STAGE)

        # Essences
        essences = bin(read(ESSENCES_COLLECTED)).count("1")
        if essences > self._essences_collected:
            self._essences_collected = essences
            logger.info("MILESTONE: Essence collected (total: %d)", essences)

        # Dungeon keys
        keys = read(DUNGEON_KEYS)
        if keys > self._dungeon_keys:
            self._dungeon_keys = keys

        # Dungeon floor
        dungeon_floor = info.get("dungeon_floor", 0)
        if active_group in (4, 5) and dungeon_floor > self._max_dungeon_floor:
            self._max_dungeon_floor = dungeon_floor

        # Dungeon entry
        if (active_group in (4, 5) and self._baseline_group not in (4, 5)
                and 1 <= dungeon_idx < 0xFF and not self._entered_dungeon):
            self._entered_dungeon = True
            self._episode_worthy = True
            self._capture_milestone_state("entered_dungeon", self._episode_total_reward)
            logger.info(
                "MILESTONE: Dungeon entry! index=%d group=%d room=%d",
                dungeon_idx, active_group, room_id,
            )

        # Snow region
        if (self._has_gnarled_key and not self._entered_snow_region
                and active_group == 0):
            cur_row = room_id // 16
            cur_col = room_id % 16
            if (self._snow_region_min_row <= cur_row <= self._snow_region_max_row
                    and self._snow_region_min_col <= cur_col <= self._snow_region_max_col):
                self._entered_snow_region = True
                self._milestone_entered_snow_region = True
                self._episode_worthy = True
                self._capture_milestone_state("entered_snow_region", self._episode_total_reward)
                logger.info("MILESTONE: Entered snow region at (%d,%d)", cur_row, cur_col)

        # D1 entrance room
        if (not self._reached_d1_entrance and active_group == 0
                and room_id == self._d1_entrance_room):
            self._reached_d1_entrance = True
            self._milestone_reached_d1_entrance = True
            self._episode_worthy = True
            self._capture_milestone_state("reached_d1_entrance", self._episode_total_reward)
            logger.info("MILESTONE: Reached D1 entrance room 0x%02X", room_id)

        # Dialog advance — reward A-presses during active dialog
        dialog_active = info.get("dialog_active", False)
        if dialog_active and self._dialog_advance_count < self._dialog_cap:
            if self._current_button == ButtonAction.A:
                self._dialog_advance_count += 1
        self._prev_dialog_active = dialog_active

    # ------------------------------------------------------------------
    # Phase detection (for info dict compatibility, NOT reward gating)
    # ------------------------------------------------------------------

    def _detect_phase(self) -> str:
        """Simple phase string for logging/monitoring only."""
        if self._entered_dungeon:
            return "dungeon"
        if self._entered_snow_region:
            return "snow_region"
        if self._has_gnarled_key:
            return "post_key"
        if self._gate_slashed:
            return "post_gate"
        if self._got_sword or self._baseline_sword > 0:
            return "pre_maku"
        return "pre_sword"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_breakdown(self, info: dict) -> None:
        """Log reward breakdown at episode end."""
        state = self._prev_state_dict or self.get_game_state_value(info)
        components = " | ".join(f"{k}={v:.1f}" for k, v in state.items() if v != 0)
        logger.info(
            "REWARD BREAKDOWN: total=%.1f, state_value=%.1f, tiles=%d, rooms=%d, "
            "phase=%s | %s",
            self._episode_total_reward,
            self._total_value,
            self._coverage.total_tiles,
            self._coverage.unique_rooms,
            self._detect_phase(),
            components,
        )

    # ------------------------------------------------------------------
    # Milestone state capture
    # ------------------------------------------------------------------

    def _capture_milestone_state(self, milestone_name: str, reward_so_far: float) -> None:
        """Capture PyBoy emulator state when a milestone fires."""
        if not self._milestone_state_dir:
            return
        if milestone_name in self._captured_milestones:
            return
        if not hasattr(self.env, "_pyboy") or self.env._pyboy is None:
            return

        self._captured_milestones.add(milestone_name)

        buf = io.BytesIO()
        self.env._pyboy.save_state(buf)
        state_bytes = buf.getvalue()

        env_id = id(self) % 100000
        filename = f"{milestone_name}_{env_id}_{self._epoch}.state"
        state_path = os.path.join(self._milestone_state_dir, filename)

        os.makedirs(self._milestone_state_dir, exist_ok=True)
        with open(state_path, "wb") as f:
            f.write(state_bytes)

        room_id = getattr(self.env, "room_id", -1) if hasattr(self.env, "room_id") else -1
        meta = {
            "milestone": milestone_name,
            "epoch": self._epoch,
            "room_id": room_id,
            "reward_so_far": reward_so_far,
            "unique_rooms": self._coverage.unique_rooms,
            "unique_tiles": self._coverage.total_tiles,
        }
        with open(state_path + ".json", "w") as f:
            json.dump(meta, f)

        logger.info(
            "Captured milestone state: %s (room=%d, reward=%.1f, tiles=%d)",
            milestone_name, room_id, reward_so_far, meta["unique_tiles"],
        )

    # ------------------------------------------------------------------
    # Episode export
    # ------------------------------------------------------------------

    def _init_exporter(
        self, s3_config: dict[str, str] | None, deferred: bool = False,
    ) -> None:
        try:
            from agent.evaluator.exporter import EpisodeExporter
            from agent.utils.s3 import S3Client
            from agent.utils.config import S3Config

            if s3_config:
                cfg = S3Config(**s3_config)
            else:
                cfg = S3Config()
            s3 = S3Client(cfg)
            s3.ensure_bucket(cfg.episodes_bucket)
            self._exporter = EpisodeExporter(
                s3_client=s3,
                bucket=cfg.episodes_bucket,
                frames_per_segment=300,
                png_interval=int(os.getenv("PNG_INTERVAL", "60")),
                epoch=self._epoch,
                deferred=deferred,
            )
        except Exception as e:
            logger.warning("Episode export disabled: %s", e)
            self._exporter = None

    def _load_reward_model(self, path: str) -> None:
        try:
            import tempfile
            from agent.evaluator.reward_model import RewardModel

            local_path = path
            if not os.path.exists(local_path):
                try:
                    from agent.utils.s3 import S3Client
                    from agent.utils.config import S3Config

                    s3 = S3Client(S3Config())
                    rm_data = s3.download_bytes("zelda-models", path)
                    os.makedirs("/tmp/reward_model", exist_ok=True)
                    safe_name = path.replace("/", "_")
                    local_path = f"/tmp/reward_model/{safe_name}"
                    if not os.path.exists(local_path):
                        tmp_fd, tmp_path = tempfile.mkstemp(
                            dir="/tmp/reward_model", suffix=".pt"
                        )
                        try:
                            os.write(tmp_fd, rm_data)
                            os.close(tmp_fd)
                            os.rename(tmp_path, local_path)
                        except Exception:
                            os.close(tmp_fd)
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                            raise
                    logger.info("Downloaded reward model from s3 (%d bytes)", len(rm_data))
                except Exception as s3_err:
                    logger.warning("Could not download reward model from s3: %s", s3_err)
                    self._reward_model = None
                    return

            self._reward_model = RewardModel()
            self._reward_model.load(local_path)
            logger.info("Loaded reward model from %s", local_path)
        except Exception as e:
            logger.warning("Could not load reward model: %s", e)
            self._reward_model = None
