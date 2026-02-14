"""Reward wrapper: integrates coverage, game events, RND, and potential shaping.

Wraps ZeldaEnv to compute composite rewards from:
  1. Game event rewards (health, rupees, keys, rooms, sword, death)
  2. Coverage rewards (tile exploration, new rooms)
  3. RND curiosity bonus (clamped)
  4. Potential-based shaping from RLAIF reward model (when available)

Also handles episode export to MinIO for the judge/eval pipeline.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np

from agent.env.ram_addresses import (
    ACTIVE_GROUP,
    DUNGEON_FLOOR,
    DUNGEON_INDEX,
    DUNGEON_KEYS,
    ESSENCES_COLLECTED,
    GNARLED_KEY_GIVEN_FLAG,
    GNARLED_KEY_GIVEN_MASK,
    GNARLED_KEY_OBTAINED,
    GNARLED_KEY_OBTAINED_MASK,
    RUPEES,
    SWORD_LEVEL,
)
from agent.rl.rewards import CoverageReward, PotentialShaping, RNDCuriosity

logger = logging.getLogger(__name__)


class RewardWrapper(gym.Wrapper):
    """Wraps ZeldaEnv with composite reward computation and episode export."""

    def __init__(
        self,
        env: gym.Env,
        reward_config: dict[str, float] | None = None,
        enable_rnd: bool = True,
        enable_shaping: bool = False,
        reward_model_path: str | None = None,
        enable_export: bool = True,
        s3_config: dict[str, str] | None = None,
        epoch: int = 0,
        stagnation_limit: int = 5000,
    ):
        super().__init__(env)
        cfg = reward_config or {}

        # Game event reward scales — flattened to reduce dynamic range
        # (Pokemon Red projects use ~1000:1 max ratio vs our old ~50000:1)
        self._rupee_scale = cfg.get("rupee", 0.01)
        self._key_scale = cfg.get("key", 0.5)
        self._death_penalty = cfg.get("death", -1.0)
        self._health_loss_scale = cfg.get("health_loss", -0.005)
        self._time_penalty = cfg.get("time_penalty", 0.0)
        self._sword_bonus = cfg.get("sword", 15.0)
        self._dungeon_bonus = cfg.get("dungeon", 15.0)
        self._maku_tree_bonus = cfg.get("maku_tree", 15.0)

        # Progression reward scales
        self._dungeon_entry_bonus = cfg.get("dungeon_entry", 15.0)
        self._maku_tree_visit_bonus = cfg.get("maku_tree_visit", 15.0)
        self._indoor_entry_bonus = cfg.get("indoor_entry", 5.0)
        self._dungeon_floor_bonus = cfg.get("dungeon_floor", 2.0)

        # Dialog interaction reward — teaches the agent that NPC dialog is
        # valuable (required for quest progression: Maku Tree gives Gnarled Key)
        self._dialog_bonus = cfg.get("dialog_bonus", 3.0)
        self._dialog_rooms: set[int] = set()
        self._prev_dialog_active = False

        # Maku Tree quest milestone rewards — one-time bonuses for
        # critical quest progression (oracles-disasm confirmed addresses)
        self._maku_dialog_bonus = cfg.get("maku_dialog", 30.0)
        self._gnarled_key_bonus = cfg.get("gnarled_key", 30.0)
        self._prev_maku_dialog = False
        self._prev_gnarled_key = False

        # Distance-from-start exploration bonus — rewards agent for reaching
        # rooms further from the starting position (pokemonred_puffer style).
        # Only active in overworld (group 0) to avoid reward spikes from
        # non-overworld room IDs that are in completely different ranges.
        self._distance_bonus = cfg.get("distance_bonus", 5.0)
        self._max_distance = 0
        self._start_row = 0
        self._start_col = 0

        # Directional exploration bonus — biases exploration toward the Maku
        # Tree which is east then north of the starting position.
        # Decays per step (0.999^step) so it's strong early (guiding toward
        # Maku Tree) but fades by mid-episode, allowing westward movement
        # to reach Dungeon 1 after completing the Maku Tree sequence.
        self._directional_bonus = cfg.get("directional_bonus", 8.0)
        self._directional_decay = cfg.get("directional_decay", 0.999)
        self._min_row_reached = 0
        self._max_col_reached = 0

        # Stagnation-based truncation — end episode early if agent hasn't
        # discovered any new TILES for this many steps.  Tile-based (not
        # room-based) so the agent can transit through known rooms.
        self._stagnation_limit = stagnation_limit
        self._steps_since_discovery = 0
        self._prev_total_tiles = 0

        # Menu management — allow brief menu use for item switching,
        # but penalize camping and suppress exploration rewards while open
        self._menu_steps = 0
        self._menu_grace = 30     # Steps allowed in menu without penalty
        self._menu_max = 60       # Auto-dismiss menu after this many steps
        self._menu_penalty = -0.05 # Per-step penalty after grace period

        # Exit-seeking shaping — continuous reward for moving toward FRONTIER exits
        self._exit_seeking_scale = cfg.get("exit_seeking", 0.5)
        self._prev_exit_dist = 0

        # Area-based exploration boost — multiplies coverage reward by
        # active_group to guide exploration toward key progression areas.
        # Inspired by pokemonred_puffer's map-specific exploration weights.
        # Group 0 (overworld) = 1.0x, Group 2 (maku tree) = 3.0x,
        # Group 3 (indoors/NPCs) = 1.5x, Group 4-5 (dungeons) = 2.0x.
        self._area_boost = {
            0: cfg.get("area_boost_overworld", 1.0),
            1: cfg.get("area_boost_subrosia", 1.5),
            2: cfg.get("area_boost_maku", 3.0),
            3: cfg.get("area_boost_indoors", 1.5),
            4: cfg.get("area_boost_dungeon", 2.0),
            5: cfg.get("area_boost_dungeon", 2.0),
        }

        # Backtrack penalty — discourage re-entering recently visited rooms
        self._recent_rooms: deque[int] = deque(maxlen=5)
        self._backtrack_penalty = cfg.get("backtrack_penalty", 0.0)
        self._prev_room_id = -1

        # Sub-reward modules
        self._coverage = CoverageReward(
            bonus_per_tile=cfg.get("grid_exploration", 0.005),
            bonus_per_room=cfg.get("new_room", 10.0),
            coord_decay_factor=cfg.get("coord_decay_factor", 0.9998),
            coord_decay_floor=cfg.get("coord_decay_floor", 0.60),
        )

        self._rnd = RNDCuriosity() if enable_rnd else None

        # Potential-based shaping from RLAIF reward model
        self._shaping = PotentialShaping(lam=0.01) if enable_shaping else None
        self._reward_model = None
        if enable_shaping and reward_model_path:
            self._load_reward_model(reward_model_path)

        # Tracking
        self._prev_health = 0
        self._prev_rupees = 0
        self._prev_keys = 0
        self._prev_sword = 0
        self._prev_essences = 0
        self._prev_group = 0
        self._prev_dungeon_floor = 0
        self._epoch = epoch
        self._episode_id = ""

        # Progression milestones — track what the agent achieves each episode
        self._milestone_got_sword = False
        self._milestone_entered_dungeon = False
        self._milestone_visited_maku_tree = False
        self._milestone_essences = 0
        self._milestone_dungeon_keys = 0
        self._milestone_max_rooms = 0

        # Episode export (must be after self._epoch is set)
        # Only export from ~10% of environments to reduce MinIO storage usage.
        # The judge only needs 30 segments per epoch — no need for all 100 envs
        # to export.
        self._exporter = None
        self._enable_export = enable_export
        import random
        if enable_export and random.random() < 0.10:
            self._init_exporter(s3_config)

    def _load_reward_model(self, path: str) -> None:
        try:
            import tempfile

            from agent.evaluator.reward_model import RewardModel

            local_path = path
            # If local file doesn't exist, try downloading from MinIO
            if not os.path.exists(local_path):
                try:
                    from agent.utils.s3 import S3Client
                    from agent.utils.config import S3Config

                    s3 = S3Client(S3Config())
                    # path is the S3 key like "reward_model/epoch_0/rm.pt"
                    rm_data = s3.download_bytes("zelda-models", path)
                    os.makedirs("/tmp/reward_model", exist_ok=True)
                    # Use epoch-specific filename to avoid stale cache:
                    # previously all epochs wrote to the same rm.pt and
                    # the exists() check skipped newer models.
                    safe_name = path.replace("/", "_")
                    local_path = f"/tmp/reward_model/{safe_name}"
                    # Atomic write: write to temp file then rename to avoid
                    # race conditions when multiple envs init in same process
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

    def _init_exporter(self, s3_config: dict[str, str] | None) -> None:
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
                png_interval=60,
                epoch=self._epoch,
            )
        except Exception as e:
            logger.warning("Episode export disabled: %s", e)
            self._exporter = None

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)

        # Snapshot baseline state
        self._prev_health = info.get("health", 0)
        self._prev_rupees = 0
        self._prev_keys = 0
        self._prev_sword = self.env._read(SWORD_LEVEL) if hasattr(self.env, "_read") else 0
        self._prev_essences = 0
        self._prev_group = info.get("active_group", 0)
        self._prev_dungeon_floor = info.get("dungeon_floor", 0)

        # Distance tracking — overworld grid is 16 columns wide
        start_room = info.get("room_id", 0)
        self._start_row = start_room // 16
        self._start_col = start_room % 16
        self._max_distance = 0
        self._min_row_reached = self._start_row
        self._max_col_reached = self._start_col

        # Reset sub-modules
        self._coverage.reset()
        self._steps_since_discovery = 0
        self._prev_total_tiles = 0
        self._menu_steps = 0
        self._recent_rooms.clear()
        self._prev_room_id = info.get("room_id", -1)
        self._dialog_rooms.clear()
        self._prev_dialog_active = False
        self._prev_maku_dialog = False
        self._prev_gnarled_key = False

        # Reset milestones for new episode
        self._milestone_got_sword = False
        self._milestone_entered_dungeon = False
        self._milestone_visited_maku_tree = False
        self._milestone_essences = 0
        self._milestone_dungeon_keys = 0
        self._milestone_max_rooms = 0
        if self._shaping:
            self._shaping.reset()

        # Expose visited rooms set to base env for state_encoder access
        self.env._visited_rooms_set = self._coverage._visited_rooms

        # Initialize frontier exit distance (all rooms unvisited at start)
        self._prev_exit_dist = self.env.frontier_exit_dist(self._coverage._visited_rooms)

        # Start episode recording
        if self._exporter:
            self._episode_id = self._exporter.begin_episode()
            info["episode_id"] = self._episode_id

        info["epoch"] = self._epoch
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, _, terminated, truncated, info = self.env.step(action)

        # Track rooms before reward computation for stagnation detection
        prev_rooms = self._coverage.unique_rooms

        reward = self._compute_reward(obs, info, terminated)

        # Tile-based stagnation — reset counter when ANY new tile is found.
        # Room-based stagnation was too aggressive: the agent got truncated
        # while transiting through known rooms to reach new areas.  Tile-based
        # allows transit (new tiles are found even in visited rooms) while
        # still ending episodes where the agent circles the same tiles.
        # Dialog steps don't count toward stagnation — NPC dialog is
        # productive (quest progression) and shouldn't trigger truncation.
        new_rooms = self._coverage.unique_rooms
        new_tiles = self._coverage.total_tiles
        dialog_active = info.get("dialog_active", False)
        if new_tiles > self._prev_total_tiles:
            self._steps_since_discovery = 0
        elif not dialog_active:
            self._steps_since_discovery += 1
        self._prev_total_tiles = new_tiles

        if self._stagnation_limit > 0 and self._steps_since_discovery >= self._stagnation_limit:
            truncated = True

        # Record frame for export
        if self._exporter:
            screen = self.env.render() if hasattr(self.env, "render") else None
            self._exporter.record_frame(
                step=info.get("step", 0),
                state=info,
                action=action,
                reward=reward,
                screen_array=screen,
            )

        # Flush on episode end
        if (terminated or truncated) and self._exporter:
            self._exporter.end_episode()

        # Update milestones from reward computation state
        self._milestone_max_rooms = max(self._milestone_max_rooms, new_rooms)
        if hasattr(self.env, "_read"):
            sword = self.env._read(SWORD_LEVEL)
            if sword > 0:
                self._milestone_got_sword = True
            keys = self.env._read(DUNGEON_KEYS)
            self._milestone_dungeon_keys = max(self._milestone_dungeon_keys, keys)
            essences = bin(self.env._read(ESSENCES_COLLECTED)).count("1")
            self._milestone_essences = max(self._milestone_essences, essences)
        active_group = info.get("active_group", 0)
        if active_group in (4, 5):
            self._milestone_entered_dungeon = True
        if active_group == 2:
            self._milestone_visited_maku_tree = True

        info["epoch"] = self._epoch
        info["stagnation_steps"] = self._steps_since_discovery
        info["unique_rooms"] = new_rooms
        info["unique_tiles"] = self._coverage.total_tiles
        info["seen_coords"] = self._coverage.total_tiles  # Pokemon Red-style coord count
        info["max_distance"] = self._max_distance

        # Progression milestones (available every step, reported at episode end)
        info["milestone_got_sword"] = float(self._milestone_got_sword)
        info["milestone_entered_dungeon"] = float(self._milestone_entered_dungeon)
        info["milestone_visited_maku_tree"] = float(self._milestone_visited_maku_tree)
        info["milestone_essences"] = float(self._milestone_essences)
        info["milestone_dungeon_keys"] = float(self._milestone_dungeon_keys)
        info["milestone_max_rooms"] = float(self._milestone_max_rooms)
        info["milestone_maku_dialog"] = float(self._prev_maku_dialog)
        info["milestone_gnarled_key"] = float(self._prev_gnarled_key)

        return obs, reward, terminated, truncated, info

    def _compute_reward(
        self, obs: np.ndarray, info: dict[str, Any], terminated: bool
    ) -> float:
        reward = 0.0

        # --- Game event rewards ---
        health = info.get("health", 0)
        health_delta = health - self._prev_health
        if health_delta < 0:
            reward += health_delta * abs(self._health_loss_scale)
        self._prev_health = health

        # Death
        if terminated:
            reward += self._death_penalty

        # Time penalty
        reward += self._time_penalty

        # Rupees (read from env if available)
        if hasattr(self.env, "_read16"):
            rupees = self.env._read16(RUPEES)
            rupee_delta = rupees - self._prev_rupees
            if rupee_delta > 0:
                reward += rupee_delta * self._rupee_scale
            self._prev_rupees = rupees

        # Keys
        if hasattr(self.env, "_read"):
            keys = self.env._read(DUNGEON_KEYS)
            if keys > self._prev_keys:
                reward += (keys - self._prev_keys) * self._key_scale
            self._prev_keys = keys

            # Sword upgrade
            sword = self.env._read(SWORD_LEVEL)
            if sword > self._prev_sword:
                reward += self._sword_bonus
            self._prev_sword = sword

            # Essences
            essences = bin(self.env._read(ESSENCES_COLLECTED)).count("1")
            if essences > self._prev_essences:
                reward += self._dungeon_bonus
            self._prev_essences = essences

        # --- Progression rewards ---
        active_group = info.get("active_group", 0)
        dungeon_floor = info.get("dungeon_floor", 0)

        # Dungeon entry bonus: ACTIVE_GROUP changed to 4 or 5
        if active_group in (4, 5) and self._prev_group not in (4, 5):
            reward += self._dungeon_entry_bonus

        # Maku Tree visit bonus: ACTIVE_GROUP changed to 2
        if active_group == 2 and self._prev_group != 2:
            reward += self._maku_tree_visit_bonus

        # Indoor area bonus: ACTIVE_GROUP changed to 3
        if active_group == 3 and self._prev_group != 3:
            reward += self._indoor_entry_bonus

        # Dungeon floor change bonus (deeper exploration)
        if dungeon_floor != self._prev_dungeon_floor and active_group in (4, 5):
            reward += self._dungeon_floor_bonus

        self._prev_group = active_group
        self._prev_dungeon_floor = dungeon_floor

        # Dialog interaction bonus — first dialog trigger per room.
        # Teaches the agent that talking to NPCs leads to quest progression
        # (e.g., Maku Tree gives Gnarled Key needed for Dungeon 1).
        dialog_active = info.get("dialog_active", False)
        room_id = info.get("room_id", 0)
        if dialog_active and not self._prev_dialog_active:
            if room_id not in self._dialog_rooms:
                reward += self._dialog_bonus
                self._dialog_rooms.add(room_id)
        self._prev_dialog_active = dialog_active

        # Maku Tree quest milestones — massive one-time bonuses.
        # These are the critical quest progression gates that unlock Dungeon 1.
        if hasattr(self.env, "_read"):
            # Check GLOBALFLAG_GNARLED_KEY_GIVEN (Maku Tree gave the quest)
            maku_dialog = bool(self.env._read(GNARLED_KEY_GIVEN_FLAG) & GNARLED_KEY_GIVEN_MASK)
            if maku_dialog and not self._prev_maku_dialog:
                reward += self._maku_dialog_bonus
                logger.info("MILESTONE: Maku Tree gave Gnarled Key quest! (+%.0f)", self._maku_dialog_bonus)
            self._prev_maku_dialog = maku_dialog

            # Check TREASURE_GNARLED_KEY obtained (picked up the key item)
            gnarled_key = bool(self.env._read(GNARLED_KEY_OBTAINED) & GNARLED_KEY_OBTAINED_MASK)
            if gnarled_key and not self._prev_gnarled_key:
                reward += self._gnarled_key_bonus
                logger.info("MILESTONE: Gnarled Key obtained! (+%.0f)", self._gnarled_key_bonus)
            self._prev_gnarled_key = gnarled_key

        # --- Menu management ---
        # Allow brief menu access for item switching, but suppress exploration
        # rewards while menu is open and penalize camping
        menu_active = info.get("menu_active", False)
        if menu_active:
            self._menu_steps += 1
            # Penalty after grace period
            if self._menu_steps > self._menu_grace:
                reward += self._menu_penalty
            # Auto-dismiss after max steps
            if self._menu_steps >= self._menu_max and hasattr(self.env, "_dismiss_menu"):
                self.env._dismiss_menu()
        else:
            self._menu_steps = 0

        # --- Exploration rewards (only when NOT in menu) ---
        if not menu_active:
            # Backtrack penalty: penalize re-entering recently visited rooms
            room_id = info.get("room_id", 0)
            if room_id != self._prev_room_id:
                if room_id in self._recent_rooms:
                    reward += self._backtrack_penalty
                self._recent_rooms.append(room_id)
            self._prev_room_id = room_id

            # Exit-seeking shaping: reward moving toward FRONTIER (unvisited) exits
            visited = self._coverage._visited_rooms
            cur_exit_dist = self.env.frontier_exit_dist(visited)
            exit_delta = self._prev_exit_dist - cur_exit_dist
            if exit_delta != 0:
                reward += exit_delta * self._exit_seeking_scale
            self._prev_exit_dist = cur_exit_dist

            # Distance-from-start bonus (overworld only).
            # Non-overworld rooms use different room ID ranges, causing
            # artificial huge Manhattan distances that destabilize training.
            room_id = info.get("room_id", 0)
            cur_row = room_id // 16
            cur_col = room_id % 16
            if active_group == 0:
                distance = abs(cur_row - self._start_row) + abs(cur_col - self._start_col)
                if distance > self._max_distance:
                    self._max_distance = distance
                    reward += self._distance_bonus * distance

                # Directional progress bonus — the Maku Tree is east then
                # north of start. Reward both eastward (higher col) and
                # northward (lower row) progress. Decays over the episode
                # so the agent can head west to Dungeon 1 after the Maku Tree.
                dir_bonus = self._directional_bonus * (
                    self._directional_decay ** self.env.step_count
                )
                if cur_col > self._max_col_reached:
                    cols_east = cur_col - self._max_col_reached
                    reward += dir_bonus * cols_east
                    self._max_col_reached = cur_col
                if cur_row < self._min_row_reached:
                    rows_north = self._min_row_reached - cur_row
                    reward += dir_bonus * rows_north
                    self._min_row_reached = cur_row

            # Coverage reward with area-based boost
            coverage = self._coverage.step(
                info.get("room_id", 0),
                info.get("pixel_x", 0),
                info.get("pixel_y", 0),
            )
            area_mult = self._area_boost.get(active_group, 1.0)
            reward += coverage * area_mult

            # RND curiosity
            if self._rnd is not None:
                curiosity = self._rnd.compute(obs, reward)
                reward += curiosity

        # --- Potential-based shaping ---
        if self._shaping is not None and self._reward_model is not None:
            phi = self._reward_model.predict(obs)
            reward = self._shaping.shape(reward, phi)

        return reward
