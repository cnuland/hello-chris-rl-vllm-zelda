"""Reward wrapper: integrates coverage, game events, and potential shaping.

Wraps ZeldaEnv to compute composite rewards from:
  1. Game event rewards (health, keys, sword, death, milestones)
  2. Coverage rewards (binary tile exploration, new rooms)
  3. LLM advisor directives (seek_room, trigger_action, avoid_region)
  4. Potential-based shaping from RLAIF reward model (when available)

Also handles episode export to MinIO for the judge/eval pipeline.
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

# The Maku Tree gate is on the OVERWORLD (group 0) at room 0xD9 (row=13, col=9).
# When slashed, bit 7 (0x80) is set at OVERWORLD_ROOM_FLAGS + 0xD9 = 0xC7D9.
MAKU_GATE_ROOM = 0xD9
from agent.env.phase_rewards import PhaseManager
from agent.rl.rewards import CoverageReward, PotentialShaping

logger = logging.getLogger(__name__)


class RewardWrapper(gym.Wrapper):
    """Wraps ZeldaEnv with composite reward computation and episode export."""

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

        # Game event reward scales
        self._key_scale = cfg.get("key", 0.5)
        self._death_penalty = cfg.get("death", -1.0)
        self._health_loss_scale = cfg.get("health_loss", -0.005)
        self._sword_bonus = cfg.get("sword", 15.0)
        self._dungeon_bonus = cfg.get("dungeon", 100.0)
        self._maku_tree_bonus = cfg.get("maku_tree", 100.0)

        # Progression reward scales
        self._dungeon_entry_bonus = cfg.get("dungeon_entry", 100.0)
        self._maku_tree_visit_bonus = cfg.get("maku_tree_visit", 100.0)
        self._indoor_entry_bonus = cfg.get("indoor_entry", 5.0)
        self._dungeon_floor_bonus = cfg.get("dungeon_floor", 10.0)

        # Dialog interaction reward — teaches the agent that NPC dialog is
        # valuable (required for quest progression: Maku Tree gives Gnarled Key)
        self._dialog_bonus = cfg.get("dialog_bonus", 10.0)
        self._dialog_rooms: set[int] = set()
        self._prev_dialog_active = False

        # Dialog progression reward — pressing A while dialog is active
        # advances text boxes.  This is critical for quest progression:
        # the Maku Tree has a multi-box dialog that gives the Gnarled Key
        # quest.  Without this, the agent starts dialog but never finishes.
        self._dialog_advance_bonus = cfg.get("dialog_advance", 25.0)
        self._dialog_advance_count = 0
        self._dialog_advance_cap = 20  # max rewarded A-presses per episode
        self._prev_dialog_value = 0    # raw DIALOG_STATE byte for tracking

        # Maku Tree quest milestone rewards — MASSIVE one-time bonuses for
        # critical quest progression. These must dominate room discovery (50/room)
        # so the agent locks on once it finds the Maku Tree path.
        self._maku_dialog_bonus = cfg.get("maku_dialog", 500.0)
        self._gnarled_key_bonus = cfg.get("gnarled_key", 500.0)
        self._maku_seed_bonus = cfg.get("maku_seed", 1000.0)
        self._prev_maku_dialog = False
        self._prev_gnarled_key = False
        self._prev_maku_seed = False

        # Maku Tree sub-event rewards — intermediate milestones that guide
        # the agent through the required interaction sequence:
        #   1. Slash the gate (room flag 0x80)  →  gate_slash_bonus
        #   2. Reach new rooms in group 2       →  maku_room_bonus (per room)
        #   3. Maku Tree stage changes (0xCC39) →  maku_stage_bonus
        self._gate_slash_bonus = cfg.get("gate_slash", 250.0)
        self._maku_room_bonus = cfg.get("maku_room", 100.0)
        self._maku_stage_bonus = cfg.get("maku_stage", 300.0)
        self._gate_slashed = False
        self._maku_rooms_visited: set[int] = set()
        self._prev_maku_stage = 0

        # Pixel position tracking — used to gate coverage reward behind
        # actual movement (prevents standing-still farming)
        self._prev_pixel_x = 0
        self._prev_pixel_y = 0

        # Milestone-triggered flag for this step
        self._milestone_achieved_this_step = False

        # Exit-seeking reward — per-step gradient toward room exits that
        # lead to unvisited rooms.  Uses frontier_exit_dist() from ZeldaEnv
        # which returns Manhattan distance to nearest walkable exit tile
        # leading to an unvisited neighbor room.  Reward = scale * (prev - cur),
        # positive when getting closer, negative when moving away.
        self._exit_seeking_scale = cfg.get("exit_seeking", 0.0)
        self._prev_frontier_dist = 0
        self._prev_room_for_exit = -1

        # Directional bonus — default OFF (0.0).  The LLM reward advisor
        # can enable it by setting directional_bonus > 0 in the config
        # along with a target room (row, col).  This is the key LLM-driven
        # reward signal: the advisor analyzes agent behavior and decides
        # where to direct exploration.
        self._directional_bonus = cfg.get("directional_bonus", 0.0)
        self._directional_target_row = int(cfg.get("directional_target_row",
                                                    os.getenv("DIRECTIONAL_TARGET_ROW", "5")))
        self._directional_target_col = int(cfg.get("directional_target_col",
                                                    os.getenv("DIRECTIONAL_TARGET_COL", "12")))
        self._directional_target_scale = float(cfg.get("directional_target_scale", "1.0"))
        self._min_target_distance = 999

        # Post-Gnarled-Key phase: once the agent has the key, suppress all
        # Maku Tree farming rewards and activate directional guidance toward
        # Dungeon 1 at (row=10, col=4).
        self._post_key_directional_activated = False
        self._maku_loiter_penalty = cfg.get("maku_loiter_penalty", 1.0)

        # Phase-driven reward management — replaces scattered
        # ``if has_gnarled_key_now:`` checks with declarative profiles.
        self._phase_manager = PhaseManager()
        phase_overrides = cfg.get("phase_overrides", {})
        for phase_name, overrides in phase_overrides.items():
            self._phase_manager.merge_advisor_overrides(phase_name, overrides)

        # Snow region milestone — massive one-time bonus for reaching the
        # snowy northwest area (path to Dungeon 1 / Gnarled Root).
        # Only fires after Gnarled Key is obtained.
        self._snow_region_bonus = cfg.get("snow_region", 0.0)
        self._snow_region_max_row = int(cfg.get("snow_region_max_row",
                                                 os.getenv("SNOW_REGION_MAX_ROW", "11")))
        self._snow_region_max_col = int(cfg.get("snow_region_max_col",
                                                 os.getenv("SNOW_REGION_MAX_COL", "5")))
        self._entered_snow_region = False

        # Structured directives from the LLM reward advisor — processed as
        # one-time bonuses or per-step penalties based on area/action conditions.
        self._directives = cfg.get("directives", [])
        self._triggered_directives: set[str] = set()

        # Sword interaction bonus — per-room reward for pressing A near
        # obstacles or in key areas.  Default off (0.0).
        self._sword_use_bonus = cfg.get("sword_use", 0.0)
        self._a_press_rooms: set[tuple[int, int]] = set()
        self._current_button = 0  # ButtonAction.A=0, ButtonAction.B=1

        # Area-based exploration boost — multiplies coverage reward by
        # active_group to guide exploration toward key progression areas.
        self._area_boost = {
            0: cfg.get("area_boost_overworld", 1.0),
            1: cfg.get("area_boost_subrosia", 1.5),
            2: cfg.get("area_boost_maku", 3.0),
            3: cfg.get("area_boost_indoors", 1.5),
            4: cfg.get("area_boost_dungeon", 2.0),
            5: cfg.get("area_boost_dungeon", 2.0),
        }

        # Sub-reward modules
        self._coverage = CoverageReward(
            bonus_per_tile=cfg.get("grid_exploration", 0.1),
            bonus_per_room=cfg.get("new_room", 10.0),
        )
        self._cumulative_coverage_reward = 0.0

        # Potential-based shaping from RLAIF reward model (lambda decays with epoch)
        self._shaping = PotentialShaping(lam=0.01, epoch=epoch) if enable_shaping else None
        self._reward_model = None
        if enable_shaping and reward_model_path:
            self._load_reward_model(reward_model_path)

        # Tracking
        self._prev_health = 0
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
        self._milestone_entered_snow_region = False
        self._milestone_essences = 0
        self._milestone_dungeon_keys = 0
        self._milestone_max_rooms = 0

        # Advancing checkpoint: capture PyBoy save states at milestone moments.
        self._milestone_state_dir = os.getenv("MILESTONE_STATE_DIR", "")
        self._captured_milestones: set[str] = set()

        # Episode export
        self._exporter = None
        self._enable_export = enable_export
        if enable_export:
            import random
            export_prob = max(0.10, 1.0 / max(int(os.getenv("NUM_ENVS", "1")), 1))
            if random.random() < export_prob:
                self._init_exporter(s3_config)

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

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)

        # Snapshot baseline state from actual RAM
        self._prev_health = info.get("health", 0)
        self._prev_group = info.get("active_group", 0)
        self._prev_dungeon_floor = info.get("dungeon_floor", 0)
        if hasattr(self.env, "_read"):
            self._prev_keys = self.env._read(DUNGEON_KEYS)
            self._prev_sword = self.env._read(SWORD_LEVEL)
            self._prev_essences = bin(self.env._read(ESSENCES_COLLECTED)).count("1")
            self._prev_maku_dialog = bool(
                self.env._read(GNARLED_KEY_GIVEN_FLAG) & GNARLED_KEY_GIVEN_MASK
            )
            self._prev_gnarled_key = bool(
                self.env._read(GNARLED_KEY_OBTAINED) & GNARLED_KEY_OBTAINED_MASK
            )
            self._prev_maku_seed = bool(
                self.env._read(MAKU_SEED_FLAG) & MAKU_SEED_MASK
            )
            self._prev_maku_stage = self.env._read(MAKU_TREE_STAGE)
        else:
            self._prev_keys = 0
            self._prev_sword = 0
            self._prev_essences = 0
            self._prev_maku_dialog = False
            self._prev_gnarled_key = False
            self._prev_maku_seed = False

        # Reset sub-modules (per-episode)
        self._coverage.reset()
        self._cumulative_coverage_reward = 0.0
        self._dialog_rooms.clear()
        self._prev_dialog_active = False
        self._dialog_advance_count = 0
        self._prev_dialog_value = 0
        self._prev_pixel_x = info.get("pixel_x", 0)
        self._prev_pixel_y = info.get("pixel_y", 0)

        # Exit-seeking tracking — reset on episode start
        self._prev_frontier_dist = 0
        self._prev_room_for_exit = -1

        # Directional bonus tracking — initialize from starting position
        self._start_room_id = info.get("room_id", 0)
        self._start_group = info.get("active_group", 0)
        start_row = self._start_room_id // 16
        start_col = self._start_room_id % 16
        self._min_target_distance = (abs(start_row - self._directional_target_row) +
                                     abs(start_col - self._directional_target_col))

        # Reset triggered directives for new episode
        self._triggered_directives.clear()

        # Log starting position once per epoch for debugging
        if not hasattr(self, "_logged_start"):
            logger.info(
                "Episode start: room_id=%d (row=%d, col=%d), group=%d, "
                "pixel=(%d,%d), sword=%d",
                self._start_room_id,
                self._start_room_id // 16, self._start_room_id % 16,
                info.get("active_group", -1),
                info.get("pixel_x", 0), info.get("pixel_y", 0),
                self._prev_sword,
            )
            self._logged_start = True

        # Reset milestones for new episode
        self._milestone_got_sword = False
        self._milestone_entered_dungeon = False
        self._milestone_visited_maku_tree = False
        self._milestone_entered_snow_region = False
        self._milestone_essences = 0
        self._milestone_dungeon_keys = 0
        self._milestone_max_rooms = 0
        self._baseline_sword = self._prev_sword
        self._baseline_essences = self._prev_essences
        self._baseline_maku_dialog = self._prev_maku_dialog
        self._baseline_gnarled_key = self._prev_gnarled_key
        self._baseline_maku_seed = self._prev_maku_seed
        self._baseline_maku_stage = self._prev_maku_stage
        self._baseline_group = self._prev_group
        self._gate_slashed = False
        self._entered_snow_region = False
        self._maku_rooms_visited.clear()
        self._captured_milestones.clear()
        self._a_press_rooms.clear()
        self._current_button = 0
        self._post_key_directional_activated = False

        # Detect initial phase from save state baseline and apply
        # the profile's directional target if applicable.
        initial_phase = self._phase_manager.reset(
            sword_level=self._prev_sword,
            has_gnarled_key=self._prev_gnarled_key,
            active_group=self._prev_group,
            baseline_sword=self._baseline_sword,
            baseline_gnarled_key=self._baseline_gnarled_key,
        )
        profile = self._phase_manager.active_profile
        if profile.directional_target is not None:
            self._directional_target_row = profile.directional_target[0]
            self._directional_target_col = profile.directional_target[1]
            if profile.directional_bonus > 0 and self._directional_bonus == 0.0:
                self._directional_bonus = profile.directional_bonus

        if self._shaping:
            self._shaping.reset()

        # Expose visited rooms set to base env for state_encoder access.
        self.env._visited_rooms_set = self._coverage._visited_rooms

        # Start episode recording
        if self._exporter:
            self._episode_id = self._exporter.begin_episode()
            info["episode_id"] = self._episode_id

        info["epoch"] = self._epoch
        return obs, info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Extract button action for sword interaction bonus
        action_arr = np.asarray(action).flatten()
        self._current_button = int(action_arr[1]) if len(action_arr) > 1 else 0

        obs, _, terminated, truncated, info = self.env.step(action)

        self._milestone_achieved_this_step = False
        reward = self._compute_reward(obs, info, terminated)

        # Log new room discoveries
        new_rooms = self._coverage.unique_rooms
        room_id = info.get("room_id", 0)
        active_group = info.get("active_group", 0)
        if new_rooms > getattr(self, "_prev_logged_rooms", 0):
            logger.info(
                "NEW ROOM: group=%d room=%d (row=%d, col=%d) | total=%d | bonus=%.1f",
                active_group, room_id, room_id // 16, room_id % 16,
                new_rooms, self._coverage.bonus_per_room,
            )
        self._prev_logged_rooms = new_rooms

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

        # Update milestones
        self._milestone_max_rooms = max(self._milestone_max_rooms, new_rooms)
        if hasattr(self.env, "_read"):
            sword = self.env._read(SWORD_LEVEL)
            if sword > self._baseline_sword:
                self._milestone_got_sword = True
            keys = self.env._read(DUNGEON_KEYS)
            self._milestone_dungeon_keys = max(self._milestone_dungeon_keys, keys)
            essences = bin(self.env._read(ESSENCES_COLLECTED)).count("1")
            self._milestone_essences = max(self._milestone_essences, essences)
        if active_group in (4, 5) and self._baseline_group not in (4, 5):
            self._milestone_entered_dungeon = True
        if active_group == 2 and self._baseline_group != 2:
            self._milestone_visited_maku_tree = True

        info["epoch"] = self._epoch
        info["unique_rooms"] = new_rooms
        info["unique_tiles"] = self._coverage.total_tiles
        info["seen_coords"] = self._coverage.total_tiles
        info["current_phase"] = self._phase_manager.current_phase

        # Progression milestones
        info["milestone_got_sword"] = float(self._milestone_got_sword)
        info["milestone_entered_dungeon"] = float(self._milestone_entered_dungeon)
        info["milestone_visited_maku_tree"] = float(self._milestone_visited_maku_tree)
        info["milestone_essences"] = float(self._milestone_essences)
        info["milestone_dungeon_keys"] = float(self._milestone_dungeon_keys)
        info["milestone_max_rooms"] = float(self._milestone_max_rooms)
        info["milestone_maku_dialog"] = float(
            self._prev_maku_dialog and not self._baseline_maku_dialog
        )
        info["milestone_gnarled_key"] = float(
            self._prev_gnarled_key and not self._baseline_gnarled_key
        )
        info["milestone_maku_seed"] = float(
            self._prev_maku_seed and not self._baseline_maku_seed
        )
        info["milestone_gate_slashed"] = float(self._gate_slashed)
        info["milestone_entered_snow_region"] = float(self._milestone_entered_snow_region)
        info["milestone_maku_rooms"] = float(len(self._maku_rooms_visited))
        info["milestone_maku_stage"] = float(
            self._prev_maku_stage > self._baseline_maku_stage
        )

        # Save-state baseline inventory
        info["baseline_has_sword"] = float(self._baseline_sword > 0)
        info["baseline_has_maku_dialog"] = float(self._baseline_maku_dialog)
        info["baseline_has_gnarled_key"] = float(self._baseline_gnarled_key)
        info["baseline_has_maku_seed"] = float(self._baseline_maku_seed)
        info["baseline_maku_stage"] = float(self._baseline_maku_stage)
        info["baseline_in_maku"] = float(self._baseline_group == 2)
        info["baseline_in_dungeon"] = float(self._baseline_group in (4, 5))

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
                self._milestone_achieved_this_step = True
                self._capture_milestone_state("got_sword", reward)
            self._prev_sword = sword

            # Essences
            essences = bin(self.env._read(ESSENCES_COLLECTED)).count("1")
            if essences > self._prev_essences:
                reward += self._dungeon_bonus
                self._milestone_achieved_this_step = True
            self._prev_essences = essences

        # --- Progression rewards ---
        active_group = info.get("active_group", 0)
        dungeon_floor = info.get("dungeon_floor", 0)

        # Post-Gnarled-Key reward gating: once the agent has the key,
        # suppress ALL Maku Tree farming rewards so the agent pivots
        # toward Dungeon 1 instead of looping between gate and tree.
        has_gnarled_key_now = False
        if hasattr(self.env, "_read"):
            has_gnarled_key_now = bool(
                self.env._read(GNARLED_KEY_OBTAINED) & GNARLED_KEY_OBTAINED_MASK
            )

        # Dungeon entry bonus
        if active_group in (4, 5) and self._prev_group not in (4, 5):
            reward += self._dungeon_entry_bonus
            self._milestone_achieved_this_step = True
            self._capture_milestone_state("entered_dungeon", reward)

        # Maku Tree visit bonus — suppressed in post-key phases
        if (active_group == 2 and self._prev_group != 2
                and not self._phase_manager.is_reward_suppressed("maku_tree_visit")):
            reward += self._maku_tree_visit_bonus
            self._milestone_achieved_this_step = True
            self._capture_milestone_state("visited_maku_tree", reward)

        # Indoor area bonus
        if active_group == 3 and self._prev_group != 3:
            reward += self._indoor_entry_bonus

        # Dungeon floor change bonus
        if dungeon_floor != self._prev_dungeon_floor and active_group in (4, 5):
            reward += self._dungeon_floor_bonus

        self._prev_group = active_group
        self._prev_dungeon_floor = dungeon_floor

        # Dialog interaction bonus — first dialog trigger per room
        dialog_active = info.get("dialog_active", False)
        room_id = info.get("room_id", 0)
        if dialog_active and not self._prev_dialog_active:
            if room_id not in self._dialog_rooms:
                reward += self._dialog_bonus
                self._dialog_rooms.add(room_id)
        self._prev_dialog_active = dialog_active

        # Dialog progression reward — reward A-presses during active dialog,
        # but ONLY in quest-relevant areas (Maku Tree group 2, indoors group 3).
        # This prevents farming random overworld NPCs while teaching the agent
        # to press A to advance through the Maku Tree's multi-box dialog
        # (required to get the Gnarled Key quest).
        if hasattr(self.env, "_read"):
            dialog_value = self.env._read(DIALOG_STATE)
            # Phase-driven dialog area restriction — each phase defines which
            # area groups qualify for dialog advance rewards.
            in_quest_area = active_group in self._phase_manager.get_dialog_advance_groups()
            if (dialog_active and in_quest_area
                    and self._dialog_advance_count < self._dialog_advance_cap):
                # Reward A-presses during dialog — directly teaches "press A to advance text"
                if self._current_button == ButtonAction.A:
                    reward += self._dialog_advance_bonus
                    self._dialog_advance_count += 1
                    logger.info(
                        "DIALOG ADVANCE: A-press in group=%d room=%d count=%d/%d (+%.0f)",
                        active_group, room_id, self._dialog_advance_count,
                        self._dialog_advance_cap, self._dialog_advance_bonus,
                    )
                # Also reward dialog value changes (text state transitions)
                elif (dialog_value != self._prev_dialog_value
                      and self._prev_dialog_value != 0
                      and dialog_value != 0):
                    reward += self._dialog_advance_bonus * 0.5
                    self._dialog_advance_count += 1
                    logger.info(
                        "DIALOG STATE CHANGE: group=%d 0x%02X → 0x%02X room=%d (+%.0f)",
                        active_group, self._prev_dialog_value, dialog_value,
                        room_id, self._dialog_advance_bonus * 0.5,
                    )
            self._prev_dialog_value = dialog_value
        else:
            self._prev_dialog_value = 0

        # Maku Tree quest milestones — massive one-time bonuses
        if hasattr(self.env, "_read"):
            maku_dialog = bool(self.env._read(GNARLED_KEY_GIVEN_FLAG) & GNARLED_KEY_GIVEN_MASK)
            if maku_dialog and not self._prev_maku_dialog:
                reward += self._maku_dialog_bonus
                self._milestone_achieved_this_step = True
                logger.info("MILESTONE: Maku Tree gave Gnarled Key quest! (+%.0f)", self._maku_dialog_bonus)
                self._capture_milestone_state("maku_dialog", reward)
            self._prev_maku_dialog = maku_dialog

            gnarled_key = bool(self.env._read(GNARLED_KEY_OBTAINED) & GNARLED_KEY_OBTAINED_MASK)
            if gnarled_key and not self._prev_gnarled_key:
                reward += self._gnarled_key_bonus
                self._milestone_achieved_this_step = True
                logger.info("MILESTONE: Gnarled Key obtained! (+%.0f)", self._gnarled_key_bonus)
                self._capture_milestone_state("gnarled_key", reward)
            self._prev_gnarled_key = gnarled_key

            maku_seed = bool(self.env._read(MAKU_SEED_FLAG) & MAKU_SEED_MASK)
            if maku_seed and not self._prev_maku_seed:
                reward += self._maku_seed_bonus
                self._milestone_achieved_this_step = True
                logger.info("MILESTONE: Maku Seed obtained! (+%.0f)", self._maku_seed_bonus)
            self._prev_maku_seed = maku_seed

        # --- Gate slash detection (overworld group 0, room 0xD9) ---
        # The Maku Tree gate is on the OVERWORLD — check the overworld room
        # flags at a fixed address, not the Maku/Subrosia group flags.
        if hasattr(self.env, "_read") and not self._gate_slashed:
            gate_flags = self.env._read(OVERWORLD_ROOM_FLAGS + MAKU_GATE_ROOM)
            if gate_flags & ROOMFLAG_GATE_HIT:
                self._gate_slashed = True
                reward += self._gate_slash_bonus
                self._milestone_achieved_this_step = True
                logger.info(
                    "MILESTONE: Maku Tree gate slashed! "
                    "addr=0x%04X flags=0x%02X (+%.0f)",
                    OVERWORLD_ROOM_FLAGS + MAKU_GATE_ROOM,
                    gate_flags, self._gate_slash_bonus,
                )
                self._capture_milestone_state("gate_slashed", reward)

        # --- Maku Tree sub-event rewards (group 2 interior) ---
        # Phase-driven suppression replaces hardcoded has_gnarled_key_now check.
        if active_group == 2 and hasattr(self.env, "_read"):
            room_id = info.get("room_id", 0)

            # New rooms within group 2 — suppressed in post-key phases
            if (room_id not in self._maku_rooms_visited
                    and not self._phase_manager.is_reward_suppressed("maku_room")):
                self._maku_rooms_visited.add(room_id)
                reward += self._maku_room_bonus
                self._milestone_achieved_this_step = True
                logger.info(
                    "MILESTONE: New Maku Tree room %d (total: %d) (+%.0f)",
                    room_id, len(self._maku_rooms_visited), self._maku_room_bonus,
                )

            # Maku Tree stage change — suppressed in post-key phases
            maku_stage = self.env._read(MAKU_TREE_STAGE)
            if (maku_stage != self._prev_maku_stage
                    and maku_stage > self._baseline_maku_stage
                    and not self._phase_manager.is_reward_suppressed("maku_stage")):
                reward += self._maku_stage_bonus
                self._milestone_achieved_this_step = True
                logger.info(
                    "MILESTONE: Maku Tree stage %d → %d (+%.0f)",
                    self._prev_maku_stage, maku_stage, self._maku_stage_bonus,
                )
            self._prev_maku_stage = maku_stage

        # --- Phase-driven loiter penalty ---
        # Replaces hardcoded ``if has_gnarled_key_now and active_group == 2``
        # with profile-based penalty lookup.  Falls back to env-var-configured
        # _maku_loiter_penalty for backward compat.
        phase_loiter = self._phase_manager.get_loiter_penalty(active_group)
        if phase_loiter > 0:
            reward -= phase_loiter
        elif has_gnarled_key_now and active_group == 2 and self._maku_loiter_penalty > 0:
            # Legacy fallback when loiter_penalties not set in profile
            reward -= self._maku_loiter_penalty

        # --- Snow region milestone (post-Gnarled-Key) ---
        # Massive one-time bonus for reaching the snowy northwest area,
        # which is on the path to Dungeon 1 / Gnarled Root.
        if (has_gnarled_key_now
                and self._snow_region_bonus > 0
                and not self._entered_snow_region
                and active_group == 0):
            room_id = info.get("room_id", 0)
            cur_row = room_id // 16
            cur_col = room_id % 16
            if cur_row <= self._snow_region_max_row and cur_col <= self._snow_region_max_col:
                self._entered_snow_region = True
                self._milestone_entered_snow_region = True
                self._milestone_achieved_this_step = True
                reward += self._snow_region_bonus
                logger.info(
                    "MILESTONE: Entered snow region at room (%d,%d)! (+%.0f)",
                    cur_row, cur_col, self._snow_region_bonus,
                )
                self._capture_milestone_state("entered_snow_region", reward)

        # --- Phase re-detection on milestone events ---
        # When any milestone fires, re-detect the game phase and apply
        # the new profile's directional target if it changed.
        if self._milestone_achieved_this_step:
            phase_changed = self._phase_manager.update_phase(
                sword_level=self._prev_sword,
                has_gnarled_key=has_gnarled_key_now,
                active_group=active_group,
                entered_snow_region=self._entered_snow_region,
                baseline_sword=self._baseline_sword,
                baseline_gnarled_key=self._baseline_gnarled_key,
                step=info.get("step", 0),
            )
            if phase_changed:
                profile = self._phase_manager.active_profile
                if profile.directional_target is not None:
                    self._directional_target_row = profile.directional_target[0]
                    self._directional_target_col = profile.directional_target[1]
                    if profile.directional_bonus > 0 and self._directional_bonus == 0.0:
                        self._directional_bonus = profile.directional_bonus
                    # Reset min distance for new target
                    room_id = info.get("room_id", 0)
                    self._min_target_distance = (
                        abs(room_id // 16 - self._directional_target_row)
                        + abs(room_id % 16 - self._directional_target_col)
                    )

        # --- Sword interaction bonus ---
        if self._sword_use_bonus > 0 and self._current_button == ButtonAction.A:
            room_key = (active_group, room_id)
            if room_key not in self._a_press_rooms:
                self._a_press_rooms.add(room_key)
                bonus = self._sword_use_bonus
                if active_group == 2:
                    bonus *= 3.0
                elif hasattr(self.env, "_read"):
                    pushing = self.env._read(LINK_PUSHING_DIRECTION)
                    if pushing != 0xFF and not info.get("transitioning", False):
                        bonus *= 2.0
                reward += bonus

        # --- LLM Advisor Directives ---
        for directive in self._directives:
            dtype = directive.get("type")
            if dtype == "seek_room":
                tgt_group = directive.get("target_group")
                bonus = directive.get("bonus", 0)
                key = f"seek_{tgt_group}"
                if (tgt_group is not None and active_group == tgt_group
                        and key not in self._triggered_directives):
                    reward += bonus
                    self._triggered_directives.add(key)
                    logger.info("DIRECTIVE: seek_room group=%d triggered (+%.1f)", tgt_group, bonus)
            elif dtype == "trigger_action":
                condition = directive.get("condition", "")
                bonus = directive.get("bonus", 0)
                if condition == "dialog_in_group_2" and dialog_active and active_group == 2:
                    if "dialog_g2" not in self._triggered_directives:
                        reward += bonus
                        self._triggered_directives.add("dialog_g2")
                        logger.info("DIRECTIVE: dialog_in_group_2 triggered (+%.1f)", bonus)
            elif dtype == "avoid_region":
                tgt_group = directive.get("target_group")
                if tgt_group is not None and active_group == tgt_group:
                    reward += directive.get("bonus", 0)

        # --- Exploration rewards (only when NOT transitioning) ---
        is_transitioning = info.get("transitioning", False)
        if not is_transitioning:
            # Coverage reward with area-based boost — only when moved
            cur_pixel_x = info.get("pixel_x", 0)
            cur_pixel_y = info.get("pixel_y", 0)
            actually_moved = (cur_pixel_x != self._prev_pixel_x or
                              cur_pixel_y != self._prev_pixel_y)
            if actually_moved:
                raw_room = info.get("room_id", 0)
                if active_group == 0:
                    qualified_room = raw_room
                else:
                    qualified_room = active_group * 256 + raw_room
                coverage = self._coverage.step(
                    qualified_room,
                    cur_pixel_x,
                    cur_pixel_y,
                )
                area_mult = self._area_boost.get(active_group, 1.0)
                coverage_reward = coverage * area_mult

                # Phase-driven coverage cap — prevents exploration reward
                # from drowning out milestone rewards (gate slash, etc.)
                cap = self._phase_manager.active_profile.coverage_reward_cap
                if cap is not None:
                    remaining = max(0.0, cap - self._cumulative_coverage_reward)
                    coverage_reward = min(coverage_reward, remaining)
                self._cumulative_coverage_reward += coverage_reward

                reward += coverage_reward
            # --- Exit-seeking reward (guides agent toward frontier exits) ---
            # Provides a per-step gradient toward room exits that lead to
            # unvisited rooms.  Only fires when the agent actually moves and
            # stays in the same room (room transitions reset the baseline).
            if self._exit_seeking_scale > 0 and actually_moved:
                cur_frontier_dist = self.env.frontier_exit_dist(
                    self._coverage._visited_rooms
                )
                cur_room = info.get("room_id", 0)
                if (self._prev_room_for_exit == cur_room
                        and self._prev_frontier_dist > 0):
                    dist_delta = self._prev_frontier_dist - cur_frontier_dist
                    reward += dist_delta * self._exit_seeking_scale
                self._prev_frontier_dist = cur_frontier_dist
                self._prev_room_for_exit = cur_room

            self._prev_pixel_x = cur_pixel_x
            self._prev_pixel_y = cur_pixel_y

        # --- Directional bonus (LLM-controlled, default OFF) ---
        # Only active when the LLM advisor sets directional_bonus > 0.
        if not is_transitioning and self._directional_bonus > 0:
            room_id = info.get("room_id", 0)
            cur_row = room_id // 16
            cur_col = room_id % 16
            target_dist = (abs(cur_row - self._directional_target_row) +
                          abs(cur_col - self._directional_target_col))
            if target_dist < self._min_target_distance:
                delta = self._min_target_distance - target_dist
                reward += delta * self._directional_bonus * self._directional_target_scale
                self._min_target_distance = target_dist

        # --- Potential-based shaping ---
        if self._shaping is not None and self._reward_model is not None:
            phi = self._reward_model.predict(obs)
            reward = self._shaping.shape(reward, phi)

        return reward
