"""Reward wrapper: integrates coverage, game events, RND, and potential shaping.

Wraps ZeldaEnv to compute composite rewards from:
  1. Game event rewards (health, rupees, keys, rooms, sword, death)
  2. Coverage rewards (tile exploration, new rooms)
  3. RND curiosity bonus (clamped)
  4. Potential-based shaping from RLAIF reward model (when available)

Also handles episode export to MinIO for the judge/eval pipeline.
"""

from __future__ import annotations

import io
import json
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
    LINK_PUSHING_DIRECTION,
    MAKU_ROOM_FLAGS,
    MAKU_SEED_FLAG,
    MAKU_SEED_MASK,
    MAKU_TREE_STAGE,
    ROOMFLAG_GATE_HIT,
    ROOMFLAG_ITEM_OBTAINED,
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
        self._wall_collision_penalty = cfg.get("wall_collision", -0.01)
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
        # critical quest progression (oracles-disasm confirmed addresses).
        # These must be LARGE relative to exploration rewards (~300/episode)
        # to incentivize the specific actions needed at the Maku Tree:
        # slash gate → enter grove → pop bubble → dialog → pick up key.
        self._maku_dialog_bonus = cfg.get("maku_dialog", 100.0)
        self._gnarled_key_bonus = cfg.get("gnarled_key", 100.0)
        self._maku_seed_bonus = cfg.get("maku_seed", 200.0)
        self._prev_maku_dialog = False
        self._prev_gnarled_key = False
        self._prev_maku_seed = False

        # Maku Tree sub-event rewards — intermediate milestones that guide
        # the agent through the required interaction sequence:
        #   1. Slash the gate (room flag 0x80)  →  gate_slash_bonus
        #   2. Reach new rooms in group 2       →  maku_room_bonus (per room)
        #   3. Maku Tree stage changes (0xCC39) →  maku_stage_bonus
        # These bridge the gap between "entered Maku Tree area" and "got key."
        self._gate_slash_bonus = cfg.get("gate_slash", 50.0)
        self._maku_room_bonus = cfg.get("maku_room", 15.0)
        self._maku_stage_bonus = cfg.get("maku_stage", 75.0)
        self._gate_slashed = False
        self._maku_rooms_visited: set[int] = set()
        self._prev_maku_stage = 0

        # Stagnation-based truncation — end episode early if agent hasn't
        # discovered any new TILES for this many steps.  Tile-based (not
        # room-based) so the agent can transit through known rooms.
        self._stagnation_limit = stagnation_limit
        self._steps_since_discovery = 0
        self._prev_total_tiles = 0

        # Pixel position tracking — used to gate coverage reward behind
        # actual movement (prevents standing-still and wall-bump farming)
        self._prev_pixel_x = 0
        self._prev_pixel_y = 0

        # Milestone-triggered stagnation reset — when a milestone fires,
        # reset the stagnation counter so the agent can backtrack through
        # known territory without being truncated.
        self._milestone_achieved_this_step = False

        # Menu management — allow brief menu use for item switching,
        # but penalize camping and suppress exploration rewards while open
        self._menu_steps = 0
        self._menu_grace = 30     # Steps allowed in menu without penalty
        self._menu_max = 60       # Auto-dismiss menu after this many steps
        self._menu_penalty = -0.05 # Per-step penalty after grace period

        # Exit-seeking shaping — continuous reward for moving toward FRONTIER exits
        self._exit_seeking_scale = cfg.get("exit_seeking", 0.5)

        # Distance bonus — reward for moving far from the starting position.
        # Critical for multi-room exploration: without this, the agent has no
        # incentive to push through rooms.  Uses Manhattan distance in the
        # 16×16 room grid (row, col) from the starting room.
        self._distance_bonus = cfg.get("distance_bonus", 5.0)
        self._start_room_id = -1
        self._max_distance_achieved = 0

        # Directional bonus — reward for moving toward a configurable target
        # room on the 16×16 grid.  Uses Manhattan distance reduction: each
        # new minimum distance from the target earns a one-time bonus.
        # Default target = Maku Tree area (row=5, col=12), configurable by
        # the LLM reward advisor each epoch.
        self._directional_bonus = cfg.get("directional_bonus", 10.0)
        self._directional_target_row = int(cfg.get("directional_target_row",
                                                    os.getenv("DIRECTIONAL_TARGET_ROW", "5")))
        self._directional_target_col = int(cfg.get("directional_target_col",
                                                    os.getenv("DIRECTIONAL_TARGET_COL", "12")))
        self._directional_target_scale = float(cfg.get("directional_target_scale", "1.0"))
        self._min_target_distance = 999
        self._prev_exit_dist = 0

        # Structured directives from the LLM reward advisor — processed as
        # one-time bonuses or per-step penalties based on area/action conditions.
        self._directives = cfg.get("directives", [])
        self._triggered_directives: set[str] = set()

        # Sword interaction bonus — per-room reward for pressing A near
        # obstacles or in key areas.  Prevents button entropy collapse by
        # giving the agent a gradient signal that "A button does something."
        # Capped to once per (group, room) to prevent spamming.
        self._sword_use_bonus = cfg.get("sword_use", 0.5)
        self._a_press_rooms: set[tuple[int, int]] = set()
        self._current_button = 0  # 0=NOP, 1=A, 2=B

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

        # Backtrack penalty — discourage re-entering recently visited rooms.
        # Default 0.0: disabled because -0.3 made room transitions net-negative,
        # causing the policy to converge on "never leave the current room."
        # Coverage's 1/sqrt(N) diminishing returns already naturally discourages
        # revisiting the same tiles without penalizing necessary backtracking.
        self._recent_rooms: deque[int] = deque(maxlen=5)
        self._backtrack_penalty = cfg.get("backtrack_penalty", 0.0)
        self._prev_room_id = -1

        # Sub-reward modules
        self._coverage = CoverageReward(
            bonus_per_tile=cfg.get("grid_exploration", 0.1),
            bonus_per_room=cfg.get("new_room", 10.0),
        )

        self._rnd = RNDCuriosity() if enable_rnd else None

        # Potential-based shaping from RLAIF reward model (lambda decays with epoch)
        self._shaping = PotentialShaping(lam=0.01, epoch=epoch) if enable_shaping else None
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

        # Advancing checkpoint: capture PyBoy save states at milestone moments.
        # Workers write .state files to MILESTONE_STATE_DIR; the training loop
        # reads them at epoch end to decide whether to advance the save state.
        self._milestone_state_dir = os.getenv("MILESTONE_STATE_DIR", "")
        self._captured_milestones: set[str] = set()

        # Episode export (must be after self._epoch is set)
        # Only export from a fraction of environments to reduce MinIO storage.
        # The judge only needs 30 segments per epoch — no need for all envs
        # to export.  Use env ID (hash of id(self)) to deterministically pick
        # exporters so at least one env always exports, even with small counts.
        self._exporter = None
        self._enable_export = enable_export
        if enable_export:
            import random
            # Guarantee at least 1 exporter: use max(10%, 1/num_envs) probability.
            # For <=10 envs every env exports; for 24 envs ~10%; for 100 envs ~10%.
            export_prob = max(0.10, 1.0 / max(int(os.getenv("NUM_ENVS", "1")), 1))
            if random.random() < export_prob:
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

    def _capture_milestone_state(self, milestone_name: str, reward_so_far: float) -> None:
        """Capture PyBoy emulator state when a milestone fires.

        Writes the state bytes to a temp file that the training loop
        reads at epoch end for advancing checkpoint selection.
        Only captures once per milestone per episode.
        """
        if not self._milestone_state_dir:
            return
        if milestone_name in self._captured_milestones:
            return
        if not hasattr(self.env, "_pyboy") or self.env._pyboy is None:
            return

        self._captured_milestones.add(milestone_name)

        # Capture current emulator state
        buf = io.BytesIO()
        self.env._pyboy.save_state(buf)
        state_bytes = buf.getvalue()

        # Write state file
        env_id = id(self) % 100000
        filename = f"{milestone_name}_{env_id}_{self._epoch}.state"
        state_path = os.path.join(self._milestone_state_dir, filename)

        os.makedirs(self._milestone_state_dir, exist_ok=True)
        with open(state_path, "wb") as f:
            f.write(state_bytes)

        # Write metadata sidecar
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

        # Snapshot baseline state from actual RAM — critical for curriculum
        # learning where different save states start with different items,
        # rupees, keys, and quest flags already set.
        self._prev_health = info.get("health", 0)
        self._prev_group = info.get("active_group", 0)
        self._prev_dungeon_floor = info.get("dungeon_floor", 0)
        if hasattr(self.env, "_read"):
            self._prev_rupees = self.env._read16(RUPEES)
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
            self._prev_rupees = 0
            self._prev_keys = 0
            self._prev_sword = 0
            self._prev_essences = 0
            self._prev_maku_dialog = False
            self._prev_gnarled_key = False
            self._prev_maku_seed = False

        # Reset sub-modules
        self._coverage.reset()
        self._steps_since_discovery = 0
        self._prev_total_tiles = 0
        self._menu_steps = 0
        self._recent_rooms.clear()
        self._prev_room_id = info.get("room_id", -1)
        self._dialog_rooms.clear()
        self._prev_dialog_active = False
        self._prev_pixel_x = info.get("pixel_x", 0)
        self._prev_pixel_y = info.get("pixel_y", 0)

        # Distance/directional bonus tracking
        self._start_room_id = info.get("room_id", 0)
        self._start_group = info.get("active_group", 0)
        self._max_distance_achieved = 0

        # Initialize min target distance from starting position
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

        # Reset milestones for new episode — record what the save state
        # already has so milestone reporting only counts NEW achievements.
        self._milestone_got_sword = False
        self._milestone_entered_dungeon = False
        self._milestone_visited_maku_tree = False
        self._milestone_essences = 0
        self._milestone_dungeon_keys = 0
        self._milestone_max_rooms = 0
        # Save-state baseline: milestones already present at episode start
        self._baseline_sword = self._prev_sword
        self._baseline_essences = self._prev_essences
        self._baseline_maku_dialog = self._prev_maku_dialog
        self._baseline_gnarled_key = self._prev_gnarled_key
        self._baseline_maku_seed = self._prev_maku_seed
        self._baseline_maku_stage = self._prev_maku_stage
        self._baseline_group = self._prev_group
        self._gate_slashed = False
        self._maku_rooms_visited.clear()
        self._captured_milestones.clear()
        self._a_press_rooms.clear()
        self._current_button = 0
        if self._shaping:
            self._shaping.reset()

        # Expose visited rooms set to base env for state_encoder access.
        # Filter to overworld rooms (room_id < 256) since state_encoder and
        # frontier_exit_dist operate on the 16×16 overworld grid.
        self.env._visited_rooms_set = self._coverage._visited_rooms

        # Initialize frontier exit distance (all rooms unvisited at start)
        self._prev_exit_dist = self.env.frontier_exit_dist(set())

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

        # Track rooms before reward computation for stagnation detection
        prev_rooms = self._coverage.unique_rooms

        self._milestone_achieved_this_step = False
        reward = self._compute_reward(obs, info, terminated)

        # Log new room discoveries — helps diagnose exploration bottlenecks
        new_rooms = self._coverage.unique_rooms
        if new_rooms > prev_rooms:
            room_id = info.get("room_id", 0)
            active_group = info.get("active_group", 0)
            logger.info(
                "NEW ROOM: group=%d room=%d (row=%d, col=%d) | total=%d | bonus=%.1f",
                active_group, room_id, room_id // 16, room_id % 16,
                new_rooms, self._coverage.bonus_per_room,
            )

        # Tile-based stagnation — reset counter when ANY new tile is found.
        # Room-based stagnation was too aggressive: the agent got truncated
        # while transiting through known rooms to reach new areas.  Tile-based
        # allows transit (new tiles are found even in visited rooms) while
        # still ending episodes where the agent circles the same tiles.
        # Dialog steps don't count toward stagnation — NPC dialog is
        # productive (quest progression) and shouldn't trigger truncation.
        new_tiles = self._coverage.total_tiles
        dialog_active = info.get("dialog_active", False)
        is_transitioning = info.get("transitioning", False)
        active_group = info.get("active_group", 0)
        if new_tiles > self._prev_total_tiles:
            self._steps_since_discovery = 0
        elif not dialog_active and not is_transitioning and active_group != 2:
            # Freeze stagnation counter in Maku Tree area (group 2).
            # The agent needs many steps of non-movement interaction
            # (slashing gate, popping bubble, advancing dialog) to complete
            # quest objectives.  These actions don't discover new tiles but
            # ARE productive, so don't count them toward stagnation.
            self._steps_since_discovery += 1
        self._prev_total_tiles = new_tiles

        # Reset stagnation on milestone — the agent needs time to backtrack
        # from milestone locations (Maku Tree, dungeons) through known territory
        if self._milestone_achieved_this_step:
            self._steps_since_discovery = 0

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

        # Update milestones from reward computation state — only count
        # achievements BEYOND what the save state already had (baseline).
        self._milestone_max_rooms = max(self._milestone_max_rooms, new_rooms)
        if hasattr(self.env, "_read"):
            sword = self.env._read(SWORD_LEVEL)
            if sword > self._baseline_sword:
                self._milestone_got_sword = True
            keys = self.env._read(DUNGEON_KEYS)
            self._milestone_dungeon_keys = max(self._milestone_dungeon_keys, keys)
            essences = bin(self.env._read(ESSENCES_COLLECTED)).count("1")
            self._milestone_essences = max(self._milestone_essences, essences)
        active_group = info.get("active_group", 0)
        if active_group in (4, 5) and self._baseline_group not in (4, 5):
            self._milestone_entered_dungeon = True
        if active_group == 2 and self._baseline_group != 2:
            self._milestone_visited_maku_tree = True

        info["epoch"] = self._epoch
        info["stagnation_steps"] = self._steps_since_discovery
        info["unique_rooms"] = new_rooms
        info["unique_tiles"] = self._coverage.total_tiles
        info["seen_coords"] = self._coverage.total_tiles  # Pokemon Red-style coord count

        # Progression milestones (available every step, reported at episode end)
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
        info["milestone_maku_rooms"] = float(len(self._maku_rooms_visited))
        info["milestone_maku_stage"] = float(
            self._prev_maku_stage > self._baseline_maku_stage
        )

        # Save-state baseline inventory — reported every step so the training
        # loop can record what the agent *starts* with.  The phase detector
        # uses this to skip phases the save state has already completed
        # (e.g. don't target Hero's Cave if the save state already has sword).
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

        # Time penalty
        reward += self._time_penalty

        # Wall collision penalty — wLinkPushingDirection (oracles-disasm)
        # is $FF when not pushing, else equals Link's direction. Penalize
        # to discourage wasting steps pressing into walls/obstacles.
        if hasattr(self.env, "_read"):
            pushing = self.env._read(LINK_PUSHING_DIRECTION)
            if pushing != 0xFF and not info.get("transitioning", False):
                reward += self._wall_collision_penalty

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

        # Dungeon entry bonus: ACTIVE_GROUP changed to 4 or 5
        if active_group in (4, 5) and self._prev_group not in (4, 5):
            reward += self._dungeon_entry_bonus
            self._milestone_achieved_this_step = True
            self._capture_milestone_state("entered_dungeon", reward)

        # Maku Tree visit bonus: ACTIVE_GROUP changed to 2
        if active_group == 2 and self._prev_group != 2:
            reward += self._maku_tree_visit_bonus
            self._milestone_achieved_this_step = True
            self._capture_milestone_state("visited_maku_tree", reward)

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
                self._milestone_achieved_this_step = True
                logger.info("MILESTONE: Maku Tree gave Gnarled Key quest! (+%.0f)", self._maku_dialog_bonus)
                self._capture_milestone_state("maku_dialog", reward)
            self._prev_maku_dialog = maku_dialog

            # Check TREASURE_GNARLED_KEY obtained (picked up the key item)
            gnarled_key = bool(self.env._read(GNARLED_KEY_OBTAINED) & GNARLED_KEY_OBTAINED_MASK)
            if gnarled_key and not self._prev_gnarled_key:
                reward += self._gnarled_key_bonus
                self._milestone_achieved_this_step = True
                logger.info("MILESTONE: Gnarled Key obtained! (+%.0f)", self._gnarled_key_bonus)
                self._capture_milestone_state("gnarled_key", reward)
            self._prev_gnarled_key = gnarled_key

            # Check GLOBALFLAG_GOT_MAKU_SEED (end-game, after all 8 essences)
            maku_seed = bool(self.env._read(MAKU_SEED_FLAG) & MAKU_SEED_MASK)
            if maku_seed and not self._prev_maku_seed:
                reward += self._maku_seed_bonus
                self._milestone_achieved_this_step = True
                logger.info("MILESTONE: Maku Seed obtained! (+%.0f)", self._maku_seed_bonus)
            self._prev_maku_seed = maku_seed

        # --- Maku Tree sub-event rewards ---
        # Guide the agent through the interaction sequence within group 2:
        #   slash gate → enter grove → pop bubble → dialog → pick up key
        # Room flags at $C800+room_id track persistent per-room state.
        if active_group == 2 and hasattr(self.env, "_read"):
            room_id = info.get("room_id", 0)

            # 1. Gate slash: room flag bit 7 (0x80) set by makuTreeScript_gateHit
            room_flags = self.env._read(MAKU_ROOM_FLAGS + room_id)
            if (room_flags & ROOMFLAG_GATE_HIT) and not self._gate_slashed:
                self._gate_slashed = True
                reward += self._gate_slash_bonus
                self._milestone_achieved_this_step = True
                logger.info(
                    "MILESTONE: Maku Tree gate slashed! room=%d flags=0x%02X (+%.0f)",
                    room_id, room_flags, self._gate_slash_bonus,
                )
                self._capture_milestone_state("gate_slashed", reward)

            # 2. New rooms within group 2 — each new room means deeper
            #    progress into the Maku Tree area
            if room_id not in self._maku_rooms_visited:
                self._maku_rooms_visited.add(room_id)
                reward += self._maku_room_bonus
                self._milestone_achieved_this_step = True
                logger.info(
                    "MILESTONE: New Maku Tree room %d (total: %d) (+%.0f)",
                    room_id, len(self._maku_rooms_visited), self._maku_room_bonus,
                )

            # 3. Maku Tree stage change — ws_cc39 increases as the tree
            #    interaction progresses (bubble pop → dialog → key)
            maku_stage = self.env._read(MAKU_TREE_STAGE)
            if maku_stage != self._prev_maku_stage and maku_stage > self._baseline_maku_stage:
                reward += self._maku_stage_bonus
                self._milestone_achieved_this_step = True
                logger.info(
                    "MILESTONE: Maku Tree stage %d → %d (+%.0f)",
                    self._prev_maku_stage, maku_stage, self._maku_stage_bonus,
                )
            self._prev_maku_stage = maku_stage

        # --- Sword interaction bonus ---
        # Reward pressing A (sword) in new rooms to encourage button usage.
        # Without this, the button policy head collapses to NOP since movement
        # dominates reward.  Per-(group, room) cap prevents spamming.
        if self._current_button == 0:  # A button pressed (ButtonAction.A=0)
            room_key = (active_group, room_id)
            if room_key not in self._a_press_rooms:
                self._a_press_rooms.add(room_key)
                bonus = self._sword_use_bonus
                # Triple bonus in Maku Tree area — gate needs slashing
                if active_group == 2:
                    bonus *= 3.0
                # Double bonus when pushing against obstacle (slash it!)
                elif hasattr(self.env, "_read"):
                    pushing = self.env._read(LINK_PUSHING_DIRECTION)
                    if pushing != 0xFF and not info.get("transitioning", False):
                        bonus *= 2.0
                reward += bonus

        # --- LLM Advisor Directives ---
        # Process structured directives from the reward advisor:
        #   seek_room: one-time bonus when entering a target area group
        #   trigger_action: one-time bonus for performing a specific action
        #   avoid_region: per-step penalty for being in an undesirable area
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
                    reward += directive.get("bonus", 0)  # negative for penalties

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

        # --- Exploration rewards (only when NOT in menu and NOT transitioning) ---
        is_transitioning = info.get("transitioning", False)
        if not menu_active and not is_transitioning:
            # Backtrack penalty: penalize re-entering recently visited rooms
            room_id = info.get("room_id", 0)
            if room_id != self._prev_room_id:
                if room_id in self._recent_rooms:
                    reward += self._backtrack_penalty
                self._recent_rooms.append(room_id)
            self._prev_room_id = room_id

            # Exit-seeking shaping: reward moving toward FRONTIER (unvisited) exits.
            # Delta is clamped to [-2, +2] to prevent large reward spikes when
            # frontier_exit_dist jumps (e.g., entering a new room where the
            # nearest frontier exit is suddenly much closer or farther).
            # frontier_exit_dist works on the raw overworld room grid (0-255).
            # Non-overworld rooms are stored as group*256+room_id (≥256) so
            # they naturally don't interfere with the overworld grid lookup.
            cur_exit_dist = self.env.frontier_exit_dist(
                self._coverage._visited_rooms
            )
            exit_delta = max(min(self._prev_exit_dist - cur_exit_dist, 2.0), -2.0)
            if exit_delta != 0:
                reward += exit_delta * self._exit_seeking_scale
            self._prev_exit_dist = cur_exit_dist

            # Coverage reward with area-based boost — only when the agent
            # actually moved (prevents standing-still and wall-bump farming).
            # Uses group-qualified room_id (group * 256 + room_id) so rooms
            # from different groups (overworld, indoors, dungeons) are tracked
            # independently.  Raw room_id alone would conflate Hero's Cave
            # room 1 with overworld room 1.
            cur_pixel_x = info.get("pixel_x", 0)
            cur_pixel_y = info.get("pixel_y", 0)
            actually_moved = (cur_pixel_x != self._prev_pixel_x or
                              cur_pixel_y != self._prev_pixel_y)
            if actually_moved:
                # Overworld (group 0) rooms stay 0-255 for frontier/neighbor
                # compatibility.  Non-overworld rooms get offset (group*256+id)
                # to avoid conflating Hero's Cave room 1 with overworld room 1.
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
                reward += coverage * area_mult
            self._prev_pixel_x = cur_pixel_x
            self._prev_pixel_y = cur_pixel_y

            # RND curiosity
            if self._rnd is not None:
                curiosity = self._rnd.compute(obs, reward)
                reward += curiosity

        # --- Distance and directional exploration bonuses ---
        # Reward reaching new rooms based on room_id distance from the starting
        # room.  Works across ALL area groups (overworld, indoors, dungeons) so
        # the agent always has an incentive to push further.  The 16×16 room grid
        # math is imprecise for indoor rooms, but still gives a useful signal for
        # "room_id changed → you went somewhere new."
        if not is_transitioning:
            room_id = info.get("room_id", 0)
            start_row = self._start_room_id // 16
            start_col = self._start_room_id % 16
            cur_row = room_id // 16
            cur_col = room_id % 16

            # Distance bonus — one-time reward per new max Manhattan distance
            # from starting room.  Encourages spreading outward in all directions.
            manhattan = abs(cur_row - start_row) + abs(cur_col - start_col)
            if manhattan > self._max_distance_achieved:
                delta = manhattan - self._max_distance_achieved
                reward += delta * self._distance_bonus
                self._max_distance_achieved = manhattan

            # Directional bonus — one-time reward per new minimum Manhattan
            # distance toward the target room.  Target is configurable by the
            # LLM reward advisor (default: Maku Tree at row=5, col=12).
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
