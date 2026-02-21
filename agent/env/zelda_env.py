"""Gymnasium wrapper for Zelda: Oracle of Seasons via PyBoy.

Provides reset/save/load state, fixed step, deterministic seeding,
and RAM taps for room_id, tile_x/y, dialog flags, OAM presence.
"""

from __future__ import annotations

import io
import logging
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)


class ZeldaAction(IntEnum):
    """Game Boy button mapping (legacy Discrete(7) reference)."""

    NOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    A = 5
    B = 6
    START = 7  # Opens inventory for item switching


class MovementAction(IntEnum):
    """Movement dimension of MultiDiscrete action."""

    NOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class ButtonAction(IntEnum):
    """Button dimension of MultiDiscrete action.

    NOP removed — the agent should always be pressing A (sword) or B (item).
    Without NOP, button entropy collapse is impossible and the agent will
    naturally discover sword interactions at the Maku Tree gate.
    """

    A = 0
    B = 1


# Map MultiDiscrete enums to ZeldaAction for press/release event lookup
_MOVEMENT_TO_ZELDA = {
    MovementAction.NOP: ZeldaAction.NOP,
    MovementAction.UP: ZeldaAction.UP,
    MovementAction.DOWN: ZeldaAction.DOWN,
    MovementAction.LEFT: ZeldaAction.LEFT,
    MovementAction.RIGHT: ZeldaAction.RIGHT,
}
_BUTTON_TO_ZELDA = {
    ButtonAction.A: ZeldaAction.A,
    ButtonAction.B: ZeldaAction.B,
}

# RAM addresses — oracles-disasm confirmed (Seasons-specific)
# Link object struct at w1Link = $D000, SpecialObjectStruct layout
_PLAYER_X = 0xD00D       # w1Link + $0D (xh - pixel X)
_PLAYER_Y = 0xD00B       # w1Link + $0B (yh - pixel Y)
_PLAYER_DIR = 0xD008     # w1Link + $08 (direction)
_PLAYER_ROOM = 0xCC4C    # wActiveRoom (Seasons)
_HEALTH = 0xC6A2         # wLinkHealth (Seasons) — quarter-hearts
_MAX_HEALTH = 0xC6A3     # wLinkMaxHealth (Seasons)
_DIALOG_STATE = 0xCBA0   # wTextIsActive (0 = no text)
_DUNGEON_FLOOR = 0xCC57  # wDungeonFloor (Seasons)
_DEATH_COUNT = 0xC61E    # 2 bytes LE
_PUZZLE_FLAGS = 0xCC58   # wDungeonRoomProperties
_SCREEN_TRANSITION = 0xCD00  # wScrollMode
_LOADING = 0xC2F2
_MENU_STATE = 0xCBCB         # wOpenedMenuType (non-zero = menu open)
_ACTIVE_GROUP = 0xCC49       # wActiveGroup (0=overworld, 2=maku, 4-5=dungeons)
_DUNGEON_INDEX = 0xCC55      # wDungeonIndex ($FF = overworld)
_KEYS_PRESSED = 0xC481       # wKeysPressed (currently held buttons)
_KEYS_JUST_PRESSED = 0xC482  # wKeysJustPressed (buttons pressed this frame)

# Room collision data — populated per-room, 16 cols × 12 rows = 192 bytes
_ROOM_COLLISIONS = 0xCE00
_ACTIVE_TILE_TYPE = 0xCCB6

# Tile types that Link can walk on (from oracles-disasm tileTypes.s).
# CRITICAL: every navigation observation feature (edge exits, ray-casts,
# collision map 5×4, frontier distance, exit-seeking) depends on this set.
# Missing a walkable type causes the agent to "see" walls where there are
# paths, creating invisible barriers in the observation that prevent
# exploration even though the game engine allows walking there.
_WALKABLE_TILES = frozenset({
    0x00,  # TILETYPE_NORMAL
    0x03,  # TILETYPE_CRACKEDFLOOR (breaks after standing)
    0x04,  # TILETYPE_VINES (climbable)
    0x05,  # TILETYPE_GRASS (cutable, walkable)
    0x06,  # TILETYPE_STAIRS (slower movement)
    0x09,  # TILETYPE_UPCONVEYOR (pushes Link up)
    0x0A,  # TILETYPE_RIGHTCONVEYOR (pushes Link right)
    0x0B,  # TILETYPE_DOWNCONVEYOR (pushes Link down)
    0x0C,  # TILETYPE_LEFTCONVEYOR (pushes Link left)
    0x0D,  # TILETYPE_SPIKE (damages but walkable)
    0x0E,  # TILETYPE_CRACKED_ICE (Seasons: breaks after walking)
    0x0F,  # TILETYPE_ICE (slippery)
    0x11,  # TILETYPE_PUDDLE (shallow water, splash effect)
})


class ZeldaEnv(gym.Env):
    """Gymnasium env wrapping PyBoy for Oracle of Seasons.

    Observation: 128-D float32 vector (from StateEncoder).
    Action: Discrete(7) — NOP, UP, DOWN, LEFT, RIGHT, A, B.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        rom_path: str,
        headless: bool = True,
        frame_skip: int = 4,
        max_steps: int = 30_000,
        save_state_path: str | None = None,
        alt_save_state_path: str | None = None,
        alt_save_state_ratio: float = 0.5,
        render_mode: str | None = None,
        seed: int | None = None,
        god_mode: bool = False,
    ):
        super().__init__()
        self.rom_path = rom_path
        self._headless = headless
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self._save_state_path = save_state_path or None
        self._alt_save_state_path = alt_save_state_path or None
        self._alt_save_state_ratio = alt_save_state_ratio
        self.render_mode = render_mode
        self._seed = seed
        self._god_mode = god_mode

        # Lazy PyBoy init — deferred to first reset()
        self._pyboy = None
        self._initial_state: bytes | None = None
        self._alt_initial_state: bytes | None = None
        self._rng = np.random.RandomState(seed or 0)

        # Spaces — MultiDiscrete allows simultaneous movement + button press.
        # Dimension 0: movement (5) — NOP, UP, DOWN, LEFT, RIGHT
        # Dimension 1: button (2) — A, B (no NOP — always press a button)
        self.action_space = spaces.MultiDiscrete([5, 2])
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(128,), dtype=np.float32
        )

        # Episode bookkeeping
        self.step_count = 0
        self.episode_count = 0
        self._initial_deaths = 0
        self._last_movement: ZeldaAction | None = None
        self._last_button: ZeldaAction | None = None
        self._prev_room = 0
        self._last_valid_obs: np.ndarray | None = None

        # WindowEvent mappings (proven to work with PyBoy)
        self._press_events: dict[ZeldaAction, Any] = {}
        self._release_events: dict[ZeldaAction, Any] = {}

    # ------------------------------------------------------------------
    # PyBoy lifecycle
    # ------------------------------------------------------------------

    def _ensure_pyboy(self) -> None:
        """Lazily create PyBoy instance."""
        if self._pyboy is not None:
            return
        try:
            from pyboy import PyBoy
            from pyboy.utils import WindowEvent
        except ImportError as exc:
            raise ImportError(
                "pyboy is required: pip install pyboy>=2.6.0"
            ) from exc

        # Build WindowEvent mappings
        self._press_events = {
            ZeldaAction.NOP: None,
            ZeldaAction.UP: WindowEvent.PRESS_ARROW_UP,
            ZeldaAction.DOWN: WindowEvent.PRESS_ARROW_DOWN,
            ZeldaAction.LEFT: WindowEvent.PRESS_ARROW_LEFT,
            ZeldaAction.RIGHT: WindowEvent.PRESS_ARROW_RIGHT,
            ZeldaAction.A: WindowEvent.PRESS_BUTTON_A,
            ZeldaAction.B: WindowEvent.PRESS_BUTTON_B,
            ZeldaAction.START: WindowEvent.PRESS_BUTTON_START,
        }
        self._release_events = {
            ZeldaAction.UP: WindowEvent.RELEASE_ARROW_UP,
            ZeldaAction.DOWN: WindowEvent.RELEASE_ARROW_DOWN,
            ZeldaAction.LEFT: WindowEvent.RELEASE_ARROW_LEFT,
            ZeldaAction.RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
            ZeldaAction.A: WindowEvent.RELEASE_BUTTON_A,
            ZeldaAction.B: WindowEvent.RELEASE_BUTTON_B,
            ZeldaAction.START: WindowEvent.RELEASE_BUTTON_START,
        }

        # Store SELECT release event to prevent spamming (START is a valid action)
        self._release_select_event = WindowEvent.RELEASE_BUTTON_SELECT

        # Suppress PyBoy sound buffer overrun spam
        logging.getLogger("pyboy.core.sound").setLevel(logging.CRITICAL + 1)
        logging.getLogger("pyboy").setLevel(logging.WARNING)

        window = "null" if self._headless else "SDL2"
        self._pyboy = PyBoy(self.rom_path, window=window, sound_emulated=False)
        # Tick frames to get past the boot logo
        for _ in range(1000):
            self._pyboy.tick()
        # Load save state if provided
        if self._save_state_path:
            with open(self._save_state_path, "rb") as f:
                self._pyboy.load_state(f)
        # Capture initial state for deterministic resets
        buf = io.BytesIO()
        self._pyboy.save_state(buf)
        self._initial_state = buf.getvalue()

        # Load alternate save state for curriculum diversity
        if self._alt_save_state_path:
            with open(self._alt_save_state_path, "rb") as f:
                self._pyboy.load_state(f)
            alt_buf = io.BytesIO()
            self._pyboy.save_state(alt_buf)
            self._alt_initial_state = alt_buf.getvalue()
            # Restore primary state after capturing alt
            self._pyboy.load_state(io.BytesIO(self._initial_state))
            logger.info(
                "Loaded alt save state: %s (ratio=%.0f%%)",
                self._alt_save_state_path,
                self._alt_save_state_ratio * 100,
            )

    # ------------------------------------------------------------------
    # RAM helpers
    # ------------------------------------------------------------------

    def _read(self, addr: int) -> int:
        """Read a single byte from Game Boy memory."""
        return self._pyboy.memory[addr]

    def _read16(self, addr: int) -> int:
        """Read 16-bit little-endian value."""
        return self._read(addr) | (self._read(addr + 1) << 8)

    @property
    def room_id(self) -> int:
        return self._read(_PLAYER_ROOM)

    @property
    def tile_x(self) -> int:
        return self._read(_PLAYER_X) // 16

    @property
    def tile_y(self) -> int:
        return self._read(_PLAYER_Y) // 16

    @property
    def pixel_x(self) -> int:
        return self._read(_PLAYER_X)

    @property
    def pixel_y(self) -> int:
        return self._read(_PLAYER_Y)

    @property
    def player_dir(self) -> int:
        return self._read(_PLAYER_DIR)

    @property
    def health(self) -> int:
        return self._read(_HEALTH) // 4

    @property
    def max_health(self) -> int:
        return self._read(_MAX_HEALTH) // 4

    @property
    def dialog_active(self) -> bool:
        return self._read(_DIALOG_STATE) != 0

    @property
    def dungeon_floor(self) -> int:
        return self._read(_DUNGEON_FLOOR)

    @property
    def puzzle_flags(self) -> int:
        return self._read(_PUZZLE_FLAGS)

    @property
    def is_transitioning(self) -> bool:
        # wScrollMode bit 7 is set during active screen scroll.
        # Values 0x01 (normal) and 0x02 (dungeon) are NOT transitions.
        # Values 0x81, 0x82, etc. indicate active scrolling.
        scroll = self._read(_SCREEN_TRANSITION)
        return (scroll & 0x80) != 0

    # ------------------------------------------------------------------
    # OAM sprite presence
    # ------------------------------------------------------------------

    def oam_sprite_count(self) -> int:
        """Count active sprites in OAM (y != 0 and y < 160)."""
        count = 0
        for i in range(40):
            y = self._read(0xFE00 + i * 4)
            if 0 < y < 160:
                count += 1
        return count

    # ------------------------------------------------------------------
    # Collision / navigation
    # ------------------------------------------------------------------

    def log_collision_edge_types(self, room_id: int) -> None:
        """Log tile type values at room edges (diagnostic).

        Called once per new room to reveal which tile types appear at
        screen edges.  Helps diagnose "invisible barrier" issues where
        the agent sees no exit but the game allows walking there.
        """
        top_types = [self._read(_ROOM_COLLISIONS + 0 * 16 + x) for x in range(10)]
        bot_types = [self._read(_ROOM_COLLISIONS + 7 * 16 + x) for x in range(10)]
        left_types = [self._read(_ROOM_COLLISIONS + y * 16 + 0) for y in range(8)]
        right_types = [self._read(_ROOM_COLLISIONS + y * 16 + 9) for y in range(8)]

        # Only log unknown types (not in _WALKABLE_TILES) to reduce noise
        unknown_top = {v for v in top_types if v not in _WALKABLE_TILES and v != 0}
        unknown_bot = {v for v in bot_types if v not in _WALKABLE_TILES and v != 0}
        unknown_left = {v for v in left_types if v not in _WALKABLE_TILES and v != 0}
        unknown_right = {v for v in right_types if v not in _WALKABLE_TILES and v != 0}

        if unknown_top or unknown_bot or unknown_left or unknown_right:
            row, col = room_id // 16, room_id % 16
            logger.info(
                "COLLISION EDGE room=%d (row=%d,col=%d) | "
                "top=%s unkn=%s | bot=%s unkn=%s | "
                "left=%s unkn=%s | right=%s unkn=%s",
                room_id, row, col,
                [f"0x{v:02X}" for v in top_types],
                {f"0x{v:02X}" for v in unknown_top} if unknown_top else "{}",
                [f"0x{v:02X}" for v in bot_types],
                {f"0x{v:02X}" for v in unknown_bot} if unknown_bot else "{}",
                [f"0x{v:02X}" for v in left_types],
                {f"0x{v:02X}" for v in unknown_left} if unknown_left else "{}",
                [f"0x{v:02X}" for v in right_types],
                {f"0x{v:02X}" for v in unknown_right} if unknown_right else "{}",
            )

    def check_edge_exits(self) -> tuple[float, float, float, float]:
        """Check if any tile on each screen edge is walkable (potential exit).

        Returns (up, down, left, right) as floats 0.0 or 1.0.
        Checks the standard 10x8 metatile playable area.
        """
        up = 0.0
        down = 0.0
        left = 0.0
        right = 0.0

        for x in range(10):
            if self._read(_ROOM_COLLISIONS + 0 * 16 + x) in _WALKABLE_TILES:
                up = 1.0
            if self._read(_ROOM_COLLISIONS + 7 * 16 + x) in _WALKABLE_TILES:
                down = 1.0
        for y in range(8):
            if self._read(_ROOM_COLLISIONS + y * 16 + 0) in _WALKABLE_TILES:
                left = 1.0
            if self._read(_ROOM_COLLISIONS + y * 16 + 9) in _WALKABLE_TILES:
                right = 1.0

        return up, down, left, right

    def ray_cast_distances(self) -> tuple[float, float, float, float]:
        """Cast rays in 4 cardinal directions from player position.

        Returns (up, down, left, right) — number of walkable tiles before
        hitting an obstacle, normalized by max possible distance.
        """
        tx, ty = self.tile_x, self.tile_y

        up = 0
        for y in range(ty - 1, -1, -1):
            if self._read(_ROOM_COLLISIONS + y * 16 + tx) in _WALKABLE_TILES:
                up += 1
            else:
                break

        down = 0
        for y in range(ty + 1, 12):
            if self._read(_ROOM_COLLISIONS + y * 16 + tx) in _WALKABLE_TILES:
                down += 1
            else:
                break

        left = 0
        for x in range(tx - 1, -1, -1):
            if self._read(_ROOM_COLLISIONS + ty * 16 + x) in _WALKABLE_TILES:
                left += 1
            else:
                break

        right = 0
        for x in range(tx + 1, 16):
            if self._read(_ROOM_COLLISIONS + ty * 16 + x) in _WALKABLE_TILES:
                right += 1
            else:
                break

        return up / 8.0, down / 8.0, left / 10.0, right / 10.0

    def exit_distances(self) -> tuple[float, float, float, float, float, float]:
        """Compute Manhattan distance to nearest walkable exit on each edge.

        Returns (dist_up, dist_down, dist_left, dist_right, dir_x, dir_y).
        Distances normalized to [0, 1] where 0 = at the exit, 1 = far away.
        dir_x/dir_y encode the direction to the overall nearest exit,
        mapped to [0, 1] where 0.5 = no displacement.
        """
        tx, ty = self.tile_x, self.tile_y
        max_dist = 16.0

        best_overall = float("inf")
        best_dx, best_dy = 0.0, 0.0

        # Top edge (y=0)
        best_up = float("inf")
        for x in range(10):
            if self._read(_ROOM_COLLISIONS + 0 * 16 + x) in _WALKABLE_TILES:
                d = abs(tx - x) + ty
                if d < best_up:
                    best_up = d
                if d < best_overall:
                    best_overall = d
                    best_dx = float(x - tx)
                    best_dy = float(0 - ty)

        # Bottom edge (y=7)
        best_down = float("inf")
        for x in range(10):
            if self._read(_ROOM_COLLISIONS + 7 * 16 + x) in _WALKABLE_TILES:
                d = abs(tx - x) + abs(7 - ty)
                if d < best_down:
                    best_down = d
                if d < best_overall:
                    best_overall = d
                    best_dx = float(x - tx)
                    best_dy = float(7 - ty)

        # Left edge (x=0)
        best_left = float("inf")
        for y in range(8):
            if self._read(_ROOM_COLLISIONS + y * 16 + 0) in _WALKABLE_TILES:
                d = tx + abs(ty - y)
                if d < best_left:
                    best_left = d
                if d < best_overall:
                    best_overall = d
                    best_dx = float(0 - tx)
                    best_dy = float(y - ty)

        # Right edge (x=9)
        best_right = float("inf")
        for y in range(8):
            if self._read(_ROOM_COLLISIONS + y * 16 + 9) in _WALKABLE_TILES:
                d = abs(9 - tx) + abs(ty - y)
                if d < best_right:
                    best_right = d
                if d < best_overall:
                    best_overall = d
                    best_dx = float(9 - tx)
                    best_dy = float(y - ty)

        # Normalize distances (0 = at exit, 1 = far/no exit)
        dist_up = min(best_up / max_dist, 1.0) if best_up < float("inf") else 1.0
        dist_down = min(best_down / max_dist, 1.0) if best_down < float("inf") else 1.0
        dist_left = min(best_left / max_dist, 1.0) if best_left < float("inf") else 1.0
        dist_right = min(best_right / max_dist, 1.0) if best_right < float("inf") else 1.0

        # Direction to nearest exit: normalize and map [-1,1] → [0,1]
        if best_overall < float("inf") and best_overall > 0:
            mag = abs(best_dx) + abs(best_dy)
            dir_x = (best_dx / mag + 1.0) / 2.0
            dir_y = (best_dy / mag + 1.0) / 2.0
        else:
            dir_x = 0.5
            dir_y = 0.5

        return dist_up, dist_down, dist_left, dist_right, dir_x, dir_y

    def collision_map_5x4(self) -> tuple:
        """Return 5×4 downsampled collision map (20 floats).

        The 10×8 playable tile grid is reduced to 5×4 using 2×2 max-pooling:
        a block is 1.0 if ANY tile in the 2×2 region is walkable, else 0.0.
        Gives the agent a spatial map of the room layout.
        """
        result = []
        for by in range(4):
            for bx in range(5):
                walkable = 0.0
                for dy in range(2):
                    for dx in range(2):
                        x = bx * 2 + dx
                        y = by * 2 + dy
                        if self._read(_ROOM_COLLISIONS + y * 16 + x) in _WALKABLE_TILES:
                            walkable = 1.0
                            break
                    if walkable:
                        break
                result.append(walkable)
        return tuple(result)

    @property
    def active_tile_type(self) -> int:
        return self._read(_ACTIVE_TILE_TYPE)

    # ------------------------------------------------------------------
    # Input safety
    # ------------------------------------------------------------------

    def _release_select(self) -> None:
        """Force-release SELECT to prevent spamming (START is a valid action)."""
        if self._pyboy is None:
            return
        self._pyboy.send_input(self._release_select_event)

    def _clear_input_registers(self) -> None:
        """Zero out software input registers to flush stale button state."""
        self._pyboy.memory[_KEYS_PRESSED] = 0x00
        self._pyboy.memory[_KEYS_JUST_PRESSED] = 0x00

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._ensure_pyboy()

        # Reload from saved state — randomly choose between primary and alt
        # to add curriculum diversity (e.g., 50% advanced state, 50% default)
        if self._initial_state is not None:
            if (self._alt_initial_state is not None
                    and self._rng.random() < self._alt_save_state_ratio):
                self._pyboy.load_state(io.BytesIO(self._alt_initial_state))
            else:
                self._pyboy.load_state(io.BytesIO(self._initial_state))

        # Flush stale button state from save state
        self._clear_input_registers()

        # Tick a few frames to let the game engine settle after state load
        for _ in range(10):
            self._pyboy.tick()
        self._prev_room = self.room_id

        # Force-release SELECT to prevent spamming
        self._release_select()

        self.step_count = 0
        self.episode_count += 1
        self._initial_deaths = self._read16(_DEATH_COUNT)
        self._last_movement = None
        self._last_button = None

        obs = self._get_obs()
        self._last_valid_obs = obs.copy()
        info = self._get_info()
        return obs, info

    def _dismiss_menu(self) -> None:
        """Auto-dismiss menu via direct RAM write (avoids registering START input)."""
        if self._read(_MENU_STATE) == 0:
            return
        self._pyboy.memory[_MENU_STATE] = 0
        logger.debug("Menu dismissed via RAM write")

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # MultiDiscrete: action = [movement, button]
        action = np.asarray(action).flatten()
        move_idx = int(action[0])
        btn_idx = int(action[1])

        move_act = _MOVEMENT_TO_ZELDA[MovementAction(move_idx)]
        btn_act = _BUTTON_TO_ZELDA[ButtonAction(btn_idx)]

        # Force-release SELECT to prevent spamming
        self._release_select()

        # Release previous movement
        if self._last_movement is not None and self._last_movement != ZeldaAction.NOP:
            release_event = self._release_events.get(self._last_movement)
            if release_event:
                self._pyboy.send_input(release_event)

        # Release previous button
        if self._last_button is not None and self._last_button != ZeldaAction.NOP:
            release_event = self._release_events.get(self._last_button)
            if release_event:
                self._pyboy.send_input(release_event)

        # Press new movement
        if move_act != ZeldaAction.NOP:
            press_event = self._press_events.get(move_act)
            if press_event:
                self._pyboy.send_input(press_event)

        # Press new button
        if btn_act != ZeldaAction.NOP:
            press_event = self._press_events.get(btn_act)
            if press_event:
                self._pyboy.send_input(press_event)

        # Advance emulator — natural screen transitions handled by
        # the game engine (clean save state, no mid-fade hacks needed).
        for _ in range(self.frame_skip):
            self._pyboy.tick()
            # God mode: write max health every tick to prevent death.
            # Isolates exploration learning from survival (curriculum).
            if self._god_mode:
                self._pyboy.memory[_HEALTH] = self._pyboy.memory[_MAX_HEALTH]
            # Suppress START/SELECT — the agent doesn't need inventory
            # management yet.  The START menu wastes thousands of steps as
            # the agent gets stuck for the visual feedback.  Mask out START
            # (bit 3) and SELECT (bit 2) from input registers AND force-close
            # any menu that managed to open.
            self._pyboy.memory[_KEYS_PRESSED] &= ~0x0C
            self._pyboy.memory[_KEYS_JUST_PRESSED] &= ~0x0C
            if self._pyboy.memory[_MENU_STATE] != 0:
                self._pyboy.memory[_MENU_STATE] = 0

        self._last_movement = move_act
        self._last_button = btn_act

        self.step_count += 1
        obs = self._get_obs()
        terminated = self._check_terminated()
        truncated = self.step_count >= self.max_steps
        info = self._get_info()

        # During screen transitions, return cached observation.
        # The room_id has changed but x/y coords still reflect the old room,
        # so a fresh observation would encode incorrect position data.
        if self.is_transitioning:
            if self._last_valid_obs is not None:
                obs = self._last_valid_obs.copy()
            info["transitioning"] = True
        else:
            self._last_valid_obs = obs.copy()
            info["transitioning"] = False

        # Reward is computed externally by the RL wrapper
        reward = 0.0
        return obs, reward, terminated, truncated, info

    def _check_terminated(self) -> bool:
        """Link died if death counter incremented."""
        return self._read16(_DEATH_COUNT) > self._initial_deaths

    def _get_obs(self) -> np.ndarray:
        """Build 128-D observation vector from RAM."""
        from agent.env.state_encoder import encode_vector

        return encode_vector(self)

    @property
    def nearest_exit_dist(self) -> int:
        """Manhattan distance to the nearest walkable edge tile."""
        tx, ty = self.tile_x, self.tile_y
        best = 999
        for x in range(10):
            if self._read(_ROOM_COLLISIONS + 0 * 16 + x) in _WALKABLE_TILES:
                best = min(best, abs(tx - x) + ty)
            if self._read(_ROOM_COLLISIONS + 7 * 16 + x) in _WALKABLE_TILES:
                best = min(best, abs(tx - x) + abs(7 - ty))
        for y in range(8):
            if self._read(_ROOM_COLLISIONS + y * 16 + 0) in _WALKABLE_TILES:
                best = min(best, tx + abs(ty - y))
            if self._read(_ROOM_COLLISIONS + y * 16 + 9) in _WALKABLE_TILES:
                best = min(best, abs(9 - tx) + abs(ty - y))
        return best

    def frontier_exit_dist(self, visited_rooms: set[int]) -> int:
        """Manhattan distance to nearest exit leading to an UNVISITED room.

        For each screen edge, checks whether the neighbor room in that
        direction has been visited. Only counts walkable edge tiles that
        lead to frontier (unvisited) rooms. Falls back to nearest_exit_dist
        when all neighbors are visited.
        """
        tx, ty = self.tile_x, self.tile_y
        room = self.room_id
        row, col = room // 16, room % 16

        best_frontier = 999
        best_any = 999

        # Top edge (y=0) → north neighbor
        north_id = (row - 1) * 16 + col if row > 0 else -1
        for x in range(10):
            if self._read(_ROOM_COLLISIONS + 0 * 16 + x) in _WALKABLE_TILES:
                d = abs(tx - x) + ty
                best_any = min(best_any, d)
                if north_id >= 0 and north_id not in visited_rooms:
                    best_frontier = min(best_frontier, d)

        # Bottom edge (y=7) → south neighbor
        south_id = (row + 1) * 16 + col if row < 15 else -1
        for x in range(10):
            if self._read(_ROOM_COLLISIONS + 7 * 16 + x) in _WALKABLE_TILES:
                d = abs(tx - x) + abs(7 - ty)
                best_any = min(best_any, d)
                if south_id >= 0 and south_id not in visited_rooms:
                    best_frontier = min(best_frontier, d)

        # Left edge (x=0) → west neighbor
        west_id = row * 16 + (col - 1) if col > 0 else -1
        for y in range(8):
            if self._read(_ROOM_COLLISIONS + y * 16 + 0) in _WALKABLE_TILES:
                d = tx + abs(ty - y)
                best_any = min(best_any, d)
                if west_id >= 0 and west_id not in visited_rooms:
                    best_frontier = min(best_frontier, d)

        # Right edge (x=9) → east neighbor
        east_id = row * 16 + (col + 1) if col < 15 else -1
        for y in range(8):
            if self._read(_ROOM_COLLISIONS + y * 16 + 9) in _WALKABLE_TILES:
                d = abs(9 - tx) + abs(ty - y)
                best_any = min(best_any, d)
                if east_id >= 0 and east_id not in visited_rooms:
                    best_frontier = min(best_frontier, d)

        return best_frontier if best_frontier < 999 else best_any

    def neighbor_room_visited(self) -> tuple[float, float, float, float]:
        """Check if each adjacent room (N/S/E/W) has been visited this episode.

        Returns (north, south, east, west) as 0.0 (unvisited) or 1.0 (visited).
        Reads from _visited_rooms_set, which is set by RewardWrapper.
        """
        visited = getattr(self, "_visited_rooms_set", set())
        room = self.room_id
        row, col = room // 16, room % 16

        north = 1.0 if row > 0 and ((row - 1) * 16 + col) in visited else 0.0
        south = 1.0 if row < 15 and ((row + 1) * 16 + col) in visited else 0.0
        west = 1.0 if col > 0 and (row * 16 + (col - 1)) in visited else 0.0
        east = 1.0 if col < 15 and (row * 16 + (col + 1)) in visited else 0.0

        return north, south, east, west

    def _get_info(self) -> dict[str, Any]:
        return {
            "room_id": self.room_id,
            "tile_x": self.tile_x,
            "tile_y": self.tile_y,
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "health": self.health,
            "max_health": self.max_health,
            "dialog_active": self.dialog_active,
            "menu_active": self._read(_MENU_STATE) != 0,
            "dungeon_floor": self.dungeon_floor,
            "active_group": self._read(_ACTIVE_GROUP),
            "dungeon_index": self._read(_DUNGEON_INDEX),
            "nearest_exit_dist": self.nearest_exit_dist,
            "step": self.step_count,
            "episode": self.episode_count,
            "sprites": self.oam_sprite_count(),
        }

    def render(self):
        if self.render_mode == "rgb_array" and self._pyboy is not None:
            frame = np.array(self._pyboy.screen.ndarray)
            # PyBoy returns RGBA (144, 160, 4); convert to RGB
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            return frame
        return None

    def close(self):
        if self._pyboy is not None:
            try:
                self._pyboy.stop(save=False)
            except Exception:
                pass
            self._pyboy = None
