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
    """Game Boy button mapping."""

    NOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    A = 5
    B = 6
    START = 7


# PyBoy button name mapping
_PYBOY_BUTTONS = {
    ZeldaAction.NOP: None,
    ZeldaAction.UP: "up",
    ZeldaAction.DOWN: "down",
    ZeldaAction.LEFT: "left",
    ZeldaAction.RIGHT: "right",
    ZeldaAction.A: "a",
    ZeldaAction.B: "b",
    ZeldaAction.START: "start",
}

# RAM addresses — Data Crystal confirmed
_PLAYER_X = 0xC4AC
_PLAYER_Y = 0xC4AD
_PLAYER_DIR = 0xC4AE
_PLAYER_ROOM = 0xC63B
_HEALTH = 0xC021  # quarter-hearts
_MAX_HEALTH = 0xC022
_DIALOG_STATE = 0xC2EF
_DUNGEON_FLOOR = 0xC63D
_DEATH_COUNT = 0xC61E  # 2 bytes LE
_PUZZLE_FLAGS = 0xC6C0
_SCREEN_TRANSITION = 0xC2F1
_LOADING = 0xC2F2


class ZeldaEnv(gym.Env):
    """Gymnasium env wrapping PyBoy for Oracle of Seasons.

    Observation: 128-D float32 vector (from StateEncoder).
    Action: Discrete(8) — NOP, UP, DOWN, LEFT, RIGHT, A, B, START.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        rom_path: str,
        headless: bool = True,
        frame_skip: int = 4,
        max_steps: int = 30_000,
        save_state_path: str | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.rom_path = rom_path
        self._headless = headless
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self._save_state_path = save_state_path or None
        self.render_mode = render_mode
        self._seed = seed

        # Lazy PyBoy init — deferred to first reset()
        self._pyboy = None
        self._initial_state: bytes | None = None

        # Spaces
        self.action_space = spaces.Discrete(len(ZeldaAction))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(128,), dtype=np.float32
        )

        # Episode bookkeeping
        self.step_count = 0
        self.episode_count = 0
        self._initial_deaths = 0

    # ------------------------------------------------------------------
    # PyBoy lifecycle
    # ------------------------------------------------------------------

    def _ensure_pyboy(self) -> None:
        """Lazily create PyBoy instance."""
        if self._pyboy is not None:
            return
        try:
            from pyboy import PyBoy
        except ImportError as exc:
            raise ImportError(
                "pyboy is required: pip install pyboy>=2.6.0"
            ) from exc

        # Suppress PyBoy sound buffer overrun spam
        logging.getLogger("pyboy.core.sound").setLevel(logging.CRITICAL + 1)
        logging.getLogger("pyboy").setLevel(logging.WARNING)

        window = "null" if self._headless else "SDL2"
        self._pyboy = PyBoy(self.rom_path, window=window, sound_emulated=False)
        # Tick a few frames to get past the boot logo
        for _ in range(300):
            self._pyboy.tick(count=1, render=not self._headless)
        # Capture initial state for deterministic resets
        if self._save_state_path:
            with open(self._save_state_path, "rb") as f:
                self._pyboy.load_state(f)
        buf = io.BytesIO()
        self._pyboy.save_state(buf)
        self._initial_state = buf.getvalue()

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
        return self._read(_SCREEN_TRANSITION) != 0 or self._read(_LOADING) != 0

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
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._ensure_pyboy()

        # Deterministic reload from saved state
        if self._initial_state is not None:
            self._pyboy.load_state(io.BytesIO(self._initial_state))

        self.step_count = 0
        self.episode_count += 1
        self._initial_deaths = self._read16(_DEATH_COUNT)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        act = ZeldaAction(action)
        btn = _PYBOY_BUTTONS[act]

        # Press button and tick for frame_skip frames
        if btn is not None:
            self._pyboy.button(btn)
        for _ in range(self.frame_skip):
            self._pyboy.tick(count=1, render=not self._headless)
        if btn is not None:
            self._pyboy.button_release(btn)

        self.step_count += 1
        obs = self._get_obs()
        terminated = self._check_terminated()
        truncated = self.step_count >= self.max_steps
        info = self._get_info()

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
            "dungeon_floor": self.dungeon_floor,
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
            self._pyboy.stop()
            self._pyboy = None
