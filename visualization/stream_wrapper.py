"""Gymnasium wrapper that streams agent position data over WebSocket.

Wraps the Zelda Oracle of Seasons environment to capture position data
and broadcast it for real-time map visualization, without affecting the
environment's observations, rewards, or done signals.

Architecture inspired by:
  - PokemonRedExperiments stream_agent_wrapper.py
    (https://github.com/PWhiddy/PokemonRedExperiments)
    Original WebSocket streaming approach for Pokemon Red RL agents.
  - LinkMapViz / LADXExperiments StreamWrapper
    (https://github.com/Xe-Xo/LinkMapViz)
    (https://github.com/Xe-Xo/LADXExperiments)
    Real-time PixiJS visualization for Link's Awakening DX RL agents.

Memory addresses derived from oracles-disasm:
  https://github.com/Stewmath/oracles-disasm

The coordinate system matches the overworld.png layout:
  world_x = (room_id % 16) * 10 + tile_x   (0-159)
  world_y = (room_id // 16) * 8  + tile_y   (0-127)
  world_z = active_group                     (0=overworld, 1=subrosia, etc.)

License: MIT
"""

import asyncio
import json
import logging
import time
from typing import Any

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)

# Oracle of Seasons RAM addresses (Seasons column from oracles-disasm)
_ACTIVE_GROUP = 0xCC49
_ACTIVE_ROOM = 0xCC4C
_PLAYER_X = 0xD00D
_PLAYER_Y = 0xD00B
_PLAYER_DIR = 0xD008
_SCROLL_MODE = 0xCD00
_ROOM_STATE_MOD = 0xCC4E
_OVERWORLD_FLAGS = 0xC700

# Room flag bits
_FLAG_GATE_HIT = 0x80
_FLAG_ITEM_OBTAINED = 0x40
_FLAG_VISITED = 0x10


class StreamWrapper(gym.Wrapper):
    """Transparent wrapper that streams position telemetry over WebSocket.

    Does NOT modify observations, rewards, or done signals. The training
    loop is completely unaffected — this only reads RAM and sends data.

    Data format sent over WebSocket:
    {
        "name": "stream",
        "version": 1,
        "metadata": { "user": "...", "env_id": 0, "color": "#..." },
        "pos_data": [
            {"x": 45, "y": 32, "z": 0, "notable": ""},
            {"x": 46, "y": 32, "z": 0, "notable": "gate_slash"},
            ...
        ]
    }
    """

    # Notable event types (matching LinkMapViz icon system)
    NOTABLE_GATE = "gate_slash"
    NOTABLE_ITEM = "item_obtained"
    NOTABLE_NEW_ROOM = "new_room"

    def __init__(
        self,
        env: gym.Env,
        ws_address: str = "ws://localhost:3344/broadcast",
        stream_metadata: dict[str, Any] | None = None,
        upload_interval: int = 500,
        csv_log_path: str | None = None,
    ):
        """Initialize the stream wrapper.

        Args:
            env: The Zelda environment to wrap.
            ws_address: WebSocket URL for the relay server.
            stream_metadata: Metadata dict (user, env_id, color, etc.).
            upload_interval: Steps between WebSocket uploads.
            csv_log_path: If set, also log coordinates to this CSV file
                          for offline rendering.
        """
        super().__init__(env)
        self.ws_address = ws_address
        self.metadata = stream_metadata or {
            "user": "zelda-rl",
            "env_id": 0,
            "color": "#44aa77",
        }
        self.upload_interval = upload_interval

        # Internal state
        self._step_counter = 0
        self._coord_buffer: list[dict] = []
        self._visited_rooms: set[int] = set()
        self._prev_room_flags: dict[int, int] = {}

        # WebSocket
        self._loop = asyncio.new_event_loop()
        self._websocket = None
        self._connect_ws()

        # Optional CSV logging for offline rendering
        self._csv_file = None
        self._csv_path = csv_log_path
        if csv_log_path:
            self._csv_file = open(csv_log_path, "w")
            self._csv_file.write("step,world_x,world_y,world_z,room_id,tile_x,tile_y,direction,notable\n")

    def _connect_ws(self):
        """Attempt WebSocket connection (non-blocking, fail-safe)."""
        try:
            import websockets.sync.client as ws_client
            self._websocket = ws_client.connect(self.ws_address)
            logger.info(f"StreamWrapper connected to {self.ws_address}")
        except Exception as e:
            logger.debug(f"StreamWrapper WebSocket connect failed: {e}")
            self._websocket = None

    def _get_pyboy(self):
        """Get the PyBoy instance from the wrapped env."""
        env = self.env
        # Walk the wrapper chain to find the PyBoy instance
        while hasattr(env, "env"):
            if hasattr(env, "_pyboy"):
                return env._pyboy
            env = env.env
        if hasattr(env, "_pyboy"):
            return env._pyboy
        return None

    def _read_position(self) -> dict | None:
        """Read current position from Game Boy RAM.

        Returns:
            Dict with world coordinates and notable events, or None.
        """
        pyboy = self._get_pyboy()
        if pyboy is None:
            return None

        group = pyboy.memory[_ACTIVE_GROUP]
        room_id = pyboy.memory[_ACTIVE_ROOM]
        pixel_x = pyboy.memory[_PLAYER_X]
        pixel_y = pyboy.memory[_PLAYER_Y]
        direction = pyboy.memory[_PLAYER_DIR]
        scroll = pyboy.memory[_SCROLL_MODE]

        # Skip during transitions (position is invalid)
        if (scroll & 0x80) != 0:
            return None

        # Convert to tile coordinates
        tile_x = pixel_x // 16
        tile_y = pixel_y // 16

        # Convert to world coordinates (matching overworld.png layout)
        room_col = room_id % 16
        room_row = room_id // 16
        world_x = room_col * 10 + min(tile_x, 9)
        world_y = room_row * 8 + min(tile_y, 7)

        # Detect notable events
        notable = ""
        if group == 0:  # Overworld
            flags = pyboy.memory[_OVERWORLD_FLAGS + room_id]
            prev_flags = self._prev_room_flags.get(room_id, 0)

            if (flags & _FLAG_GATE_HIT) and not (prev_flags & _FLAG_GATE_HIT):
                notable = self.NOTABLE_GATE
            elif (flags & _FLAG_ITEM_OBTAINED) and not (prev_flags & _FLAG_ITEM_OBTAINED):
                notable = self.NOTABLE_ITEM
            elif room_id not in self._visited_rooms:
                notable = self.NOTABLE_NEW_ROOM

            self._prev_room_flags[room_id] = flags
            self._visited_rooms.add(room_id)

        return {
            "x": world_x,
            "y": world_y,
            "z": group,
            "room_id": room_id,
            "tile_x": tile_x,
            "tile_y": tile_y,
            "direction": direction,
            "notable": notable,
        }

    def _flush_buffer(self):
        """Send buffered coordinates over WebSocket."""
        if not self._coord_buffer:
            return

        message = json.dumps({
            "name": "stream",
            "version": 1,
            "metadata": {
                **self.metadata,
                "extra": f"rooms: {len(self._visited_rooms)}",
            },
            "pos_data": [
                {"x": c["x"], "y": c["y"], "z": c["z"], "notable": c["notable"]}
                for c in self._coord_buffer
            ],
        })

        try:
            if self._websocket is None:
                self._connect_ws()
            if self._websocket is not None:
                self._websocket.send(message)
        except Exception:
            self._websocket = None

        self._coord_buffer.clear()

    def step(self, action):
        """Step the environment and stream position data.

        The wrapper is fully transparent — it returns exactly what the
        inner environment returns.
        """
        result = self.env.step(action)
        self._step_counter += 1

        # Read and buffer position
        pos = self._read_position()
        if pos is not None:
            self._coord_buffer.append(pos)

            # CSV logging for offline rendering
            if self._csv_file is not None:
                self._csv_file.write(
                    f"{self._step_counter},{pos['x']},{pos['y']},{pos['z']},"
                    f"{pos['room_id']},{pos['tile_x']},{pos['tile_y']},"
                    f"{pos['direction']},{pos['notable']}\n"
                )

        # Flush buffer at upload interval
        if self._step_counter % self.upload_interval == 0:
            self._flush_buffer()
            if self._csv_file is not None:
                self._csv_file.flush()

        return result

    def reset(self, **kwargs):
        """Reset the environment and clear tracking state."""
        result = self.env.reset(**kwargs)
        self._visited_rooms.clear()
        self._prev_room_flags.clear()
        self._step_counter = 0
        self._coord_buffer.clear()
        return result

    def close(self):
        """Flush remaining data and close connections."""
        self._flush_buffer()
        if self._websocket is not None:
            try:
                self._websocket.close()
            except Exception:
                pass
        if self._csv_file is not None:
            self._csv_file.close()
        super().close()
