"""Tests for state encoder: shape checks and JSON schema validation."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from agent.env.state_encoder import VECTOR_SIZE, encode_json, encode_vector


class FakeEnv:
    """Minimal mock of ZeldaEnv for state encoder tests."""

    def __init__(self):
        self._memory = {i: 0 for i in range(0x10000)}
        # Set some realistic defaults
        self._memory[0xC4AC] = 80  # player X
        self._memory[0xC4AD] = 72  # player Y
        self._memory[0xC4AE] = 2  # facing down
        self._memory[0xC63B] = 0x10  # room
        self._memory[0xC021] = 12  # 3 hearts (12 quarter-hearts)
        self._memory[0xC022] = 12  # 3 max hearts
        self._memory[0xC6A5] = 50  # 50 rupees
        self._memory[0xC668] = 1  # sword level 1
        self._memory[0xC680] = 0  # spring

    def _read(self, addr: int) -> int:
        return self._memory.get(addr, 0)

    def _read16(self, addr: int) -> int:
        return self._read(addr) | (self._read(addr + 1) << 8)


class TestEncodeVector:
    def test_shape(self):
        env = FakeEnv()
        v = encode_vector(env)
        assert v.shape == (VECTOR_SIZE,)
        assert v.dtype == np.float32

    def test_values_normalized(self):
        env = FakeEnv()
        v = encode_vector(env)
        assert np.all(v >= 0.0)
        assert np.all(v <= 1.0)

    def test_player_position_encoded(self):
        env = FakeEnv()
        v = encode_vector(env)
        # Player X at index 0: 80/255 ≈ 0.314
        assert 0.3 < v[0] < 0.35
        # Player Y at index 1: 72/255 ≈ 0.282
        assert 0.25 < v[1] < 0.3

    def test_determinism(self):
        env = FakeEnv()
        v1 = encode_vector(env)
        v2 = encode_vector(env)
        np.testing.assert_array_equal(v1, v2)


class TestEncodeJson:
    def test_has_required_keys(self):
        env = FakeEnv()
        state = encode_json(env)
        assert "player" in state
        assert "room_id" in state
        assert "inventory" in state
        assert "flags" in state
        assert "interactables" in state

    def test_player_fields(self):
        env = FakeEnv()
        state = encode_json(env)
        player = state["player"]
        assert "x" in player
        assert "y" in player
        assert "dir" in player
        assert "hp" in player
        assert "max_hp" in player
        assert player["x"] == 80
        assert player["y"] == 72
        assert player["hp"] == 3

    def test_flags_are_boolean(self):
        env = FakeEnv()
        state = encode_json(env)
        flags = state["flags"]
        assert isinstance(flags["dialog"], bool)
        assert isinstance(flags["puzzle"], bool)
        assert isinstance(flags["cutscene"], bool)

    def test_inventory_has_numeric_values(self):
        env = FakeEnv()
        state = encode_json(env)
        inv = state["inventory"]
        assert isinstance(inv["sword"], int)
        assert isinstance(inv["rupees"], int)
        assert inv["sword"] == 1
        assert inv["rupees"] == 50

    def test_interactables_is_list(self):
        env = FakeEnv()
        state = encode_json(env)
        assert isinstance(state["interactables"], list)
