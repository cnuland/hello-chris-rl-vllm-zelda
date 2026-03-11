"""Reward shaping: coverage with decay and potential-based RLAIF.

Coverage reward: exploration bonus per unique tile and room, with
optional decay to incentivize continuous exploration (Pokemon Red style).

RLAIF shaping: r' = r + gamma * phi(s') - phi(s) (potential-based).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CoverageReward:
    """Track coordinate coverage with decaying exploration values.

    Each tile starts at ``exploration_inc`` when first visited. Over time,
    tile values decay by ``decay_factor`` every ``decay_frequency`` steps,
    with a floor of ``decay_floor``. This incentivizes the agent to keep
    exploring rather than exhausting nearby tiles and stopping.

    Room values work the same way — first visit = ``room_inc``, then decay.

    The total exploration value is ``sum(tile_values) + sum(room_values)``.
    In a delta-based reward system, the step reward from exploration is
    the change in this total, which can be negative if decay outpaces
    new discovery.
    """

    exploration_inc: float = 1.0      # Value assigned to newly visited tile
    room_inc: float = 1.0             # Value assigned to newly visited room
    decay_factor: float = 0.9995      # Multiply values by this each decay tick
    decay_frequency: int = 10         # Decay every N steps
    decay_floor: float = 0.15         # Minimum non-zero value (never fully forgotten)

    # Internal state
    _tile_values: dict[tuple[int, int, int], float] = field(default_factory=dict)
    _room_values: dict[int, float] = field(default_factory=dict)
    _step_count: int = 0

    def step(self, room_id: int, pixel_x: int, pixel_y: int) -> None:
        """Update coverage for this step (does NOT return reward directly).

        In delta-based mode, the reward is computed externally from
        the change in total_value().
        """
        self._step_count += 1

        # New room
        if room_id not in self._room_values:
            self._room_values[room_id] = self.room_inc

        # Fine-grained coordinate: (room_id, tile_x, tile_y)
        tile_x = pixel_x // 16
        tile_y = pixel_y // 16
        coord = (room_id, tile_x, tile_y)

        if coord not in self._tile_values:
            self._tile_values[coord] = self.exploration_inc

        # Periodic decay
        if self._step_count % self.decay_frequency == 0:
            self._decay()

    def _decay(self) -> None:
        """Apply decay to all tracked tile and room values."""
        for coord in self._tile_values:
            val = self._tile_values[coord] * self.decay_factor
            self._tile_values[coord] = max(self.decay_floor, val)
        for room_id in self._room_values:
            val = self._room_values[room_id] * self.decay_factor
            self._room_values[room_id] = max(self.decay_floor, val)

    def total_tile_value(self) -> float:
        """Sum of all tile exploration values (with decay applied)."""
        return sum(self._tile_values.values())

    def total_room_value(self) -> float:
        """Sum of all room exploration values (with decay applied)."""
        return sum(self._room_values.values())

    def reset(self) -> None:
        self._tile_values.clear()
        self._room_values.clear()
        self._step_count = 0

    @property
    def unique_rooms(self) -> int:
        return len(self._room_values)

    @property
    def total_tiles(self) -> int:
        return len(self._tile_values)

    @property
    def _visited_rooms(self) -> set[int]:
        """Compatibility: set of visited room IDs."""
        return set(self._room_values.keys())


@dataclass
class PotentialShaping:
    """Potential-based reward shaping from RLAIF reward model.

    r' = r + gamma * phi(s') - phi(s)
    where phi(s) = lambda * R_phi(s).

    This preserves optimal policy (Ng et al. 1999).

    Lambda decays with epoch: effective_lam = lam * decay_rate^epoch.
    This gradually fades the shaping signal as the extrinsic rewards
    become more reliable (the agent learns to hit milestones consistently).
    """

    gamma: float = 0.99
    lam: float = 0.05
    epoch: int = 0
    decay_rate: float = 0.95
    _prev_potential: float = 0.0

    @property
    def effective_lam(self) -> float:
        """Lambda with epoch-based decay."""
        return self.lam * (self.decay_rate ** self.epoch)

    def shape(self, extrinsic: float, phi_s_prime: float) -> float:
        potential = self.effective_lam * phi_s_prime
        shaped = extrinsic + self.gamma * potential - self._prev_potential
        self._prev_potential = potential
        return shaped

    def reset(self) -> None:
        self._prev_potential = 0.0
