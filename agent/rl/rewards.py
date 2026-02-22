"""Reward shaping: coverage and potential-based RLAIF.

Coverage reward: binary exploration bonus per unique tile and room.
RLAIF shaping: r' = r + gamma * phi(s') - phi(s) (potential-based).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CoverageReward:
    """Track coordinate coverage with binary exploration rewards.

    Uses fine-grained (room_id, tile_x, tile_y) coordinate tracking.
    Each 16×16-pixel tile position is a unique coordinate, giving ~80
    coords per room (10 cols × 8 rows).

    Binary exploration: first visit to a tile = bonus_per_tile,
    all subsequent visits = 0.  This is the same approach used by
    PokemonRedExperiments and LADXExperiments — simple, no decay,
    no diminishing returns.

    New room discovery gives a one-time bonus_per_room.
    """

    bonus_per_tile: float = 0.1
    bonus_per_room: float = 10.0
    # coord → visited flag (set membership)
    _visited_coords: set[tuple[int, int, int]] = field(default_factory=set)
    _visited_rooms: set[int] = field(default_factory=set)

    def step(self, room_id: int, pixel_x: int, pixel_y: int) -> float:
        """Return coverage reward for this step."""
        reward = 0.0

        # New room bonus — flat per room, one-time only
        if room_id not in self._visited_rooms:
            self._visited_rooms.add(room_id)
            reward += self.bonus_per_room

        # Fine-grained coordinate: (room_id, tile_x, tile_y)
        tile_x = pixel_x // 16
        tile_y = pixel_y // 16
        coord = (room_id, tile_x, tile_y)

        # Binary: first visit = bonus, revisit = 0
        if coord not in self._visited_coords:
            self._visited_coords.add(coord)
            reward += self.bonus_per_tile

        return reward

    def reset(self) -> None:
        self._visited_coords.clear()
        self._visited_rooms.clear()

    def reset_tiles(self) -> None:
        """Reset tile coverage but keep room history.

        This makes room bonuses one-shot within an epoch — once you visit
        a room, you only get the bonus the first time.  Tile coverage
        resets each episode so the agent still has per-step exploration
        reward in known rooms.  Mirrors PokemonRed's persistent seen_coords.
        """
        self._visited_coords.clear()

    @property
    def unique_rooms(self) -> int:
        return len(self._visited_rooms)

    @property
    def total_tiles(self) -> int:
        return len(self._visited_coords)


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
    lam: float = 0.05  # lambda weight for R_phi (conservative to avoid destabilization)
    epoch: int = 0
    decay_rate: float = 0.95  # lam decays by 5% per epoch
    _prev_potential: float = 0.0

    @property
    def effective_lam(self) -> float:
        """Lambda with epoch-based decay."""
        return self.lam * (self.decay_rate ** self.epoch)

    def shape(self, extrinsic: float, phi_s_prime: float) -> float:
        """Apply potential-based shaping.

        Args:
            extrinsic: Raw extrinsic reward.
            phi_s_prime: R_phi(s') from the reward model.

        Returns:
            Shaped reward.
        """
        potential = self.effective_lam * phi_s_prime
        shaped = extrinsic + self.gamma * potential - self._prev_potential
        self._prev_potential = potential
        return shaped

    def reset(self) -> None:
        self._prev_potential = 0.0
