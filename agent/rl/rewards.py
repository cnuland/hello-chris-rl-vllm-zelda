"""Reward shaping: coverage, RND curiosity, potential-based RLAIF.

Coverage reward: +0.2 per first-visit tile per room (8×8 bin).
RND curiosity: clamped to <= 30% of extrinsic reward.
RLAIF shaping: r' = r + lambda * R_phi (potential-based).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CoverageReward:
    """Track tile coverage per room and award first-visit bonuses.

    Tiles are binned into 8×8 grids within each room (160×144 screen
    → 20×18 tiles → 8×8 bins ≈ 2-3 tiles per bin).
    """

    bonus_per_tile: float = 0.2
    bonus_per_room: float = 20.0
    revisit_penalty: float = -0.5
    _visited: dict[int, set[tuple[int, int]]] = field(default_factory=dict)
    _visited_rooms: set[int] = field(default_factory=set)

    def step(self, room_id: int, pixel_x: int, pixel_y: int) -> float:
        """Return coverage reward for this step."""
        reward = 0.0

        # New room bonus
        if room_id not in self._visited_rooms:
            self._visited_rooms.add(room_id)
            reward += self.bonus_per_room

        # Tile bin (8×8 grid)
        bin_x = pixel_x // 20  # 160 / 8 = 20
        bin_y = pixel_y // 18  # 144 / 8 = 18
        tile = (bin_x, bin_y)

        if room_id not in self._visited:
            self._visited[room_id] = set()

        if tile not in self._visited[room_id]:
            self._visited[room_id].add(tile)
            reward += self.bonus_per_tile
        else:
            reward += self.revisit_penalty

        return reward

    def reset(self) -> None:
        self._visited.clear()
        self._visited_rooms.clear()

    @property
    def unique_rooms(self) -> int:
        return len(self._visited_rooms)

    @property
    def total_tiles(self) -> int:
        return sum(len(tiles) for tiles in self._visited.values())


class RNDCuriosity:
    """Random Network Distillation curiosity bonus.

    Maintains a fixed random target network and a trainable predictor.
    Curiosity = MSE between predictor and target outputs.
    Clamped to <= 30% of extrinsic reward magnitude.
    """

    def __init__(
        self,
        obs_dim: int = 128,
        embed_dim: int = 64,
        max_ratio: float = 0.3,
        learning_rate: float = 1e-3,
    ):
        self.max_ratio = max_ratio
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self._lr = learning_rate
        self._initialized = False

        # Lazy init to avoid torch import at module level
        self._target_net = None
        self._predictor_net = None
        self._optimizer = None

    def _lazy_init(self) -> None:
        if self._initialized:
            return
        import torch
        import torch.nn as nn

        self._target_net = nn.Sequential(
            nn.Linear(self.obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.embed_dim),
        )
        for p in self._target_net.parameters():
            p.requires_grad = False

        self._predictor_net = nn.Sequential(
            nn.Linear(self.obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.embed_dim),
        )
        self._optimizer = torch.optim.Adam(self._predictor_net.parameters(), lr=self._lr)
        self._initialized = True

    def compute(self, obs: np.ndarray, extrinsic_reward: float) -> float:
        """Compute clamped curiosity bonus and update predictor."""
        self._lazy_init()
        import torch

        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            target = self._target_net(obs_t)
        pred = self._predictor_net(obs_t)
        mse = ((pred - target) ** 2).mean()

        # Train predictor
        self._optimizer.zero_grad()
        mse.backward()
        self._optimizer.step()

        curiosity = mse.item()
        # Clamp to max_ratio of extrinsic
        cap = abs(extrinsic_reward) * self.max_ratio
        return min(curiosity, cap) if cap > 0 else min(curiosity, 0.1)


@dataclass
class PotentialShaping:
    """Potential-based reward shaping from RLAIF reward model.

    r' = r + gamma * phi(s') - phi(s)
    where phi(s) = lambda * R_phi(s).

    This preserves optimal policy (Ng et al. 1999).
    """

    gamma: float = 0.99
    lam: float = 0.15  # lambda weight for R_phi
    _prev_potential: float = 0.0

    def shape(self, extrinsic: float, phi_s_prime: float) -> float:
        """Apply potential-based shaping.

        Args:
            extrinsic: Raw extrinsic reward.
            phi_s_prime: R_phi(s') from the reward model.

        Returns:
            Shaped reward.
        """
        potential = self.lam * phi_s_prime
        shaped = extrinsic + self.gamma * potential - self._prev_potential
        self._prev_potential = potential
        return shaped

    def reset(self) -> None:
        self._prev_potential = 0.0
