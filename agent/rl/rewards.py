"""Reward shaping: coverage, RND curiosity, potential-based RLAIF.

Coverage reward: count-based exploration bonus per tile per room.
RND curiosity: clamped to <= 30% of extrinsic reward, with obs normalization.
RLAIF shaping: r' = r + lambda * R_phi (potential-based).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CoverageReward:
    """Track coordinate coverage with count-based diminishing returns.

    Uses fine-grained (room_id, tile_x, tile_y) coordinate tracking.
    Each 16×16-pixel tile position is a unique coordinate, giving ~80
    coords per room (10 cols × 8 rows).

    Count-based exploration: each coord tracks its visit count N.
    Reward per visit = bonus / sqrt(N).  This means:
      - 1st visit: full bonus (1/sqrt(1) = 1.0)
      - 2nd visit: 71% bonus (1/sqrt(2) ≈ 0.707)
      - 4th visit: 50% bonus (1/sqrt(4) = 0.5)
      - 9th visit: 33% bonus (1/sqrt(9) ≈ 0.333)
      - 100th visit: 10% bonus (1/sqrt(100) = 0.1)

    Unlike coord decay, this never recovers — revisiting the same tile
    always yields less reward, preventing circular exploitation.
    O(1) per step.
    """

    bonus_per_tile: float = 0.1
    bonus_per_room: float = 10.0
    # coord → visit count
    _visit_counts: dict[tuple[int, int, int], int] = field(default_factory=dict)
    _visited_rooms: set[int] = field(default_factory=set)

    def step(self, room_id: int, pixel_x: int, pixel_y: int) -> float:
        """Return coverage reward for this step."""
        reward = 0.0

        # New room bonus — flat per room
        if room_id not in self._visited_rooms:
            self._visited_rooms.add(room_id)
            reward += self.bonus_per_room

        # Fine-grained coordinate: (room_id, tile_x, tile_y)
        tile_x = pixel_x // 16
        tile_y = pixel_y // 16
        coord = (room_id, tile_x, tile_y)

        # Count-based: reward = bonus / sqrt(visit_count)
        count = self._visit_counts.get(coord, 0) + 1
        self._visit_counts[coord] = count
        reward += self.bonus_per_tile / math.sqrt(count)

        return reward

    def reset(self) -> None:
        self._visit_counts.clear()
        self._visited_rooms.clear()

    @property
    def unique_rooms(self) -> int:
        return len(self._visited_rooms)

    @property
    def total_tiles(self) -> int:
        return len(self._visit_counts)


class RNDCuriosity:
    """Random Network Distillation curiosity bonus.

    Maintains a fixed random target network and a trainable predictor.
    Curiosity = MSE between predictor and target outputs.
    Clamped to <= 30% of extrinsic reward magnitude.

    Includes running observation normalization for stable RND predictions.
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

        # Running observation normalization
        self._obs_mean = np.zeros(obs_dim, dtype=np.float64)
        self._obs_var = np.ones(obs_dim, dtype=np.float64)
        self._obs_count = 0

    def _lazy_init(self) -> None:
        if self._initialized:
            return
        import torch
        import torch.nn as nn

        # Force CPU: RND runs inside PufferLib worker processes which
        # are forked and cannot access the parent's CUDA context.
        self._target_net = nn.Sequential(
            nn.Linear(self.obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.embed_dim),
        ).cpu()
        for p in self._target_net.parameters():
            p.requires_grad = False

        self._predictor_net = nn.Sequential(
            nn.Linear(self.obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.embed_dim),
        ).cpu()
        self._optimizer = torch.optim.Adam(self._predictor_net.parameters(), lr=self._lr)
        self._initialized = True

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Running normalization for stable RND predictions."""
        self._obs_count += 1
        delta = obs - self._obs_mean
        self._obs_mean += delta / self._obs_count
        delta2 = obs - self._obs_mean
        self._obs_var += (delta * delta2 - self._obs_var) / self._obs_count
        std = np.sqrt(self._obs_var + 1e-8)
        return (obs - self._obs_mean) / std

    def compute(self, obs: np.ndarray, extrinsic_reward: float) -> float:
        """Compute clamped curiosity bonus and update predictor."""
        self._lazy_init()
        import torch

        normed = self._normalize_obs(obs.astype(np.float64)).astype(np.float32)
        obs_t = torch.from_numpy(normed).float().unsqueeze(0)
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
    lam: float = 0.05  # lambda weight for R_phi (conservative to avoid destabilization)
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
