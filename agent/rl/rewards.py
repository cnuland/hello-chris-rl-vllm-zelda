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
    """Random Network Distillation curiosity bonus (pure numpy).

    Maintains a fixed random target network and a trainable predictor.
    Curiosity = MSE between predictor and target outputs.
    Clamped to <= 30% of extrinsic reward magnitude.

    Pure numpy implementation — no PyTorch dependency.  This is critical
    because PufferLib forks worker processes and PyTorch's autograd engine
    is fundamentally incompatible with fork-based multiprocessing.

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

        # Running observation normalization
        self._obs_mean = np.zeros(obs_dim, dtype=np.float64)
        self._obs_var = np.ones(obs_dim, dtype=np.float64)
        self._obs_count = 0

        # Kaiming uniform initialization (matches PyTorch nn.Linear default)
        def _kaiming(out_dim: int, in_dim: int) -> np.ndarray:
            bound = 1.0 / np.sqrt(in_dim)
            return np.random.uniform(-bound, bound, (out_dim, in_dim)).astype(np.float32)

        # Target network (fixed): obs_dim -> 128 -> embed_dim
        self._t_w1 = _kaiming(128, obs_dim)
        self._t_b1 = np.zeros(128, dtype=np.float32)
        self._t_w2 = _kaiming(embed_dim, 128)
        self._t_b2 = np.zeros(embed_dim, dtype=np.float32)

        # Predictor network (trainable): obs_dim -> 128 -> 128 -> embed_dim
        self._p_w1 = _kaiming(128, obs_dim)
        self._p_b1 = np.zeros(128, dtype=np.float32)
        self._p_w2 = _kaiming(128, 128)
        self._p_b2 = np.zeros(128, dtype=np.float32)
        self._p_w3 = _kaiming(embed_dim, 128)
        self._p_b3 = np.zeros(embed_dim, dtype=np.float32)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    def _target_forward(self, x: np.ndarray) -> np.ndarray:
        h = self._relu(self._t_w1 @ x + self._t_b1)
        return self._t_w2 @ h + self._t_b2

    def _predictor_forward(self, x: np.ndarray) -> tuple:
        """Forward pass returning output and intermediates for backprop."""
        h1 = self._p_w1 @ x + self._p_b1
        a1 = self._relu(h1)
        h2 = self._p_w2 @ a1 + self._p_b2
        a2 = self._relu(h2)
        out = self._p_w3 @ a2 + self._p_b3
        return out, (x, h1, a1, h2, a2)

    def _predictor_backward(self, d_out: np.ndarray, cache: tuple) -> None:
        """Manual backprop + SGD update for the predictor."""
        x, h1, a1, h2, a2 = cache

        # Layer 3: out = W3 @ a2 + b3
        self._p_w3 -= self._lr * np.outer(d_out, a2)
        self._p_b3 -= self._lr * d_out
        d_a2 = self._p_w3.T @ d_out

        # ReLU 2
        d_h2 = d_a2 * (h2 > 0).astype(np.float32)

        # Layer 2: h2 = W2 @ a1 + b2
        self._p_w2 -= self._lr * np.outer(d_h2, a1)
        self._p_b2 -= self._lr * d_h2
        d_a1 = self._p_w2.T @ d_h2

        # ReLU 1
        d_h1 = d_a1 * (h1 > 0).astype(np.float32)

        # Layer 1: h1 = W1 @ x + b1
        self._p_w1 -= self._lr * np.outer(d_h1, x)
        self._p_b1 -= self._lr * d_h1

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
        normed = self._normalize_obs(obs.astype(np.float64)).astype(np.float32)
        # Clip to prevent overflow in matmul
        normed = np.clip(normed, -5.0, 5.0)

        target = self._target_forward(normed)
        pred, cache = self._predictor_forward(normed)

        diff = pred - target
        mse = float(np.mean(diff ** 2))

        # NaN guard — skip update if numerics broke
        if not np.isfinite(mse):
            return 0.0

        # Backprop: d_loss/d_pred = 2 * diff / N
        d_out = (2.0 * diff / len(diff)).astype(np.float32)
        self._predictor_backward(d_out, cache)

        # Clamp to max_ratio of extrinsic
        cap = abs(extrinsic_reward) * self.max_ratio
        return min(mse, cap) if cap > 0 else min(mse, 0.1)


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
