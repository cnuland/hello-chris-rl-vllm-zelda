"""Preference-based reward model (Bradley-Terry) + SIL/AWR self-imitation.

Pipeline:
  1. Build pairwise preferences from judge scores.
  2. Train R_phi via Bradley-Terry logistic loss.
  3. Wrap as potential-based shaping for the next RL burst.
  4. Maintain top-K segment buffer for self-imitation (SIL/AWR).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RewardModel:
    """Bradley-Terry reward model trained on pairwise preferences.

    R_phi: observation → scalar reward estimate.
    Training: P(a > b) = sigmoid(R(a) - R(b)).
    """

    def __init__(self, obs_dim: int = 128, hidden_dim: int = 64, lr: float = 1e-3):
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self._lr = lr
        self._net = None
        self._optimizer = None
        self._initialized = False

    def _lazy_init(self) -> None:
        if self._initialized:
            return
        import torch
        import torch.nn as nn

        self._net = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr)
        self._initialized = True

    def predict(self, obs: np.ndarray) -> float:
        """Predict reward for a single observation."""
        self._lazy_init()
        import torch

        with torch.no_grad():
            x = torch.from_numpy(obs).float().unsqueeze(0)
            return self._net(x).item()

    def train_on_preferences(
        self, prefs: list[tuple[np.ndarray, np.ndarray, float]]
    ) -> float:
        """Train on pairwise preferences.

        Args:
            prefs: List of (obs_a, obs_b, label) where label=1 means a>b, 0 means b>a.

        Returns:
            Average training loss.
        """
        self._lazy_init()
        import torch
        import torch.nn.functional as F

        total_loss = 0.0
        for obs_a, obs_b, label in prefs:
            a = torch.from_numpy(obs_a).float().unsqueeze(0)
            b = torch.from_numpy(obs_b).float().unsqueeze(0)
            target = torch.tensor([label], dtype=torch.float32)

            r_a = self._net(a)
            r_b = self._net(b)
            # Bradley-Terry: P(a>b) = sigmoid(R(a) - R(b))
            logit = r_a - r_b
            loss = F.binary_cross_entropy_with_logits(logit.squeeze(), target.squeeze())

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(prefs), 1)

    def save(self, path: str) -> None:
        import torch

        if self._net:
            torch.save(self._net.state_dict(), path)

    def load(self, path: str) -> None:
        import torch

        self._lazy_init()
        self._net.load_state_dict(torch.load(path, weights_only=True))


def build_preferences(
    scores: list[dict[str, Any]],
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Build pairwise preferences from judge scores.

    Compares all pairs; the one with higher weighted_score wins.
    Returns synthetic observation pairs (using segment-level features).
    """
    prefs = []
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            a = scores[i]
            b = scores[j]
            wa = a.get("weighted_score", 0.0)
            wb = b.get("weighted_score", 0.0)

            if abs(wa - wb) < 0.05:
                continue  # skip ties

            obs_a = _segment_to_obs(a)
            obs_b = _segment_to_obs(b)
            label = 1.0 if wa > wb else 0.0
            prefs.append((obs_a, obs_b, label))

    return prefs


def _segment_to_obs(segment: dict[str, Any]) -> np.ndarray:
    """Convert segment scores to a pseudo-observation for the reward model."""
    obs = np.zeros(128, dtype=np.float32)
    scores = segment.get("scores", {})
    for i, key in enumerate(["progress", "dialog", "puzzle", "novelty", "efficiency"]):
        obs[i] = scores.get(key, 0.0)
    obs[5] = segment.get("weighted_score", 0.0)
    return obs


class SelfImitationBuffer:
    """Top-K segment buffer for SIL/AWR self-imitation learning.

    Keeps the K best-scored segments. Periodically provides them
    for imitation updates in the PPO training loop.
    """

    def __init__(self, capacity: int = 100):
        self._capacity = capacity
        self._buffer: list[dict[str, Any]] = []

    def add(self, segment: dict[str, Any]) -> None:
        """Add a scored segment, maintaining top-K by weighted_score."""
        self._buffer.append(segment)
        self._buffer.sort(key=lambda s: s.get("weighted_score", 0.0), reverse=True)
        if len(self._buffer) > self._capacity:
            self._buffer = self._buffer[: self._capacity]

    def sample(self, k: int = 10) -> list[dict[str, Any]]:
        """Sample up to k segments from the buffer."""
        return self._buffer[: min(k, len(self._buffer))]

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def min_score(self) -> float:
        if not self._buffer:
            return 0.0
        return self._buffer[-1].get("weighted_score", 0.0)
