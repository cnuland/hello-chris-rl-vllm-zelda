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
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 1),
        )
        self._optimizer = torch.optim.Adam(
            self._net.parameters(), lr=self._lr, weight_decay=1e-4,
        )
        self._initialized = True

    def predict(self, obs: np.ndarray) -> float:
        """Predict reward for a single observation."""
        self._lazy_init()
        import torch

        with torch.no_grad():
            x = torch.from_numpy(obs).float().unsqueeze(0)
            return self._net(x).item()

    def train_on_preferences(
        self, prefs: list[tuple[np.ndarray, np.ndarray, float]],
        n_epochs: int = 5,
    ) -> float:
        """Train on pairwise preferences with multiple epochs.

        Args:
            prefs: List of (obs_a, obs_b, label) where label=1 means a>b, 0 means b>a.
            n_epochs: Number of training passes over the data.

        Returns:
            Average training loss from the final epoch.
        """
        if not prefs:
            return 0.0

        self._lazy_init()
        import random

        import torch
        import torch.nn.functional as F

        final_loss = 0.0
        for epoch in range(n_epochs):
            self._net.train()
            epoch_loss = 0.0
            shuffled = list(prefs)
            random.shuffle(shuffled)
            for obs_a, obs_b, label in shuffled:
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
                epoch_loss += loss.item()

            final_loss = epoch_loss / len(prefs)
            logger.info("Reward model epoch %d/%d loss: %.4f", epoch + 1, n_epochs, final_loss)

        self._net.eval()
        return final_loss

    def save(self, path: str) -> None:
        import torch

        if self._net:
            torch.save(self._net.state_dict(), path)

    def load(self, path: str) -> None:
        import torch

        self._lazy_init()
        self._net.load_state_dict(torch.load(path, weights_only=False, map_location="cpu"))


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
    """Convert segment data to a 128-D observation matching state_encoder format.

    Reconstructs actual game state features from the segment's state data
    so the reward model trains on the same input distribution it sees at
    inference time in reward_wrapper.py.
    """
    obs = np.zeros(128, dtype=np.float32)

    # Use mid-point state from the segment's trajectory
    states = segment.get("states", [])
    if not states:
        # Fallback: just use judge scores in reserved dims
        scores = segment.get("scores", {})
        for i, key in enumerate(["progress", "dialog", "puzzle", "novelty", "efficiency"]):
            obs[90 + i] = scores.get(key, 0.0)
        return obs

    mid = states[len(states) // 2]
    st = mid.get("state", {})

    # --- Dims 0-3: Position & room (matches state_encoder layout) ---
    obs[0] = st.get("pixel_x", 0) / 255.0
    obs[1] = st.get("pixel_y", 0) / 255.0
    # dim 2 (direction) not in segment state, leave 0
    obs[3] = st.get("room_id", 0) / 255.0

    # --- Dims 4-6: Health ---
    health = st.get("health", 0)
    max_health = st.get("max_health", 1)
    obs[4] = health / 20.0
    obs[5] = max_health / 20.0
    obs[6] = health / max(max_health, 1)

    # --- Dims 26-29: Dungeon progress ---
    obs[28] = st.get("dungeon_floor", 0) / 10.0
    # essences not in segment state

    # --- Dims 30-33: Flags ---
    obs[30] = 1.0 if st.get("dialog_active") else 0.0
    obs[32] = 1.0  # transition always set mid-segment

    # --- Dims 34-36: Entity counts ---
    obs[34] = min(st.get("sprites", 0) / 10.0, 1.0)

    # --- Segment-level aggregate features (reserved dims 90+) ---
    # Unique rooms visited across the segment
    rooms = set()
    dialog_count = 0
    for s in states:
        frame_st = s.get("state", {})
        rooms.add(frame_st.get("room_id", 0))
        if frame_st.get("dialog_active"):
            dialog_count += 1
    obs[90] = min(len(rooms) / 10.0, 1.0)
    obs[91] = min(dialog_count / len(states), 1.0)
    obs[92] = st.get("active_group", 0) / 5.0  # 0=overworld, 4-5=dungeon

    # Judge scores (so model can learn from them too)
    scores = segment.get("scores", {})
    for i, key in enumerate(["progress", "dialog", "puzzle", "novelty", "efficiency"]):
        obs[95 + i] = scores.get(key, 0.0)
    obs[100] = segment.get("weighted_score", 0.0)

    return np.clip(obs, 0.0, 1.0)


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
