"""PufferLib-compatible policy for Zelda RL.

Plain PyTorch nn.Module with encode_observations/decode_actions split
to support PufferLib's recurrent mode.

Architecture matches agent/rl/model.py ZeldaMLPModel:
  Policy trunk: 3x Linear(128->256)+ReLU -> Linear(256->num_actions)
  Value trunk:  3x Linear(128->256)+ReLU -> Linear(256->1)
  Separate trunks (no shared layers) — prevents value gradients from
  corrupting the policy when reward variance is high.
  Orthogonal init: trunk sqrt(2), policy head 0.01, value head 1.0.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ZeldaPolicy(nn.Module):
    def __init__(self, env, hidden_size: int = 256):
        super().__init__()
        obs_size = int(np.prod(env.single_observation_space.shape))  # 128
        num_actions = env.single_action_space.n  # 7

        # Policy trunk (encoder for actions)
        self.policy_encoder = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, num_actions)

        # Value trunk (separate from policy)
        self.value_encoder = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_size, 1)

        # Orthogonal initialization matching ZeldaMLPModel
        for layer in self.policy_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

        for layer in self.value_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def encode_observations(self, obs):
        """Encode observations into hidden representation.

        When using PufferLib's recurrent wrapper, this is called first,
        the hidden is passed through LSTM, then decode_actions is called.

        We cache raw observations so the value trunk can use them directly
        (separate trunks — value does not share the policy encoder).
        """
        batch_size = obs.shape[0]
        flat = obs.reshape(batch_size, -1)
        self._obs_cache = flat
        return self.policy_encoder(flat)

    def decode_actions(self, hidden, lookup=None):
        """Decode hidden state into action logits and value.

        Args:
            hidden: output of encode_observations (or recurrent layer output)
            lookup: PufferLib lookup for action masking (unused)

        Returns:
            (logits, value) tuple
        """
        logits = self.policy_head(hidden)
        value_hidden = self.value_encoder(self._obs_cache)
        value = self.value_head(value_hidden).squeeze(-1)
        return logits, value

    def forward(self, obs):
        hidden = self.encode_observations(obs)
        return self.decode_actions(hidden)
