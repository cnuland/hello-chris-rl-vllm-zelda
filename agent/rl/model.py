"""Custom MLP policy/value model for Ray RLlib."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class ZeldaMLPModel(TorchModelV2, nn.Module):
    """Simple 3-layer MLP with separate policy and value heads.

    Input: 128-D vector observation.
    Shared: 3×Linear(256) + ReLU.
    Policy head: Linear(256) → num_actions logits.
    Value head: Linear(256) → 1 scalar.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        in_size = int(np.prod(obs_space.shape))
        hidden = model_config.get("fcnet_hiddens", [256])[0]

        self.shared = nn.Sequential(
            nn.Linear(in_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, num_outputs)
        self.value_head = nn.Linear(hidden, 1)

        # Orthogonal init
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

        self._features = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        self._features = self.shared(obs)
        logits = self.policy_head(self._features)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "forward() must be called first"
        return self.value_head(self._features).squeeze(-1)
