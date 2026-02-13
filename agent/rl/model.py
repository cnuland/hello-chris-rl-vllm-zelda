"""Custom MLP policy/value model for Ray RLlib."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class ZeldaMLPModel(TorchModelV2, nn.Module):
    """MLP with separate policy and value networks.

    Input: 128-D vector observation.
    Policy net: 3×Linear(256) + ReLU → num_actions logits.
    Value net:  3×Linear(256) + ReLU → 1 scalar.

    Separate trunks prevent value gradients from corrupting the policy,
    which is critical when reward variance is high (room bonuses are
    sparse and large).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        in_size = int(np.prod(obs_space.shape))
        hidden = model_config.get("fcnet_hiddens", [256])[0]

        # Separate policy network
        self.policy_net = nn.Sequential(
            nn.Linear(in_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, num_outputs)

        # Separate value network
        self.value_net = nn.Sequential(
            nn.Linear(in_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden, 1)

        # Orthogonal init — policy net
        for layer in self.policy_net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

        # Orthogonal init — value net
        for layer in self.value_net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

        self._obs = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._obs = input_dict["obs_flat"].float()
        features = self.policy_net(self._obs)
        logits = self.policy_head(features)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._obs is not None, "forward() must be called first"
        features = self.value_net(self._obs)
        return self.value_head(features).squeeze(-1)
