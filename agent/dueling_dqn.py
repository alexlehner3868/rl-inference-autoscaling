from __future__ import annotations

import torch as th
from torch import nn

from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork


class DuelingQNetwork(QNetwork):
    def __init__(
        self,
        observation_space,
        action_space,
        features_extractor,
        features_dim,
        net_arch=None,
        activation_fn=nn.ReLU,
        normalize_images=True,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
        )

        del self.q_net

        if net_arch is None:
            net_arch = [64, 64]

        layers = []
        input_dim = features_dim

        for hidden_dim in net_arch:
            layers.append(
                nn.Linear(
                    input_dim,
                    hidden_dim,
                )
            )

            layers.append(
                activation_fn()
            )

            input_dim = hidden_dim

        self.shared_net = nn.Sequential(
            *layers
        )

        self.value_net = nn.Linear(
            input_dim,
            1,
        )

        self.advantage_net = nn.Linear(
            input_dim,
            int(action_space.n),
        )

    def forward(
        self,
        obs: PyTorchObs,
    ) -> th.Tensor:
        features = self.extract_features(
            obs,
            self.features_extractor,
        )

        latent = self.shared_net(
            features
        )

        value = self.value_net(
            latent
        )

        advantage = self.advantage_net(
            latent
        )

        return (
            value
            + advantage
            - advantage.mean(
                dim=1,
                keepdim=True,
            )
        )


class DuelingDQNPolicy(DQNPolicy):
    def make_q_net(
        self,
    ) -> DuelingQNetwork:
        net_args = self._update_features_extractor(
            self.net_args,
            features_extractor=None,
        )

        return DuelingQNetwork(
            **net_args
        ).to(
            self.device
        )