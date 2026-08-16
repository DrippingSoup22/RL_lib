"""Neural policy models for discrete action spaces."""

import torch
import torch.nn as nn


class DiscretePolicyNetwork(nn.Module):
    """Map observations to one unnormalized logit per discrete action."""

    def __init__(
        self,
        observation_size: int,
        number_of_actions: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()

        if observation_size <= 0:
            raise ValueError("Observation size must be greater than 0")
        if number_of_actions <= 0:
            raise ValueError("Number of actions must be greater than 0")
        if any(hidden_size <= 0 for hidden_size in hidden_sizes):
            raise ValueError("All hidden sizes must be greater than 0")

        self.observation_size = observation_size
        self.number_of_actions = number_of_actions
        self.hidden_sizes = hidden_sizes

        input_size = observation_size
        layers = []

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, self.number_of_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)
