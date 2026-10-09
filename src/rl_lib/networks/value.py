"""Small multilayer perceptrons for state and action values."""

import torch
import torch.nn as nn

from rl_lib.networks.activations import hidden_activation


class StateValueNetwork(nn.Module):
    """Map one observation, or a batch of observations, to state values.

    ``activation`` is the hidden layers' activation, ``"relu"`` or ``"tanh"``.
    """

    def __init__(
        self,
        observation_size: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        activation: str = "relu",
    ) -> None:
        super().__init__()

        if observation_size <= 0:
            raise ValueError("Observation size must be greater than 0")
        if any(hidden_size <= 0 for hidden_size in hidden_sizes):
            raise ValueError("All hidden sizes must be greater than 0")
        activation_class = hidden_activation(activation)

        self.observation_size = observation_size
        self.hidden_sizes = hidden_sizes

        input_size = observation_size
        layers = []

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(activation_class())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation).squeeze(-1)


class ActionValueNetwork(nn.Module):
    """Map observations to one action value per discrete action."""

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
