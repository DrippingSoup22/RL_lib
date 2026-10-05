"""Policy networks for categorical and continuous action spaces."""

import numpy as np
import torch
import torch.nn as nn


class CategoricalPolicyNetwork(nn.Module):
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


class GaussianPolicyNetwork(nn.Module):
    """Map observations to a diagonal Gaussian's mean and standard deviation."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        initial_std: float = 1.0,
        std_mode: str = "state_dependent",
    ) -> None:
        super().__init__()

        if observation_size <= 0:
            raise ValueError("Observation size must be greater than 0")
        if action_size <= 0:
            raise ValueError("Action size must be greater than 0")
        if any(hidden_size <= 0 for hidden_size in hidden_sizes):
            raise ValueError("All hidden sizes must be greater than 0")
        if not np.isfinite(initial_std) or initial_std <= 0:
            raise ValueError("Initial std must be finite and greater than 0")
        if std_mode not in ("state_dependent", "global"):
            raise ValueError("std_mode must be 'state_dependent' or 'global'")

        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.std_mode = std_mode

        mean_input_size = observation_size
        mean_layers = []

        for hidden_size in hidden_sizes:
            mean_layers.append(nn.Linear(mean_input_size, hidden_size))
            mean_layers.append(nn.ReLU())
            mean_input_size = hidden_size
        mean_layers.append(nn.Linear(mean_input_size, action_size))

        self.mean_network = nn.Sequential(*mean_layers)

        self.log_std_network: nn.Sequential | None
        self.log_std: nn.Parameter | None
        if std_mode == "state_dependent":
            std_input_size = observation_size
            log_std_layers = []
            for hidden_size in hidden_sizes:
                log_std_layers.append(nn.Linear(std_input_size, hidden_size))
                log_std_layers.append(nn.ReLU())
                std_input_size = hidden_size

            log_std_output = nn.Linear(std_input_size, action_size)
            nn.init.zeros_(log_std_output.weight)
            nn.init.constant_(log_std_output.bias, float(np.log(initial_std)))
            log_std_layers.append(log_std_output)
            self.log_std_network = nn.Sequential(*log_std_layers)
            self.log_std = None
        else:
            self.log_std_network = None
            self.log_std = nn.Parameter(
                torch.full((action_size,), float(np.log(initial_std)))
            )

    def forward(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_network(observation)
        # Bounding log standard deviation prevents numerical overflow or a
        # nearly deterministic zero-variance Normal distribution.
        if self.log_std_network is not None:
            log_standard_deviation = self.log_std_network(observation)
        else:
            assert self.log_std is not None
            log_standard_deviation = self.log_std.expand_as(mean)
        log_standard_deviation = torch.clamp(log_standard_deviation, min=-20.0, max=2.0)
        standard_deviation = torch.exp(log_standard_deviation)

        return mean, standard_deviation
