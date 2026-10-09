"""Policy networks for categorical and continuous action spaces."""

import math

import torch
import torch.nn as nn

from rl_lib.networks.activations import hidden_activation


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
    """Map observations to a diagonal Gaussian's mean and standard deviation.

    ``activation`` is the hidden layers' activation, ``"relu"`` or ``"tanh"``.
    ``mean_output_scale`` multiplies the starting weights and bias of the layer
    that outputs the mean. 1 keeps PyTorch's initialization, whose mean varies
    from one observation to the next. A small value such as 0.01 starts the mean
    near zero for every observation, so that the first actions differ only by
    their noise. Andrychowicz et al. (ICLR 2021) found that this start matters
    surprisingly much, and recommend 0.01.
    """

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        initial_std: float = 1.0,
        std_mode: str = "state_dependent",
        activation: str = "relu",
        mean_output_scale: float = 1.0,
    ) -> None:
        super().__init__()

        if observation_size <= 0:
            raise ValueError("Observation size must be greater than 0")
        if action_size <= 0:
            raise ValueError("Action size must be greater than 0")
        if any(hidden_size <= 0 for hidden_size in hidden_sizes):
            raise ValueError("All hidden sizes must be greater than 0")
        if not math.isfinite(initial_std) or initial_std <= 0:
            raise ValueError("Initial std must be finite and greater than 0")
        if std_mode not in ("state_dependent", "global"):
            raise ValueError("std_mode must be 'state_dependent' or 'global'")
        if not math.isfinite(mean_output_scale) or mean_output_scale <= 0:
            raise ValueError("Mean output scale must be finite and greater than 0")
        activation_class = hidden_activation(activation)

        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.std_mode = std_mode

        mean_input_size = observation_size
        mean_layers = []

        for hidden_size in hidden_sizes:
            mean_layers.append(nn.Linear(mean_input_size, hidden_size))
            mean_layers.append(activation_class())
            mean_input_size = hidden_size
        mean_output = nn.Linear(mean_input_size, action_size)
        with torch.no_grad():
            mean_output.weight.mul_(mean_output_scale)
            mean_output.bias.mul_(mean_output_scale)
        mean_layers.append(mean_output)

        self.mean_network = nn.Sequential(*mean_layers)

        self.log_std_network: nn.Sequential | None
        self.log_std: nn.Parameter | None
        if std_mode == "state_dependent":
            std_input_size = observation_size
            log_std_layers = []
            for hidden_size in hidden_sizes:
                log_std_layers.append(nn.Linear(std_input_size, hidden_size))
                log_std_layers.append(activation_class())
                std_input_size = hidden_size

            log_std_output = nn.Linear(std_input_size, action_size)
            nn.init.zeros_(log_std_output.weight)
            nn.init.constant_(log_std_output.bias, math.log(initial_std))
            log_std_layers.append(log_std_output)
            self.log_std_network = nn.Sequential(*log_std_layers)
            self.log_std = None
        else:
            self.log_std_network = None
            self.log_std = nn.Parameter(
                torch.full((action_size,), math.log(initial_std))
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
