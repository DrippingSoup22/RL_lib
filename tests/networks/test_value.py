import pytest
import torch

from rl_lib.networks import ActionValueNetwork, StateValueNetwork

WEIGHTS = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
BIASES = torch.tensor([0.5, -0.5])


def test_linear_value_networks_compute_expected_values() -> None:
    action_values = ActionValueNetwork(2, 2, hidden_sizes=())
    state_values = StateValueNetwork(2, hidden_sizes=())
    with torch.no_grad():
        action_values.network[0].weight.copy_(WEIGHTS)
        action_values.network[0].bias.copy_(BIASES)
        state_values.network[0].weight.copy_(WEIGHTS[:1])
        state_values.network[0].bias.copy_(BIASES[:1])
    observations = torch.tensor([[2.0, 3.0], [0.0, 0.0]])

    torch.testing.assert_close(
        action_values(observations), torch.tensor([[8.5, -1.0], [0.5, -0.5]])
    )
    # One value per observation, without a trailing dimension of size 1.
    torch.testing.assert_close(state_values(observations), torch.tensor([8.5, 0.5]))


def test_value_networks_reject_invalid_dimensions() -> None:
    for constructor in (
        lambda: StateValueNetwork(0),
        lambda: StateValueNetwork(2, hidden_sizes=(4, 0)),
        lambda: ActionValueNetwork(2, 0),
    ):
        with pytest.raises(ValueError):
            constructor()
