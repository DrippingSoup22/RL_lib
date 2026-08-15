import pytest
import torch

from rl_lib.models import ActionValueNetwork, StateValueNetwork


def test_value_network_output_shapes() -> None:
    state_model = StateValueNetwork(2, hidden_sizes=(4,))
    action_model = ActionValueNetwork(2, 3, hidden_sizes=(4,))

    assert state_model(torch.zeros(2)).shape == torch.Size([])
    assert state_model(torch.zeros(5, 2)).shape == torch.Size([5])
    assert action_model(torch.zeros(2)).shape == torch.Size([3])
    assert action_model(torch.zeros(5, 2)).shape == torch.Size([5, 3])


def test_linear_action_value_network_computes_expected_values() -> None:
    model = ActionValueNetwork(2, 2, hidden_sizes=())
    layer = model.network[0]
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0], [-1.0, 0.5]]))
        layer.bias.copy_(torch.tensor([0.5, -0.5]))

    values = model(torch.tensor([2.0, 3.0]))

    torch.testing.assert_close(values, torch.tensor([8.5, -1.0]))


@pytest.mark.parametrize(
    "constructor",
    (
        lambda: StateValueNetwork(0),
        lambda: StateValueNetwork(2, hidden_sizes=(4, 0)),
        lambda: ActionValueNetwork(2, 0),
    ),
)
def test_value_networks_reject_invalid_dimensions(constructor: object) -> None:
    with pytest.raises(ValueError):
        constructor()
