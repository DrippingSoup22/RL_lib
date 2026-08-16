import pytest
import torch

from rl_lib.models import DiscretePolicyNetwork


def test_policy_network_output_shapes() -> None:
    model = DiscretePolicyNetwork(4, 2, hidden_sizes=(8,))

    assert model(torch.zeros(4)).shape == torch.Size([2])
    assert model(torch.zeros(5, 4)).shape == torch.Size([5, 2])


def test_linear_policy_network_computes_expected_logits() -> None:
    model = DiscretePolicyNetwork(2, 2, hidden_sizes=())
    layer = model.network[0]
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0], [-1.0, 0.5]]))
        layer.bias.copy_(torch.tensor([0.5, -0.5]))

    logits = model(torch.tensor([2.0, 3.0]))

    torch.testing.assert_close(logits, torch.tensor([8.5, -1.0]))


@pytest.mark.parametrize(
    "constructor",
    (
        lambda: DiscretePolicyNetwork(0, 2),
        lambda: DiscretePolicyNetwork(2, 0),
        lambda: DiscretePolicyNetwork(2, 2, hidden_sizes=(4, 0)),
    ),
)
def test_policy_network_rejects_invalid_dimensions(constructor: object) -> None:
    with pytest.raises(ValueError):
        constructor()
