import pytest
import torch

from rl_lib.models import DiscretePolicyNetwork, GaussianPolicyNetwork


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


def test_gaussian_policy_network_output_shapes_and_initial_std() -> None:
    model = GaussianPolicyNetwork(4, 2, hidden_sizes=(8,), initial_std=0.5)

    mean, standard_deviation = model(torch.zeros(4))
    batch_mean, batch_standard_deviation = model(torch.zeros(5, 4))

    assert mean.shape == standard_deviation.shape == torch.Size([2])
    assert batch_mean.shape == batch_standard_deviation.shape == torch.Size([5, 2])
    torch.testing.assert_close(standard_deviation, torch.full((2,), 0.5))
    torch.testing.assert_close(batch_standard_deviation, torch.full((5, 2), 0.5))


def test_linear_gaussian_policy_network_computes_expected_parameters() -> None:
    model = GaussianPolicyNetwork(2, 2, hidden_sizes=())
    mean_layer = model.mean_network[0]
    log_std_layer = model.log_std_network[0]
    with torch.no_grad():
        mean_layer.weight.copy_(torch.tensor([[1.0, 2.0], [-1.0, 0.5]]))
        mean_layer.bias.copy_(torch.tensor([0.5, -0.5]))
        log_std_layer.weight.copy_(torch.tensor([[0.1, 0.0], [0.0, -0.2]]))
        log_std_layer.bias.zero_()

    mean, standard_deviation = model(torch.tensor([2.0, 3.0]))

    torch.testing.assert_close(mean, torch.tensor([8.5, -1.0]))
    torch.testing.assert_close(
        standard_deviation,
        torch.exp(torch.tensor([0.2, -0.6])),
    )


def test_gaussian_policy_network_bounds_log_standard_deviation() -> None:
    model = GaussianPolicyNetwork(1, 2, hidden_sizes=())
    log_std_layer = model.log_std_network[0]
    with torch.no_grad():
        log_std_layer.weight.zero_()
        log_std_layer.bias.copy_(torch.tensor([100.0, -100.0]))

    _, standard_deviation = model(torch.zeros(1))

    torch.testing.assert_close(
        standard_deviation,
        torch.exp(torch.tensor([2.0, -20.0])),
    )


@pytest.mark.parametrize(
    "constructor",
    (
        lambda: GaussianPolicyNetwork(0, 2),
        lambda: GaussianPolicyNetwork(2, 0),
        lambda: GaussianPolicyNetwork(2, 2, hidden_sizes=(4, 0)),
        lambda: GaussianPolicyNetwork(2, 2, initial_std=0.0),
        lambda: GaussianPolicyNetwork(2, 2, initial_std=-1.0),
        lambda: GaussianPolicyNetwork(2, 2, initial_std=float("inf")),
        lambda: GaussianPolicyNetwork(2, 2, initial_std=float("nan")),
    ),
)
def test_gaussian_policy_network_rejects_invalid_configuration(
    constructor: object,
) -> None:
    with pytest.raises(ValueError):
        constructor()
