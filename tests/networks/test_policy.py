import pytest
import torch

from rl_lib.networks import CategoricalPolicyNetwork, GaussianPolicyNetwork


def test_linear_categorical_network_computes_expected_logits() -> None:
    network = CategoricalPolicyNetwork(2, 2, hidden_sizes=())
    layer = network.network[0]
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0], [-1.0, 0.5]]))
        layer.bias.copy_(torch.tensor([0.5, -0.5]))

    logits = network(torch.tensor([2.0, 3.0]))

    torch.testing.assert_close(logits, torch.tensor([8.5, -1.0]))


def test_categorical_network_rejects_invalid_dimensions() -> None:
    for arguments in ((0, 2), (2, 0), (2, 2, (4, 0))):
        with pytest.raises(ValueError):
            CategoricalPolicyNetwork(*arguments)


def test_linear_gaussian_network_computes_expected_parameters() -> None:
    network = GaussianPolicyNetwork(2, 2, hidden_sizes=())
    mean_layer = network.mean_network[0]
    log_std_layer = network.log_std_network[0]
    with torch.no_grad():
        mean_layer.weight.copy_(torch.tensor([[1.0, 2.0], [-1.0, 0.5]]))
        mean_layer.bias.copy_(torch.tensor([0.5, -0.5]))
        log_std_layer.weight.copy_(torch.tensor([[0.1, 0.0], [0.0, -0.2]]))
        log_std_layer.bias.zero_()

    mean, standard_deviation = network(torch.tensor([2.0, 3.0]))

    torch.testing.assert_close(mean, torch.tensor([8.5, -1.0]))
    torch.testing.assert_close(standard_deviation, torch.exp(torch.tensor([0.2, -0.6])))


def test_gaussian_network_bounds_log_standard_deviation() -> None:
    network = GaussianPolicyNetwork(1, 2, hidden_sizes=())
    log_std_layer = network.log_std_network[0]
    with torch.no_grad():
        log_std_layer.weight.zero_()
        log_std_layer.bias.copy_(torch.tensor([100.0, -100.0]))

    _, standard_deviation = network(torch.zeros(1))

    torch.testing.assert_close(
        standard_deviation, torch.exp(torch.tensor([2.0, -20.0]))
    )


def test_gaussian_network_can_learn_one_global_standard_deviation() -> None:
    network = GaussianPolicyNetwork(
        3, 2, hidden_sizes=(4,), initial_std=0.5, std_mode="global"
    )

    _, standard_deviation = network(torch.zeros(5, 3))

    assert network.log_std_network is None
    torch.testing.assert_close(standard_deviation, torch.full((5, 2), 0.5))
    standard_deviation.sum().backward()
    assert network.log_std.grad is not None


def test_gaussian_network_rejects_invalid_configuration() -> None:
    for arguments, settings in (
        ((0, 2), {}),
        ((2, 2), {"hidden_sizes": (4, 0)}),
        ((2, 2), {"initial_std": 0.0}),
        ((2, 2), {"initial_std": float("nan")}),
        ((2, 2), {"std_mode": "invalid"}),
    ):
        with pytest.raises(ValueError):
            GaussianPolicyNetwork(*arguments, **settings)
