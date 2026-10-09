import math

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


def test_gaussian_network_applies_its_hidden_activation() -> None:
    # One hidden unit that receives -1: ReLU passes on 0, tanh passes on tanh(-1).
    for activation, expected_mean in (("relu", 0.0), ("tanh", math.tanh(-1.0))):
        network = GaussianPolicyNetwork(
            1, 1, hidden_sizes=(1,), std_mode="global", activation=activation
        )
        hidden_layer, _, mean_layer = network.mean_network
        with torch.no_grad():
            hidden_layer.weight.fill_(-1.0)
            hidden_layer.bias.zero_()
            mean_layer.weight.fill_(1.0)
            mean_layer.bias.zero_()

        mean, _ = network(torch.ones(1))

        torch.testing.assert_close(mean, torch.tensor([expected_mean]))


def test_a_small_mean_output_scale_starts_the_mean_near_zero_everywhere() -> None:
    torch.manual_seed(0)
    default = GaussianPolicyNetwork(8, 3, hidden_sizes=(16,))
    torch.manual_seed(0)
    scaled = GaussianPolicyNetwork(8, 3, hidden_sizes=(16,), mean_output_scale=0.01)
    observations = torch.randn(100, 8)

    with torch.no_grad():
        default_mean, default_std = default(observations)
        scaled_mean, scaled_std = scaled(observations)

    # The same starting network, but its mean a hundred times smaller.
    torch.testing.assert_close(scaled_mean, default_mean * 0.01)
    torch.testing.assert_close(scaled_std, default_std)
    assert default_mean.abs().max() > 0.1 > 10 * scaled_mean.abs().max()


def test_gaussian_network_rejects_invalid_configuration() -> None:
    for arguments, settings in (
        ((0, 2), {}),
        ((2, 2), {"hidden_sizes": (4, 0)}),
        ((2, 2), {"initial_std": 0.0}),
        ((2, 2), {"initial_std": float("nan")}),
        ((2, 2), {"std_mode": "invalid"}),
        ((2, 2), {"activation": "sigmoid"}),
        ((2, 2), {"mean_output_scale": 0.0}),
        ((2, 2), {"mean_output_scale": float("inf")}),
    ):
        with pytest.raises(ValueError):
            GaussianPolicyNetwork(*arguments, **settings)
