import pytest
import torch
import torch.multiprocessing as mp

from rl_lib.optimization import (
    clip_gradients,
    share_optimizer_state,
    validate_max_gradient_norm,
)


def increment_tensor(tensor: torch.Tensor) -> None:
    tensor.add_(1.0)


def shared_parameter() -> torch.nn.Parameter:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.share_memory_()
    return parameter


def test_share_optimizer_state_initializes_shared_adam_state() -> None:
    parameter = shared_parameter()
    optimizer = torch.optim.AdamW((parameter,), lr=0.1)

    share_optimizer_state(optimizer)

    state = optimizer.state[parameter]
    assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
    assert all(value.is_shared() for value in state.values())
    assert state["step"].item() == 0.0
    assert torch.count_nonzero(state["exp_avg"]).item() == 0
    assert torch.count_nonzero(state["exp_avg_sq"]).item() == 0

    parameter.grad = torch.tensor([0.5])
    optimizer.step()
    assert state["step"].item() == 1.0


def test_shared_optimizer_state_is_visible_in_a_spawned_process() -> None:
    parameter = shared_parameter()
    optimizer = torch.optim.AdamW((parameter,), lr=0.1)
    share_optimizer_state(optimizer)
    step = optimizer.state[parameter]["step"]

    process = mp.get_context("spawn").Process(target=increment_tensor, args=(step,))
    process.start()
    process.join(timeout=10)

    assert process.exitcode == 0
    assert step.item() == 1.0


def test_sgd_is_shared_without_state_but_not_with_momentum() -> None:
    optimizer = torch.optim.SGD((shared_parameter(),), lr=0.1)
    share_optimizer_state(optimizer)
    assert not optimizer.state

    optimizer = torch.optim.SGD((shared_parameter(),), lr=0.1, momentum=0.9)
    with pytest.raises(ValueError, match="momentum"):
        share_optimizer_state(optimizer)


def test_clip_gradients_limits_the_joint_norm() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([3.0, 4.0])

    clip_gradients((parameter,), 1.0)

    torch.testing.assert_close(parameter.grad, torch.tensor([0.6, 0.8]))


def test_gradient_norm_validation_rejects_invalid_limits() -> None:
    for value in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="gradient norm"):
            validate_max_gradient_norm(value)
