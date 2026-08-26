import pytest
import torch
import torch.multiprocessing as mp

from rl_lib.optimizers import share_optimizer_state


def increment_tensor(tensor: torch.Tensor) -> None:
    tensor.add_(1.0)


def shared_parameter() -> torch.nn.Parameter:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.share_memory_()
    return parameter


@pytest.mark.parametrize("optimizer_class", (torch.optim.Adam, torch.optim.AdamW))
def test_share_optimizer_state_initializes_shared_adam_state(
    optimizer_class: type[torch.optim.Optimizer],
) -> None:
    parameter = shared_parameter()
    optimizer = optimizer_class((parameter,), lr=0.1)

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


def test_share_optimizer_state_accepts_stateless_sgd() -> None:
    parameter = shared_parameter()
    optimizer = torch.optim.SGD((parameter,), lr=0.1)

    share_optimizer_state(optimizer)

    assert not optimizer.state


def test_share_optimizer_state_rejects_sgd_momentum() -> None:
    parameter = shared_parameter()
    optimizer = torch.optim.SGD((parameter,), lr=0.1, momentum=0.9)

    with pytest.raises(ValueError, match="momentum"):
        share_optimizer_state(optimizer)


def test_share_optimizer_state_requires_shared_parameters() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW((parameter,), lr=0.1)

    with pytest.raises(ValueError, match="shared memory"):
        share_optimizer_state(optimizer)
