"""Prepare standard PyTorch optimizer state for multiprocessing."""

import torch


def _optimizer_parameters(
    optimizer: torch.optim.Optimizer,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        parameter for group in optimizer.param_groups for parameter in group["params"]
    )


def _validate_shared_cpu_parameters(optimizer: torch.optim.Optimizer) -> None:
    for parameter in _optimizer_parameters(optimizer):
        if parameter.device.type != "cpu":
            raise ValueError("Shared optimizer parameters must be on the CPU")
        if not parameter.is_shared():
            raise ValueError("Optimizer parameters must already be in shared memory")


def _initialize_adam_state(optimizer: torch.optim.Optimizer) -> None:
    step_dtype = (
        torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32
    )
    for group in optimizer.param_groups:
        amsgrad = bool(group["amsgrad"])
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            if not state:
                state["step"] = torch.zeros((), dtype=step_dtype)
                state["exp_avg"] = torch.zeros_like(
                    parameter,
                    memory_format=torch.preserve_format,
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    parameter,
                    memory_format=torch.preserve_format,
                )
                if amsgrad:
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )

            for value in state.values():
                if not isinstance(value, torch.Tensor):
                    raise ValueError("Shared optimizer state values must be tensors")
                if value.device.type != "cpu":
                    raise ValueError("Shared optimizer state must be on the CPU")
                value.share_memory_()


def share_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Initialize and share the state of a supported CPU optimizer.

    Model parameters must be placed in shared memory before calling this
    function. Stateless SGD needs no additional state; Adam and AdamW create
    their normally lazy state tensors here so workers inherit shared storage.
    """
    _validate_shared_cpu_parameters(optimizer)

    if isinstance(optimizer, torch.optim.SGD):
        if any(float(group["momentum"]) != 0.0 for group in optimizer.param_groups):
            raise ValueError("Shared SGD does not support momentum")
        return

    if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        _initialize_adam_state(optimizer)
        return

    raise TypeError("Shared optimizer must be SGD, Adam, or AdamW")
