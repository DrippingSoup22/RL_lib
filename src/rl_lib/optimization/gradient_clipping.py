"""Small gradient utilities shared by neural algorithms."""

import math
from collections.abc import Iterable

import torch


def validate_max_gradient_norm(value: float | None) -> float | None:
    """Validate and normalize an optional gradient-norm limit."""
    if value is None:
        return None
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Maximum gradient norm must be finite and positive")
    return float(value)


def clip_gradients(
    parameters: Iterable[torch.nn.Parameter],
    max_gradient_norm: float | None,
) -> None:
    """Clip one model's gradient norm when a limit is configured."""
    if max_gradient_norm is not None:
        torch.nn.utils.clip_grad_norm_(tuple(parameters), max_gradient_norm)
