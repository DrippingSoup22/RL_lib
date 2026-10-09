"""The activations a network's hidden layers may use."""

import torch.nn as nn

HIDDEN_ACTIVATIONS: dict[str, type[nn.Module]] = {"relu": nn.ReLU, "tanh": nn.Tanh}


def hidden_activation(name: str) -> type[nn.Module]:
    """The activation called ``name``, checked once where a network is built."""
    if name not in HIDDEN_ACTIVATIONS:
        raise ValueError(
            f"Hidden activation must be one of: {', '.join(HIDDEN_ACTIVATIONS)}"
        )
    return HIDDEN_ACTIVATIONS[name]
