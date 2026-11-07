"""Optimizer and learning-rate scheduler registry.

This module provides simple string-based lookup utilities for selecting
PyTorch optimizers and learning-rate schedulers. It is primarily used to
decouple configuration files from direct PyTorch class imports.
"""

from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

name_to_optimizer = {
    "Adam": Adam,
    "AdamW": AdamW,
    "SGD": SGD,
}

name_to_scheduler = {
    "Step": StepLR,
    "CosineAnnealing": CosineAnnealingLR,
}


def get_optimizer(name):
    """Retrieve an optimizer class by name.

    Args:
        name: Name of the optimizer. Must be one of
            {"Adam", "AdamW", "SGD"}.

    Returns:
        A PyTorch optimizer class.
    """
    return name_to_optimizer[name]


def get_scheduler(name):
    """Retrieve a learning-rate scheduler class by name.

    Args:
        name: Name of the scheduler. Must be one of
            {"Step", "CosineAnnealing"}.

    Returns:
        A PyTorch learning-rate scheduler class.
    """
    return name_to_scheduler[name]
