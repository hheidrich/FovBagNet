"""Loss functions used during training.

This module provides task-specific loss functions as well as a factory
function for constructing the training criterion from the configuration.

Supported losses include:

- BCE: Binary cross-entropy with logits.
- CCE: Multi-class cross-entropy.
- IoU: Soft intersection-over-union loss.
- MSE: Mean squared error applied to sigmoid probabilities.

The custom losses expect logits as predictions and internally apply a
sigmoid transformation before computing the loss.
"""

import torch
from torch.nn import functional as F


def iou_loss(pred, y):
    """Compute a soft IoU loss.

    The predictions are interpreted as logits and converted to
    probabilities using a sigmoid activation. The loss is defined as

        1 - IoU

    where IoU denotes the soft intersection-over-union between
    prediction and target.

    Note:
        The union is computed as ``n - TN`` rather than explicitly as
        ``TP + FP + FN``.

    Args:
        pred: Predicted logits.
        y: Binary target tensor.

    Returns:
        Scalar loss value.
    """
    n = y.numel()
    pred = torch.sigmoid(pred)

    a = (pred * y).sum()
    b = ((1 - pred) * (1 - y)).sum()
    iou_true = a / (n - b)

    return 1.0 - iou_true


def mse_loss(pred, y):
    """Compute mean squared error on sigmoid probabilities.

    The predictions are interpreted as logits and converted to
    probabilities using a sigmoid activation before computing the MSE.

    Args:
        pred: Predicted logits.
        y: Target tensor.

    Returns:
        Scalar loss value.
    """
    pred = torch.sigmoid(pred)
    return ((pred - y) ** 2).mean()


def get_criterion(cfg):
    """Create the training loss function from the configuration.

    Supported criterion names are:

    - ``"BCE"``: Binary cross-entropy with logits.
    - ``"CCE"``: Cross-entropy loss.
    - ``"IoU"``: Soft IoU loss.
    - ``"MSE"``: Mean squared error on sigmoid probabilities.

    Args:
        cfg: Configuration object containing
            ``cfg.training.criterion``.

    Returns:
        Loss function or loss module corresponding to the configured
        criterion.
    """
    criteria = {
        "BCE": F.binary_cross_entropy_with_logits,
        "CCE": torch.nn.CrossEntropyLoss(),
        "IoU": iou_loss,
        "MSE": mse_loss,
    }

    return criteria[cfg.training.criterion]
