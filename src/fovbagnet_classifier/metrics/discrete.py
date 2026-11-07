"""Discrete evaluation metrics for classification tasks.

This module provides metric implementations for classification problems,
including confusion matrix-based metrics, and top-k accuracy.

Metrics are stateful and accumulate predictions across batches. Final
evaluation scores are computed at epoch end via torchmetrics or custom
statistical aggregations.
"""

import numpy as np
import torch
from torchmetrics import ConfusionMatrix as TorchMetricsConfusionMatrix

from .abstract import Metric


class ConfusionMatrix(Metric):
    """Confusion matrix-based classification metrics.

    This metric accumulates classification predictions and targets using
    ``torchmetrics.ConfusionMatrix`` and exposes several commonly used
    evaluation measures derived from the resulting confusion matrix:

    - Confusion Matrix
    - Accuracy
    - Per-class Intersection over Union (IoU)
    - Mean Intersection over Union (mIoU)

    The implementation supports both binary and multiclass classification
    tasks.

    Attributes:
        num_classes: Number of target classes.
        _confusion_matrix: TorchMetrics confusion matrix accumulator.
    """

    def __init__(self, num_classes):
        """Initialize the metric.

        Args:
            num_classes: Number of classes in the classification task.
                Must be at least 2.
        """
        assert num_classes >= 2
        super().__init__()

        if num_classes == 2:
            task = "binary"
        else:
            task = "multiclass"

        self.num_classes = num_classes
        self._confusion_matrix = TorchMetricsConfusionMatrix(num_classes=num_classes, task=task)

    def reset(self):
        """Reset accumulated confusion matrix statistics."""
        self._confusion_matrix.reset()

    def update(self, _, groundtruth, prediction):
        """Update the metric with a new batch.

        Args:
            _: Input data (unused by this metric).
            groundtruth: Ground-truth class labels.
            prediction: Predicted class labels or logits compatible with
                ``torchmetrics.ConfusionMatrix``.
        """
        self._confusion_matrix.update(preds=prediction, target=groundtruth)

    def get_confusion_matrix(self):
        """Return the accumulated confusion matrix.

        Returns:
            Tensor of shape ``(C, C)`` containing class-wise counts,
            where ``C`` is the number of classes.
        """
        return self._confusion_matrix.compute()

    def get_accuracy(self):
        """Compute overall classification accuracy.

        Returns:
            Fraction of correctly classified samples.
        """
        confusion_counts = self.get_confusion_matrix()
        return confusion_counts.diag().sum() / confusion_counts.sum()

    def get_ious(self):
        """Compute per-class Intersection over Union (IoU).

        IoU for class c is defined as:

            TP / (TP + FP + FN)

        Returns:
            Tensor containing one IoU value per class.
        """
        confusion_counts = self.get_confusion_matrix()
        col_sums = confusion_counts.sum(axis=0)
        row_sums = confusion_counts.sum(axis=1)
        diag = confusion_counts.diag()
        return diag / (row_sums + col_sums - diag)

    def get_miou(self):
        """Compute mean Intersection over Union (mIoU).

        Returns:
            Average IoU across all classes.
        """
        return self.get_ious().mean()

    def _get_key_to_eval_func(self):
        """Return available metric outputs."""
        return {
            "Confusion Matrix": self.get_confusion_matrix,
            "Accuracy": self.get_accuracy,
            "IoUs": self.get_ious,
            "Mean IoU": self.get_miou,
        }

    @classmethod
    def from_cfg(cls, cfg):
        """Instantiate the metric from a configuration object.

        Args:
            cfg: Configuration containing a ``num_classes`` field.

        Returns:
            Configured ConfusionMatrix instance.
        """
        return cls(num_classes=cfg["num_classes"])


class AccuracyTopK(Metric):
    """Top-k classification accuracy metric.

    This metric tracks how often the ground-truth class appears among the
    k highest-scoring predicted classes. Counts are accumulated across
    batches and converted to top-k accuracy scores at evaluation time.

    Attributes:
        num_classes: Number of target classes.
        max_top_k: Largest k value tracked.
        counter: Number of processed samples.
        top_k_counts: Number of correct predictions at each rank.
    """

    max_top_k = 5

    def __init__(self, num_classes, max_top_k=5):
        """Initialize the metric.

        Args:
            num_classes: Number of classes in the classification task.
            max_top_k: Maximum rank to track for top-k accuracy.
        """
        assert num_classes >= 2
        super().__init__()
        self.max_top_k = max_top_k
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset accumulated statistics."""
        self.counter = 0
        self.top_k_counts = torch.nn.Parameter(torch.zeros(self.max_top_k, dtype=torch.int64), requires_grad=False)

    def update(self, _, groundtruth, prediction):
        """Update the metric with a new batch.

        Args:
            _: Input data (unused by this metric).
            groundtruth: Ground-truth class labels of shape ``(N,)``.
            prediction: Prediction scores or logits of shape
                ``(N, num_classes)``.
        """
        (n,) = groundtruth.shape
        self.counter += n
        self.top_k_counts += (
            prediction.topk(
                k=self.max_top_k,
                sorted=True,
                largest=True,
                dim=-1,
            ).indices
            == groundtruth.unsqueeze(1)
        ).sum(dim=0)

    def get_top_k_accuracy(self, top_k):
        """Compute top-k accuracy.

        A prediction is considered correct if the ground-truth class is
        among the k highest-scoring predicted classes.

        Args:
            top_k: Number of highest-ranked predictions to consider.

        Returns:
            Top-k accuracy.
        """
        cumulative_counts = self.top_k_counts.cumsum(dim=0)
        return cumulative_counts[top_k - 1] / self.counter

    def _get_key_to_eval_func(self):
        """Return available metric outputs."""
        return {f"Top_{k}_Accuracy": lambda: self.get_top_k_accuracy(top_k=k) for k in range(1, self.max_top_k + 1)}

    def save_cumulative_counts(self, path):
        """Save accumulated statistics to disk.

        Args:
            path: Output file path for the NumPy archive.
        """
        np.savez(
            file=path,
            num_classes=self.num_classes,
            counter=self.counter,
            top_k_counts=self.top_k_counts.data.cpu().numpy(),
        )

    @classmethod
    def load_from_cumulative_counts(cls, path):
        """Load a metric instance from saved statistics.

        Args:
            path: Path to a file created by
                ``save_cumulative_counts``.

        Returns:
            Reconstructed AccuracyTopK instance with restored state.
        """
        with np.load(path) as data:
            num_classes = int(data["num_classes"])
            counter = int(data["counter"])
            top_k_counts = torch.from_numpy(data["top_k_counts"])

        (max_top_k,) = top_k_counts.shape
        self = cls(
            num_classes=num_classes,
            max_top_k=max_top_k,
        )
        self.counter = counter
        self.top_k_counts[:] = top_k_counts
        return self
