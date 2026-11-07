"""Metric registry and aggregation utilities.

This module defines a unified Metrics container that aggregates multiple
metric implementations (e.g., accuracy, AUROC, regression errors) into a
single interface. It also provides a registry for mapping metric names to
their corresponding implementations.

Metrics are updated and reset jointly, enabling consistent evaluation
across training and validation loops in PyTorch Lightning workflows.
"""

from torch.nn import ModuleList

from .abstract import Metric
from .continuous import MeanAbsoluteError, MeanSquaredError
from .discrete import AccuracyTopK, ConfusionMatrix

_metrics = [
    MeanAbsoluteError,
    MeanSquaredError,
    ConfusionMatrix,
    AccuracyTopK,
]

_key_to_metric = {key: metric for metric in _metrics for key in metric.get_keys()}


class Metrics(ModuleList, Metric):
    """Container for multiple metrics with unified update and reset logic.

    This class wraps multiple Metric instances and exposes a shared interface
    for updating, resetting, and retrieving evaluation functions. It is used
    to compute multiple evaluation metrics simultaneously during training
    and validation.
    """

    def __init__(self, metrics, keys):
        """Initializes a container for multiple metrics.

        Wraps a list of Metric instances and registers them as a ModuleList,
        enabling unified handling of metric updates, resets, and evaluation
        function retrieval across all contained metrics.

        Args:
            metrics: A list of Metric instances to be aggregated. Each metric
                is expected to implement `update`, `reset`, and
                `_get_key_to_eval_func`.
            keys: A list of string identifiers specifying which metric outputs
                should be tracked and exposed during evaluation.
        """
        super().__init__(metrics)
        self.keys = keys

    def reset(self):
        """Reset all contained metrics to their initial state."""
        for metric in self:
            metric.reset()

    def update(self, x, groundtruth, prediction):
        """Update all metrics with a new batch of predictions.

        Args:
            x: Input data (e.g., images or features).
            groundtruth: Ground truth labels or targets.
            prediction: Model predictions.
        """
        for metric in self:
            metric.update(x, groundtruth, prediction)

    def _get_key_to_eval_func(self, keys=None):
        """Retrieve evaluation functions for selected metric keys.

        Builds a mapping from metric keys to their corresponding evaluation
        functions across all contained metrics.

        Args:
            keys: Optional subset of metric keys to include. If None, all
                configured keys are used.

        Returns:
            Dictionary mapping metric keys to evaluation functions.
        """
        if keys is None:
            keys = self.keys
        return {
            key: eval_function
            for metric in self
            for key, eval_function in metric._get_key_to_eval_func().items()
            if key in keys
        }


def get_metrics(cfg):
    """Construct a Metrics container from configuration.

    Instantiates all metric classes specified in the configuration and
    wraps them in a unified Metrics container for joint evaluation.

    Args:
        cfg: Configuration dictionary containing a "metrics" field with
            a list of metric names.

    Returns:
        A Metrics instance aggregating the selected metrics.
    """
    metrics = {_key_to_metric[key] for key in cfg["metrics"]}
    return Metrics([metric.from_cfg(cfg) for metric in metrics], cfg["metrics"])
