"""Abstract metric interface.

This module defines the base ``Metric`` class used throughout the project.
All evaluation metrics inherit from this class and implement methods for
state management, batch-wise updates, and metric computation.

The interface is designed to support:
- Stateful accumulation across batches
- Multiple metric outputs per metric instance
- Integration with PyTorch modules and training loops
"""

from abc import ABC, abstractmethod

from torch.nn import Module


class Metric(Module, ABC):
    """Abstract base class for all evaluation metrics.

    This class defines the interface for metrics used in training and
    evaluation. Each metric is responsible for maintaining internal state
    during updates and exposing one or more computed outputs via key-based
    evaluation functions.

    Metrics are designed to be:
    - Stateful (accumulate information across batches)
    - Multi-output (a single metric may expose multiple keys)
    - Compatible with PyTorch Modules

    Subclasses must implement reset, update, and _get_key_to_eval_func.

    Attributes:
        None (state is defined by subclasses).
    """

    def __init__(self, **kwargs):
        """Initialize the metric base class.

        Args:
            **kwargs: Optional keyword arguments passed by subclasses or
                configuration utilities (currently unused in base class).
        """
        super().__init__()

    def get_out_dict(self, keys=None):
        """Compute metric outputs as a dictionary of scalar values.

        Evaluates all registered metric functions and returns their results
        as a dictionary keyed by metric name.

        Args:
            keys: Optional subset of metric keys to compute. If None, all
                available metric keys are evaluated.

        Returns:
            Dictionary mapping metric keys to computed scalar values.
        """
        key_to_eval_func = self._get_key_to_eval_func()
        if keys is None:
            keys = key_to_eval_func.keys()
        return {key: key_to_eval_func[key]() for key in keys}

    @classmethod
    def get_keys(cls):
        """Return all metric output keys defined by the class.

        This method instantiates the metric class without calling __init__
        to inspect available evaluation keys.

        Returns:
            Iterable of metric output keys.
        """
        return cls.__new__(cls)._get_key_to_eval_func().keys()

    @classmethod
    def from_cfg(cls, cfg):
        """Instantiate a metric from configuration.

        Default implementation ignores the configuration and returns a
        default-initialized metric instance. Subclasses may override this
        method to support parameterized metrics.

        Args:
            cfg: Configuration object or dictionary.

        Returns:
            An instance of the metric class.
        """
        return cls()

    @abstractmethod
    def reset(self):
        """Reset internal metric state.

        This method is called at the beginning of each epoch to clear
        accumulated statistics.
        """
        pass

    @abstractmethod
    def update(self, x, groundtruth, prediction):
        """Update metric state with a new batch of data.

        Args:
            x: Input data (e.g., images or features).
            groundtruth: Ground truth labels or targets.
            prediction: Model predictions.
        """
        pass

    @abstractmethod
    def _get_key_to_eval_func(self):
        """Return mapping from metric keys to evaluation functions.

        Each metric must expose one or more callable functions that compute
        scalar outputs from the accumulated internal state.

        Returns:
            Dictionary mapping metric keys (str) to callables returning scalars.
        """
        pass
