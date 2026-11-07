"""Metrics for continuous-valued prediction tasks.

This module provides evaluation metrics for regression and image
reconstruction tasks. Metrics accumulate statistics across batches and
compute scalar performance measures over an entire evaluation epoch.

Currently implemented metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
"""

import torch

from .abstract import Metric


class MeanAbsoluteError(Metric):
    """Compute the mean absolute error (MAE).

    This metric accumulates the sum of absolute differences between
    predictions and ground-truth values across batches and computes
    the average error over all observed samples.

    Attributes:
        _abs_error: Accumulated absolute error.
        _num_samples: Total number of evaluated elements.
    """

    def __init__(self):
        """Initialize the metric."""
        super().__init__()

    def reset(self):
        """Reset accumulated statistics."""
        self._abs_error = 0.0
        self._num_samples = 0

    def update(self, _, groundtruth, prediction):
        """Update the metric with a new batch.

        Args:
            _: Input data (unused by this metric).
            groundtruth: Ground-truth target values.
            prediction: Predicted target values.
        """
        pred_flat = prediction.reshape(-1)
        gt_flat = groundtruth.reshape(-1)
        self._abs_error += torch.abs(pred_flat - gt_flat).sum()
        self._num_samples += len(pred_flat)

    def get_mae(self):
        """Return the mean absolute error.

        Returns:
            Mean absolute error as a scalar tensor, or None if no
            samples have been observed.
        """
        if self._num_samples > 0:
            return self._abs_error / self._num_samples
        else:
            return None

    def _get_key_to_eval_func(self):
        """Return available metric outputs."""
        return {"MAE": self.get_mae}


class MeanSquaredError(Metric):
    """Compute the mean squared error (MSE) and PSNR.

    This metric accumulates squared prediction errors across batches and
    computes both the mean squared error and the corresponding peak
    signal-to-noise ratio (PSNR).

    Attributes:
        _squared_error: Accumulated squared error.
        _num_samples: Total number of evaluated elements.
    """

    def __init__(self):
        """Initialize the metric."""
        super().__init__()

    def reset(self):
        """Reset accumulated statistics."""
        self._squared_error = 0.0
        self._num_samples = 0

    def update(self, _, groundtruth, prediction):
        """Update the metric with a new batch.

        Args:
            _: Input data (unused by this metric).
            groundtruth: Ground-truth target values.
            prediction: Predicted target values.
        """
        pred_flat = prediction.reshape(-1)
        gt_flat = groundtruth.reshape(-1)
        self._squared_error += torch.sum((pred_flat - gt_flat) ** 2)
        self._num_samples += len(pred_flat)

    def get_mse(self):
        """Return the mean squared error.

        Returns:
            Mean squared error as a scalar tensor, or None if no
            samples have been observed.
        """
        if self._num_samples > 0:
            return self._squared_error / self._num_samples
        else:
            return None

    def get_psnr(self):
        """Return the peak signal-to-noise ratio (PSNR).

        PSNR is computed from the accumulated mean squared error as:

            PSNR = -10 * log10(MSE)

        Returns:
            Peak signal-to-noise ratio as a scalar tensor, or None if
            no samples have been observed.
        """
        if self._num_samples > 0:
            return -10 * torch.log10(self.get_mse())
        else:
            return None

    def _get_key_to_eval_func(self):
        """Return available metric outputs."""
        return {
            "MSE": self.get_mse,
            "PSNR": self.get_psnr,
        }
