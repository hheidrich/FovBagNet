"""Utility classes and functions for training, checkpointing, and deployment.

This module provides helper components used throughout the project,
including PyTorch Lightning callbacks, dataset utilities, model export
functionality, and compatibility helpers for distributed training.

The utilities include custom progress reporting, TorchScript export and
checkpoint integration, dataset composition helpers, and conversion
routines for models that use synchronized batch normalization.
"""

import copy
import os
from collections.abc import Sequence
from os import path as osp
from typing import Any, Tuple

import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch.utils.data import Dataset


class LitProgressBar(RichProgressBar):
    """Custom Rich progress bar for PyTorch Lightning training.

    This progress bar extends the default Lightning ``RichProgressBar`` by modifying the displayed metrics during
    training. In particular, it removes the version number from the displayed progress information.
    """

    def __init__(self):  # noqa: D107
        super().__init__()
        self.enable = True

    def get_metrics(self, trainer, pl_module):
        """Return formatted metrics for the progress bar display.

        This method overrides the default implementation to exclude the Lightning version number ("v_num") from the
        displayed metrics.

        Args:
            trainer: The PyTorch Lightning Trainer instance.
            pl_module: The Lightning module being trained.

        Returns:
            Dictionary of metrics to display in the progress bar.
        """
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)  # remove the version number
        return items


def revert_sync_batchnorm_(module: torch.nn.Module) -> None:
    """Recursively replace SyncBatchNorm layers with standard BatchNorm layers.

    This function traverses a PyTorch model in-place and converts all ``torch.nn.SyncBatchNorm`` layers to standard
    ``torch.nn.BatchNorm`` layers of the appropriate dimensionality. All weights, biases, and running statistics
    are preserved in the conversion.

    This is useful for TorchScript export or scenarios where synchronized batch normalization is not supported or
    required (e.g., inference or single-GPU training).

    Args:
        module: The root module to process. All children are traversed
            recursively, and ``SyncBatchNorm`` layers are replaced in-place.

    Notes:
        - The operation modifies the input module directly and does not return a new copy.
        - Only ``torch.nn.SyncBatchNorm`` instances are replaced; other types of batch normalization layers are left
          unchanged.
    """
    for name, child in module.named_children():
        if isinstance(child, torch.nn.SyncBatchNorm):
            # Convert to BatchNorm of appropriate dimension
            new_bn = torch.nn.BatchNorm2d(
                child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats,
            )
            # Copy weights
            if child.affine:
                with torch.no_grad():
                    new_bn.weight.copy_(child.weight)
                    new_bn.bias.copy_(child.bias)

            # Copy running statistics
            with torch.no_grad():
                new_bn.running_mean.copy_(child.running_mean)
                new_bn.running_var.copy_(child.running_var)

            setattr(module, name, new_bn)
        else:
            revert_sync_batchnorm_(child)


def save_torchscript(module: torch.nn.Module, filepath: str, original_module: torch.nn.Module | None = None) -> None:
    """Export a PyTorch model to TorchScript and save it to disk.

    This function creates a TorchScript-compatible copy of a model and serializes it to ``filepath``.

    If ``original_module`` is provided, it is used as a structural reference model. In this case, the function compares
    its state dict with the (potentially wrapped or compiled) ``module`` after removing common framework prefixes
    (e.g., "_orig_mod.").

    Batch normalization layers are converted back from synchronized versions to ensure TorchScript compatibility before
    scripting.

    Args:
        module: Trained PyTorch model (possibly wrapped or compiled).
        filepath: Destination path for the saved TorchScript model.
        original_module: Optional clean reference model used for validation
            and reconstruction.

    Raises:
        ValueError: If the state dictionaries of the reconstructed model and
            the provided module do not match after normalization.
    """
    # Create a clean copy of the model to be scripted
    if original_module is None:
        module_clone = copy.deepcopy(module)
    else:
        module_clone = copy.deepcopy(original_module)

        # Align state dict keys with wrapped/compiled module format
        module_state = {key.removeprefix("_orig_mod."): val for key, val in module.state_dict().items()}

        clone_state = module_clone.state_dict()

        # Validate parameter consistency
        for key, expected_value in clone_state.items():
            if key not in module_state:
                raise ValueError(f"Missing key in state_dict: {key}")

            if not torch.equal(module_state[key], expected_value):
                raise ValueError(f"Mismatch in parameter: {key}")

        # Optional recovery path (kept explicit instead of hidden behavior)
        # module_clone.load_state_dict(module_state)

    # Ensure TorchScript compatibility
    revert_sync_batchnorm_(module_clone)

    # Script and save
    scripted_model = torch.jit.script(module_clone)
    scripted_model.save(filepath)


class TorchScriptModelCheckpoint(ModelCheckpoint):
    """Checkpoint callback that additionally exports TorchScript models.

    Whenever a PyTorch Lightning checkpoint is saved, this callback exports
    the current model as a standalone TorchScript ``.pt`` file using the same
    naming convention as the checkpoint.

    The callback also removes TorchScript artifacts associated with
    checkpoints that are no longer retained by the configured top-k
    checkpoint policy.
    """

    def __init__(  # noqa: D107
        self,
        dirpath=None,
        filename=None,
        monitor=None,
        verbose=False,
        save_last=None,
        save_top_k=1,
        save_weights_only=False,
        mode="min",
        auto_insert_metric_name=True,
        every_n_train_steps=None,
        train_time_interval=None,
        every_n_epochs=None,
        save_on_train_epoch_end=None,
        enable_version_counter=True,
    ):
        super(TorchScriptModelCheckpoint, self).__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
            enable_version_counter=enable_version_counter,
        )
        self.last_kth_best_model_path = ""

    @rank_zero_only
    def on_save_checkpoint(self, trainer, pl_module, checkpoint: dict) -> dict:
        """Export the model as TorchScript when a checkpoint is saved.

        This hook is called during checkpoint creation. It exports the current
        model as a TorchScript ``.pt`` file, stores the generated file path in
        the checkpoint metadata, and removes any TorchScript artifact
        corresponding to a checkpoint that has fallen outside the retained
        top-k checkpoints.

        Args:
            trainer: The active PyTorch Lightning trainer instance.
            pl_module: The Lightning module being checkpointed.
            checkpoint: Checkpoint state dictionary to be saved.

        Returns:
            The checkpoint dictionary with an additional
            ``"torchscript_model_path"`` entry containing the path to the
            exported TorchScript model.
        """
        if not osp.exists(self.dirpath):
            os.mkdir(self.dirpath)
        callback_metrics = {key: int(val) if key == "step" else val for key, val in trainer.callback_metrics.items()}
        callback_metrics["epoch"] = trainer.current_epoch
        filename, _ = os.path.splitext(self.format_checkpoint_name(callback_metrics, self.filename))
        torchscript_model_path = f"{filename}.pt"

        # Save the model
        (original_model,) = (
            pl_module.original_model
        )  # Unpack original model because pl_module.original_model is a 1-tuple.
        save_torchscript(pl_module.model, torchscript_model_path, original_model)

        # Optionally, you can include the TorchScript model path in the checkpoint (if you want)
        checkpoint["torchscript_model_path"] = torchscript_model_path

        # Remove (k+1)th model
        if self.last_kth_best_model_path != "":
            filename_to_delete, _ = os.path.splitext(self.last_kth_best_model_path)
            os.remove(f"{filename_to_delete}.pt")

        # Update deathlist
        self.last_kth_best_model_path = self.kth_best_model_path

        return checkpoint


class ZipDataset(Dataset[Tuple[Any, ...]]):
    """Combine multiple datasets by zipping their samples together.

    This dataset behaves similarly to Python's built-in ``zip`` function, returning tuples containing samples from
    multiple datasets at the same index. All datasets must have identical lengths.

    Example:
        >>> dataset = ZipDataset(images, labels)
        >>> image, label = dataset[0]

    Attributes:
        datasets: The datasets wrapped by this class.
    """

    datasets: Sequence[Dataset[Any]]
    __length: int

    def __init__(self, *datasets: Dataset[Any]) -> None:
        """Initialize the zipped dataset.

        Args:
            *datasets: Variable number of datasets to combine.

        Raises:
            ValueError: If no datasets are provided or if their lengths differ.
        """
        if not datasets:
            raise ValueError("At least one dataset must be provided.")

        length = len(datasets[0])
        for i, dataset in enumerate(datasets[1:], start=1):
            if len(dataset) != length:
                raise ValueError(
                    f"All datasets must have the same length. "
                    f"Dataset 0 has length {length}, but dataset {i} has length {len(dataset)}."
                )

        self.__length = length
        self.datasets = datasets

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.__length

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        """Retrieve samples from all datasets at a given index.

        Args:
            index: Sample index.

        Returns:
            A tuple containing one sample from each wrapped dataset.
        """
        return tuple(dataset[index] for dataset in self.datasets)
