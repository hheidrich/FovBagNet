"""PyTorch Lightning training module.

This module provides the project's primary LightningModule implementation, which encapsulates model construction,
optimization, metric tracking, logging, and training/validation workflows.
"""

import os

import lightning.pytorch as pl
import numpy as np
import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torchvision.transforms import v2

from .criterion import get_criterion
from .metrics import get_metrics
from .models import get_model
from .optimizers import get_optimizer, get_scheduler
from .utils import save_torchscript


class LightningWrapper(pl.LightningModule):
    """PyTorch Lightning wrapper for model training and evaluation.

    This class integrates model creation, loss computation, metric tracking, optimization, and logging into a single
    LightningModule. It supports optional model compilation, learning-rate warmup, configurable optimizers and
    schedulers, and automatic metric aggregation across training and validation epochs.

    Args:
        cfg: Project configuration object containing model, data, optimization, and scheduler settings.
    """

    def __init__(self, cfg):
        """Initialize the training module from a project configuration.

        Builds the model, loss function, metrics, optimizer configuration, and
        data preprocessing pipeline. The model is also validated for TorchScript
        exportability during initialization.

        Args:
            cfg: Project configuration object containing model, data,
                optimization, and scheduler settings.
        """
        super(LightningWrapper, self).__init__()
        self.save_hyperparameters()

        self.normalize_images = v2.Normalize(
            mean=cfg.data.normalization.mean,
            std=cfg.data.normalization.std,
        )

        self.model = get_model(cfg)
        self.original_model = (self.model,) if cfg.model.compile else (None,)

        # Sanity check: Make sure that model is scriptable
        save_torchscript(module=self.model, filepath=os.devnull)

        self.criterion = get_criterion(cfg)

        self.image_resolution = cfg.data.image_resolution

        self.optimizer_name = cfg.optimizer.name
        self.optimizer_kwargs = cfg.optimizer.kwargs

        self.scheduler_name = cfg.scheduler.name
        self.scheduler_kwargs = cfg.scheduler.kwargs

        self.num_warmup_epochs = cfg.scheduler.num_warmup_epochs

        self.modes = ["train", "val"]

        self.losses = {mode: [] for mode in self.modes}
        self.metrics = torch.nn.ModuleList([get_metrics(cfg) for mode in self.modes])
        self.mode_to_metrics = {mode: metric for mode, metric in zip(self.modes, self.metrics)}

    def configure_model(self):
        """Configure the underlying model for training.

        If model compilation is enabled in the configuration, the model is compiled using ``torch.compile`` before
        training begins.

        Returns:
            The configured model instance.
        """
        (original_model,) = self.original_model
        self.model = torch.compile(self.model, dynamic=False) if original_model is not None else self.model
        return self.model

    def configure_optimizers(self):
        """Create optimizer and learning-rate scheduler configuration.

        Constructs the optimizer specified in the configuration and optionally attaches a warmup scheduler and a
        primary learning-rate scheduler. When both are present, they are combined using ``SequentialLR``.

        Returns:
            A Lightning-compatible optimizer configuration consisting of an optimizer alone, an optimizer and scheduler
            pair, or an optimizer with a sequential scheduler stack.
        """
        optimizer = get_optimizer(self.optimizer_name)(self.parameters(), **self.optimizer_kwargs)

        schedulers = []

        if self.num_warmup_epochs > 0:
            schedulers.append(
                LinearLR(optimizer, start_factor=1 / self.num_warmup_epochs, total_iters=self.num_warmup_epochs)
            )

        if self.scheduler_name is not None:
            schedulers.append(get_scheduler(self.scheduler_name)(optimizer, **self.scheduler_kwargs))

        if len(schedulers) == 0:
            return optimizer
        elif len(schedulers) == 1:
            return [optimizer], schedulers
        else:
            scheduler = SequentialLR(optimizer, schedulers=schedulers, milestones=[self.num_warmup_epochs])
            return [optimizer], [scheduler]

    def forward(self, x):
        """Run a forward pass through the model.

        Args:
            x: Input tensor containing a batch of images.

        Returns:
            Model predictions for the input batch.
        """
        y = self.model(x)
        return y

    def on_train_epoch_start(self) -> None:  # noqa: D102
        self.__reset_metrics("train")

    def on_validation_epoch_start(self) -> None:  # noqa: D102
        self.__reset_metrics("val")

    def __reset_metrics(self, mode: str) -> None:
        self.losses[mode] = []
        metrics = self.mode_to_metrics[mode]
        metrics.reset()

    def training_step(self, batch, batch_idx):
        """Perform a single training step.

        Args:
            batch: Training batch consisting of images and labels.
            batch_idx: Index of the current batch.

        Returns:
            The computed training loss.
        """
        return self.__step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step.

        Args:
            batch: Validation batch consisting of images and labels.
            batch_idx: Index of the current batch.

        Returns:
            The computed validation loss.
        """
        return self.__step(batch, batch_idx, "val")

    def __step(self, batch, batch_idx, mode):  # mode is one of ['train','test','val']

        imgs, labels = batch

        n, c, h, w = imgs.shape
        assert h == w == self.image_resolution

        normalized_imgs = self.normalize_images(imgs)
        logits = self(normalized_imgs)

        loss = self.criterion(logits, labels)

        with torch.no_grad():
            metrics = self.mode_to_metrics[mode]
            metrics.update(imgs, labels, logits)

        if torch.isnan(loss.cpu()):
            raise Exception("loss is Nan.")
        else:
            self.losses[mode].append(loss.detach().cpu().item())

        return loss

    def on_train_epoch_end(self) -> None:  # noqa: D102
        self.__log_metrics(mode="train")

    def on_validation_epoch_end(self) -> None:  # noqa: D102
        self.__log_metrics(mode="val")

    def __log_metrics(self, mode):
        metrics = self.mode_to_metrics[mode]
        metrics_out = metrics.get_out_dict()
        logs = {f"{mode} {key}": val for key, val in metrics_out.items()}
        logs[f"{mode} loss"] = np.mean(self.losses[mode])
        if mode == "train":
            logs["learning_rate"] = self.optimizers().param_groups[0]["lr"]
        logs["step"] = float(self.current_epoch)
        self.log_dict(logs, on_step=False, on_epoch=True, sync_dist=True)
