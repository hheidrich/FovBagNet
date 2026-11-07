"""Entry point for training image classification models.

This module provides the command-line interface (CLI) for configuring and running the training of several neural
network models on ImageNet's classification task. The CLI is designed as a hierarchy with three levels of ascending
priority:
    1. All available configuration parameters are defined in the standard configuration file located at
       configs/default.yaml.
    2. You can use the "--cfg path/to/another_config.yaml" argument to specify another (partial) config file that
       overrides the default parameters.
    3. The third level allows parameters to be set directly from the command line using
       "--some/parameter desired_value".

Example:
    Train a ResNet50 model:

        python train.py --cfg configs/ablations/resnet50.yaml

    Train a ResNet18 model using the same configuration parameters:

        python train.py --cfg configs/ablations/resnet50.yaml --model/name resnet18

For a full list of available command-line options, refer to configs/default.yaml or take a look at the other files in
the configs directory.
"""

import os

import lightning.pytorch as pl
import torch

from fovbagnet_classifier.config import ConfigMode, get_config
from fovbagnet_classifier.data import get_dataloader
from fovbagnet_classifier.training import LightningWrapper
from fovbagnet_classifier.utils import LitProgressBar, TorchScriptModelCheckpoint

PATH_TO_DEFAULT_CFG = "configs/default.yaml"


def main():
    """Parse config from command-line arguments and runs the model training."""
    cfg = get_config(PATH_TO_DEFAULT_CFG, config_mode=ConfigMode.START_TRAINING)
    module = LightningWrapper(cfg)
    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)

    os.makedirs(cfg.training.out_dir, exist_ok=True)

    accelerator = cfg.accelerator
    if accelerator == "auto":
        if torch.cuda.is_available():
            accelerator = "gpu"
        else:
            accelerator = "cpu"

    if accelerator == "gpu":
        if cfg.devices == "auto" or cfg.devices == -1:
            devices = num_devices = torch.cuda.device_count()
        elif type(cfg.devices) is list:
            devices = cfg.devices
            num_devices = len(cfg.devices)
        elif type(cfg.devices) is int:
            devices = cfg.devices
            num_devices = cfg.devices

    elif accelerator == "cpu":
        devices = 1
        num_devices = 0

    else:
        raise Exception(f"Accelerator {accelerator} is not supported.")

    if num_devices > 1:
        module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=(pl.strategies.DDPStrategy(find_unused_parameters=False) if num_devices > 1 else "auto"),
        max_epochs=cfg.max_epochs,
        logger=pl.loggers.TensorBoardLogger(
            save_dir=cfg.training.out_dir,
            default_hp_metric=False,
        ),
        callbacks=[
            LitProgressBar(),
            TorchScriptModelCheckpoint(
                save_top_k=cfg.training.checkpoints.save_top_k,
                monitor=cfg.training.checkpoints.monitor,
                mode=cfg.training.checkpoints.mode,
                filename=cfg.training.checkpoints.filename,
            ),
        ],
        default_root_dir=cfg.training.out_dir,  # path to checkpoints,
        log_every_n_steps=1,
        val_check_interval=None,
        check_val_every_n_epoch=1,
        precision=cfg.training.precision,
        sync_batchnorm=num_devices > 1,
        benchmark=accelerator == "gpu",
        gradient_clip_val=cfg.optimizer.gradient_clip_val,
    )
    trainer.fit(
        module,
        train_dataloaders=get_dataloader(cfg, mode="train"),
        val_dataloaders=get_dataloader(cfg, mode="val"),
    )


if __name__ == "__main__":
    main()
