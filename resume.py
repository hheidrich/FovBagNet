"""Entry point for resuming image classification model training.

This module provides the command-line interface (CLI) for resuming the training
of previously trained neural network models from a checkpoint. Configuration is
handled using the same hierarchical system as in train.py:

1. All available configuration parameters are defined in the standard
   configuration file located at ``configs/default.yaml``.
2. Your checkpoint lives in a path/to/checkpoints folder and it is assumed that
   path/to/hparams.yaml exists containing the original configuration of the
   checkpoint. Note that path/to/hparams.yaml is written at the start of each
   training process.
3. Individual parameters can be overridden directly from the command line
   using ``--some/parameter desired_value``. This is required if a model was
   initially defined to train for 100 epochs but should now be trained for
   another 100 epochs. In that case set --max_epochs 200.

In addition, this script requires a checkpoint path specifying the training
state to restore before continuing optimization.

Example:
Resume training from a checkpoint saved after 50 epochs:

    python resume.py --ckpt path/to/checkpoints/epoch=50.ckpt

For a full list of available command-line options, refer to
configs/default.yaml or the other configuration files in the configs
directory.
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
    """Parse config from command-line arguments and resumes the model training from a given checkpoint file."""
    cfg = get_config(PATH_TO_DEFAULT_CFG, config_mode=ConfigMode.RESUME_TRAINING)
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
        sync_batchnorm=num_devices > 1,
        benchmark=accelerator == "gpu",
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
        gradient_clip_val=cfg.optimizer.gradient_clip_val,
    )
    trainer.fit(
        module,
        ckpt_path=cfg.ckpt,
        train_dataloaders=get_dataloader(cfg, mode="train"),
        val_dataloaders=get_dataloader(cfg, mode="val"),
        weights_only=False,
    )


if __name__ == "__main__":
    main()
