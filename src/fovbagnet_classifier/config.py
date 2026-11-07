"""Configuration management module.

This module provides utilities to load, merge, and override configuration
for training and evaluation scripts. Configurations can come from multiple
sources:

1. Default YAML configuration files (`default.yaml`).
2. Optional user-specified YAML config files (`--cfg` argument).
3. Command-line arguments (`--<key>`).

The precedence order for merging configurations is as follows:
    cfg_default < cfg_special < cfg_args
Later sources override earlier ones.

Key functionalities include:
- Loading YAML config files into Python dictionaries (`load_yaml_config`).
- Flattening and nesting dictionary keys for CLI argument compatibility.
- Merging multiple configuration sources with nested key support.
- Building final OmegaConf configurations for training and resuming from checkpoints.

Example usage:
    cfg = get_config("path/to/default.yaml")
    cfg_resume = get_resume_config("path/to/default.yaml")
"""

import argparse
from enum import Enum, auto
from os import path as osp
from typing import Any, Dict, List, Optional, Tuple

import yaml
from omegaconf import DictConfig, OmegaConf


def flatten(xs: list) -> list:
    """Flatten a nested list of arbitrary depth.

    Args:
        xs: A possibly nested list.

    Returns:
        A flat list containing all elements of xs.
    """
    ys = []
    for x in xs:
        if type(x) is list:
            ys = ys + flatten(x)
        else:
            ys.append(x)
    return ys


def nest_dict(x: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a flat dictionary with '/'-separated keys into a nested dictionary.

    Args:
        x: Dictionary with keys like "a/b/c".

    Returns:
        Nested dictionary where keys are split by '/'.
    """
    y: Dict[str, Any] = {}

    def deep_insert(y: dict, key: Any, val: Any) -> None:
        split = key.split("/")
        if len(split) == 1:
            y[key] = val
        else:
            prefix = split[0]
            suffix = "/".join(split[1:])
            if prefix not in y:
                y[prefix] = {}
            deep_insert(y[prefix], suffix, val)

    for key, val in x.items():
        deep_insert(y, key, val)
    return y


def extract_argument_keys(cfg: Dict[str, Any], prefix: Optional[str] = None) -> List[Tuple[str, Any]]:
    """Extract argument keys and their default values from a config dictionary.

    Args:
        cfg: Config dictionary to process.
        prefix: Optional prefix to prepend to argument keys.

    Returns:
        List of tuples of the form (key, default_value).
    """
    if prefix is None:
        prefix = ""
    if prefix != "":
        prefix = f"{prefix}/"

    def f(key, val):
        if isinstance(val, dict):
            return extract_argument_keys(val, prefix=f"{prefix}{key}")
        else:
            return f"{prefix}{key}", val

    return [f(key, val) for key, val in cfg.items()]


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries with ascending priority.

    Args:
        *dicts: Dictionaries to merge, later dictionaries overwrite earlier ones.

    Returns:
        Merged dictionary.
    """
    y: Dict[str, Any] = {}

    def deep_update(a, b):
        """Dict a is updated with the values of dict b."""
        for key, val in b.items():
            if key not in a:
                a[key] = {}
            if isinstance(val, dict):
                deep_update(a[key], val)
            else:
                a[key] = val

    for x in dicts:
        deep_update(y, x)
    return y


def __my_bool(val: str) -> bool:
    """Custom argparse type for boolean values.

    Args:
        val: String to convert to bool.

    Returns:
        Boolean value corresponding to val.

    Raises:
        argparse.ArgumentTypeError: If val is not "True" or "False".
    """
    if val == "True":
        return True
    elif val == "False":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value must be True or False.")


def load_yaml_config(path_to_config: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path_to_config: Path to the YAML config file.

    Returns:
        Dictionary containing the YAML configuration.
    """
    with open(path_to_config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


class ConfigMode(Enum):
    """Execution modes supported by ``get_config``."""

    START_TRAINING = auto()
    RESUME_TRAINING = auto()
    EVALUATE_CHECKPOINT = auto()


def get_config(path_to_default_cfg: str, config_mode: ConfigMode = ConfigMode.START_TRAINING) -> DictConfig:
    """Build the final configuration by merging multiple configuration sources.

    The configuration is assembled from different sources depending on
    config_mode. Later sources override earlier ones.

    For ConfigMode.START_TRAINING the precedence is:

    cfg_default < cfg_special < cfg_args

    where

    cfg_default: Base configuration loaded from default.yaml.
    cfg_special: Optional configuration file specified via --cfg.
    cfg_args: Command-line overrides.

    For ConfigMode.RESUME_TRAINING and
    ConfigMode.EVALUATE_CHECKPOINT the precedence is:

    cfg_default < cfg_checkpoint < cfg_args

    where

    cfg_default: Base configuration loaded from default.yaml.
    cfg_checkpoint: Configuration stored in the checkpoint's
    hparams.yaml file.
    cfg_args: Command-line overrides.

    Args:
    path_to_default_cfg: Path to the default YAML configuration file.
    config_mode: Determines how the configuration is initialized and
    which command-line arguments are available.

        - ``ConfigMode.START_TRAINING``: Start a new training run.
        Accepts an optional ``--cfg`` argument.
        - ``ConfigMode.RESUME_TRAINING``: Resume training from a
        checkpoint. Requires ``--ckpt``.
        - ``ConfigMode.EVALUATE_CHECKPOINT``: Evaluate a checkpoint.
        Requires ``--ckpt`` and enables additional evaluation-specific
        command-line arguments.

    Returns:
    OmegaConf.DictConfig: The final merged configuration.
    """
    # load defaul config file (default.yaml)
    cfg_default = load_yaml_config(path_to_default_cfg)

    # extract argument keys and their default values from the default config
    arguments = dict(flatten(extract_argument_keys(cfg_default)))

    # build parser
    parser = argparse.ArgumentParser()
    if config_mode is ConfigMode.START_TRAINING:
        help_cfg = "Path where the config yaml file is located."
        parser.add_argument("--cfg", help=help_cfg)
    else:
        help_ckpt = """
        Path where the model checkpoint is located.
        The path must end with '.../version_{}/checkpoints/epoch={}.ckpt'.
        Also '.../version_{}/hparams.yaml' must exist.
        """
        parser.add_argument(
            "--ckpt",
            help=help_ckpt,
            required=True,
        )

    if config_mode is ConfigMode.EVALUATE_CHECKPOINT:
        help_evaluate_on_masks = """
        If set to True, use the ImageNet-S validation dataset for evaluation. Default False.
        """
        parser.add_argument(
            "--evaluate_on_masks",
            help=help_evaluate_on_masks,
            required=False,
            type=__my_bool,
            default=True,
        )

        help_path_to_annotations_root = """
        Path to the annotations' root directory.
        If evaluate_on_masks this should lead to the ImageNet-S root containing the 'validation-segmentation' directory.
        Otherwise it should lead to a folder with the val directory containing the ImageNet xml annotation files.
        """
        parser.add_argument(
            "--path_to_annotations_root",
            help=help_path_to_annotations_root,
            required=True,
            type=str,
        )

    for argument_key, default_value in arguments.items():
        parser.add_argument(
            f"--{argument_key}",
            help=f"Default: {default_value}",
            type=__my_bool if (t := type(default_value)) is bool else t,
        )

    # parse arguments
    args = parser.parse_args()
    cfg_args = nest_dict({key: val for key, val in vars(args).items() if val is not None})

    if config_mode is ConfigMode.START_TRAINING:
        # load special config file
        path_to_special_cfg = args.cfg
        if path_to_special_cfg is not None:
            cfg_special = load_yaml_config(path_to_special_cfg)
        else:
            cfg_special = {}

        # create the final configuration
        cfg = merge_dicts(cfg_default, cfg_special, cfg_args)
    else:
        # load checkpoint config file hparams.yaml
        path_to_checkpoint = args.ckpt
        path_to_checkpoints_dir, checkpoint_name = osp.split(path_to_checkpoint)
        path_to_version, _ = osp.split(path_to_checkpoints_dir)
        path_to_hparams = osp.join(path_to_version, "hparams.yaml")
        cfg_checkpoint = load_yaml_config(path_to_hparams)["cfg"]

        # create the final configuration
        cfg = merge_dicts(cfg_default, cfg_checkpoint, cfg_args)

    # return the final configuration as OmegaConf instance
    return OmegaConf.create(cfg)
