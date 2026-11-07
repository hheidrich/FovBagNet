"""Model factory utilities.

This module provides a centralized interface for constructing models from
configuration files. Supported architectures include standard ResNets,
BagNet variants with different receptive field sizes and depths, and
custom architectures developed for explainable computer vision research.

Supported architectures:

* ResNet-18
* ResNet-50
* BagNet-9
* BagNet-17
* BagNet-33
* BagNet26-9
* BagNet26-17
* BagNet26-33
* Multiscale Dilation Model
* High-Resolution ResNet-50
* High-Resolution ResNet-26 Ensemble

Typical usage:

    >>> model = get_model(cfg)

where ``cfg.model.name`` specifies the architecture identifier and
``cfg.model.kwargs`` contains the corresponding constructor arguments.
"""

import omegaconf
import torch.nn as nn
from torchvision.models import resnet18, resnet50

from .bagnet import bagnet9, bagnet17, bagnet33
from .bagnet26 import bagnet26_9, bagnet26_17, bagnet26_33
from .fovbagnet import MultiscaleDilationModel
from .high_resolution_resnet import get_highresnet26_ensemble, get_highresnet50


def get_model(cfg):
    """Constructs a model from a configuration object.

    The model architecture is selected using ``cfg.model.name`` and
    instantiated with keyword arguments from ``cfg.model.kwargs``.
    After construction, convolutional and batch-normalization layers
    are reinitialized using standard initialization schemes commonly
    used for deep convolutional networks.

    Supported model names include:

    * ``resnet18``
    * ``resnet50``
    * ``bagnet9``
    * ``bagnet17``
    * ``bagnet33``
    * ``bagnet26_9``
    * ``bagnet26_17``
    * ``bagnet26_33``
    * ``bagnet50_9``
    * ``bagnet50_17``
    * ``bagnet50_33``
    * ``multiscale_dilation_model``
    * ``highresnet50``
    * ``highresnet26_ensemble``

    Args:
        cfg (omegaconf.DictConfig): Configuration object containing model
            specification parameters. The following fields are expected:

            * ``cfg.model.name``: Model identifier.
            * ``cfg.model.kwargs``: Keyword arguments passed to the model
              constructor.

    Returns:
        nn.Module: Instantiated and initialized model.

    Raises:
        KeyError: If ``cfg.model.name`` does not correspond to a supported
            architecture.
    """
    model = {
        "resnet18": resnet18,
        "resnet50": resnet50,
        "bagnet9": bagnet9,
        "bagnet17": bagnet17,
        "bagnet33": bagnet33,
        "bagnet26_9": bagnet26_9,
        "bagnet26_17": bagnet26_17,
        "bagnet26_33": bagnet26_33,
        "bagnet50_9": bagnet9,
        "bagnet50_17": bagnet17,
        "bagnet50_33": bagnet33,
        "multiscale_dilation_model": MultiscaleDilationModel,
        "highresnet50": get_highresnet50,
        "highresnet26_ensemble": get_highresnet26_ensemble,
    }[cfg.model.name](**omegaconf.OmegaConf.to_container(cfg.model.kwargs, resolve=True))

    # Initialize weights using He initialization
    pretrained = cfg.model.kwargs.get("pretrained", False)
    if not pretrained:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    return model
