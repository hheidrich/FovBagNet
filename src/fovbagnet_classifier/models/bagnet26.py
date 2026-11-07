"""Reduced-depth BagNet architectures.

This module provides BagNet variants with a shallower backbone than the
standard BagNet models defined in :mod:`bagnet`. Whereas the original
BagNet architectures use a ResNet-50-style stage configuration
``[3, 4, 6, 3]``, the models implemented here use a reduced-depth
configuration ``[2, 2, 2, 2]`` corresponding to a 26-layer network.

These models preserve the locality constraints and receptive field
characteristics of their standard counterparts while reducing the number
of parameters and computational cost. They are primarily intended for
ablation studies, rapid experimentation, and resource-constrained
training scenarios.

Available variants:

* ``bagnet26_9``  – Maximal receptive field size of 9×9 pixels.
* ``bagnet26_17`` – Maximal receptive field size of 17×17 pixels.
* ``bagnet26_33`` – Maximal receptive field size of 33×33 pixels.
"""

from .bagnet import BagNet, Bottleneck

__all__ = ["bagnet26_9", "bagnet26_17", "bagnet26_33"]


def bagnet26_33(
    pretrained: bool = False, strides: list[int] = [2, 2, 2, 1], num_classes: int = 1000, build_fc: bool = True
) -> BagNet:
    """Constructs a depth-26 BagNet-33 model.

    This variant uses a reduced-depth residual backbone with stage
    configuration ``[2, 2, 2, 2]`` while maintaining the receptive field
    characteristics of BagNet-33 through the use of 3×3 convolutions in
    all residual stages.

    Args:
        pretrained: If ``True``, loads ImageNet-pretrained weights.
            Defaults to ``False``.
        strides: Stage-wise downsampling factors. Defaults to ``[2, 2, 2, 1]``.
        num_classes: Number of output classes. Defaults to 1000.
        build_fc: Whether to include the final classification layer.
            Defaults to ``True``.

    Returns:
        Initialized BagNet-33 model.
    """
    if pretrained:
        raise Exception("No pretrained models available yet")  # TODO

    model = BagNet(
        Bottleneck, [2, 2, 2, 2], strides=strides, kernel3=[1, 1, 1, 1], num_classes=num_classes, build_fc=build_fc
    )

    return model


def bagnet26_17(
    pretrained: bool = False, strides: list[int] = [2, 2, 2, 1], num_classes: int = 1000, build_fc: bool = True
) -> BagNet:
    """Constructs a depth-26 BagNet-33 model.

    This variant uses a reduced-depth residual backbone with stage
    configuration ``[2, 2, 2, 2]`` while maintaining the receptive field
    characteristics of BagNet-33 through the use of 3×3 convolutions in
    all residual stages.

    Args:
        pretrained: If ``True``, loads ImageNet-pretrained weights.
            Defaults to ``False``.
        strides: Stage-wise downsampling factors. Defaults to ``[2, 2, 2, 1]``.
        num_classes: Number of output classes. Defaults to 1000.
        build_fc: Whether to include the final classification layer.
            Defaults to ``True``.

    Returns:
        Initialized BagNet-33 model.
    """
    if pretrained:
        raise Exception("No pretrained models available yet")  # TODO

    model = BagNet(
        Bottleneck, [2, 2, 2, 2], strides=strides, kernel3=[1, 1, 1, 0], num_classes=num_classes, build_fc=build_fc
    )

    return model


def bagnet26_9(
    pretrained: bool = False, strides: list[int] = [2, 2, 2, 1], num_classes: int = 1000, build_fc: bool = True
) -> BagNet:
    """Constructs a depth-26 BagNet-33 model.

    This variant uses a reduced-depth residual backbone with stage
    configuration ``[2, 2, 2, 2]`` while maintaining the receptive field
    characteristics of BagNet-33 through the use of 3×3 convolutions in
    all residual stages.

    Args:
        pretrained: If ``True``, loads ImageNet-pretrained weights.
            Defaults to ``False``.
        strides: Stage-wise downsampling factors. Defaults to ``[2, 2, 2, 1]``.
        num_classes: Number of output classes. Defaults to 1000.
        build_fc: Whether to include the final classification layer.
            Defaults to ``True``.

    Returns:
        Initialized BagNet-33 model.
    """
    if pretrained:
        raise Exception("No pretrained models available yet")  # TODO

    model = BagNet(
        Bottleneck, [2, 2, 2, 2], strides=strides, kernel3=[1, 1, 0, 0], num_classes=num_classes, build_fc=build_fc
    )

    return model
