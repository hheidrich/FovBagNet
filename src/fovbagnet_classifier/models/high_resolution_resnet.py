"""High-resolution ResNet models for image classification.

This module implements a variant of ResNet designed to maintain higher spatial
resolution in deeper layers by carefully controlling strides, dilations, and
kernel sizes. The `HighResolutionResNet` class allows fine-grained adjustment
of these parameters, making it suitable for tasks that require preserving
spatial detail, such as dense prediction or fine-grained classification.

The module also provides a factory function for constructing a ResNet-50-like
high-resolution model.
"""

import math

import torch
import torch.nn as nn

from ..model_footprint_reduction import fix_strides, reduce_model_stride
from .bagnet import Bottleneck


class HighResolutionResNet(nn.Module):
    """ResNet variant preserving high-resolution features.

    This model allows flexible control over strides, dilations, and kernel sizes
    across all stages, enabling high-resolution feature maps in deep layers. It
    supports optional fully connected layers for classification.

    Args:
        block: Bottleneck block class to use in residual layers.
        layers: Number of blocks in each residual layer.
        strides: Stride for each convolutional stage. Defaults to [1, 1, 2, 2, 2, 1].
        dilations: Dilation for each convolutional stage. Defaults to [1, 1, 1, 1, 1, 1].
        kernel3: Number of 3x3 convolutional layers in each residual layer. Defaults to [0, 0, 0, 0].
        num_classes: Number of output classes. Defaults to 1000.
        build_fc: Whether to include a fully connected classification layer. Defaults to True.

    Attributes:
        conv1, conv2 (nn.Conv2d): Initial convolutional layers.
        bn1 (nn.BatchNorm2d): Batch normalization after conv layers.
        relu (nn.ReLU): Activation function.
        layer1, layer2, layer3, layer4 (nn.Sequential): Residual layers.
        fc (nn.Linear or None): Fully connected classification layer.
        block (nn.Module): Bottleneck block used in layers.
    """

    def __init__(  # noqa: D107
        self,
        block: nn.Module,
        layers: list[int],
        strides: list[int] | None = None,
        dilations: list[int] | None = None,
        kernel3: list[int] | None = None,
        num_classes: int = 1000,
        build_fc: bool = True,
    ):
        if strides is None:
            strides = [1, 1, 2, 2, 2, 1]
        if dilations is None:
            dilations = [1, 1, 1, 1, 1, 1]
        if kernel3 is None:
            kernel3 = [0, 0, 0, 0]

        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=strides[0],
            padding=3 * dilations[0],
            dilation=dilations[0],
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            stride=strides[1],
            padding=dilations[1],
            dilation=dilations[1],
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, 64, layers[0], stride=strides[2], dilation=dilations[2], kernel3=kernel3[0], prefix="layer1"
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=strides[3], dilation=dilations[3], kernel3=kernel3[1], prefix="layer2"
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=strides[4], dilation=dilations[4], kernel3=kernel3[2], prefix="layer3"
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=strides[5], dilation=dilations[5], kernel3=kernel3[3], prefix="layer4"
        )
        self.fc = nn.Linear(512 * block.expansion, num_classes) if build_fc else None
        self.block = block

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(
        self,
        block: nn.Module,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
        kernel3: int = 0,
        prefix: str = "",
    ) -> nn.Sequential:
        """Constructs a residual layer with the given parameters.

        Args:
            block: Bottleneck block class to use.
            planes: Number of output channels for this layer.
            blocks: Number of bottleneck blocks in this layer.
            stride: Stride for the first block in the layer. Defaults to 1.
            dilation: Dilation for convolutional layers. Defaults to 1.
            kernel3: Number of 3x3 convolutions in the layer. Remaining blocks use 1x1. Defaults to 0.
            prefix: Optional string prefix for layer naming. Defaults to "".

        Returns:
            Residual layer containing `blocks` bottleneck modules.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        kernel, dil = (1, 1) if kernel3 == 0 else (3, dilation)

        layers.append(
            block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                dilation=dil,
                downsample=downsample,
                kernel_size=kernel,
                padding=dil * (kernel // 2),
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel, dil = (1, 1) if kernel3 <= i else (3, dilation)
            layers.append(
                block(
                    inplanes=self.inplanes, planes=planes, dilation=dil, kernel_size=kernel, padding=dil * (kernel // 2)
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, avg_pool: bool = True) -> torch.Tensor:
        """Forward pass of the HighResolutionResNet.

        Args:
        x: Input image batch of shape (N, C, H, W).
        avg_pool: Whether to apply global average pooling before the fully connected layer.
          If False, the feature map is returned in (N, H, W, C) format. Defaults to True.

        Returns:
        If avg_pool is True, returns (N, num_classes) logits. Otherwise,
        returns high-resolution feature maps of shape (N, H, W, C).
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if avg_pool:
            x = nn.functional.avg_pool2d(x, x.size()[2], stride=1)
            x = x.view(x.size(0), -1)
        else:
            x = x.permute(0, 2, 3, 1)

        if self.fc is not None:
            x = self.fc(x)

        return x


def get_highresnet50(num_classes: int = 1000, stride: int = 8) -> HighResolutionResNet:
    """Constructs a high-resolution ResNet-50 model.

    The model uses Bottleneck blocks and modified strides/dilations to maintain
    higher spatial resolution compared to standard ResNet-50. This is useful
    for dense prediction or tasks where spatial detail is important.

    Args:
        num_classes: Number of output classes. Defaults to 1000.
        stride: Final stride of the network. Adjusts intermediate strides and
          dilations to achieve the desired output resolution. Defaults to 8.

    Returns:
        Configured high-resolution ResNet-50 model.
    """
    layers = [3, 4, 6, 3]
    strides = [1, 2, 2, 2, 1]
    # our model lacks original ResNet's first two pooling steps hence we need dilations to resemble its behaviour.
    dilations = [2, 4, 4, 4, 4]
    kernel3 = [1] + layers  # all bottleneck layers use 3x3 conv.
    strides_adapted, dilations_adapted = reduce_model_stride(
        strides=[1] + strides, dilations=[1] + dilations, final_stride=stride
    )
    initial_stride, strides_fixed, dilations_fixed, offset_left, receptive_field = fix_strides(
        kernel_sizes=[7] + [2 * k + 1 for k in kernel3],
        strides=strides_adapted,
        dilations=dilations_adapted,
    )
    print(
        ",".join(
            [
                f"receptive_field {receptive_field}",
                f"offset_left {offset_left}",
                f"dilations {[1] + dilations}",
                f"dilations_fixed {dilations_fixed}",
                f"strides {[1] + strides}",
                f"strides_fixed {strides_fixed}",
                f"initial_stride {initial_stride}",
            ]
        )
    )
    model = HighResolutionResNet(
        Bottleneck,
        layers=layers,
        strides=strides_fixed,
        dilations=dilations_fixed,
        kernel3=layers,
        num_classes=num_classes,
        build_fc=True,
    )
    return model


class HighResolutionResNetEnsemble(nn.Module):
    """Ensemble of ResNet variants preserving high-resolution features.

    Args:
        block: Bottleneck block class to use in residual layers.
        layers: Number of blocks in each residual layer.
        strides: Stride for each convolutional stage. Defaults to [1, 1, 2, 2, 2, 1].
        dilations: Dilation for each convolutional stage. Defaults to [1, 1, 1, 1, 1, 1].
        kernel3: Number of 3x3 convolutional layers in each residual layer. Defaults to [0, 0, 0, 0].
        num_classes: Number of output classes. Defaults to 1000.
        num_models: Number of model in the ensemble. Defaults to 1.
        num_model_aggregation_blocks: Number of blocks after model aggregation. Defaults to 2.

    Attributes:
        conv1, conv2 (nn.Conv2d): Initial convolutional layers.
        bn1 (nn.BatchNorm2d): Batch normalization after conv layers.
        relu (nn.ReLU): Activation function.
        layer1, layer2, layer3, layer4 (nn.Sequential): Residual layers.
        fc (nn.Linear or None): Fully connected classification layer.
        block (nn.Module): Bottleneck block used in layers.
    """

    def __init__(  # noqa: D107
        self,
        block: nn.Module,
        layers: list[int],
        strides: list[int] | None = None,
        dilations: list[int] | None = None,
        kernel3: list[int] | None = None,
        num_classes: int = 1000,
        num_models: int = 1,
        num_model_aggregation_blocks: int = 2,
    ):
        if strides is None:
            strides = [1, 1, 2, 2, 2, 1]
        if dilations is None:
            dilations = [1, 1, 1, 1, 1, 1]
        if kernel3 is None:
            kernel3 = [0, 0, 0, 0]

        super().__init__()

        self.models = nn.ModuleList(
            [
                HighResolutionResNet(
                    block=block,
                    layers=layers,
                    strides=strides,
                    dilations=dilations,
                    kernel3=kernel3,
                    num_classes=num_classes,
                    build_fc=False,
                )
                for _ in range(num_models)
            ]
        )

        self.model_aggregation_blocks = nn.ModuleList(
            [Bottleneck(2048, 512) for _ in range(num_model_aggregation_blocks)]
        )
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor, avg_pool: bool = True) -> torch.Tensor:
        """Forward pass of the HighResolutionResNetEnsemble.

        Args:
        x: Input image batch of shape (N, C, H, W).
        avg_pool: Whether to apply global average pooling before the fully connected layer.
            If False, the feature map is returned in (N, H, W, C) format. Defaults to True.

        Returns:
        If avg_pool is True, returns (N, num_classes) logits. Otherwise,
        returns high-resolution feature maps of shape (N, H, W, C).
        """
        x = torch.stack([model(x, avg_pool=False).permute(0, 3, 1, 2) for model in self.models], dim=0).sum(dim=0)

        for block in self.model_aggregation_blocks:
            x = block(x)

        n, num_features, h, w = x.shape
        if avg_pool:
            x = nn.functional.avg_pool2d(x, (h, w), stride=1)
            x = x.view(x.size(0), -1)
        else:
            x = x.permute(0, 2, 3, 1)
        return self.fc(x)


def get_highresnet26_ensemble(
    num_classes: int = 1000,
    stride: int = 8,
    num_models: int = 1,
    num_model_aggregation_blocks: int = 2,
) -> HighResolutionResNetEnsemble:
    """Constructs an Ensemble of high-resolution ResNet-26 models.

    Args:
        num_classes: Number of output classes. Defaults to 1000.
        stride: Final stride of the network. Adjusts intermediate strides and
          dilations to achieve the desired output resolution. Defaults to 8.
        num_models: Number of model in the ensemble.
        num_model_aggregation_blocks: Number of blocks after model aggregation.

    Returns:
        Configured high-resolution ResNet-50 model.
    """
    layers = [2, 2, 2, 2]
    strides = [1, 2, 2, 2, 1]
    dilations = [
        2,
        4,
        4,
        4,
        4,
    ]  # our model lacks original ResNet's first two pooling steps hence we need dilations to resemble its behaviour.
    kernel3 = [1] + layers  # all bottleneck layers use 3x3 conv.
    strides_adapted, dilations_adapted = reduce_model_stride(
        strides=[1] + strides, dilations=[1] + dilations, final_stride=stride
    )
    initial_stride, strides_fixed, dilations_fixed, offset_left, receptive_field = fix_strides(
        kernel_sizes=[7] + [2 * k + 1 for k in kernel3],
        strides=strides_adapted,
        dilations=dilations_adapted,
    )
    print(
        ",".join(
            [
                f"receptive_field {receptive_field}",
                f"offset_left {offset_left}",
                f"dilations {[1] + dilations}",
                f"dilations_fixed {dilations_fixed}",
                f"strides {[1] + strides}",
                f"strides_fixed {strides_fixed}",
                f"initial_stride {initial_stride}",
            ]
        )
    )
    model = HighResolutionResNetEnsemble(
        Bottleneck,
        layers=layers,
        strides=strides_fixed,
        dilations=dilations_fixed,
        kernel3=layers,
        num_classes=num_classes,
        num_models=num_models,
        num_model_aggregation_blocks=num_model_aggregation_blocks,
    )
    return model
