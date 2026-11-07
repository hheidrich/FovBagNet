"""BagNet architectures for interpretable image classification.

This module implements the BagNet family of convolutional neural networks
introduced by Brendel and Bethge (2019). BagNets are ResNet-like models
that deliberately restrict the receptive field of deep feature maps,
encouraging predictions to be based on local image evidence rather than
global object shape.

The implementation is adapted from the official BagNet repository and
provides configurable BagNet variants with maximum receptive field sizes
of 9×9, 17×17, and 33×33 pixels. The models can be used either as
standard image classifiers or as dense feature extractors for
explainability and localization studies.

Unlike standard ResNets, BagNets replace many spatial convolutions with
1×1 convolutions, thereby limiting the amount of spatial context
available to each prediction. This property makes them particularly
useful for research on:

* Explainable artificial intelligence (XAI)
* Local feature attribution
* Weakly supervised localization
* Shortcut learning and dataset bias analysis
* Receptive field ablation studies

References:
    Brendel, W., & Bethge, M. (2019).
    "Approximating CNNs with Bag-of-local-Features Models Works
    Surprisingly Well on ImageNet."
    International Conference on Learning Representations (ICLR).

Original implementation:
    https://github.com/wielandbrendel/bag-of-local-features-models
"""

import math

import torch
import torch.nn as nn
from torch.utils import model_zoo

__all__ = ["bagnet9", "bagnet17", "bagnet33"]

model_urls = {
    "bagnet9": "https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet8-34f4ccd2.pth.tar",
    "bagnet17": "https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet16-105524de.pth.tar",
    "bagnet33": "https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet32-2ddd53ed.pth.tar",
}


class Bottleneck(nn.Module):
    """Residual bottleneck block used in BagNet architectures.

    This block follows the ResNet bottleneck design consisting of a
    1×1 reduction convolution, a spatial convolution (either 1×1 or 3×3),
    and a final 1×1 expansion convolution. Unlike standard ResNet blocks,
    BagNet uses carefully controlled spatial kernels to restrict the
    receptive field and encourage locality-based feature extraction.

    When the spatial convolution uses a kernel larger than 1×1, the
    residual branch is cropped to align with the centered receptive field
    of the main branch before residual addition.

    Attributes:
        expansion (int): Channel expansion factor applied by the final
            1×1 convolution.
    """

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
        kernel_size: int = 1,
        padding=0,
    ):
        """Initializes a BagNet bottleneck block.

        Args:
            inplanes: Number of input channels.
            planes: Number of channels in the bottleneck representation.
            stride: Stride of the spatial convolution. Defaults to 1.
            dilation: Dilation factor of the spatial convolution. Defaults to 1.
            downsample: Residual projection used to match spatial dimensions and channel count.
                Defaults to None i.e. the identity mapping.
            kernel_size: Kernel size of the spatial convolution. Typically 1 or 3. Defaults to 1.
            padding: Padding applied to the spatial convolution. Defaults to 0.
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass of the bottleneck block.

        Args:
            x: Input tensor of shape ``(batch_size, channels, height, width)``.

        Returns:
            Output feature map after residual addition and activation.
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.kernel_size > 1:
            # Align height and width of residual to match the centering pixel of the conv2 kernel.
            n, c, h, w = x.shape
            k = self.dilation * (self.kernel_size // 2) - self.padding
            residual = x[:, :, k : h - k, k : w - k]
        else:
            residual = x

        if self.downsample is not None:
            # If channels differ or conv2 is strided, use a 1x1 conv to match residual to out's shape.
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class BagNet(nn.Module):
    """BagNet architecture with configurable receptive field size.

    BagNet is a ResNet-like convolutional network that limits the receptive
    field of deep features by replacing many spatial convolutions with 1×1
    convolutions. Depending on the chosen configuration (e.g., BagNet-9,
    BagNet-17, or BagNet-33), only a subset of residual blocks contains
    3×3 convolutions.

    The architecture can operate either as a standard image classifier
    using global average pooling or as a dense feature extractor that
    preserves spatial predictions.

    References:
        Brendel, W. and Bethge, M. (2019).
        "Approximating CNNs with Bag-of-local-Features Models Works
        Surprisingly Well on ImageNet".
    """

    def __init__(
        self,
        block: type[nn.Module],
        layers: list[int],
        strides: list[int] = [2, 2, 2, 1],
        dilations: list[int] = [1, 1, 1, 1],
        kernel3: list[int] = [0, 0, 0, 0],
        num_classes: int = 1000,
        build_fc: bool = True,
    ):
        """Initializes a BagNet model.

        Args:
            block: Residual block class used to construct the network.
            layers: Number of residual blocks in each stage.
            strides: Stage-wise downsampling factors. Defaults to ``[2, 2, 2, 1]``.
            dilations: Dilation factors for each stage. Defaults to ``[1, 1, 1, 1]``.
            kernel3: Number of initial blocks in each stage that use 3×3 convolutions.
                Remaining blocks use 1×1 convolutions. Defaults to ``[0, 0, 0, 0]``.
            num_classes : Number of output classes. Defaults to 1000.
            build_fc: Whether to construct the final classification layer.
                If ``False``, the model outputs pooled or dense feature representations.
                Defaults to ``True``.

        Notes:
            Variant configuration:

            * BagNet-33: kernel3=[1, 1, 1, 1]
            * BagNet-17: kernel3=[1, 1, 1, 0]
            * BagNet-9:  kernel3=[1, 1, 0, 0]

            These settings determine how many residual stages contain 3×3
            convolutions and therefore control the effective receptive field.
        """
        self.inplanes = 64
        super(BagNet, self).__init__()

        if len(strides) == 4:
            strides = [1] + strides
        if len(dilations) == 4:
            dilations = [1] + dilations
        if len(kernel3) == 4:
            kernel3 = [1] + kernel3

        stride_conv2, strides = strides[0], strides[1:]
        dilation_conv_2, dilations = dilations[0], dilations[1:]
        kernel3_conv2, kernel3 = kernel3[0], kernel3[1:]

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            64,
            64,
            kernel_size=3 if kernel3_conv2 else 1,
            stride=stride_conv2,
            padding=0,
            dilation=dilation_conv_2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, 64, layers[0], stride=strides[0], dilation=dilations[0], kernel3=kernel3[0], prefix="layer1"
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=strides[1], dilation=dilations[1], kernel3=kernel3[1], prefix="layer2"
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=strides[2], dilation=dilations[2], kernel3=kernel3[2], prefix="layer3"
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=strides[3], dilation=dilations[3], kernel3=kernel3[3], prefix="layer4"
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
        block: type[nn.Module],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
        kernel3: int = 0,
        prefix: str | None = None,
    ) -> nn.Sequential:
        """Builds a residual stage of the network.

        Args:
            block: Residual block class.
            planes: Number of bottleneck channels in the stage.
            blocks: Number of residual blocks in the stage.
            stride: Stride of the first block in the stage. Defaults to 1.
            dilation: Dilation factor applied to spatial convolutions. Defaults to 1.
            kernel3: Number of blocks that use 3×3 convolutions.
                Remaining blocks use 1×1 convolutions. Defaults to 0.
            prefix: Stage identifier retained for backward compatibility. Defaults to "".

        Returns:
            nn.Sequential: Residual stage composed of bottleneck blocks.
        """
        if prefix is None:
            prefix = ""

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
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel, dil = (1, 1) if kernel3 <= i else (3, dilation)
            layers.append(block(inplanes=self.inplanes, planes=planes, dilation=dil, kernel_size=kernel))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, avg_pool: bool = True) -> torch.Tensor:
        """Runs a forward pass through the network.

        Args:
            x: Input image batch of shape ``(batch_size, 3, height, width)``.
            avg_pool: If ``True``, applies global average pooling before the classifier,
                producing one prediction per image. If ``False``, preserves spatial
                dimensions and returns dense feature representations or dense class logits.
                Defaults to ``True``.

        Returns:
            If ``avg_pool=True``, returns a tensor of shape
            ``(batch_size, num_classes)`` when a classifier is present,
            otherwise ``(batch_size, feature_dim)``.

            If ``avg_pool=False``, returns spatial features or logits of
            shape ``(batch_size, height, width, channels_or_classes)``.
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


def bagnet33(
    pretrained: bool = False, strides: list[int] = [2, 2, 2, 1], num_classes: int = 1000, build_fc: bool = True
) -> BagNet:
    """Constructs a BagNet-33 model.

    BagNet-33 has a maximum receptive field size of 33×33 pixels.

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
    model = BagNet(
        Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1, 1, 1, 1], num_classes=num_classes, build_fc=build_fc
    )
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["bagnet33"]))
    return model


def bagnet17(
    pretrained: bool = False, strides: list[int] = [2, 2, 2, 1], num_classes: int = 1000, build_fc: bool = True
) -> BagNet:
    """Constructs a BagNet-17 model.

    BagNet-17 has a maximum receptive field size of 17×17 pixels.

    Args:
        pretrained: If ``True``, loads ImageNet-pretrained weights.
            Defaults to ``False``.
        strides: Stage-wise downsampling factors. Defaults to ``[2, 2, 2, 1]``.
        num_classes: Number of output classes. Defaults to 1000.
        build_fc: Whether to include the final classification layer.
            Defaults to ``True``.

    Returns:
        Initialized BagNet-17 model.
    """
    model = BagNet(
        Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1, 1, 1, 0], num_classes=num_classes, build_fc=build_fc
    )
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["bagnet17"]))
    return model


def bagnet9(
    pretrained: bool = False, strides: list[int] = [2, 2, 2, 1], num_classes: int = 1000, build_fc: bool = True
) -> BagNet:
    """Constructs a BagNet-9 model.

    BagNet-9 has a maximum receptive field size of 9×9 pixels.

    Args:
        pretrained: If ``True``, loads ImageNet-pretrained weights.
            Defaults to ``False``.
        strides: Stage-wise downsampling factors. Defaults to ``[2, 2, 2, 1]``.
        num_classes: Number of output classes. Defaults to 1000.
        build_fc: Whether to include the final classification layer.
            Defaults to ``True``.

    Returns:
        Initialized BagNet-9 model.
    """
    model = BagNet(
        Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1, 1, 0, 0], num_classes=num_classes, build_fc=build_fc
    )
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["bagnet9"]))
    return model
