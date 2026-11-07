"""Multiscale dilation-based BagNet model for image classification.

This module implements a multi-scale BagNet-like architecture that applies
dilated convolutions across multiple Gaussian pyramid levels. It supports
evaluation at several scales (levels), aggregates feature maps with optional
Bottleneck blocks, and produces class logits.

Input image (N, C, H, W)
            │
            ▼
    Gaussian scale-space
            │
    ┌───────┼───────┐
    │       │       │
    ▼       ▼       ▼
 Level 0  Level 1  Level 2   ...
 σ=σ₀     σ=2σ₀    σ=4σ₀
    │       │       │
    │       │       │
    ▼       ▼       ▼
 BagNet    BagNet   BagNet
 d=1       d=2      d=4
 RF≈9      RF≈17    RF≈33
    │       │       │
    └───────┼───────┘
            ▼
    Sum across levels
            │
            ▼
 Bottleneck aggregation
        blocks
            │
            ▼
  Global average pool
       (optional)
            │
            ▼
     Fully connected
            │
            ▼
         Logits

"""

import math

import numpy as np
import torch
from torch import nn

from ..model_footprint_reduction import fix_strides, reduce_model_stride
from ..scalespace_tools import GaussFilter, get_gauss_stack
from .bagnet import BagNet, Bottleneck


class MultiscaleDilationModel(nn.Module):
    """Multi-scale BagNet model with dilated convolutions.

    This model evaluates an image at multiple scales using Gaussian pyramid
    filtering, applies BagNet backbones per scale, and aggregates features
    across levels using Bottleneck blocks. The final classification head
    produces class logits.

    Attributes:
        levels_to_evaluate: List of pyramid levels to evaluate.
        num_levels: Total number of levels, derived from `levels_to_evaluate`.
        patch_size: Receptive field size of BagNet backbone.
        stride: Network output stride.
        level_to_initial_stride (dict[int, int]): Maps each level to its initial stride in the backbone.
        models: List of BagNet instances, one per level.
        level_aggregation_blocks: Bottleneck blocks for cross-level feature aggregation.
        fc: Final fully connected layer for classification.
    """

    levels_to_evaluate: list[int]
    num_levels: int
    patch_size: int
    stride: int
    level_to_initial_stride: dict[int, int]
    # The following attributes' type definitions are commented due to torchscript typing:
    # models: nn.ModuleList
    # level_aggregation_blocks: nn.ModuleList
    # fc: nn.Linear

    def __init__(
        self,
        pretrained: bool = False,
        num_classes: int = 1000,
        levels_to_evaluate: int | list[int] = 2,
        sigma: float | None = None,
        patch_size: int = 9,
        stride: int = 8,
        num_network_layers: int = 26,
        share_params_across_scales: bool = False,
        num_level_aggregation_blocks: int = 2,
    ):
        """Initializes a multi-scale BagNet model with dilated convolutions.

        Constructs one BagNet backbone per evaluated pyramid level, adapts the
        backbone strides and dilations to achieve the requested output stride,
        and creates feature aggregation and classification layers.

        Args:
            pretrained: If True, attempts to load pretrained weights.
                Pretrained models are not currently available.
            num_classes: Number of output classes for the final classifier.
            levels_to_evaluate: Pyramid levels to evaluate. If an integer is
                provided, evaluates levels ``range(levels_to_evaluate)``.
                If a list is provided, evaluates the specified levels.
            sigma: Standard deviation of the Gaussian filter used to build the
                image pyramid. If None, defaults to ``sqrt(2)``.
            patch_size: BagNet receptive field size. Must be one of
                ``{9, 17, 33, 121}``.
            stride: Desired final output stride of each backbone after stride
                reduction. Defaults to 8.
            num_network_layers: Number of layers in each BagNet backbone.
                Supported values are ``26`` and ``50``.
            share_params_across_scales: Whether to share backbone parameters
                across pyramid levels. Not currently implemented.
            num_level_aggregation_blocks: Number of Bottleneck blocks used to
                aggregate features across pyramid levels.

        Raises:
            Exception: If ``pretrained=True``.
            NotImplementedError: If ``share_params_across_scales=True``.
        """
        super().__init__()
        if pretrained:
            raise Exception("No pretrained models available yet")  # TODO
        if share_params_across_scales:
            raise NotImplementedError()

        layers = {
            26: [2, 2, 2, 2],
            50: [3, 4, 6, 3],
        }[num_network_layers]

        kernel3 = {
            9: [1, 1, 1, 0, 0],
            17: [1, 1, 1, 1, 0],
            33: [1, 1, 1, 1, 1],
            121: [1] + layers,
        }[patch_size]
        kernel_sizes = [2 * k + 1 for k in kernel3]
        strides: list[int] = [1, 2, 2, 2, 1]

        self.patch_size = patch_size
        self.stride = stride
        if sigma is None:
            sigma = np.sqrt(2)
        self.gauss_filter = GaussFilter(sigma=sigma)

        if isinstance(levels_to_evaluate, int):
            levels_to_evaluate = list(range(levels_to_evaluate))
        self.levels_to_evaluate: list[int] = levels_to_evaluate
        self.num_levels = max(levels_to_evaluate) + 1
        self.level_to_initial_stride = {}

        models = []
        for level in levels_to_evaluate:
            dilations = [1 if kernel_size == 1 else int(2**level) for kernel_size in kernel_sizes]
            strides_adapted, dilations_adapted = reduce_model_stride(
                strides=strides, dilations=dilations, final_stride=stride
            )
            initial_stride, strides_fixed, dilations_fixed, offset_left, receptive_field = fix_strides(
                kernel_sizes=kernel_sizes,
                strides=strides_adapted,
                dilations=dilations_adapted,
            )
            model = BagNet(
                block=Bottleneck,
                layers=layers,
                strides=strides_fixed,
                dilations=dilations_fixed,
                kernel3=kernel3,
                build_fc=False,
            )
            models.append(model)
            self.level_to_initial_stride[level] = initial_stride

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

        self.share_params_across_scales = share_params_across_scales

        self.models = nn.ModuleList(models)

        self.level_aggregation_blocks = nn.ModuleList(
            [Bottleneck(2048, 512) for _ in range(num_level_aggregation_blocks)]
        )
        self.fc = nn.Linear(2048, num_classes)

    @torch.jit.export
    def convert_levels_to_evaluate(self, levels_to_evaluate: list[int] | int | None = None) -> tuple[list[int], int]:
        """Standardizes the levels to evaluate.

        Args:
            levels_to_evaluate (int or list[int] or None, optional): Levels to evaluate.
                - If None, returns the default `self.levels_to_evaluate`.
                - If int, returns list(range(levels_to_evaluate)).
                - If list[int], returns it unchanged.

        Returns:
            tuple[list[int], int]: A tuple of standardized levels list and total number of levels.
        """
        if levels_to_evaluate is None:
            return self.levels_to_evaluate, self.num_levels
        elif isinstance(levels_to_evaluate, int):
            return list(range(levels_to_evaluate)), levels_to_evaluate
        else:
            return levels_to_evaluate, max(levels_to_evaluate) + 1

    @torch.jit.export
    def evaluate_levels(
        self, images: torch.Tensor, levels_to_evaluate: list[int] | int | None = None, avg_pool: bool = False
    ) -> list[torch.Tensor]:
        """Evaluates the input images at multiple pyramid levels.

        Applies Gaussian pyramid filtering, passes each level through its
        corresponding BagNet, and optionally applies global pooling per level.

        Args:
            images (torch.Tensor): Input image batch of shape (N, C, H, W).
            levels_to_evaluate (int, list[int], or None, optional): Levels to evaluate. Defaults to None.
            avg_pool (bool, optional): If True, applies global average pooling to each level. Defaults to False.

        Returns:
                List of feature maps for each evaluated level.
                Shape per tensor: (N, C, H_level, W_level) if `avg_pool=False`,
                or (N, num_features, 1, 1) if `avg_pool=True`.
        """
        levels_to_evaluate, num_levels = self.convert_levels_to_evaluate(levels_to_evaluate)
        with torch.no_grad():
            gauss_stacks = get_gauss_stack(
                images, num_levels=num_levels, gauss_filter=self.gauss_filter, num_dimensions=2
            )
        n, c, h, w = gauss_stacks[0].shape
        k = self.patch_size // 2
        results = []
        for key, model in enumerate(self.models):
            level = self.levels_to_evaluate[key]
            if level in levels_to_evaluate:
                initial_stride = self.level_to_initial_stride[level]

                pre_stride_padding = int(k * ((2**level) - 1))

                post_stride_padding_left = post_stride_padding_top = math.ceil(pre_stride_padding / initial_stride)
                post_stride_padding_right = math.ceil(
                    (pre_stride_padding - ((pre_stride_padding + w) % initial_stride)) / initial_stride
                )
                post_stride_padding_bot = math.ceil(
                    (pre_stride_padding - ((pre_stride_padding + h) % initial_stride)) / initial_stride
                )
                shift = pre_stride_padding % initial_stride
                with torch.no_grad():
                    gauss_padded = torch.nn.functional.pad(
                        gauss_stacks[level][:, :, shift::initial_stride, shift::initial_stride],
                        pad=(
                            post_stride_padding_left,
                            post_stride_padding_right,
                            post_stride_padding_top,
                            post_stride_padding_bot,
                        ),
                    )
                results.append(model(gauss_padded, avg_pool=avg_pool).permute(0, 3, 1, 2))
        return results

    def forward(
        self, images: torch.Tensor, levels_to_evaluate: list[int] | None = None, avg_pool: bool = True
    ) -> torch.Tensor:
        """Forward pass of the MultiscaleDilationModel.

        Steps:
            1. Evaluate input images at multiple Gaussian pyramid levels.
            2. Aggregate level-wise feature maps by summation.
            3. Apply Bottleneck blocks for cross-level feature aggregation.
            4. Optionally apply global average pooling.
            5. Compute final class logits via fully connected layer.

        Args:
            images: Input image batch of shape (N, C, H, W).
            levels_to_evaluate: Pyramid levels to evaluate. Defaults to None.
            avg_pool: If True, applies global average pooling before the final classification layer. Defaults to True.

        Returns:
            Output logits of shape (N, num_classes), or high-resolution feature map if `avg_pool=False`.
        """
        n, c, _, _ = images.shape
        xs = self.evaluate_levels(images, levels_to_evaluate, avg_pool=False)
        xs = torch.stack(xs, dim=0) if len(xs) > 0 else torch.empty(1, n, c, 1, 1)
        num_levels, n, num_features, h, w = xs.shape
        x = xs.sum(dim=0)
        for block in self.level_aggregation_blocks:
            x = block(x)
        if avg_pool:
            x = nn.functional.avg_pool2d(x, (h, w), stride=1)
            x = x.view(x.size(0), -1)
        else:
            x = x.permute(0, 2, 3, 1)

        return self.fc(x)
