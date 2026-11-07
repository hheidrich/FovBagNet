"""Gaussian filtering and multi-scale signal processing utilities.

This module provides Gaussian smoothing operations for one- and
multi-dimensional tensors, including support for reflective boundary
conditions, Gaussian pyramids, and Gaussian stacks with increasing
receptive fields.

The provided utilities are implemented using PyTorch and can be used as
building blocks for feature extraction, scale-space analysis, and
multi-resolution processing pipelines.
"""

import torch
import torch.nn as nn


def reflect1d(x, padding: int = 0):
    """Apply one-dimensional reflection padding.

    Extends the last dimension of the input tensor using reflective
    boundary conditions without repeating edge values.

    Args:
        x: Input tensor.
        padding: Number of values to append on each side of the last
            dimension.

    Returns:
        The padded tensor.
    """
    if padding == 0:
        return x

    n = x.shape[-1]

    a_left = x[..., :-1]
    a_right = a_left.flip(-1)
    b_right = x[..., 1:]
    b_left = b_right.flip(-1)

    ab_left = [a_left, b_left]
    ab_right = [a_right, b_right]

    q = padding // (n - 1)

    # How many elements remain?
    remainder = padding % (n - 1)

    # How often does ab fit?
    ab_fit = q // 2

    left = ab_fit * ab_left
    right = ab_fit * ab_right

    # Do we have another single fit of b_left and a_right?
    if q % 2:
        # ab_fit is odd -> yes, we have another single fit of b_left and a_right
        left.insert(0, b_left)
        right.append(a_right)

        # We append the remainders taken from a_left and b_right
        if remainder > 0:
            left.insert(0, a_left[..., -remainder:])
            right.append(b_right[..., :remainder])

    else:
        # ab_fit is even -> no, we append the remainders taken from b_left and a_right directly
        if remainder > 0:
            left.insert(0, b_left[..., -remainder:])
            right.append(a_right[..., :remainder])

    # put everything together
    return torch.cat(left + [x] + right, dim=-1)


def reflect2d(x, padding: int = 0):
    """Apply two-dimensional reflection padding.

    Extends the last two dimension of the input tensor using reflective
    boundary conditions without repeating edge values.

    Args:
        x: Input tensor.
        padding: Number of values to append on each side of the last
            dimension.

    Returns:
        The padded tensor.
    """
    return reflect1d(reflect1d(x.transpose(-1, -2), padding=padding).transpose(-1, -2), padding=padding)


class GaussFilter(nn.Module):
    """Gaussian smoothing filter for one- and multi-dimensional tensors.

    The filter is implemented as a separable one-dimensional Gaussian
    convolution that is applied independently along one or more tensor
    dimensions.

    Args:
        sigma: Standard deviation of the Gaussian kernel.
        k: Half-width of the kernel. If not specified, a value of
            ``round(2 * sigma)`` is used.
    """

    def __init__(self, sigma, k=None):
        """Initialize a Gaussian smoothing kernel.

        Constructs a 1D Gaussian kernel with standard deviation ``sigma`` and
        normalizes it to sum to 1. The kernel is stored as a non-trainable
        parameter and used for separable convolution.

        Args:
            sigma: Standard deviation of the Gaussian kernel.
            k: Half-width of the kernel. If not provided, it is set to
                ``round(2 * sigma)``.
        """
        super().__init__()
        if k is None:
            k = round(2.0 * sigma)

        kernel = torch.exp(-0.5 * torch.arange(-k, k + 1) ** 2 / sigma**2).reshape(1, 1, 2 * k + 1)
        self.kernel = nn.Parameter(kernel / kernel.sum(), requires_grad=False)
        self.k = k

    def forward(
        self, x, num_dimensions: int = 1, stride: int = 1, padding: int | None = None, dilation: int = 1
    ) -> torch.Tensor:
        """Apply Gaussian smoothing to one or more dimensions.

        The filter is applied separably across the last
        ``num_dimensions`` dimensions of the input tensor.

        Args:
            x: Input tensor.
            num_dimensions: Number of trailing dimensions over which the
                Gaussian filter is applied.
            stride: Convolution stride.
            padding: Reflection padding applied before convolution. If
                ``None``, padding is chosen to preserve the input size.
            dilation: Convolution dilation factor.

        Returns:
            The filtered tensor.
        """
        if padding is None:
            padding = self.k * dilation

        x = x.unsqueeze(-2)
        for dimension in range(num_dimensions):
            if dimension > 0:
                x = x.swapaxes(-dimension - 2, -1)
            x_shape = x.shape
            x = torch.conv1d(
                reflect1d(x.reshape((-1,) + x_shape[-2:]), padding),
                self.kernel,
                stride=stride,
                padding=0,
                dilation=dilation,
            ).view(x_shape[:-1] + (-1,))
        x = x.squeeze(-2)

        x_dim = len(x.shape)
        permutation = torch.cat(
            [
                torch.arange(x.dim() - num_dimensions),
                torch.tensor([x.dim() - 1]),
                torch.arange(x.dim() - num_dimensions, x.dim() - 1),
            ]
        )
        permutation = list(range(x_dim - num_dimensions)) + [x_dim - 1] + list(range(x_dim - num_dimensions, x_dim - 1))
        return x.permute(permutation)


def get_gauss_pyramid(
    xs: torch.Tensor,
    num_levels: int,
    gauss_filter: GaussFilter,
    num_dimensions: int,
) -> list[torch.Tensor]:
    """Construct a Gaussian pyramid.

    Successively smooths and downsamples the input tensor by a factor of
    two to generate a multi-resolution representation.

    Args:
        xs: Input tensor.
        num_levels: Number of pyramid levels.
        gauss_filter: Gaussian filter used for smoothing.
        num_dimensions: Number of trailing dimensions to process.

    Returns:
        A list of tensors ordered from highest to lowest resolution.
    """
    gauss_pyramids = [xs]
    gauss_in = xs
    for _ in range(num_levels - 1):
        gauss_out = gauss_filter.forward(
            gauss_in,
            num_dimensions=num_dimensions,
            stride=2,
        )
        gauss_pyramids.append(gauss_out)
        gauss_in = gauss_out

    return gauss_pyramids


def get_gauss_stack(
    xs: torch.Tensor,
    num_levels: int,
    gauss_filter: GaussFilter,
    num_dimensions: int,
    apply_padding: bool = True,
) -> list[torch.Tensor]:
    """Construct a Gaussian stack with increasing receptive fields.

    Successively applies Gaussian smoothing with exponentially increasing
    dilation while preserving the spatial resolution of the input.

    Args:
        xs: Input tensor.
        num_levels: Number of stack levels.
        gauss_filter: Gaussian filter used for smoothing.
        num_dimensions: Number of trailing dimensions to process.
        apply_padding: Use reflection padding in each stage to maintain input size if True.
            No padding is applied when set to False.
            Defaults to True.

    Returns:
        A list of tensors ordered from smallest to largest effective
        smoothing scale.
    """
    gauss_stacks = [xs]
    gauss_in = xs
    dilation = 1
    for _ in range(num_levels - 1):
        gauss_out = gauss_filter.forward(
            gauss_in,
            num_dimensions=num_dimensions,
            dilation=dilation,
            padding=None if apply_padding else 0,
        )
        gauss_stacks.append(gauss_out)
        gauss_in = gauss_out
        dilation *= 2

    return gauss_stacks
