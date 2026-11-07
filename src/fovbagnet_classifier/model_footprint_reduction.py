"""Utilities for manipulating and analyzing convolutional receptive field footprints.

This module provides tools to:
- Represent receptive field “footprints” of CNN layers
- Expand and reduce footprint representations across strides
- Compute sequential receptive field growth through a network
- Adjust strides/dilations while preserving spatial topology

The core idea:
A convolutional network induces a mapping between output pixels and input
pixel positions. This mapping can be represented as a set of offsets modulo
strides ("footprints"). This module provides algebraic tools to manipulate
these mappings.

These utilities are used at model initialization time to ensure consistent
spatial alignment and to optionally reduce effective stride while preserving
receptive field structure.
"""

import math
from typing import Dict, List, Set, Tuple

import numpy as np

Offset = int
Stride = int
Footprint = Dict[Stride, Set[Offset]]


def expand_footprint_to_factor(footprint: Footprint, factor: int, check_factor: bool = True) -> Footprint:
    """Expand a modular footprint representation to a common stride factor.

    Each footprint is represented as:
        {stride: set(offsets mod stride)}

    This function lifts all strides into a common modulus ("factor"),
    producing a unified representation.

    Args:
        footprint:
            Dictionary mapping stride → set of valid offsets.

        factor:
            Target stride resolution to expand to.

        check_factor:
            If True, asserts that all strides divide `factor`.

    Returns:
        A single-stride footprint representation:
            {factor: set(expanded offsets)}
    """
    if check_factor:
        for stride in footprint.keys():
            assert factor % stride == 0
    offsets_factor = set()
    for stride, offsets in footprint.items():
        for offset in offsets:
            offsets_factor = offsets_factor.union({i + offset for i in range(0, factor, stride)})
    return {factor: offsets_factor}


def expand_footprint(footprint: Footprint) -> Footprint:
    """Expand a footprint to a common resolution using the least common multiple (LCM).

    This ensures all stride periodicities are represented on a shared grid.

    Args:
        footprint: Original footprint mapping.

    Returns:
        Expanded footprint with a single stride equal to the LCM.
    """
    lcm = math.lcm(*footprint.keys())
    return expand_footprint_to_factor(footprint, lcm, check_factor=False)


def get_first_prime_factor(n: int) -> int:
    """Return the smallest prime factor of an integer.

    Args:
        n: The input integer.

    Returns:
        The smallest prime factor of `n`.
    """
    primes = []
    for i in range(2, 1 + math.floor(math.sqrt(n))):
        i_is_prime = True
        for prime in primes:
            if i % prime == 0:
                i_is_prime = False
                break
        if i_is_prime:
            if n % i == 0:
                return i
            primes.append(i)
    return n


def _reduce_footprint_single(footprint: Footprint) -> Tuple[Footprint, Footprint]:
    """Attempt to simplify a footprint by merging stride-offset structures.

    This function finds lower-stride representations that fully
    explain higher-stride periodic patterns.

    Args:
        footprint (dict[int, set[int]]): Original footprint mapping.

    Returns:
        tuple[dict[int, set[int]], dict[int, set[int]]]:
            - Updated footprint after a single reduction step.
            - Newly added reductions to be merged.
    """
    footprint = {key: val.copy() for key, val in footprint.items()}
    footprint_to_add = {}
    for stride, offsets in footprint.items():
        if stride > 1:
            for new_stride in range(math.ceil(stride / len(offsets)), 1 + stride // get_first_prime_factor(stride)):
                if stride % new_stride == 0:
                    for j in range(new_stride):
                        all_in = True
                        for k in range(0, stride, new_stride):
                            if j + k not in offsets:
                                all_in = False
                        if all_in:
                            # add combined (offset,stride) to ordered_updated
                            if new_stride not in footprint_to_add.keys():
                                footprint_to_add[new_stride] = set()
                            footprint_to_add[new_stride].add(j)

                            # add combined (offset,stride) to ordered_updated
                            for k in range(0, stride, new_stride):
                                footprint[stride].remove(j + k)
                            return footprint, footprint_to_add
    return footprint, footprint_to_add


def reduce_footprint(footprint: Footprint) -> Footprint:
    """Iteratively reduce a footprint to a minimal equivalent representation.

    This repeatedly applies `_reduce_footprint_single` until no further
    simplification is possible.

    Returns:
        Reduced footprint dictionary.
    """
    footprint_to_add = None
    footprint = expand_footprint(footprint)
    while footprint_to_add != {}:
        footprint, footprint_to_add = _reduce_footprint_single(footprint)
        for stride, offsets in footprint_to_add.items():
            if stride not in footprint.keys():
                footprint[stride] = offsets
            else:
                footprint[stride] = footprint[stride].union(offsets)
        footprint = {key: val for key, val in footprint.items() if val != set()}

    return footprint


def compute_sequential_footprint(kernel_sizes, strides, dilations):
    """Compute receptive field footprint propagation through a CNN.

    This function tracks how a single output position maps back to input
    positions across sequential convolutional layers.

    Returns:
        offset_left:
            Left-side receptive field offset.

        receptive_field_size:
            Full receptive field size.

        accumulated_strides:
            Effective stride after each layer.

        requirements:
            Footprint representation at each layer.
    """
    offset_left = 0
    accumulated_stride = 1
    accumulated_strides = [1]
    for kernel_size, stride, dilation in zip(kernel_sizes, strides, dilations):
        dilated_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
        strided_kernel_size = dilated_kernel_size + (dilated_kernel_size - 1) * (accumulated_stride - 1)
        offset_left += strided_kernel_size // 2
        accumulated_stride *= stride
        accumulated_strides.append(accumulated_stride)
    requirements = [
        {accumulated_stride: {offset_left % accumulated_stride}}
    ]  # Starting at the last layer with a single requirement.
    for kernel_size, stride, dilation, accumulated_stride in zip(
        kernel_sizes[::-1], strides[::-1], dilations[::-1], accumulated_strides[-2::-1]
    ):
        requirements_next = {}
        for stride_req, offsets_req in requirements[-1].items():
            if stride_req not in requirements_next.keys():
                requirements_next[stride_req] = set()
            for offset_req in offsets_req:
                for i in range(kernel_size):
                    pos = accumulated_stride * dilation * (i - kernel_size // 2)
                    requirements_next[stride_req].add((offset_req + pos) % stride_req)
        requirements.append(reduce_footprint(requirements_next))

    return offset_left, 1 + 2 * offset_left, accumulated_strides, requirements


def is_subfootprint(footprint_a: Footprint, footprint_b: Footprint) -> bool:
    """Check whether footprint A is contained in footprint B.

    Both footprints are expanded to a common stride before comparison.
    """
    joint_lcm = math.lcm(*footprint_a.keys(), *footprint_b.keys())
    footprint_a_expanded = expand_footprint_to_factor(footprint_a, joint_lcm, check_factor=False)[joint_lcm]
    footprint_b_expanded = expand_footprint_to_factor(footprint_b, joint_lcm, check_factor=False)[joint_lcm]

    return footprint_a_expanded <= footprint_b_expanded


def get_hull(footprint: Footprint) -> Footprint:
    """Return the minimal stride-offset pair that covers the footprint."""
    min_key = min(footprint.keys())
    for stride in range(min_key, 0, -1):
        if min_key % stride == 0:
            for offset in range(stride):
                potential_hull = {stride: {offset}}
                if is_subfootprint(footprint, potential_hull):
                    return potential_hull


def fix_strides(
    kernel_sizes: List[int], strides: List[int], dilations: List[int]
) -> Tuple[int, List[int], List[int], int, int]:
    """Adjust strides and dilations to minimize model footprint while preserving topology.

    Returns:
        initial_stride: Adjusted initial stride.
        strides_fixed: Layer-wise fixed strides.
        dilations_fixed: Layer-wise fixed dilations.
        offset_left: Total input offset.
        receptive_field: Total receptive field size.
    """
    offset_left, receptive_field, accumulated_strides, footprints = compute_sequential_footprint(
        kernel_sizes, strides, dilations
    )
    hulls = [min(get_hull(r).keys()) for r in footprints][::-1]
    overhead_factors = [hull // stride if hull % stride == 0 else 1 for hull, stride in zip(hulls, accumulated_strides)]
    stride_diffs = 2 ** np.diff(np.log2(np.array([1] + overhead_factors)))
    initial_stride = int(stride_diffs[0])
    strides_fixed = [int(stride * stride_diff) for stride, stride_diff in zip(strides, stride_diffs[1:])]
    dilations_fixed = [
        dilation // overhead_factor if kernel_size > 1 else 1
        for dilation, overhead_factor, kernel_size in zip(dilations, overhead_factors, kernel_sizes)
    ]
    return initial_stride, strides_fixed, dilations_fixed, offset_left, receptive_field


def log2(n: int, assert_power_of_two: bool = True) -> int:
    """Compute log2 of an integer using its binary representation.

    Args:
        n: Integer to compute log2 of.
        assert_power_of_two: If True, assert that n is a power of two.

    Returns:
        The base-2 logarithm of n.
    """
    binary_string = f"{n:b}"
    if assert_power_of_two:
        assert n > 0 and binary_string.count("1") == 1
    return len(binary_string) - 1


def reduce_model_stride(strides: List[int], dilations: List[int], final_stride: int = 1) -> Tuple[List[int], List[int]]:
    """Reduce the model's total stride while preserving topology.

    Args:
        strides: Original layer-wise strides.
        dilations: Original layer-wise dilations.
        final_stride: Desired final stride (must be <= original total stride).

    Returns:
        strides_new: Adjusted strides per layer.
    """
    final_stride_log2 = log2(final_stride)

    current_stride = math.prod(strides)
    current_stride_log2 = log2(current_stride)
    assert final_stride <= current_stride

    total_reduction = current_stride_log2 - final_stride_log2
    total_reductions = []
    strides_new = []
    for stride in strides[::-1]:
        stride_log2 = log2(stride)
        reduction = min(stride_log2, total_reduction)
        stride_new_log2 = stride_log2 - reduction
        total_reduction -= reduction
        strides_new.append(2**stride_new_log2)
        total_reductions.append(total_reduction)

    dilations_new = [dilations[0]] + [
        dilation * 2**reduction for dilation, reduction in zip(dilations[1:], total_reductions[-2::-1])
    ]
    return strides_new[::-1], dilations_new
