from typing import Callable
from jax._src.tree_util import (
    tree_flatten,
    tree_unflatten,
)
from jax._src import util
from jax._src import core
from jax._src.lax import slicing
from jax._src.lax import lax
from jax._src.util import safe_map
from functools import partial
import numpy as np

_map = safe_map


def associative_scan(
    fn: Callable,
    elems,
    reverse: bool = False,
    axis: int = 0,
    inner_fn: Callable = lambda x: x,
):
    """
    A copy of jax.lax.associative_scan that applies an extra user definable function after each combine step.
    """
    if not callable(fn):
        raise TypeError("lax.associative_scan: fn argument should be callable.")
    elems_flat, tree = tree_flatten(elems)

    if reverse:
        elems_flat = [lax.rev(elem, [axis]) for elem in elems_flat]

    def combine(a_flat, b_flat):
        a = tree_unflatten(tree, a_flat)
        b = tree_unflatten(tree, b_flat)
        c = fn(a, b)
        c_flat, _ = tree_flatten(c)
        # Enforce dtype consistency with scan inputs (critical for mixed precision).
        aligned = []
        for c_leaf, a_leaf in zip(c_flat, a_flat):
            if c_leaf.dtype != a_leaf.dtype:
                c_leaf = c_leaf.astype(a_leaf.dtype)
            aligned.append(c_leaf)
        return aligned

    axis = util.canonicalize_axis(axis, elems_flat[0].ndim)

    if not core.is_constant_dim(elems_flat[0].shape[axis]):
        raise NotImplementedError(
            "associative scan over axis "
            f"of non-constant size: {elems_flat[0].shape[axis]}. You may be "
            "able to avoid this on TPU. See b/274176030."
        )
    num_elems = int(elems_flat[0].shape[axis])
    if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
        raise ValueError(
            "Array inputs to associative_scan must have the same "
            "first dimension. (saw: {})".format([elem.shape for elem in elems_flat])
        )

    def _scan(elems):
        """Perform scan on `elems`."""

        num_elems = elems[0].shape[axis]

        if num_elems < 2:
            return elems

        # Combine adjacent pairs of elements.
        reduced_elems = combine(
            [slicing.slice_in_dim(elem, 0, -1, stride=2, axis=axis) for elem in elems],
            [
                slicing.slice_in_dim(elem, 1, None, stride=2, axis=axis)
                for elem in elems
            ],
        )
        # Apply inner function after each combine step.
        # Keep original dtypes so downstream lax.concatenate/interleave receives
        # uniform dtypes (important for mixed-precision runs, e.g. bf16 compute).
        cast_reduced_elems = []
        for elem in reduced_elems:
            elem_out = inner_fn(elem)
            if elem_out.dtype != elem.dtype:
                elem_out = elem_out.astype(elem.dtype)
            cast_reduced_elems.append(elem_out)
        reduced_elems = cast_reduced_elems

        # Recursively compute scan for partially reduced tensors.
        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            even_elems = combine(
                [slicing.slice_in_dim(e, 0, -1, axis=axis) for e in odd_elems],
                [slicing.slice_in_dim(e, 2, None, stride=2, axis=axis) for e in elems],
            )
        else:
            even_elems = combine(
                odd_elems,
                [slicing.slice_in_dim(e, 2, None, stride=2, axis=axis) for e in elems],
            )

        # The first element of a scan is the same as the first element
        # of the original `elems`.
        even_elems = [
            lax.concatenate(
                [
                    slicing.slice_in_dim(elem, 0, 1, axis=axis),
                    result.astype(elem.dtype) if result.dtype != elem.dtype else result,
                ],
                dimension=axis,
            )
            for (elem, result) in zip(elems, even_elems)
        ]
        odd_elems = [
            odd.astype(even.dtype) if odd.dtype != even.dtype else odd
            for even, odd in zip(even_elems, odd_elems)
        ]
        return list(_map(partial(_interleave, axis=axis), even_elems, odd_elems))

    scans = _scan(elems_flat)

    if reverse:
        scans = [lax.rev(scanned, [axis]) for scanned in scans]

    return tree_unflatten(tree, scans)


def _interleave(a, b, axis):
    """Given two Tensors of static shape, interleave them along the first axis."""
    assert a.shape[axis] == b.shape[axis] or a.shape[axis] == b.shape[axis] + 1
    a_pad = [(0, 0, 0)] * a.ndim
    b_pad = [(0, 0, 0)] * b.ndim
    a_pad[axis] = (0, 1 if a.shape[axis] == b.shape[axis] else 0, 1)
    b_pad[axis] = (1, 0 if a.shape[axis] == b.shape[axis] else 1, 1)
    op = lax.bitwise_or if a.dtype == np.bool_ else lax.add
    return op(lax.pad(a, lax._const(a, 0), a_pad), lax.pad(b, lax._const(b, 0), b_pad))
