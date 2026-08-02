from typing import Callable

import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.tree_util import tree_flatten, tree_unflatten


def associative_scan(
    fn: Callable,
    elems,
    reverse: bool = False,
    axis: int = 0,
    inner_fn: Callable = lambda x: x,
    pass_level: bool = False,
):
    """
    Inclusive associative scan with an optional post-merge transform.

    `inner_fn` is applied after every call to `fn`, including:
      1. adjacent-pair reductions in the recursive upsweep, and
      2. merges used to reconstruct remaining prefixes.
    """
    if not callable(fn):
        raise TypeError("associative_scan: fn must be callable.")
    if not callable(inner_fn):
        raise TypeError("associative_scan: inner_fn must be callable.")

    elems_flat, tree = tree_flatten(elems)
    if not elems_flat:
        raise ValueError("associative_scan: elems must contain at least one array.")

    ndim = elems_flat[0].ndim
    axis = axis + ndim if axis < 0 else axis
    if axis < 0 or axis >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for ndim={ndim}.")
    if not all(elem.ndim == ndim for elem in elems_flat[1:]):
        raise ValueError("associative_scan: all leaves must have matching rank.")

    num_elems = int(elems_flat[0].shape[axis])
    if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
        raise ValueError(
            "Array inputs to associative_scan must have the same scan-axis "
            f"length. Saw shapes: {[elem.shape for elem in elems_flat]}"
        )

    if reverse:
        elems_flat = [lax.rev(elem, [axis]) for elem in elems_flat]

    def combine(a_flat, b_flat, reduction_level: int):
        a = tree_unflatten(tree, a_flat)
        b = tree_unflatten(tree, b_flat)
        c = fn(a, b, reduction_level) if pass_level else fn(a, b)
        c_flat, c_tree = tree_flatten(c)

        if c_tree != tree:
            raise TypeError("associative_scan: fn output pytree must match input pytree.")
        if len(c_flat) != len(a_flat):
            raise TypeError(
                "associative_scan: fn output leaf count must match input leaf count."
            )

        # Keep scan dtypes stable (important in mixed-precision execution).
        aligned = []
        for c_leaf, a_leaf in zip(c_flat, a_flat):
            if c_leaf.dtype != a_leaf.dtype:
                c_leaf = c_leaf.astype(a_leaf.dtype)
            aligned.append(c_leaf)
        return aligned

    def apply_inner(flat_elems):
        outputs = []
        for elem in flat_elems:
            out = inner_fn(elem)
            if out.shape != elem.shape:
                raise ValueError(
                    "associative_scan: inner_fn must preserve shape. "
                    f"Got {out.shape}, expected {elem.shape}."
                )
            if out.dtype != elem.dtype:
                out = out.astype(elem.dtype)
            outputs.append(out)
        return outputs

    def _scan(flat_elems, reduction_level: int = 0):
        current_len = int(flat_elems[0].shape[axis])
        if current_len < 2:
            return flat_elems

        # Pair adjacent elements: (x0, x1), (x2, x3), ...
        reduced_elems = combine(
            [lax.slice_in_dim(elem, 0, -1, stride=2, axis=axis) for elem in flat_elems],
            [lax.slice_in_dim(elem, 1, None, stride=2, axis=axis) for elem in flat_elems],
            reduction_level,
        )
        reduced_elems = apply_inner(reduced_elems)

        # Recursively scan pair reductions.
        odd_elems = _scan(reduced_elems, reduction_level + 1)

        # Reconstruct remaining prefixes.
        if current_len % 2 == 0:
            even_elems = combine(
                [lax.slice_in_dim(elem, 0, -1, axis=axis) for elem in odd_elems],
                [lax.slice_in_dim(elem, 2, None, stride=2, axis=axis) for elem in flat_elems],
                reduction_level,
            )
        else:
            even_elems = combine(
                odd_elems,
                [lax.slice_in_dim(elem, 2, None, stride=2, axis=axis) for elem in flat_elems],
                reduction_level,
            )

        # These are merge outputs too.
        even_elems = apply_inner(even_elems)

        # Prefix 0 is the original first element and has not undergone a merge.
        even_elems = [
            lax.concatenate(
                [
                    lax.slice_in_dim(elem, 0, 1, axis=axis),
                    result.astype(elem.dtype) if result.dtype != elem.dtype else result,
                ],
                dimension=axis,
            )
            for elem, result in zip(flat_elems, even_elems)
        ]

        odd_elems = [
            odd.astype(even.dtype) if odd.dtype != even.dtype else odd
            for even, odd in zip(even_elems, odd_elems)
        ]

        return [_interleave(even, odd, axis=axis) for even, odd in zip(even_elems, odd_elems)]

    scans = _scan(elems_flat, reduction_level=0)
    if reverse:
        scans = [lax.rev(scanned, [axis]) for scanned in scans]
    return tree_unflatten(tree, scans)


def _interleave(a, b, axis: int):
    """Interleave arrays `a` and `b` along `axis`."""
    if not (
        a.shape[axis] == b.shape[axis]
        or a.shape[axis] == b.shape[axis] + 1
    ):
        raise ValueError(
            "Interleave inputs must have equal lengths, or `a` must be one "
            f"element longer. Got {a.shape[axis]} and {b.shape[axis]}."
        )

    a_padding = [(0, 0, 0)] * a.ndim
    b_padding = [(0, 0, 0)] * b.ndim

    a_padding[axis] = (
        0,
        1 if a.shape[axis] == b.shape[axis] else 0,
        1,
    )
    b_padding[axis] = (
        1,
        0 if a.shape[axis] == b.shape[axis] else 1,
        1,
    )

    padded_a = lax.pad(a, jnp.asarray(0, dtype=a.dtype), a_padding)
    padded_b = lax.pad(b, jnp.asarray(0, dtype=b.dtype), b_padding)

    if a.dtype == np.bool_:
        return lax.bitwise_or(padded_a, padded_b)
    return lax.add(padded_a, padded_b)