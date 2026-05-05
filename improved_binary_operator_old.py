"""
Improved binary operators for Causal LDRU.

Usage:
    from improved_binary_operator import ConvexGatedBinaryOperator
    config.operator = ConvexGatedBinaryOperator
"""

from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp


class ConvexGatedBinaryOperator(hk.Module):
    """Convex-gated binary operator: g*x + (1-g)*y + u*(x*y)."""

    def __init__(self, embedding_size: int, activation: Callable = jax.nn.silu):
        super().__init__(name="ConvexGatedBinaryOperator")
        self.embedding_size = embedding_size
        self.activation = activation

    def __call__(self, xy: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            xy: Tensor of shape (..., 2, E).
        Returns:
            Tensor of shape (..., E).
        """
        if xy.shape[-2] != 2:
            raise ValueError(f"Expected pair dimension of size 2, got shape={xy.shape}")

        x = xy[..., 0, :]
        y = xy[..., 1, :]
        hidden_dim = x.shape[-1]

        init = hk.initializers.VarianceScaling(0.02, "fan_avg", "truncated_normal")
        features = jnp.concatenate([x, y, x - y, x * y], axis=-1)  # (..., 4E)

        h = hk.Linear(2 * hidden_dim, w_init=init)(features)
        h = self.activation(h)
        h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)

        gates = hk.Linear(2 * hidden_dim, w_init=init)(h)
        g_logits, u_logits = jnp.split(gates, 2, axis=-1)
        g = jax.nn.sigmoid(g_logits)
        u = jax.nn.sigmoid(u_logits)

        merged = g * x + (1.0 - g) * y + u * (x * y)

        # Preserve signal when one side is explicit padding.
        is_pad_x = jnp.isclose(jnp.sum(jnp.abs(x), axis=-1, keepdims=True), 0.0)
        is_pad_y = jnp.isclose(jnp.sum(jnp.abs(y), axis=-1, keepdims=True), 0.0)
        merged = jnp.where(is_pad_y, x, jnp.where(is_pad_x, y, merged))

        return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(merged)


class GRCOperator(hk.Module):
    """Haiku version of the PyTorch Cell, acting on (..., 2, E) -> (..., E)."""

    def __init__(
        self,
        embedding_size: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__(name="GRCOperator")
        self.embedding_size = embedding_size
        self.activation = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.dropout_rate = dropout_rate

    def __call__(self, xy: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
        xy: Tensor of shape (..., 2, E), where E == embedding_size.
        xy[..., 0, :] plays the role of vi, xy[..., 1, :] of hi.
        Returns:
        Tensor of shape (..., E).
        """
        hidden_size = self.embedding_size
        init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")

        # vi, hi: (..., E)
        vi = xy[..., 0, :]
        hi = xy[..., 1, :]

        # Flatten pair dim: (..., 2E) — equivalent to concat([vi, hi], -1)
        pair = jnp.reshape(xy, xy.shape[:-2] + (2 * hidden_size,))

        # MLP: 2H -> 4H -> 4H (no dropout here; add hk.dropout if needed)
        h = hk.dropout(hk.next_rng_key(), self.dropout_rate, pair)
        h = hk.Linear(4 * hidden_size, w_init=init)(h)
        h = jax.nn.silu(h)
        h = hk.dropout(hk.next_rng_key(), self.dropout_rate, h)
        h = hk.Linear(4 * hidden_size, w_init=init)(h)

        # Reshape to (..., 4, H) and slice instead of split
        h = jnp.reshape(h, h.shape[:-1] + (4, hidden_size))
        gates_raw = h[..., :3, :]  # (..., 3, H) -> vg, hg, cg
        cell = h[..., 3, :]  # (..., H)

        gates = jax.nn.sigmoid(gates_raw)  # (..., 3, H)
        vg = gates[..., 0, :]  # (..., H)
        hg = gates[..., 1, :]  # (..., H)
        cg = gates[..., 2, :]  # (..., H)

        # output = activation(vg * vi + hg * hi + cg * cell)
        out = self.activation(vg * vi + hg * hi + cg * cell)
        return out
