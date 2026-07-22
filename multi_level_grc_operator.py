"""Depth-aware GRC operator for causal LDRU.

This operator keeps the existing single-operator path untouched and only
uses the supplied reduction level when explicitly selected.
"""

from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import jax.nn as jnn


class MultiLevelGRCOperator(hk.Module):
    """GRC-style binary operator with optional depth conditioning."""

    def __init__(
        self,
        embedding_size: int,
        mlp_hidden_size: Optional[int] = None,
        expansion_factor: int = 4,
        dropout_rate: float = 0.0,
        max_reduction_levels: int = 16,
    ):
        super().__init__(name="MultiLevelGRCOperator")
        self.embedding_size = int(embedding_size)
        self.mlp_hidden_size = (
            self.embedding_size if mlp_hidden_size is None else int(mlp_hidden_size)
        )
        self.expansion_factor = int(expansion_factor)
        self.dropout_rate = float(dropout_rate)
        self.max_reduction_levels = int(max_reduction_levels)

        if self.embedding_size <= 0:
            raise ValueError("embedding_size must be > 0.")
        if self.mlp_hidden_size <= 0:
            raise ValueError("mlp_hidden_size must be > 0.")
        if self.expansion_factor <= 0:
            raise ValueError("expansion_factor must be > 0.")
        if self.max_reduction_levels <= 0:
            raise ValueError("max_reduction_levels must be > 0.")

    def __call__(
        self, xy: jnp.ndarray, reduction_level: Optional[jnp.ndarray | int] = None
    ) -> jnp.ndarray:
        if xy.shape[-2] != 2:
            raise ValueError(f"Expected pair dimension of size 2, got shape={xy.shape}")

        hidden_size = self.embedding_size
        operator_hidden_size = self.expansion_factor * self.mlp_hidden_size
        init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")

        vi = xy[..., 0, :]
        hi = xy[..., 1, :]
        pair = jnp.reshape(xy, xy.shape[:-2] + (2 * hidden_size,))

        h = hk.dropout(hk.next_rng_key(), self.dropout_rate, pair)
        h = hk.Linear(operator_hidden_size, w_init=init, name="depth_mix_0")(h)

        level = 0 if reduction_level is None else reduction_level
        level = jnp.asarray(level, dtype=jnp.int32)
        level = jnp.clip(level, 0, self.max_reduction_levels - 1)
        level_bias = hk.Embed(
            vocab_size=self.max_reduction_levels,
            embed_dim=operator_hidden_size,
            name="reduction_level_embedding",
        )(level)
        h = h + level_bias
        h = jnn.silu(h)

        h = hk.dropout(hk.next_rng_key(), self.dropout_rate, h)
        h = hk.Linear(4 * hidden_size, w_init=init, name="depth_mix_1")(h)
        h = jnp.reshape(h, h.shape[:-1] + (4, hidden_size))

        gates_raw = h[..., :3, :]
        cell = h[..., 3, :]
        gates = jax.nn.sigmoid(gates_raw)
        vg = gates[..., 0, :]
        hg = gates[..., 1, :]
        cg = gates[..., 2, :]

        out = jnn.silu(vg * vi + hg * hi + cg * cell)
        return out
