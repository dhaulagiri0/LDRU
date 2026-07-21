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
        mlp_hidden_size: int = None,
        expansion_factor: int = 4,
        dropout_rate: float = 0.0,
    ):
        super().__init__(name="GRCOperator")
        self.embedding_size = embedding_size
        self.mlp_hidden_size = (
            embedding_size if mlp_hidden_size is None else int(mlp_hidden_size)
        )
        self.expansion_factor = int(expansion_factor)
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
        operator_hidden_size = self.expansion_factor * self.mlp_hidden_size
        init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")

        # vi, hi: (..., E)
        vi = xy[..., 0, :]
        hi = xy[..., 1, :]

        # Flatten pair dim: (..., 2E) — equivalent to concat([vi, hi], -1)
        pair = jnp.reshape(xy, xy.shape[:-2] + (2 * hidden_size,))

        # MLP: 2E -> (expansion_factor * mlp_hidden_size) -> 4E
        h = hk.dropout(hk.next_rng_key(), self.dropout_rate, pair)
        h = hk.Linear(operator_hidden_size, w_init=init)(h)
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


class AblationBinaryOperator(hk.Module):
    """Binary operator with independent switches for expansion and combine stages."""

    def __init__(
        self,
        embedding_size: int,
        mlp_hidden_size: int = None,
        expansion_factor: int = 4,
        dropout_rate: float = 0.0,
        ablation_expansion_mode: str = "grc",
        ablation_combine_mode: str = "grc",
    ):
        super().__init__(name="AblationBinaryOperator")
        self.embedding_size = int(embedding_size)
        self.mlp_hidden_size = (
            self.embedding_size if mlp_hidden_size is None else int(mlp_hidden_size)
        )
        self.expansion_factor = int(expansion_factor)
        self.dropout_rate = float(dropout_rate)
        self.ablation_expansion_mode = str(ablation_expansion_mode).lower()
        self.ablation_combine_mode = str(ablation_combine_mode).lower()
        if self.ablation_expansion_mode not in {"binary", "grc"}:
            raise ValueError(
                f"ablation_expansion_mode must be one of ['binary', 'grc'], got {ablation_expansion_mode}"
            )
        if self.ablation_combine_mode not in {"binary", "grc"}:
            raise ValueError(
                f"ablation_combine_mode must be one of ['binary', 'grc'], got {ablation_combine_mode}"
            )

    def __call__(self, xy: jnp.ndarray) -> jnp.ndarray:
        if xy.shape[-2] != 2:
            raise ValueError(f"Expected pair dimension of size 2, got shape={xy.shape}")

        hidden_dim = self.embedding_size
        x = xy[..., 0, :]
        y = xy[..., 1, :]
        pair = jnp.reshape(xy, xy.shape[:-2] + (2 * hidden_dim,))

        binary_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")
        grc_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")

        binary_2e = None
        grc_4e = None
        operator_hidden_size = self.expansion_factor * self.mlp_hidden_size
        if self.ablation_expansion_mode == "binary":
            h = hk.Linear(2 * hidden_dim, w_init=binary_init, name="binary_mix_0")(pair)
            h = jax.nn.silu(h)
            h = hk.Linear(4 * hidden_dim, w_init=binary_init, name="binary_mix_1")(h)
            h = jax.nn.silu(h)
            binary_2e = hk.Linear(2 * hidden_dim, w_init=binary_init, name="binary_mix_2")(h)
            binary_2e = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True, name="binary_mix_norm"
            )(binary_2e)
        else:
            h = hk.dropout(hk.next_rng_key(), self.dropout_rate, pair)
            h = hk.Linear(operator_hidden_size, w_init=grc_init, name="grc_mix_0")(h)
            h = jax.nn.silu(h)
            h = hk.dropout(hk.next_rng_key(), self.dropout_rate, h)
            grc_4e = hk.Linear(4 * hidden_dim, w_init=grc_init, name="grc_mix_1")(h)

        if self.ablation_combine_mode == "binary":
            if binary_2e is None:
                binary_2e = hk.Linear(
                    2 * hidden_dim, w_init=binary_init, name="grc_to_binary_adapter"
                )(grc_4e)
                binary_2e = hk.LayerNorm(
                    axis=-1,
                    create_scale=True,
                    create_offset=True,
                    name="grc_to_binary_adapter_norm",
                )(binary_2e)

            weights = jnp.reshape(binary_2e, xy.shape)
            x_weight = weights[..., 0, :]
            y_weight = weights[..., 1, :]
            x_weight = jnp.where(
                jnp.all(x == 0, axis=-1, keepdims=True), jnp.zeros_like(x_weight), x_weight
            )
            y_weight = jnp.where(
                jnp.all(y == 0, axis=-1, keepdims=True), jnp.zeros_like(y_weight), y_weight
            )
            weighted_x = hk.Linear(
                hidden_dim, w_init=hk.initializers.Identity(), name="binary_out_x"
            )(x_weight * x)
            weighted_y = hk.Linear(
                hidden_dim, w_init=hk.initializers.Identity(), name="binary_out_y"
            )(y_weight * y)
            return hk.Linear(
                hidden_dim, w_init=hk.initializers.Identity(), name="binary_out"
            )(weighted_x + weighted_y)

        if grc_4e is None:
            grc_4e = hk.Linear(4 * hidden_dim, w_init=grc_init, name="binary_to_grc_adapter")(
                binary_2e
            )
        h = jnp.reshape(grc_4e, grc_4e.shape[:-1] + (4, hidden_dim))
        gates_raw = h[..., :3, :]
        cell = h[..., 3, :]
        gates = jax.nn.sigmoid(gates_raw)
        vg = gates[..., 0, :]
        hg = gates[..., 1, :]
        cg = gates[..., 2, :]
        return hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="grc_out_norm"
        )(vg * x + hg * y + cg * cell)
