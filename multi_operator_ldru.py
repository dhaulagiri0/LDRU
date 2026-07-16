"""Multi-operator LDRU binary composition with head-style partitioning."""

from __future__ import annotations

from typing import Callable, Optional

import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp


class MultiOperatorBinaryOperator(hk.Module):
    """
    Head-style composition wrapper around multiple independent binary operators.

    Each operator processes a disjoint chunk of the hidden/state dimension.
    A learned gate mixes operator outputs with a guaranteed non-zero floor weight.
    """

    def __init__(
        self,
        state_dim: int,
        num_operators: int,
        min_weight: float,
        operator_cls: Callable,
        mlp_hidden_size: Optional[int] = None,
        expansion_factor: int = 4,
        dropout_rate: float = 0.0,
        ablation_expansion_mode: str = "grc",
        ablation_combine_mode: str = "grc",
        max_reduction_levels: Optional[int] = None,
    ):
        super().__init__(name="MultiOperatorBinaryOperator")
        self.state_dim = int(state_dim)
        self.num_operators = int(num_operators)
        self.min_weight = float(min_weight)
        self.operator_cls = operator_cls
        self.mlp_hidden_size = (
            self.state_dim if mlp_hidden_size is None else int(mlp_hidden_size)
        )
        self.expansion_factor = int(expansion_factor)
        self.dropout_rate = float(dropout_rate)
        self.ablation_expansion_mode = str(ablation_expansion_mode)
        self.ablation_combine_mode = str(ablation_combine_mode)
        self.max_reduction_levels = (
            int(max_reduction_levels) if max_reduction_levels is not None else 16
        )

        if self.num_operators <= 0:
            raise ValueError("num_operators must be > 0.")
        if self.state_dim <= 0:
            raise ValueError("state_dim must be > 0.")
        if self.state_dim % self.num_operators != 0:
            raise ValueError(
                f"state_dim ({self.state_dim}) must be divisible by "
                f"num_operators ({self.num_operators})."
            )
        if self.min_weight < 0.0:
            raise ValueError("min_weight must be >= 0.")
        if self.min_weight * self.num_operators >= 1.0:
            raise ValueError(
                "min_weight must satisfy min_weight < 1 / num_operators "
                "so softmax mass remains positive."
            )
        if self.max_reduction_levels <= 0:
            raise ValueError("max_reduction_levels must be > 0.")

        self.head_dim = self.state_dim // self.num_operators
        per_head_hidden = max(1, self.mlp_hidden_size // self.num_operators)

        op_name = getattr(self.operator_cls, "__name__", "")
        self._is_ablation = op_name == "AblationBinaryOperator"

        self.head_operators = [
            self._build_head_operator(self.head_dim, per_head_hidden)
            for _ in range(self.num_operators)
        ]

    def _build_head_operator(self, head_dim: int, per_head_hidden: int):
        if self._is_ablation:
            return self.operator_cls(
                head_dim,
                mlp_hidden_size=per_head_hidden,
                expansion_factor=self.expansion_factor,
                dropout_rate=self.dropout_rate,
                ablation_expansion_mode=self.ablation_expansion_mode,
                ablation_combine_mode=self.ablation_combine_mode,
            )

        try:
            return self.operator_cls(
                head_dim,
                mlp_hidden_size=per_head_hidden,
                expansion_factor=self.expansion_factor,
                dropout_rate=self.dropout_rate,
            )
        except TypeError:
            try:
                return self.operator_cls(head_dim, dropout_rate=self.dropout_rate)
            except TypeError:
                return self.operator_cls(head_dim)

    def __call__(
        self, xy: jnp.ndarray, reduction_level: Optional[jnp.ndarray | int] = None
    ) -> jnp.ndarray:
        if xy.shape[-2] != 2:
            raise ValueError(f"Expected pair dimension of size 2, got shape={xy.shape}")
        if xy.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected last dim {self.state_dim}, got {xy.shape[-1]} for shape {xy.shape}."
            )

        x = xy[..., 0, :]
        y = xy[..., 1, :]
        x_heads = jnp.reshape(x, x.shape[:-1] + (self.num_operators, self.head_dim))
        y_heads = jnp.reshape(y, y.shape[:-1] + (self.num_operators, self.head_dim))

        per_head_outputs = []
        for idx in range(self.num_operators):
            head_pair = jnp.stack([x_heads[..., idx, :], y_heads[..., idx, :]], axis=-2)
            per_head_outputs.append(self.head_operators[idx](head_pair))
        stacked_outputs = jnp.stack(per_head_outputs, axis=-2)

        gate_input = jnp.reshape(xy, xy.shape[:-2] + (2 * self.state_dim,))
        gate_logits = hk.Linear(self.num_operators, name="operator_gate_logits")(gate_input)
        level = 0 if reduction_level is None else reduction_level
        level = jnp.asarray(level, dtype=jnp.int32)
        level = jnp.clip(level, 0, self.max_reduction_levels - 1)
        level_bias = hk.Embed(
            vocab_size=self.max_reduction_levels,
            embed_dim=self.num_operators,
            name="reduction_level_embedding",
        )(level)
        gate_logits = gate_logits + level_bias
        gate_probs = jnn.softmax(gate_logits, axis=-1)

        if self.min_weight > 0.0:
            residual_mass = 1.0 - self.num_operators * self.min_weight
            gate_weights = self.min_weight + residual_mass * gate_probs
        else:
            gate_weights = gate_probs

        mixed = stacked_outputs * gate_weights[..., None]
        return jnp.reshape(mixed, mixed.shape[:-2] + (self.state_dim,))
