"""
Modified LDRU v2 for next token prediction.
This file contains the necessary changes to make LDRU v2 work for autoregressive language modeling.
"""

import dataclasses
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import chex
from typing import Optional, Callable
import jax.nn as jnn
import math
from typing import Callable, Optional
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax


# Import transformer from supplementary_code
import sys

sys.path.append("supplementary_code-main")
from ldru.models.transformer import (
    MultiHeadDotProductAttention,
)
from ldru.models import positional_encodings as pos_encs_lib


# Copy the necessary classes from ldru_v2.py and modify them
@chex.dataclass
class CausalLDRUConfig:
    """Configuration for causal LDRU language model."""

    # Model architecture
    embedding_dim: int
    vocab_size: int
    num_layers: int = 1
    hidden_dim: int = 512

    # LDRU specific
    widening_factor: int = 4
    dropout_prob: float = 0.0
    emb_init_scale: float = 0.02

    # Causal modeling
    causal_masking: bool = True
    max_sequence_length: int = 1024
    use_positional_encoding: bool = False

    # Binary operator
    operator: Optional[Callable] = None

    # Scan method: 'default' (assoc_scan), or 'simple'
    scan_method: str = "default"

    # Whether to expand sequence to power of 2 with random zero insertion
    expand_to_power_of_2: bool = False

    # Whether to apply attention at each scan step in custom associative scan
    attention_per_scan_step: bool = False


class BinaryOperator(hk.Module):
    """Binary operator from LDRU v2 - handles (..., 2, E) input."""

    def __init__(self, embedding_size: int, activation: Callable = jax.nn.silu):
        super().__init__(name="BinaryOperator")
        self.activation = activation
        self.embedding_size = embedding_size

    def __call__(self, xy: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            xy: Tensor of shape (..., 2, E).
        Returns:
            Tensor of shape (..., E).
        """
        hidden_dim = xy.shape[-1]  # E
        # Use smaller, more stable initialization
        init = hk.initializers.VarianceScaling(0.02, "fan_avg", "truncated_normal")

        # Flatten pair dim for the MLP: (..., 2E)
        pair = jnp.reshape(xy, xy.shape[:-2] + (2 * hidden_dim,))

        # Simpler, more stable MLP
        weights = hk.Linear(2 * hidden_dim, w_init=init)(pair)  # Reduced complexity
        weights = self.activation(weights)
        weights = hk.Linear(4 * hidden_dim, w_init=init)(weights)  # Direct connection
        weights = self.activation(weights)
        weights = hk.Linear(2 * hidden_dim, w_init=init)(weights)  # Direct connection

        # Apply layer norm for stability
        weights = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(weights)

        # # Simple weighted combination instead of complex reshaping
        # x_weight = hk.Linear(1, w_init=init)(weights)  # Scalar weight for x
        # y_weight = hk.Linear(1, w_init=init)(weights)  # Scalar weight for y
        weights = jnp.reshape(weights, xy.shape)  # (..., 2, E)

        # # Apply sigmoid to keep weights bounded
        # x_weight = jax.nn.sigmoid(x_weight)
        # y_weight = jax.nn.sigmoid(y_weight)
        x_weight = weights[..., 0, :]
        y_weight = weights[..., 1, :]

        # Weighted sum with residual connection
        x = xy[..., 0, :]  # (..., E)
        y = xy[..., 1, :]  # (..., E)

        # Check if x or y are 0 vectors (for padding cases)
        # If so, skip the addition and simply preserve the non-zero input
        # This helps maintain position information when padding is involved
        x_weight = jnp.where(
            jnp.all(x == 0, axis=-1, keepdims=True), jnp.zeros_like(x_weight), x_weight
        )
        y_weight = jnp.where(
            jnp.all(y == 0, axis=-1, keepdims=True), jnp.zeros_like(y_weight), y_weight
        )
        weighted_x = hk.Linear(hidden_dim, w_init=hk.initializers.Identity())(
            x_weight * x
        )
        weighted_y = hk.Linear(hidden_dim, w_init=hk.initializers.Identity())(
            y_weight * y
        )

        result = hk.Linear(hidden_dim, w_init=hk.initializers.Identity())(
            weighted_x + weighted_y
        )

        return result
        # hidden_dim = xy.shape[-1]  # E
        # x, y = xy[..., 0, :], xy[..., 1, :]
        # init = hk.initializers.VarianceScaling(0.02, "fan_avg", "truncated_normal")

        # # Flatten pair dim for the MLP: (..., 2E)
        # pair = jnp.reshape(xy, xy.shape[:-2] + (2 * hidden_dim,))

        # mix_0 = hk.Linear(2 * hidden_dim, w_init=init)(pair)
        # mix_0 = self.activation(mix_0)
        # mix_1 = hk.Linear(4 * hidden_dim, w_init=init)(mix_0)
        # mix_1 = self.activation(mix_1)
        # weights = hk.Linear(2 * hidden_dim, w_init=init)(mix_1) + mix_0
        # weights = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(weights)
        # weights = jnp.reshape(weights, weights.shape[:-1] + (2, hidden_dim))

        # scaled = weights * xy

        # x_prime = hk.Linear(hidden_dim, w_init=hk.initializers.Identity())(
        #     scaled[..., 0, :]
        # )
        # y_prime = hk.Linear(hidden_dim, w_init=hk.initializers.Identity())(
        #     scaled[..., 1, :]
        # )

        # out = hk.Linear(hidden_dim, w_init=hk.initializers.Identity())(
        #     x_prime + y_prime
        # )
        # is_pad_y = jnp.isclose(jnp.sum(jnp.abs(y), axis=-1, keepdims=True), 0.0)
        # is_pad_x = jnp.isclose(jnp.sum(jnp.abs(x), axis=-1, keepdims=True), 0.0)

        # return jnp.where(is_pad_y, x, jnp.where(is_pad_x, y, out))


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    """Apply LayerNorm with default settings."""
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)


class CausalLDRULayer(hk.Module):
    """Modified LDRU layer that handles causal masking and position preservation."""

    def __init__(self, config: CausalLDRUConfig):
        super().__init__(name="CausalLDRULayer")
        self.config = config
        self.inner_norm = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="inner_norm"
        )
        self.init_norm = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="init_norm"
        )

        # Use stable binary operator
        operator_cls = config.operator or BinaryOperator
        self.binary_operator = operator_cls(config.embedding_dim)

        # Smaller initialization for stability
        init = hk.initializers.VarianceScaling(
            0.02, "fan_avg", "truncated_normal"  # Reduced for stability
        )
        self.fc_0 = hk.Linear(
            config.embedding_dim * 2, w_init=init  # Reduced widening factor
        )
        self.fc_1 = hk.Linear(config.embedding_dim, w_init=init)
        self.fc = hk.Sequential([self.fc_0, jnn.silu, self.fc_1])

    def _make_pure_binary_op(self, dummy_input):
        """
        Create a pure function version of binary_operator by pre-applying it once
        to materialize parameters, then capturing those params.

        Returns a pure function that can be used inside JAX control flow.
        """
        # Apply operator once to get all parameters
        _ = self.binary_operator(dummy_input)

        # Get all parameters for this operator
        # In Haiku, we need to use hk.experimental.to_module to extract the callable
        # Instead, we'll create a closure that captures the current Haiku context

        # For now, return a simple wrapper that will work in pure JAX context
        # This requires the operator to be stateless after init
        def pure_op(stacked_vals):
            return self.binary_operator(stacked_vals)

        return pure_op

    def simple_causal_scan(self, h: jnp.ndarray, binary_operator) -> jnp.ndarray:
        """
        Simplified causal scan that's more numerically stable.
        Instead of complex Blelloch scan, use a simple left-to-right accumulation.
        """
        B, L, E = h.shape

        # Initialize output
        output = jnp.zeros_like(h)

        # First position is just the input
        output = output.at[:, 0, :].set(h[:, 0, :])

        # For each subsequent position, combine with previous
        for i in range(1, L):
            # Take current element and previous accumulated result
            current = h[:, i : i + 1, :]  # [B, 1, E]
            prev = output[:, i - 1 : i, :]  # [B, 1, E]

            # Stack for binary operator
            pair = jnp.concatenate([prev, current], axis=2)  # [B, 1, 2E]
            pair = jnp.reshape(pair, (B, 1, 2, E))  # [B, 1, 2, E]

            # Apply binary operator
            combined = binary_operator(pair)  # [B, 1, E]

            # Store result
            output = output.at[:, i, :].set(combined.squeeze(1))

        return output

    def adaptive_scan_up(
        self,
        h: jnp.ndarray,  # [B, L, E]
        binary_operator: BinaryOperator = None,
        mix_in_ratio: Optional[
            float
        ] = None,  # None -> exact op; else blend with previous
    ):

        def _next_pow2(n: int) -> int:
            return 1 << (n - 1).bit_length()

        if binary_operator is None:

            def binary_operator(stacked_ab):
                # stacked_ab has shape (..., 2, E)
                return stacked_ab[..., 0, :] + stacked_ab[..., 1, :]

        B, L, E = h.shape

        # Ensure we work with float32 to avoid dtype issues
        original_dtype = h.dtype
        if h.dtype != jnp.float32:
            h = h.astype(jnp.float32)

        L_pow2 = _next_pow2(L)
        pad = L_pow2 - L
        if pad:
            h = jnp.pad(h, ((0, 0), (0, pad), (0, 0)))

        pos = jnp.arange(L_pow2, dtype=jnp.int32)  # [L_pow2]
        d = int(np.log2(L_pow2))

        @jax.jit
        def stage(i, x):
            # Compute block geometry (scalar traced values, but shapes remain fixed)
            block = jnp.int32(1 << (i + 1))
            half = jnp.int32(1 << i)

            # For each position, find the left and right endpoints of its block
            block_start = pos - (pos % block)
            left_idx = block_start + (half - 1)  # [L_pow2]
            right_idx = block_start + (block - 1)  # [L_pow2]

            # Gather values at those endpoints for every position
            left_vals = jnp.take(x, left_idx, axis=1)  # [B, L_pow2, E]
            right_vals = jnp.take(x, right_idx, axis=1)  # [B, L_pow2, E]

            # Stack left and right values for binary operator
            # BinaryOperator expects shape (..., 2, E)
            stacked_vals = jnp.stack(
                [left_vals, right_vals], axis=-2
            )  # [B, L_pow2, 2, E]

            # Compute reduced values for the "right endpoints"
            reduced = binary_operator(stacked_vals)  # [B, L_pow2, E]

            h = self.fc(reduced) + mix_in_ratio * reduced
            h = self.inner_norm(h)
            h = hk.dropout(hk.next_rng_key(), self.config.dropout_prob, h)

            upd = h

            # Update only positions that are the block's right endpoint
            is_right = (pos % block) == (block - 1)  # [L_pow2] bool
            mask = is_right[None, :, None].astype(x.dtype)  # [B, L_pow2, 1]

            # Ensure mask operations preserve dtype
            one_minus_mask = jnp.array(1.0, dtype=x.dtype) - mask
            x = x * one_minus_mask + upd * mask
            return x

        # Use lax.fori_loop for upsweep as well
        h_out = lax.fori_loop(0, d, stage, h)
        result = h_out[:, :L, :]

        # Convert back to original dtype if needed
        if original_dtype != jnp.float32:
            result = result.astype(original_dtype)

        return result

    @staticmethod
    def adaptive_scan_down(
        h: jnp.ndarray,  # [B, L, E] (output of upsweep)
        binary_operator: BinaryOperator = None,
        identity: Optional[
            jnp.ndarray
        ] = None,  # [B, E] identity for op (default zeros for addition)
    ):
        def _next_pow2(n: int) -> int:
            return 1 << (n - 1).bit_length()

        if binary_operator is None:

            def binary_operator(stacked_ab):
                # stacked_ab has shape (..., 2, E)
                return stacked_ab[..., 0, :] + stacked_ab[..., 1, :]

        B, L, E = h.shape

        # Ensure we work with float32 to avoid dtype issues
        original_dtype = h.dtype
        if h.dtype != jnp.float32:
            h = h.astype(jnp.float32)

        L_pow2 = _next_pow2(L)
        pad = L_pow2 - L
        if pad:
            h = jnp.pad(h, ((0, 0), (0, pad), (0, 0)))

        # Set root to identity (exclusive scan requirement)
        if identity is None:
            identity = jnp.zeros((B, E), dtype=h.dtype)
        else:
            identity = identity.astype(h.dtype)
        h = h.at[:, L_pow2 - 1, :].set(identity)

        pos = jnp.arange(L_pow2, dtype=jnp.int32)
        d = int(np.log2(L_pow2))

        @jax.jit
        def stage(i, x):
            block = jnp.int32(1 << (i + 1))
            half = jnp.int32(1 << i)

            block_start = pos - (pos % block)
            left_idx = block_start + (half - 1)
            right_idx = block_start + (block - 1)

            left_vals = jnp.take(x, left_idx, axis=1)  # [B, L_pow2, E]
            right_vals = jnp.take(x, right_idx, axis=1)  # [B, L_pow2, E]

            # Blelloch downsweep:
            # new_left  <- right
            # new_right <- op(left, right)
            new_left = right_vals

            # Stack for binary operator
            stacked_vals = jnp.stack(
                [left_vals, right_vals], axis=-2
            )  # [B, L_pow2, 2, E]
            new_right = binary_operator(stacked_vals)  # [B, L_pow2, E]

            is_left = (pos % block) == (half - 1)
            is_right = (pos % block) == (block - 1)

            mask_left = is_left[None, :, None].astype(x.dtype)
            mask_right = is_right[None, :, None].astype(x.dtype)

            # Ensure mask operations preserve dtype
            one_dtype = jnp.array(1.0, dtype=x.dtype)
            x = x * (one_dtype - mask_left) + new_left * mask_left
            x = x * (one_dtype - mask_right) + new_right * mask_right
            return x

        # Run stages in reverse: i = d-1... 0 using lax.fori_loop
        def rev_loop(x0):
            def body(j, y):
                return stage(d - 1 - j, y)

            return lax.fori_loop(0, d, body, x0)

        h_out = rev_loop(h)
        result = h_out[:, :L, :]

        # Convert back to original dtype if needed
        if original_dtype != jnp.float32:
            result = result.astype(original_dtype)

        return result

    def assoc_scan(self, h, binary_operator):
        # This function is not technically matching that described in the original paper
        # In the paper, the fc is applied after every layer. Here we only apply it once at the end.
        def scan_one(x):  # x: [L, E]
            h = jax.lax.associative_scan(
                lambda a, b: binary_operator(jnp.stack([a, b], axis=-2)), x
            )

            h_res = h
            h = self.fc(h)
            h = h + 0.1 * h_res  # Small residual
            h = self.inner_norm(h)
            h = hk.dropout(hk.next_rng_key(), self.config.dropout_prob, h)
            return h

        incl = jax.vmap(scan_one)(h)  # h: [B, L, E]

        # shift right by one slot and include identity at start
        identity = jnp.zeros((h.shape[0], 1, h.shape[2]), dtype=h.dtype)  # [B, 1, E]
        return jnp.concatenate((identity, incl[:, 1:, :]), axis=1)

    def assoc_scan_custom(self, h, binary_operator):
        # This function uses the custom associative scan with inner function
        # Can optionally include attention before each scan operation
        from custom_assoc_scan import associative_scan

        def inner_fn_with_attention(red):
            """Inner function that applies attention before the feedforward processing."""
            # Apply causal attention to the reduced values if configured
            if self.config.attention_per_scan_step:
                # Expand red dim by 1
                red_ex = jnp.expand_dims(red, axis=0)  # [1, L, E]
                red_attended = self._apply_causal_attention(red_ex)
                red_attended = red_attended[0]  # Remove extra dim

                # Apply layer norm to the attention output
                red_attended = hk.LayerNorm(
                    axis=-1, create_scale=True, create_offset=True
                )(red_attended)

                # Combine with residual connection
                red = red + red_attended

            # Apply the original feedforward processing
            h = self.fc(red) + red
            h = self.inner_norm(h)
            h = hk.dropout(hk.next_rng_key(), self.config.dropout_prob, h)
            return h

        def combine(a, b):
            return binary_operator(jnp.stack([a, b], axis=-2))

        def scan_one(x):  # x: [L, E]
            h = associative_scan(
                combine,
                x,
                inner_fn=inner_fn_with_attention,  # Use the attention-enhanced inner function
            )
            return h

        incl = jax.vmap(scan_one)(h)  # h: [B, L, E]

        # shift right by one slot and include identity at start
        identity = jnp.zeros((h.shape[0], 1, h.shape[2]), dtype=h.dtype)  # [B, 1, E]
        return jnp.concatenate((identity, incl[:, 1:, :]), axis=1)

    @staticmethod
    def expand_to_power_of_2_with_random_zeros(
        h: jnp.ndarray,  # [B, L, E]
        rng_key: jax.random.PRNGKey,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Expand sequence to next power of 2 by inserting zero vectors at random positions.

        Args:
            h: Input sequence of shape [B, L, E]
            rng_key: Random key for selecting insertion positions

        Returns:
            Tuple of:
            - Expanded sequence of shape [B, L_pow2, E] where L_pow2 is next power of 2 >= L
            - Boolean mask of shape [L_pow2] indicating positions of original elements (True)
        """

        def _next_pow2(n: int) -> int:
            return 1 << (n - 1).bit_length()

        B, L, E = h.shape
        L_pow2 = _next_pow2(L)

        if L == L_pow2:
            # Already a power of 2, no expansion needed
            mask = jnp.ones(L, dtype=bool)
            return h, mask

        @jax.jit
        def insert_random_zeros(seq, insert_rng):
            """Insert zero vectors at random positions using JAX-friendly operations."""
            seq_len = seq.shape[1]
            target_len = L_pow2

            # Generate random positions for the original elements in the expanded sequence
            expanded_positions = jax.random.choice(
                insert_rng, target_len, shape=(seq_len,), replace=False
            )
            expanded_positions = jnp.sort(expanded_positions)

            # Create the expanded sequence filled with zeros
            expanded_seq = jnp.zeros((B, target_len, E), dtype=seq.dtype)

            # Place original elements at their new positions
            expanded_seq = expanded_seq.at[:, expanded_positions, :].set(seq)

            # Create mask indicating positions of original elements
            mask = jnp.zeros(target_len, dtype=bool)
            mask = mask.at[expanded_positions].set(True)

            return expanded_seq, mask

        # Apply the insertion
        result, mask = insert_random_zeros(h, rng_key)

        return result, mask

    @staticmethod
    def restore_original_length(
        h_processed: jnp.ndarray,  # [B, L_pow2, E]
        original_mask: jnp.ndarray,  # [L_pow2] boolean mask
        original_length: int,  # Original sequence length L
    ) -> jnp.ndarray:
        """
        Restore original sequence length by selecting only the original (non-zero) positions.

        Args:
            h_processed: Processed sequence of shape [B, L_pow2, E]
            original_mask: Boolean mask of shape [L_pow2] indicating positions of original elements
            original_length: Original sequence length before expansion

        Returns:
            Restored sequence of shape [B, original_length, E]
        """
        B, L_pow2, E = h_processed.shape

        # Get indices of original elements
        original_indices = jnp.where(
            original_mask, size=original_length, fill_value=-1
        )[0]

        # Select only the original positions
        # Note: original_indices should have exactly original_length elements
        h_restored = h_processed[:, original_indices, :]  # [B, original_length, E]

        return h_restored

    def _create_causal_mask(self, seq_len: int) -> jnp.ndarray:
        """
        Create a causal attention mask where each position can only attend to previous positions.

        Args:
            seq_len: Sequence length

        Returns:
            Causal mask of shape [1, 1, seq_len, seq_len] where 1 means can attend, 0 means mask
        """
        # Create lower triangular matrix (including diagonal)
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))

        # Expand dimensions for broadcasting: [1, 1, seq_len, seq_len]
        # This matches the expected shape for multi-head attention [batch, heads, seq_len, seq_len]
        mask = mask[None, None, :, :]

        return mask

    def _apply_causal_attention(self, h: jnp.ndarray) -> jnp.ndarray:
        """
        Apply multi-head causal attention to the input.

        Args:
            h: Input tensor of shape [B, L, E]

        Returns:
            Attention output of shape [B, L, E]
        """
        # Only create attention layer if actually configured to use attention
        if not self.config.attention_per_scan_step:
            # Return input unchanged if attention is not configured
            return h

        seq_len = h.shape[1]

        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len)

        # Ensure embedding dimension is divisible by number of heads
        num_heads = 8
        head_dim = self.config.embedding_dim // num_heads
        if self.config.embedding_dim % num_heads != 0:
            # Adjust number of heads if embedding dim is not divisible
            num_heads = max(
                1, self.config.embedding_dim // 64
            )  # At least 64 dim per head
            head_dim = self.config.embedding_dim // num_heads

        # Create attention layer
        attention_layer = MultiHeadDotProductAttention(
            num_heads=num_heads,
            num_hiddens_per_head=head_dim,
            positional_encodings=pos_encs_lib.PositionalEncodings.SIN_COS,
            positional_encodings_params=pos_encs_lib.SinCosParams(
                max_time=self.config.max_sequence_length
            ),
            chunk_size=None,  # Full attention, no chunking
        )

        # Apply attention with causal masking
        h_attended = attention_layer(
            inputs_q=h,  # Queries
            inputs_kv=h,  # Keys and values (same as queries for self-attention)
            mask=causal_mask,  # Explicit causal mask
            causal=True,  # Additional internal causal flag
        )

        return h_attended

    def _simple_feedforward(self, h: jnp.ndarray) -> jnp.ndarray:
        """
        Simple 2-layer feedforward network with residual connection.

        Args:
            h: Input tensor of shape [..., E]

        Returns:
            Output tensor of same shape with feedforward applied
        """
        res = h
        h = hk.Linear(self.config.embedding_dim)(h)
        h = jnn.silu(h)
        h = hk.Linear(self.config.embedding_dim)(h)
        h = h + res
        h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)
        h = hk.dropout(hk.next_rng_key(), self.config.dropout_prob, h)
        return h

    def _resnet_feedforward(self, h: jnp.ndarray) -> jnp.ndarray:
        """
        ResNet18-inspired feedforward network with proper residual blocks and bottlenecks.

        Args:
            h: Input tensor of shape [..., E]

        Returns:
            Output tensor of same shape with complex feedforward applied
        """

        def residual_block(x, hidden_dim, use_bottleneck=True, block_name=""):
            """ResNet-style residual block with optional bottleneck."""
            residual = x

            if use_bottleneck:
                # Bottleneck: compress -> process -> expand
                bottleneck_dim = hidden_dim // 4

                # 1x1 compression
                x = hk.Linear(bottleneck_dim, name=f"{block_name}_compress")(x)
                x = hk.LayerNorm(
                    axis=-1,
                    create_scale=True,
                    create_offset=True,
                    name=f"{block_name}_norm1",
                )(x)
                x = jnn.silu(x)

                # Main processing
                x = hk.Linear(bottleneck_dim, name=f"{block_name}_main")(x)
                x = hk.LayerNorm(
                    axis=-1,
                    create_scale=True,
                    create_offset=True,
                    name=f"{block_name}_norm2",
                )(x)
                x = jnn.silu(x)

                # 1x1 expansion
                x = hk.Linear(hidden_dim, name=f"{block_name}_expand")(x)
                x = hk.LayerNorm(
                    axis=-1,
                    create_scale=True,
                    create_offset=True,
                    name=f"{block_name}_norm3",
                )(x)
            else:
                # Standard ResNet block
                x = hk.Linear(hidden_dim, name=f"{block_name}_conv1")(x)
                x = hk.LayerNorm(
                    axis=-1,
                    create_scale=True,
                    create_offset=True,
                    name=f"{block_name}_norm1",
                )(x)
                x = jnn.silu(x)

                x = hk.Linear(hidden_dim, name=f"{block_name}_conv2")(x)
                x = hk.LayerNorm(
                    axis=-1,
                    create_scale=True,
                    create_offset=True,
                    name=f"{block_name}_norm2",
                )(x)

            # Residual connection with projection if needed
            if residual.shape[-1] != x.shape[-1]:
                residual = hk.Linear(x.shape[-1], name=f"{block_name}_proj")(residual)

            # Add residual and final activation
            x = x + residual
            x = jnn.silu(x)

            return x

        # Initial processing (like ResNet stem)
        h = hk.Linear(self.config.embedding_dim, name="stem_conv")(h)
        h = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="stem_norm"
        )(h)
        h = jnn.silu(h)

        # Layer 1: 2 basic blocks (like ResNet conv2_x)
        h = residual_block(
            h,
            self.config.embedding_dim,
            use_bottleneck=False,
            block_name="layer1_block1",
        )
        h = residual_block(
            h,
            self.config.embedding_dim,
            use_bottleneck=False,
            block_name="layer1_block2",
        )

        # Layer 2: 2 bottleneck blocks with dimension expansion (like ResNet conv3_x)
        expanded_dim = self.config.embedding_dim * 2
        h = residual_block(
            h, expanded_dim, use_bottleneck=True, block_name="layer2_block1"
        )
        h = residual_block(
            h, expanded_dim, use_bottleneck=True, block_name="layer2_block2"
        )

        # Layer 3: 2 more bottleneck blocks (like ResNet conv4_x)
        h = residual_block(
            h, expanded_dim, use_bottleneck=True, block_name="layer3_block1"
        )
        h = residual_block(
            h, expanded_dim, use_bottleneck=True, block_name="layer3_block2"
        )

        # Final projection back to original dimension
        h = hk.Linear(self.config.embedding_dim, name="final_proj")(h)
        h = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="final_norm"
        )(h)

        # Global residual connection from input
        # Apply dropout before final residual
        h = hk.dropout(hk.next_rng_key(), self.config.dropout_prob, h)

        return h

    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        """
        Apply causal LDRU processing with improved stability.

        Args:
            h: Input tensor of shape [B, L, E]

        Returns:
            Output tensor of shape [B, L, E] with causal processing applied
        """
        B, L, E = h.shape

        # Apply layer norm first for stability
        h = self.init_norm(h)

        # Apply feedforward network (can switch between simple and ResNet-inspired)
        original_h = h  # Store for global residual connection
        h = self._resnet_feedforward(h)
        # h = self._simple_feedforward(h)

        # Add global residual connection (like in ResNet)
        h = h + original_h  # Scale down the residual for stability

        # Apply simplified causal processing
        if L > 1:
            original_length = L
            original_mask = None
            if self.config.expand_to_power_of_2:
                expansion_rng = hk.next_rng_key()
                h, original_mask = self.expand_to_power_of_2_with_random_zeros(
                    h, expansion_rng
                )
                B, L, E = h.shape  # Update L to new expanded length

            # Initialize binary operator by calling it once outside the scan
            # This materializes all parameters before entering JAX control flow
            dummy_input = jnp.zeros((B, 2, 2, E))  # [B, 2, 2, E] - smaller dummy
            _ = self.binary_operator(dummy_input)

            # Choose scan method based on configuration
            # if self.config.scan_method == "simple":
            #     # Use simple sequential scan
            #     h = self.simple_causal_scan(h, self.binary_operator)
            # else:  # default

            h_attended = h
            if self.config.attention_per_scan_step:
                h_attended = self._apply_causal_attention(h)

            # Apply layer norm and residual connection
            h_attended = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
                h_attended
            )
            h_attended = hk.dropout(
                hk.next_rng_key(), self.config.dropout_prob, h_attended
            )
            h = h + h_attended  # Small residual weight for stability

            # Apply the associative scan
            h = self.assoc_scan_custom(h, self.binary_operator)
            # h = self.assoc_scan(h, self.binary_operator)

        # Restore original sequence length if expansion was used
        if self.config.expand_to_power_of_2 and original_mask is not None:
            h = self.restore_original_length(h, original_mask, original_length)

        return h  # Return original and processed


class CausalLDRUEncoder(hk.Module):
    """Encoder with multiple causal LDRU layers."""

    def __init__(self, config: CausalLDRUConfig):
        super().__init__(name="CausalLDRUEncoder")
        self.config = config

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply multiple LDRU layers with residual connections.

        Args:
            x: Input embeddings of shape [B, L, E]

        Returns:
            Hidden states of shape [B, L, E]
        """
        h = x

        # Apply multiple LDRU layers with residual connections
        for layer_idx in range(self.config.num_layers):
            layer = CausalLDRULayer(self.config)
            h = layer(h) + 0.1 * h  # Small residual for stability

        return h


class CausalLDRULanguageModel(hk.Module):
    """Complete causal LDRU model for language modeling."""

    def __init__(self, config: CausalLDRUConfig):
        super().__init__(name="CausalLDRULanguageModel")
        self.config = config

        # Token embedding
        self.token_embedding = hk.Embed(
            config.vocab_size,
            config.embedding_dim,
            w_init=hk.initializers.TruncatedNormal(stddev=config.emb_init_scale),
        )

        # Optional positional encoding
        if config.use_positional_encoding:
            self.pos_embedding = hk.Embed(
                config.max_sequence_length,
                config.embedding_dim,
                w_init=hk.initializers.TruncatedNormal(stddev=config.emb_init_scale),
            )

        # LDRU encoder
        self.encoder = CausalLDRUEncoder(config)

        # Output projection to vocabulary
        self.lm_head = hk.Linear(
            config.vocab_size,
            w_init=hk.initializers.TruncatedNormal(stddev=config.emb_init_scale),
        )

    def __call__(self, token_ids: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        """
        Forward pass for language modeling.

        Args:
            token_ids: Token IDs of shape [B, L]
            is_training: Whether in training mode

        Returns:
            Logits of shape [B, L, vocab_size]
        """
        B, L = token_ids.shape

        # Token embeddings
        embeddings = self.token_embedding(token_ids)

        # Add positional embeddings if enabled
        if self.config.use_positional_encoding:
            positions = jnp.arange(L)[None, :]  # [1, L]
            pos_embeddings = self.pos_embedding(positions)
            embeddings = embeddings + pos_embeddings

        # Apply LDRU encoder
        hidden_states = self.encoder(embeddings)

        # Final layer norm for stability
        hidden_states = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            hidden_states
        )

        # Project to vocabulary
        logits = self.lm_head(hidden_states)

        return logits


def create_causal_ldru_model(config: CausalLDRUConfig) -> hk.Transformed:
    """Create a transformed causal LDRU model."""

    def forward_fn(token_ids: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        model = CausalLDRULanguageModel(config)
        return model(token_ids, is_training)

    return hk.transform(forward_fn)


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = CausalLDRUConfig(
        embedding_dim=128,
        vocab_size=1000,
        num_layers=2,
        max_sequence_length=64,
        use_positional_encoding=True,
    )

    # Create model
    model = create_causal_ldru_model(config)

    # Test forward pass
    batch_size, seq_len = 4, 16
    rng_key = jax.random.PRNGKey(42)

    # Sample random token IDs
    token_ids = jax.random.randint(
        rng_key, shape=(batch_size, seq_len), minval=0, maxval=config.vocab_size
    )

    # Initialize and run model
    params = model.init(rng_key, token_ids)
    logits = model.apply(params, rng_key, token_ids)

    print(f"Input shape: {token_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_len}, {config.vocab_size})")
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
