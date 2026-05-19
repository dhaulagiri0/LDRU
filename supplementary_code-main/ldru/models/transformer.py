# Transformer architecture also containing the code for RegularGPT
# Sources: Delétang et al. (2023); Chi et al. (2023)

from typing import Callable, Optional, Any
import dataclasses
import functools
import chex
import haiku as hk
import jax
import jax.random as jrandom
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import logging
import math

from ldru.models import positional_encodings as pos_encs_lib


@chex.dataclass
class TransformerConfig:
    """Hyperparameters used in the Transformer architectures."""

    # The size of the model output (i.e., the output vocabulary size).
    output_size: int
    # The dimension of the first embedding.
    embedding_dim: int
    # The number of heads per layer.
    num_heads: int
    # number of transformer layers
    num_layers: Optional[int] = 1
    # number of different layers in one transformer layer
    thickness: Optional[int] = 1
    # The number of hidden neurons per head. If None, it is set to be equal to
    # `embedding_dim // num_heads`.
    num_hiddens_per_head: Optional[int] = None
    # The probability that each element is discarded by the dropout modules.
    dropout_prob: float = 0.1
    # The parameter initialization scale for the embeddings.
    emb_init_scale: float = 0.02
    # Whether to use the embeddings rather than raw inputs.
    use_embeddings: bool = True
    # Whether to share embeddings between the Encoder and the Decoder.
    share_embeddings: bool = False
    # The size of the sliding attention window. See MultiHeadDotProductAttention.
    chunk_size: Optional[int] = None
    # The positional encoding used with default sin/cos (Vaswani et al., 2017).
    positional_encodings: pos_encs_lib.PositionalEncodings = dataclasses.field(
        default_factory=lambda: pos_encs_lib.PositionalEncodings.SIN_COS
    )
    # The maximum size of the context (used by the posiitonal encodings).
    max_time: int = 10_000
    # The parameters for the positional encodings, default sin/cos.
    positional_encodings_params: pos_encs_lib.PositionalEncodingsParams = (
        dataclasses.field(default_factory=pos_encs_lib.SinCosParams)
    )
    # How much larger the hidden layer of the feedforward network should be
    # compared to the `embedding_dim`.
    widening_factor: int = 4
    # Add mask to make causal predictions.
    causal_masking: bool = False
    # Share transformer weight across layers
    share_weight: bool = True
    # Use our special front rear shared positional embeddings
    use_front_rear_pos: bool = False
    # Optional nanoGPT-like transformer block variant.
    # When enabled, uses pre-norm residual blocks and GELU FFN activation.
    pre_norm_gelu_block: bool = False

    def __post_init__(self) -> None:
        """Sets `num_hiddens_per_head` if it is `None`."""
        if self.num_hiddens_per_head is None:
            self.num_hiddens_per_head = self.embedding_dim // self.num_heads


def layer_norm(x: chex.Array) -> chex.Array:
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)


def shift_right(x: chex.Array, output_size: int) -> chex.Array:
    """Right-shift the one-hot encoded input by padding on the temporal axis."""
    x = jnp.argmax(x, axis=-1)

    # Add a time dimension for the single-output case (i.e., `ndim == 1`).
    if x.ndim == 1:
        x = jnp.expand_dims(x, axis=1)

    padded = jnp.pad(x, ((0, 0), (1, 0)), mode="constant", constant_values=output_size)

    return jnn.one_hot(padded[:, :-1], num_classes=output_size + 1)


def compute_sliding_window_mask(
    sequence_length: int, chunk_size: int, num_heads: int
) -> chex.Array:
    """Returns a k-diagonal mask for a sliding window.

    Args:
      sequence_length: The length of the sequence, which will determine the shape
        of the output.
      chunk_size: The size of the sliding window.

    Returns:
      A symmetric matrix of shape (sequence_length, sequence_length),
      chunk_size-diagonal, with ones on the diagonal and on all the
      upper/lower diagonals up to chunk_size // 2.

    Raises:
      ValueError if chunk_size is <= 0.
    """
    if chunk_size <= 0:
        raise ValueError(f"The attention window should be > 0. Got {chunk_size}.")

    if chunk_size == 1:
        return jnp.eye(sequence_length, sequence_length)

    # if we set the name to start with 'pos_', it will be frozen in training/training.py
    rpe_bias = hk.get_parameter(
        "rpe", (num_heads, chunk_size), init=hk.initializers.RandomNormal(stddev=0.02)
    )
    # logging.info('use rpe')
    attention_masks = []
    for h in range(num_heads):
        attention_mask = jnp.sum(
            jnp.stack(
                [
                    jnp.eye(sequence_length, sequence_length, k=-k, dtype=jnp.int32)
                    * rpe_bias[h][k]
                    for k in range(chunk_size)
                ]
            ),
            axis=0,
        )
        attention_masks.append(attention_mask)
    attention_mask = jnp.stack(attention_masks)
    attention_mask = jnp.where(
        attention_mask != 0, attention_mask, jnp.finfo(jnp.float32).min
    )
    # attention_mask = jnp.where(attention_mask != 0, 0, jnp.finfo(jnp.float32).min)

    return attention_mask


class TransformerLayer(hk.Module):
    def __init__(self, config, pos_enc_params):
        super().__init__(name=None)
        self._config = config
        self._pos_enc_params = pos_enc_params

    def __call__(self, h, causal_mask):
        if self._config.use_front_rear_pos:
            # add positional embeddings to h
            for i in range(self._config.chunk_size):
                pos_vec = hk.get_parameter(
                    "pos_{}".format(i),
                    (self._config.embedding_dim,),
                    init=functools.partial(jrandom.normal, hk.next_rng_key()),
                )
                h = h.at[:, i :: self._config.chunk_size].add(pos_vec)
        sequence_length = h.shape[1]
        if self._config.pre_norm_gelu_block:
            # nanoGPT-style pre-norm residual block with GELU FFN.
            attn_input = layer_norm(h)
            attention = MultiHeadDotProductAttention(
                num_heads=self._config.num_heads,
                num_hiddens_per_head=self._config.num_hiddens_per_head,
                positional_encodings=self._config.positional_encodings,
                positional_encodings_params=self._pos_enc_params,
                chunk_size=self._config.chunk_size,
            )(
                inputs_q=attn_input,
                inputs_kv=attn_input,
                mask=causal_mask,
                causal=self._config.causal_masking,
            )
            attention = hk.dropout(
                hk.next_rng_key(), self._config.dropout_prob, attention
            )
            h = h + attention

            ff_input = layer_norm(h)
            ff = hk.Linear(self._config.embedding_dim * self._config.widening_factor)(
                ff_input
            )
            ff = jnn.gelu(ff)
            ff = hk.Linear(self._config.embedding_dim)(ff)
            ff = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, ff)
            h = h + ff
            return h

        attention = MultiHeadDotProductAttention(
            num_heads=self._config.num_heads,
            num_hiddens_per_head=self._config.num_hiddens_per_head,
            positional_encodings=self._config.positional_encodings,
            positional_encodings_params=self._pos_enc_params,
            chunk_size=self._config.chunk_size,
        )(
            inputs_q=h,
            inputs_kv=h,
            mask=causal_mask,
            causal=self._config.causal_masking,
        )
        attention = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, attention)
        attention = layer_norm(h + attention)

        # Position-wise feedforward network.
        h = hk.Linear(self._config.embedding_dim * self._config.widening_factor)(
            attention
        )
        h = jnn.relu(h)
        h = hk.Linear(self._config.embedding_dim)(h)

        h = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, h)
        h = layer_norm(h + attention)

        return h


class MultiHeadDotProductAttention(hk.Module):
    """Multi-head dot-product attention (Vaswani et al., 2017)."""

    def __init__(
        self,
        num_heads: int,
        num_hiddens_per_head: int,
        positional_encodings: pos_encs_lib.PositionalEncodings,
        positional_encodings_params: pos_encs_lib.PositionalEncodingsParams,
        chunk_size: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initializes the attention module.

        Args:
          num_heads: Number of heads to use.
          num_hiddens_per_head: Number of hidden neurons per head.
          positional_encodings: Which positional encodings to use in the attention.
          positional_encodings_params: Parameters for the positional encodings.
          chunk_size: Size of the attention sliding window. None means no
            sliding window is used (or equivalently, window=full_attention_length).
            We attend only on chunk_size tokens around a given query token. We
            attend to tokens before AND after the query token. If chunk_size
            is even, we use the value +1.
          name: Name of the module.
        """
        super().__init__(name=name)
        self._num_heads = num_heads
        self._num_hiddens_per_head = num_hiddens_per_head
        self._positional_encodings = positional_encodings
        self._chunk_size = chunk_size
        self._positional_encodings_params = positional_encodings_params

    def __call__(
        self,
        inputs_q: chex.Array,
        inputs_kv: chex.Array,
        mask: Optional[chex.Array] = None,
        causal: bool = False,
    ) -> chex.Array:
        """Returns the output of the multi-head attention."""
        batch_size, sequence_length, embedding_size = inputs_q.shape

        num_hiddens = self._num_hiddens_per_head * self._num_heads
        q = hk.Linear(num_hiddens, with_bias=False)(inputs_q)
        k = hk.Linear(num_hiddens, with_bias=False)(inputs_kv)
        v = hk.Linear(num_hiddens, with_bias=False)(inputs_kv)
        # The second (sequence) dimension is undefined since it can differ between
        # queries and keys/values when decoding.
        new_shape = (batch_size, -1, self._num_heads, self._num_hiddens_per_head)
        q = jnp.reshape(q, new_shape)
        k = jnp.reshape(k, new_shape)
        v = jnp.reshape(v, new_shape)

        # Let b=batch_size, t=seq_len, h=num_heads, and d=num_hiddens_per_head.
        if self._positional_encodings == pos_encs_lib.PositionalEncodings.RELATIVE:
            attention = pos_encs_lib.compute_attention_with_relative_encodings(
                q, k, self._positional_encodings_params.max_time, causal=causal
            )
        elif self._positional_encodings == pos_encs_lib.PositionalEncodings.ROTARY:
            q = pos_encs_lib.apply_rotary_encoding(
                q, position=jnp.arange(q.shape[1])[None, :]
            )
            k = pos_encs_lib.apply_rotary_encoding(
                k, position=jnp.arange(k.shape[1])[None, :]
            )
            attention = jnp.einsum("bthd,bThd->bhtT", q, k)
        else:
            attention = jnp.einsum("bthd,bThd->bhtT", q, k)
        attention *= 1.0 / jnp.sqrt(self._num_hiddens_per_head)

        # ALiBi encodings are not scaled with the 1 / sqrt(d_k) factor.
        if self._positional_encodings == pos_encs_lib.PositionalEncodings.ALIBI:
            attention += pos_encs_lib.compute_alibi_encodings_biases(
                attention.shape[1:]
            )

        if self._chunk_size is not None:
            # We compute the sliding attention by just applying a mask on the values
            # that are outside our window.
            attention_mask = compute_sliding_window_mask(
                sequence_length, self._chunk_size, self._num_heads
            )
            attention = attention + attention_mask

        if mask is not None:
            attention = jnp.where(mask, attention, jnp.finfo(jnp.float32).min)

        normalized_attention = jnn.softmax(attention)
        output = jnp.einsum("bhtT,bThd->bthd", normalized_attention, v)
        output = jnp.reshape(output, (batch_size, sequence_length, num_hiddens))
        return hk.Linear(embedding_size, with_bias=False)(output)


class TransformerEncoder(hk.Module):
    """Transformer Encoder (Vaswani et al., 2017)."""

    def __init__(
        self,
        config: TransformerConfig,
        shared_embeddings_fn: Optional[Callable[[chex.Array], chex.Array]] = None,
        name: Optional[str] = None,
        run_gradient_analysis: bool = False,
    ) -> None:
        """Initializes the transformer encoder.

        Args:
          config: The hyperparameters used in Transformer architectures.
          shared_embeddings_fn: Embedding function that is shared with the decoder.
          name: The name of the module.
          run_gradient_analysis: Store l2 normalized gradients and exit.
        """
        super().__init__(name=name)
        self._config = config
        self._shared_embeddings_fn = shared_embeddings_fn
        self.run_gradient_analysis = run_gradient_analysis

    def __call__(self, x: jnp.ndarray) -> chex.Array:
        """Returns the transformer encoder output, shape [B, T, E]."""
        if self._config.use_embeddings:
            if self._shared_embeddings_fn is not None:
                embeddings = self._shared_embeddings_fn(x)
            else:
                # Since `x` is one-hot encoded, using hk.Linear is equivalent to
                # hk.Embed with hk.EmbedLookupStyle.ONE_HOT.
                embs_init = hk.initializers.TruncatedNormal(
                    stddev=self._config.emb_init_scale
                )
                embeddings = hk.Linear(
                    self._config.embedding_dim, with_bias=False, w_init=embs_init
                )(x)
                if self.run_gradient_analysis:
                    dummy = hk.get_parameter("dummy", embeddings.shape, init=jnp.zeros)
                    embeddings = embeddings + dummy
            embeddings *= jnp.sqrt(self._config.embedding_dim)

        else:
            embeddings = x

        batch_size, sequence_length, embedding_size = embeddings.shape

        pos_enc_params = self._config.positional_encodings_params
        if (
            self._config.positional_encodings
            == pos_encs_lib.PositionalEncodings.SIN_COS
        ):
            pos_encodings = pos_encs_lib.sinusoid_position_encoding(
                sequence_length=sequence_length,
                hidden_size=embedding_size,
                memory_length=0,
                max_timescale=pos_enc_params.max_time,
                min_timescale=2,
                clamp_length=0,
                causal=True,
            )
            h = embeddings + pos_encodings
            h = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, h)
        else:
            h = embeddings

        # The causal mask is shared across heads.
        if self._config.causal_masking:
            causal_mask = jnp.tril(
                jnp.ones((batch_size, 1, sequence_length, sequence_length))
            )
        else:
            causal_mask = None

        if self._config.num_layers is None:
            num_layers = max(
                1, int(np.ceil(math.log(sequence_length, self._config.chunk_size)))
            )
        else:
            num_layers = self._config.num_layers

        if self._config.chunk_size:
            # construct padding vector
            pad_vec = hk.get_parameter(
                "pad_vec",
                (1, 1, embedding_size),
                init=hk.initializers.RandomNormal(stddev=0.02),
            )

        for cur_layer in range(num_layers):
            if not self._config.share_weight or cur_layer == 0:
                transformer_layers = []
                for _ in range(self._config.thickness):
                    transformer_layers.append(
                        TransformerLayer(self._config, pos_enc_params)
                    )
            if self._config.chunk_size:
                h = self.pad_input(h, pad_vec)
            for transformer_layer in transformer_layers:
                h = transformer_layer(h, causal_mask)

            if self._config.chunk_size:
                indices = list(range(h.shape[1] - 1, -1, -self._config.chunk_size))[
                    ::-1
                ]
                h = h[:, indices, :]

            if self.run_gradient_analysis:

                def save_to_files(h, cur_layer):
                    np.save(
                        open("clustering/{}.npy".format(cur_layer), "wb"), np.array(h)
                    )

                jax.debug.callback(save_to_files, h, cur_layer)
        return h

    def pad_input(self, h, pad_vec):
        batch_size, sequence_length, embedding_size = h.shape
        remaining = sequence_length % self._config.chunk_size
        remaining = (self._config.chunk_size - remaining) % self._config.chunk_size
        pad_vec = jnp.resize(pad_vec, (batch_size, remaining, embedding_size))
        h = jnp.concatenate([pad_vec, h], axis=1)
        return h


class TransformerDecoder(hk.Module):
    """Transformer Decoder (Vaswani et al., 2017)."""

    def __init__(
        self,
        config: TransformerConfig,
        shared_embeddings_fn: Optional[Callable[[chex.Array], chex.Array]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initializes the Transformer decoder.

        Args:
          config: The hyperparameters used in Transformer architectures.
          shared_embeddings_fn: Embedding function that is shared with the encoder.
          name: The name of the module.
        """
        super().__init__(name=name)
        self._config = config
        self._shared_embeddings_fn = shared_embeddings_fn

    def __call__(self, encoded: chex.Array, targets: chex.Array) -> chex.Array:
        """Returns the transformer decoder output, shape [B, T_O, E].

        Args:
          encoded: The output of the encoder, shape [B, T_I, E].
          targets: The one-hot encoded target values, shape [B, T_O, 2].
        """
        targets = shift_right(targets, self._config.output_size)

        if self._config.use_embeddings:
            if self._shared_embeddings_fn is not None:
                output_embeddings = self._shared_embeddings_fn(targets)
            else:
                # Since `x` is one-hot encoded, using hk.Linear is equivalent to
                # hk.Embed with hk.EmbedLookupStyle.ONE_HOT.
                embs_init = hk.initializers.TruncatedNormal(
                    stddev=self._config.emb_init_scale
                )
                output_embeddings = hk.Linear(
                    self._config.embedding_dim, with_bias=False, w_init=embs_init
                )(targets)

            output_embeddings *= jnp.sqrt(self._config.embedding_dim)

        else:
            output_embeddings = targets

        batch_size, output_sequence_length, embedding_size = output_embeddings.shape

        if (
            self._config.positional_encodings
            == pos_encs_lib.PositionalEncodings.SIN_COS
        ):
            pos_encodings = pos_encs_lib.sinusoid_position_encoding(
                sequence_length=output_sequence_length,
                hidden_size=embedding_size,
                memory_length=0,
                max_timescale=self._config.positional_encodings_params.max_time,
                min_timescale=2,
                clamp_length=0,
                causal=True,
            )
            h = output_embeddings + pos_encodings
            h = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, h)
        else:
            h = output_embeddings

        # The causal mask is shared across heads.
        causal_mask = jnp.tril(
            jnp.ones((batch_size, 1, output_sequence_length, output_sequence_length))
        )

        for _ in range(self._config.num_layers):
            self_attention = MultiHeadDotProductAttention(
                num_heads=self._config.num_heads,
                num_hiddens_per_head=self._config.num_hiddens_per_head,
                positional_encodings=self._config.positional_encodings,
                positional_encodings_params=self._config.positional_encodings_params,
                chunk_size=self._config.chunk_size,
            )(inputs_q=h, inputs_kv=h, mask=causal_mask, causal=True)
            self_attention = hk.dropout(
                hk.next_rng_key(), self._config.dropout_prob, self_attention
            )
            self_attention = layer_norm(h + self_attention)

            cross_attention = MultiHeadDotProductAttention(
                num_heads=self._config.num_heads,
                num_hiddens_per_head=self._config.num_hiddens_per_head,
                positional_encodings=self._config.positional_encodings,
                positional_encodings_params=self._config.positional_encodings_params,
                chunk_size=self._config.chunk_size,
            )(inputs_q=self_attention, inputs_kv=encoded, causal=True)
            cross_attention = hk.dropout(
                hk.next_rng_key(), self._config.dropout_prob, cross_attention
            )
            cross_attention = layer_norm(self_attention + cross_attention)

            # Position-wise feedforward network.
            h = hk.Linear(self._config.embedding_dim * self._config.widening_factor)(
                cross_attention
            )
            h = jnn.relu(h)
            h = hk.Linear(self._config.embedding_dim)(h)

            h = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, h)
            h = layer_norm(h + cross_attention)

        return h


class Transformer(hk.Module):
    """Transformer (Vaswani et al., 2017)."""

    def __init__(self, config: TransformerConfig, name: Optional[str] = None):
        """Initializes the Transformer.

        Args:
          config: The hyperparameters used in Transformer architectures.
          name: The name of the module.
        """
        super().__init__(name=name)
        shared_embeddings_fn = None

        if config.share_embeddings:
            shared_embeddings_fn = hk.Linear(
                config.embedding_dim,
                with_bias=False,
                w_init=hk.initializers.TruncatedNormal(stddev=config.emb_init_scale),
                name="shared_embeddings",
            )

        self._encoder = TransformerEncoder(config, shared_embeddings_fn)
        self._decoder = TransformerDecoder(config, shared_embeddings_fn)

    def __call__(self, inputs: chex.Array, targets: chex.Array) -> chex.Array:
        return self._decoder(self._encoder(inputs), targets)


def make_transformer_encoder(
    output_size: int,
    embedding_dim: int,
    dropout_prob: float,
    chunk_size: int,
    num_heads: int,
    positional_encodings: Optional[pos_encs_lib.PositionalEncodings] = None,
    positional_encodings_params: Optional[
        pos_encs_lib.PositionalEncodingsParams
    ] = None,
    return_all_outputs: bool = False,
    share_weight: bool = True,
    use_front_rear_pos: bool = False,
    run_gradient_analysis: bool = False,
    num_layers: Optional[int] = None,
    thickness: Optional[int] = 1,
    emb_init_scale: float = 0.02,
) -> Callable[[chex.Array], chex.Array]:
    """Returns a transformer encoder model."""
    config = TransformerConfig(
        output_size=output_size,
        embedding_dim=embedding_dim,
        dropout_prob=dropout_prob,
        chunk_size=chunk_size,
        num_heads=num_heads,
        positional_encodings=positional_encodings,
        positional_encodings_params=positional_encodings_params,
        share_weight=share_weight,
        use_front_rear_pos=use_front_rear_pos,
        num_layers=num_layers,
        thickness=thickness,
        emb_init_scale=emb_init_scale,
    )
    logging.info(config)

    def transformer_encoder(inputs: chex.Array) -> chex.Array:
        output = TransformerEncoder(
            config, run_gradient_analysis=run_gradient_analysis
        )(inputs)
        if not return_all_outputs:
            output = output[:, -1, :]
        return hk.Linear(output_size)(output)

    return transformer_encoder


def make_transformer(
    output_size: int,
    embedding_dim: int = 64,
    num_layers: int = 5,
    num_heads: int = 8,
    num_hiddens_per_head: Optional[int] = None,
    dropout_prob: float = 0.1,
    emb_init_scale: float = 0.02,
    use_embeddings: bool = True,
    share_embeddings: bool = False,
    chunk_size: Optional[int] = None,
    positional_encodings: Optional[pos_encs_lib.PositionalEncodings] = None,
    positional_encodings_params: Optional[
        pos_encs_lib.PositionalEncodingsParams
    ] = None,
    widening_factor: int = 4,
    return_all_outputs: bool = False,
    causal_masking: bool = False,
) -> Callable[[chex.Array, chex.Array], chex.Array]:
    """Returns a transformer model."""
    if positional_encodings is None:
        positional_encodings = pos_encs_lib.PositionalEncodings.SIN_COS
        positional_encodings_params = pos_encs_lib.SinCosParams()
    elif positional_encodings_params is None:
        raise ValueError("No parameters for positional encodings are passed.")
    config = TransformerConfig(
        output_size=output_size,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_hiddens_per_head=num_hiddens_per_head,
        dropout_prob=dropout_prob,
        emb_init_scale=emb_init_scale,
        use_embeddings=use_embeddings,
        share_embeddings=share_embeddings,
        chunk_size=chunk_size,
        positional_encodings=positional_encodings,
        positional_encodings_params=positional_encodings_params,
        widening_factor=widening_factor,
        causal_masking=causal_masking,
    )

    def transformer(inputs: chex.Array, targets: chex.Array) -> chex.Array:
        output = Transformer(config)(inputs, targets)
        if not return_all_outputs:
            output = output[:, -1, :]
        return hk.Linear(output_size)(output)

    return transformer
