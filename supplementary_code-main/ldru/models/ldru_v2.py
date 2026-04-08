from typing import Any, Optional, Callable

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrandom
import numpy as np
import logging


@chex.dataclass
class LDRUConfig:
    """Hyperparameters used in the Transformer architectures."""

    # The size of the model output (i.e., the output vocabulary size).
    output_size: int
    # The dimension of the first embedding.
    embedding_dim: int
    # The number of heads per layer.
    num_layers: Optional[int] = 1
    # number of different layers in one transformer layer
    thickness: Optional[int] = 1
    # The number of hidden neurons per head. If None, it is set to be equal to
    # `embedding_dim // num_heads`.
    num_hiddens_per_head: Optional[int] = None
    # The probability that each element is discarded by the dropout modules.
    dropout_prob: float = 0.0
    # The parameter initialization scale for the embeddings.
    emb_init_scale: float = 0.02
    # Whether to use the embeddings rather than raw inputs.
    use_embeddings: bool = True
    # Whether to share embeddings between the Encoder and the Decoder.
    share_embeddings: bool = False
    # The size of the sliding attention window. See MultiHeadDotProductAttention.
    chunk_size: Optional[int] = None
    # The maximum size of the context (used by the posiitonal encodings).
    max_time: int = 10_000
    # How much larger the hidden layer of the feedforward network should be
    # compared to the `embedding_dim`.
    widening_factor: int = 4
    # Add mask to make causal predictions.
    causal_masking: bool = False
    # Share transformer weight across layers
    share_weight: bool = False
    # Use our special front rear shared positional embeddings
    use_front_rear_pos: bool = False

    # The type of operator to use in the LDRU
    operator: Optional[Callable] = None


class BinaryOperator(hk.Module):
    """Haiku implementation of BinaryOperator that takes (..., 2, E) without concat/split."""

    def __init__(self, embedding_size: int, activation: Callable = jax.nn.relu):
        super().__init__(name="BinaryOperator")
        self.activation = activation

    def __call__(self, xy: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            xy: Tensor of shape (..., 2, E).
        Returns:
            Tensor of shape (..., E).
        """
        hidden_dim = xy.shape[-1]  # E
        init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")

        # Flatten pair dim for the MLP: (..., 2E)
        pair = jnp.reshape(xy, xy.shape[:-2] + (2 * hidden_dim,))

        # Same MLP as before, now on the flattened pair
        weights = hk.Linear(2 * hidden_dim, w_init=init)(pair)
        weights = self.activation(weights)
        weights = hk.Linear(4 * hidden_dim, w_init=init)(weights)
        weights = self.activation(weights)
        weights = hk.Linear(2 * hidden_dim, w_init=init)(weights)

        # Reshape back to (..., 2, E) and scale both branches at once
        weights = jnp.reshape(weights, xy.shape)  # (..., 2, E)
        scaled = weights * xy  # (..., 2, E)

        # Keep separate linear layers per branch (same as original semantics)
        x_prime = hk.Linear(hidden_dim, w_init=hk.initializers.Identity())(
            scaled[..., 0, :]
        )
        y_prime = hk.Linear(hidden_dim, w_init=hk.initializers.Identity())(
            scaled[..., 1, :]
        )

        return hk.Linear(hidden_dim, w_init=hk.initializers.Identity())(
            x_prime + y_prime
        )


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    """Apply a unique LayerNorm to x with default settings."""
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)


class LDRU(hk.Module):
    def __init__(self, config):
        super().__init__(name=None)
        self._config = config
        self.inner_norm = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="inner_norm"
        )
        self.init_norm = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="init_norm"
        )
        self.binary_operator = config.operator(self._config.embedding_dim)

        self.wq = hk.Linear(self._config.embedding_dim, with_bias=False)
        init = hk.initializers.VarianceScaling(
            config.emb_init_scale, "fan_avg", "truncated_normal"
        )
        self.fc_0 = hk.Linear(
            self._config.embedding_dim * self._config.widening_factor, w_init=init
        )
        self.fc_1 = hk.Linear(self._config.embedding_dim, w_init=init)
        self.fc = hk.Sequential([self.fc_0, jnn.relu, self.fc_1])

    def __call__(self, h, causal_mask):

        B, L, E = h.shape
        num_adaptive_layers = int(np.ceil(np.log2(L)))
        max_size = 2**num_adaptive_layers
        mask = jnp.arange(max_size) < L
        mask = jnp.reshape(mask, (1, max_size, 1))  # [1, max_size, 1] for broadcasting
        if L < max_size:
            # Pad to the next power of 2
            pad_width = ((0, 0), (0, max_size - L), (0, 0))  # only pad length dimension
            h = jnp.pad(h, pad_width, mode="constant", constant_values=0)
            L = max_size

        h = self.init_norm(h)
        res = h.copy()
        h = hk.Linear(self._config.embedding_dim * self._config.widening_factor)(h)
        h = jnn.relu(h)
        h = hk.Linear(self._config.embedding_dim)(h) + res
        h = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, h)

        def adaptive_scan(h):
            def body_fun(i, h):
                # needs_pad = (active_len & 1) == 1
                pairs = jnp.reshape(h, (B, L // 2, 2, E))  # [B, L // 2, 2, E]
                updates = self.binary_operator(pairs)

                h_next = jnp.concatenate([updates, jnp.zeros_like(updates)], axis=1)
                # per-layer block
                h_next = self.fc(h_next) + h_next
                h_next = self.inner_norm(h_next)
                h_next = hk.dropout(
                    hk.next_rng_key(), self._config.dropout_prob, h_next
                )
                return h_next

            h = hk.fori_loop(0, num_adaptive_layers, body_fun, h)
            # for i in range(num_adaptive_layers):
            #         h = body_fun(i, h)
            return h

        if L > 1:
            h = adaptive_scan(h)

        print(f"Final h shape before trimming: {h.shape}")
        return h


class LDRUEncoder(hk.Module):
    """Transformer Encoder (Vaswani et al., 2017)."""

    def __init__(
        self,
        config: LDRUConfig,
        name: Optional[str] = None,
    ) -> None:
        """Initializes the transformer encoder.

        Args:
          config: The hyperparameters used in Transformer architectures.
          name: The name of the module.
        """
        super().__init__(name=name)
        self._config = config

    def __call__(self, x: jnp.ndarray) -> chex.Array:
        """Returns the transformer encoder output, shape [B, T, E]."""
        if self._config.use_embeddings:
            # Since `x` is one-hot encoded, using hk.Linear is equivalent to
            # hk.Embed with hk.EmbedLookupStyle.ONE_HOT.
            embs_init = hk.initializers.TruncatedNormal(
                stddev=self._config.emb_init_scale
            )
            embeddings = hk.Linear(
                self._config.embedding_dim, with_bias=False, w_init=embs_init
            )(x)

        else:
            embeddings = x

        batch_size, sequence_length, embedding_size = embeddings.shape

        h = embeddings

        # The causal mask is shared across heads.
        if self._config.causal_masking:
            causal_mask = jnp.tril(
                jnp.ones((batch_size, 1, sequence_length, sequence_length))
            )
        else:
            causal_mask = None

        layers = []

        for _ in range(self._config.num_layers):
            layers.append(LDRU(self._config))

        for layer in layers:
            h = layer(h, causal_mask)

        return h


def make_ldru(
    output_size: int,
    embedding_dim: int,
    return_all_outputs: bool = False,
    num_layers: Optional[int] = 1,
    binary_operator: hk.Module = BinaryOperator,
    return_embeddings: bool = False,
    **kwargs,
) -> Callable[[chex.Array], chex.Array]:
    """Returns a LDRU encoder model."""
    config = LDRUConfig(
        output_size=output_size,
        embedding_dim=embedding_dim,
        dropout_prob=kwargs["dropout_prob"] if "dropout_prob" in kwargs else 0.0,
        num_layers=num_layers,
        operator=binary_operator,
    )
    logging.info(config)

    def ldru(inputs: chex.Array) -> chex.Array:
        output = LDRUEncoder(config)(inputs)

        if not return_all_outputs:
            output = output[:, 0, :]

        prediction = hk.Linear(output_size)(output)

        if return_embeddings:
            return prediction, output
        return prediction

    return ldru


if __name__ == "__main__":
    # Parameters for a toy run
    B, L, E = 256, 40, 10  # batch size, sequence length, input vocab size
    output_size = 5  # e.g. logits over 4 classes
    embedding_dim = 64

    # Fake one-hot input: shape [B, L, E]
    rng = np.random.default_rng(0)
    indices = rng.integers(low=0, high=E, size=(B, L))
    x_np = np.eye(E, dtype=np.float32)[indices]  # [B, L, E]
    x = jnp.asarray(x_np)

    # Build model
    def forward_fn(x):
        return make_ldru(
            output_size=output_size,
            embedding_dim=embedding_dim,
            num_layers=1,
            dropout_prob=0.0,
        )(x)

    forward = hk.transform(forward_fn)

    # Initialize
    rng_key = jax.random.PRNGKey(42)
    params = forward.init(rng_key, x)

    # Apply
    outputs = forward.apply(params, rng_key, x)
    print("Input shape :", x.shape)
    print("Output shape:", outputs.shape)
    print("Output sample:\n", outputs)
