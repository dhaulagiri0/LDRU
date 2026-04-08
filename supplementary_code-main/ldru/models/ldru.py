# Our LDRU architecture

from typing import Optional, Callable

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jax.nn as jnn
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
    # The probability that each element is discarded by the dropout modules.
    dropout_prob: float = 0.0
    # The parameter initialization scale for the embeddings.
    emb_init_scale: float = 0.02
    # Whether to use the embeddings rather than raw inputs.
    use_embeddings: bool = True
    # How much larger the hidden layer of the feedforward network should be
    # compared to the `embedding_dim`.
    widening_factor: int = 4
    # The type of operator to use in the LDRU
    operator: Optional[Callable] = None


class BinaryOperator(hk.Module):
    """Haiku implementation of BinaryOperator."""

    def __init__(self, embedding_size: int, activation: Callable = jax.nn.relu):
        super().__init__(name="BinaryOperator")
        self.activation = activation

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        hidden_dim = x.shape[-1]

        init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")

        concat = jnp.concatenate([x, y], axis=-1)

        weights = hk.Linear(hidden_dim * 2, w_init=init)(concat)
        weights = self.activation(weights)
        weights = hk.Linear(hidden_dim * 4, w_init=init)(weights)
        weights = self.activation(weights)
        weights = hk.Linear(hidden_dim * 2, w_init=init)(weights)
        wx, wy = jnp.split(weights, 2, axis=-1)

        x_prime = wx * x
        y_prime = wy * y

        x_prime = hk.Linear(hidden_dim, w_init=hk.initializers.Identity())(x_prime)
        y_prime = hk.Linear(hidden_dim, w_init=hk.initializers.Identity())(y_prime)

        return hk.Linear(hidden_dim, w_init=hk.initializers.Identity())(
            x_prime + y_prime
        )


class ElementwiseSum(hk.Module):
    """Elementwise combination of two inputs."""

    def __init__(self, embedding_size: int, activation: Callable = jax.nn.relu):
        super().__init__(name="ElementwiseSum")

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return x + y


class LinearCat(hk.Module):
    """Linear projection of concatenation of two inputs."""

    def __init__(self, embedding_size: int, activation: Callable = jax.nn.relu):
        super().__init__(name="LinearCat")

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Concatenate x and y, then apply a linear transformation."""
        concat = jnp.concatenate([x, y], axis=-1)
        return hk.Linear(
            x.shape[-1],
            w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal"),
        )(concat)


class GatedSum(hk.Module):
    """Gated sum of two inputs."""

    def __init__(self, embedding_size: int, activation: Callable = jax.nn.sigmoid):
        super().__init__(name="GatedSum")
        self.activation = activation

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([x, y], axis=-1)
        gate = hk.Linear(
            x.shape[-1],
            w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal"),
        )(concat)
        gate = self.activation(gate)
        return gate * x + (1 - gate) * y


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

    def __call__(self, h):

        sequence_length = h.shape[1]
        num_adaptive_layers = int(np.ceil(np.log2(sequence_length)))
        # Embedding is normalized then has a residual FFN
        h = self.init_norm(h)
        res = h.copy()
        h = hk.Linear(self._config.embedding_dim * self._config.widening_factor)(h)
        h = jnn.relu(h)
        h = hk.Linear(self._config.embedding_dim)(h) + res
        h = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, h)

        def adaptive_scan(h):
            def body_fun(i, h):

                window_size = 2 ** (i + 1)
                indices = jnp.arange(sequence_length)
                max_windows = sequence_length

                def get_end_mask(size, length):
                    indices = jnp.arange(length)
                    remainder = indices % size
                    mask = (remainder == size - 1) & (indices < length)
                    mask = mask.at[length - 1].set(True)
                    return mask

                def process_window(window_idx):
                    is_valid_window = (
                        window_idx < (sequence_length + window_size - 1) // window_size
                    )
                    start_idx = window_idx * window_size
                    end_idx = jnp.minimum(
                        start_idx + window_size - 1, sequence_length - 1
                    )
                    mid_idx = start_idx + ((window_size - 1) // 2)

                    end_mask = (indices == end_idx) & is_valid_window
                    mid_mask = (
                        (indices == mid_idx) & is_valid_window & (end_idx != mid_idx)
                    )

                    mid_tokens = jnp.sum(h * mid_mask[None, :, None], axis=1)
                    end_tokens = jnp.sum(h * end_mask[None, :, None], axis=1)

                    # Each application of \odot_\theta
                    window_update = self.binary_operator(mid_tokens, end_tokens)
                    return window_update[:, None, :] * end_mask[None, :, None]

                updates = jnp.sum(
                    jax.vmap(process_window)(jnp.arange(max_windows)), axis=0
                )
                end_token_mask = get_end_mask(window_size, sequence_length)

                h = h * (1 - end_token_mask[None, :, None]) + updates
                # Residual connection
                h = self.fc(h) + h
                # Layer normalization and dropout
                h = self.inner_norm(h)
                h = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, h)

                return h

            h = hk.fori_loop(0, num_adaptive_layers, body_fun, h)
            return h

        # Scan algorithm to apply the operator across the sequences until it reaches a single embedding
        h = adaptive_scan(h)

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
        h = embeddings

        layers = []

        for _ in range(self._config.num_layers):
            layers.append(LDRU(self._config))

        for layer in layers:
            h = layer(h)

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
            output = output[:, -1, :]

        prediction = hk.Linear(output_size)(output)

        if return_embeddings:
            return prediction, output

        return prediction

    return ldru
