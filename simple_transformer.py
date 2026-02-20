"""
Simple transformer implementation for integration with LDRU training script.
Based on the transformer from supplementary_code-main/ldru/models/transformer.py
"""

import haiku as hk
import jax
import jax.numpy as jnp
import jax.nn as jnn
from typing import Optional


def create_transformer_model(config):
    """Create a simple transformer encoder model."""

    def transformer_forward(token_ids):
        batch_size, seq_length = token_ids.shape

        # Embedding layer
        embeddings = hk.Embed(config.vocab_size, config.embedding_dim)(token_ids)
        embeddings *= jnp.sqrt(config.embedding_dim)

        # Add positional encoding (simple sinusoidal)
        positions = jnp.arange(seq_length)[None, :, None]
        pos_encoding = get_positional_encoding(positions, config.embedding_dim)
        x = embeddings + pos_encoding

        # Apply dropout to embeddings
        x = hk.dropout(hk.next_rng_key(), 0.1, x)

        # Multi-layer transformer
        for layer in range(config.num_layers):
            x = transformer_layer(x, config, layer_idx=layer)

        # Output projection
        logits = hk.Linear(config.vocab_size)(x)
        return logits

    return hk.transform(transformer_forward)


def get_positional_encoding(positions, embedding_dim):
    """Generate sinusoidal positional encodings."""
    # positions shape: [batch, seq_len, 1]
    # Create encoding for each dimension
    half_dim = embedding_dim // 2

    # Create frequency basis
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half_dim) / half_dim)

    # Apply frequencies to positions
    angles = positions * freqs[None, None, :]

    # Create sin and cos encodings
    sin_enc = jnp.sin(angles)
    cos_enc = jnp.cos(angles)

    # Concatenate sin and cos to get full embedding dimension
    pos_encoding = jnp.concatenate([sin_enc, cos_enc], axis=-1)

    # Handle odd embedding dimensions
    if embedding_dim % 2 == 1:
        pos_encoding = pos_encoding[..., :-1]

    return pos_encoding


def transformer_layer(x, config, layer_idx):
    """Single transformer layer with self-attention and feed-forward."""

    # Multi-head self-attention
    attn_output = multi_head_attention(
        x,
        x,
        x,
        num_heads=getattr(config, "num_heads", 8),
        embedding_dim=config.embedding_dim,
        causal=True,
    )

    # Add & norm for attention
    x = layer_norm(x + attn_output)

    # Feed-forward network
    ff_output = feed_forward(x, config.embedding_dim)

    # Add & norm for feed-forward
    x = layer_norm(x + ff_output)

    return x


def multi_head_attention(queries, keys, values, num_heads, embedding_dim, causal=True):
    """Multi-head self-attention mechanism."""

    head_dim = embedding_dim // num_heads

    # Linear projections
    q = hk.Linear(embedding_dim, with_bias=False, name="query")(queries)
    k = hk.Linear(embedding_dim, with_bias=False, name="key")(keys)
    v = hk.Linear(embedding_dim, with_bias=False, name="value")(values)

    batch_size, seq_len = q.shape[:2]

    # Reshape to [batch, seq_len, num_heads, head_dim]
    q = q.reshape(batch_size, seq_len, num_heads, head_dim)
    k = k.reshape(batch_size, seq_len, num_heads, head_dim)
    v = v.reshape(batch_size, seq_len, num_heads, head_dim)

    # Compute attention scores
    # [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, seq_len]
    scores = jnp.einsum("bthd,bThd->bhtT", q, k) / jnp.sqrt(head_dim)

    # Apply causal mask
    if causal:
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        scores = jnp.where(mask, scores, -jnp.inf)

    # Softmax to get attention weights
    attn_weights = jnn.softmax(scores, axis=-1)

    # Apply attention to values
    # [batch, num_heads, seq_len, seq_len] @ [batch, num_heads, seq_len, head_dim]
    attn_output = jnp.einsum("bhtT,bThd->bthd", attn_weights, v)

    # Reshape back to [batch, seq_len, embedding_dim]
    attn_output = attn_output.reshape(batch_size, seq_len, embedding_dim)

    # Final linear projection
    return hk.Linear(embedding_dim, with_bias=False, name="output")(attn_output)


def feed_forward(x, embedding_dim):
    """Position-wise feed-forward network."""
    hidden_dim = embedding_dim * 4  # Standard transformer scaling

    # Two linear layers with ReLU activation
    h = hk.Linear(hidden_dim)(x)
    h = jnn.relu(h)
    h = hk.dropout(hk.next_rng_key(), 0.1, h)
    return hk.Linear(embedding_dim)(h)


def layer_norm(x):
    """Layer normalization."""
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
