"""Compute the unique state after the first p symbols of a string."""

import functools
from typing import Mapping

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from ldru.tasks import task

class NPrefixSymbols(task.GeneralizationTask):
  """Goal to remember the first N symbols of a string
  """
  def __init__(self, num_symbols: int = 2, prefix_length: int = 1, **kwargs):
    super().__init__()
    self.num_symbols = num_symbols
    self.prefix_length = prefix_length
    self.invalid_class_idx = self.num_symbols ** self.prefix_length

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> Mapping[str, jnp.ndarray]:
    """Returns a batch of strings and the expected class."""
    sequences = jrandom.randint(
       rng, shape=(batch_size, length), minval=0, maxval=self.num_symbols)
    
    # Simple check if length is less than prefix_length
    if length < self.prefix_length:
      # If the length is less than the prefix length, all labels are the invalid class
      label_indices = jnp.full((batch_size,), self.invalid_class_idx)
    else:
      # Extract the prefix of each sequence
      prefix_sequences = sequences[:, :self.prefix_length]
      # Flatten the prefixes to create unique labels
      multiplier = jnp.power(
          self.num_symbols, 
          jnp.arange(self.prefix_length - 1, -1, -1)
      )

      label_indices = jnp.sum(prefix_sequences * multiplier, axis=1)

    
    # One-hot encode the input sequences
    one_hot_strings = jnn.one_hot(sequences, num_classes=self.num_symbols)
    
    labels = jnn.one_hot(label_indices, num_classes=self.output_size)
    
    return {"input": one_hot_strings, "output": labels}

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return self.num_symbols

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    # Output size is num_symbols^prefix_length + 1 (for the invalid class)
    return (self.num_symbols ** self.prefix_length) + 1
  

if __name__ == "__main__":
    # Example usage
    task_instance = NPrefixSymbols(num_symbols=2, prefix_length=8)
    rng = jax.random.PRNGKey(0)
    batch_size = 8
    length = 40
    for l in range(30, length+1):
        print(f"Length: {l}")
        batch = task_instance.sample_batch(rng, batch_size, l)
        # print("Input shape:", batch["input"].shape)
        print("Input:", batch["input"].argmax(-1))
        # print("Output shape:", batch["output"].shape)
        print("Output:", batch["output"].argmax(-1))
