"""Compute whether the number of 1s in a string is even."""

import functools
from typing import Mapping

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from ldru.tasks import task


class ParityCheck(task.GeneralizationTask):
  """A task which goal is to count the number of '1' in a string, modulo 2.

  The input is a string, composed of 0s and 1s. If the result is even, the class
  is 0, otherwise it's 1.

  Examples:
    1010100 -> 3 1s (odd) -> class 1
    01111 -> 4 1s (even) -> class 0

  Note that the sampling is jittable so this task is fast.
  """

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> Mapping[str, jnp.ndarray]:
    """Returns a batch of strings and the expected class."""
    #strings = jrandom.randint(
    #    rng, shape=(batch_size, length), minval=0, maxval=2)
    strings = jrandom.bernoulli(
        rng, p=0.5, shape=(batch_size, length)).astype(jnp.int32)
    n_b = jnp.sum(strings, axis=1) % 2
    n_b = jnn.one_hot(n_b, num_classes=2)
    one_hot_strings = jnn.one_hot(strings, num_classes=2)
    return {"input": one_hot_strings, "output": n_b}

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 2

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2
