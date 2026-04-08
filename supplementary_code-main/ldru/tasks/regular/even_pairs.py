"""Compute whether the number of 01's and 10's is even."""

import functools
from typing import Mapping

import jax
from jax import nn as jnn
from jax import numpy as jnp
from jax import random as jrandom

from ldru.tasks import task


class EvenPairs(task.GeneralizationTask):
  """A task which goal is to check whether the number of 01's and 10's is even.

  The input is a binary string, composed of 0s and 1s. If the result is even,
  the class is 0, otherwise it's one.

  Examples:
    001110 -> 1 '10' and 1 '01' -> class 0
    0101001 -> 2 '10' and 3 '01' -> class 1

  Note the sampling is jittable so this task is fast.
  """

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> Mapping[str, jnp.ndarray]:
    """Returns a batch of strings and the expected class."""
    strings = jrandom.randint(
        rng,
        shape=(batch_size, length),
        minval=0,
        maxval=2,
    )
    one_hot_strings = jnn.one_hot(strings, num_classes=2)
    unequal_pairs = jnp.logical_xor(strings[:, :-1], strings[:, 1:])
    odd_unequal_pairs = jnp.sum(unequal_pairs, axis=-1) % 2
    return {
        'input': one_hot_strings,
        'output': jnn.one_hot(odd_unequal_pairs, num_classes=self.output_size),
    }

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 2

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2
