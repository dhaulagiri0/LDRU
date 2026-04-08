"""Compute whether the input string is Tomita 1."""
# Source: Chi et al. (2023)

import functools
from typing import Mapping

import jax
from jax import nn as jnn
from jax import numpy as jnp
from jax import random as jrandom

from ldru.tasks import task


class Tomita1(task.GeneralizationTask):
  """A task which goal is to check whether the input string is 0*.

  The input is a binary string, composed of 0s and 1s. If they are all 0s,
  the class is 0, otherwise it's one.

  Examples:
    00000 -> class 0
    11011 -> class 1

  Note the sampling is jittable so this task is fast.
  """

  @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                   length: int) -> Mapping[str, jnp.ndarray]:
    """Returns a batch of strings and the expected class."""
    rng1, rng2 = jax.random.split(rng)
    strings = jrandom.randint(
        rng1,
        shape=(batch_size, length),
        minval=0,
        maxval=2,
    )
    strings = strings * jrandom.randint(rng2, shape=(batch_size, 1), minval=0, maxval=2)
    one_hot_strings = jnn.one_hot(strings, num_classes=2)
    ans = (jnp.sum(strings, axis=1)==0).astype(jnp.int32)
    return {
        'input': one_hot_strings,
        'output': jnn.one_hot(ans, num_classes=self.output_size),
    }

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 2

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2

