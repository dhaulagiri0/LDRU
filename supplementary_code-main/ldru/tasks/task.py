"""Base class for generalization tasks."""
# Source: Delétang et al. (2023)

import abc
from typing import TypedDict

import chex
import jax.nn as jnn
import jax.numpy as jnp

Batch = TypedDict('Batch', {'input': chex.Array, 'output': chex.Array})


class GeneralizationTask(abc.ABC):
  """A task for the generalization project.

  Exposes a sample_batch method, and some details about input/output sizes,
  losses and accuracies.
  """

  @abc.abstractmethod
  def sample_batch(self, rng: chex.PRNGKey, batch_size: int,
                   length: int) -> Batch:
    """Returns a batch of inputs/outputs."""

  def pointwise_loss_fn(self, output: chex.Array,
                        target: chex.Array) -> chex.Array:
    """Returns the pointwise loss between an output and a target."""
    return -target * jnn.log_softmax(output)

  def accuracy_fn(self, output: chex.Array, target: chex.Array) -> chex.Array:
    """Returns the accuracy between an output and a target."""
    return (jnp.argmax(output,
                       axis=-1) == jnp.argmax(target,
                                              axis=-1)).astype(jnp.float32)

  def accuracy_mask(self, target: chex.Array) -> chex.Array:
    """Returns a mask to compute the accuracies, to remove the superfluous ones."""
    # Target is a shape of shape (B, T, C) where C is the number of classes.
    # We want a mask per input (B, T), so we take this shape.
    return jnp.ones(target.shape[:-1])

  @property
  @abc.abstractmethod
  def input_size(self) -> int:
    """Returns the size of the input of the models trained on this task."""

  @property
  @abc.abstractmethod
  def output_size(self) -> int:
    """Returns the size of the output of the models trained on this task."""

  def output_length(self, input_length: int) -> int:
    """Returns the length of the output, given an input length."""
    del input_length
    return 1
