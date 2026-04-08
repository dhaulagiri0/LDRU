"""Evaluation of a network on sequences of different lengths."""
# Source: Delétang et al. (2023)

import os
import pickle
import gzip

import dataclasses
import random
from typing import Any, Callable, Dict, List, Mapping,  Optional

import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from time import time

_Batch = Mapping[str, jnp.ndarray]

def load_evaluation_data(
    data_dir: str,
    length: int = None,
    compress: bool = None
) -> Dict[int, List[_Batch]]:
    """
    Load pre-generated evaluation data.
    
    Args:
        data_dir: Directory containing the saved data
        length: Specific sequence length to load (if None, loads all data)
        compress: Whether data is compressed (auto-detect if None)
    
    Returns:
        Dictionary with length as key and list of batches as value
        If length is specified, returns {length: batches}
        If length is None, returns all data {length1: batches1, length2: batches2, ...}
    """
    # Auto-detect compression if not specified
    if compress is None:
        gz_path = os.path.join(data_dir, "evaluation_data.pkl.gz")
        pkl_path = os.path.join(data_dir, "evaluation_data.pkl")
        
        if os.path.exists(gz_path):
            compress = True
            filepath = gz_path
        elif os.path.exists(pkl_path):
            compress = False
            filepath = pkl_path
        else:
            raise FileNotFoundError(f"No evaluation data file found in {data_dir}")
    else:
        filename = "evaluation_data.pkl"
        if compress:
            filename += ".gz"
        filepath = os.path.join(data_dir, filename)
    
    # Load all data
    if compress:
        with gzip.open(filepath, 'rb') as f:
            all_data = pickle.load(f)
    else:
        with open(filepath, 'rb') as f:
            all_data = pickle.load(f)
    
    # Convert back to JAX arrays
    jax_data = {}
    for data_length, batches in all_data.items():
        jax_batches = []
        for batch in batches:
            jax_batch = {
                key: jnp.array(value) for key, value in batch.items()
            }
            jax_batches.append(jax_batch)
        jax_data[data_length] = jax_batches
    
    # Return specific length or all data
    if length is not None:
        if length not in jax_data:
            raise ValueError(f"Length {length} not found in data. Available lengths: {list(jax_data.keys())}")
        return {length: jax_data[length]}
    
    return jax_data

def load_metadata(data_dir: str) -> Dict[str, Any]:
    """Load metadata about the saved dataset."""
    metadata_path = os.path.join(data_dir, 'metadata.pkl')
    with open(metadata_path, 'rb') as f:
        return pickle.load(f)

@dataclasses.dataclass
class EvaluationParams:
  """The parameters used for range evaluation of networks."""
  model: hk.Transformed
  params: hk.Params

  accuracy_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
  sample_batch: Callable[[jnp.ndarray, int, int], _Batch]

  max_test_length: int
  total_batch_size: int
  sub_batch_size: int  # We use this to avoid memory overflow.

  sample: bool = True
  data_file_path: str = ''

def range_evaluation_sample(
    eval_params: EvaluationParams,
    use_tqdm: bool = False,
    writer: Optional[Any] = None,
) -> List[Mapping[str, Any]]:
  """Evaluates the model on longer, never seen strings and log the results.

  Args:
    eval_params: The evaluation parameters, see above.
    use_tqdm: Whether to use a progress bar with tqdm.

  Returns:
    The list of dicts containing the accuracies.
  """
  init_time = time()
  model = eval_params.model
  params = eval_params.params

  random.seed(1)
  np.random.seed(1)
  rng_seq = hk.PRNGSequence(1)
  apply_fn = jax.jit(model.apply)

  results = []
  lengths = range(1, eval_params.max_test_length + 1)
  if use_tqdm:
    lengths = tqdm.tqdm(lengths)
  for length in lengths:
    # Clear the cache every 32 steps to avoid caching too many apply calls
    if length % 32 == 0:
      jax.clear_caches()
    sub_accuracies = []
  
    for _ in range(eval_params.total_batch_size // eval_params.sub_batch_size):
      batch = eval_params.sample_batch(next(rng_seq), eval_params.sub_batch_size, length)
      outputs = apply_fn(params, next(rng_seq), batch['input'])
      sub_accuracies.append(
          float(np.mean(eval_params.accuracy_fn(outputs, batch['output']))))
      # writer is a tensorboard writer
      if writer != None:
        writer.add_scalar("range_eval/accuracy", float(np.mean(eval_params.accuracy_fn(outputs, batch['output']))), length)
    log_data = {
        'length': length,
        'accuracy': np.mean(sub_accuracies),
    }
    logging.info(log_data)
    results.append(log_data)
  print("Time taken for evaluation:", time() - init_time)
  return results

def range_evaluation_file(
    eval_params: EvaluationParams,
    use_tqdm: bool = False,
    writer: Optional[Any] = None,
) -> List[Mapping[str, Any]]:
  """Evaluates the model on longer, never seen strings using pre-generated data.

  Args:
    eval_params: The evaluation parameters, see above.
    use_tqdm: Whether to use a progress bar with tqdm.
    writer: Optional tensorboard writer.

  Returns:
    The list of dicts containing the accuracies.
  """
  init_time = time()
  model = eval_params.model
  params = eval_params.params

  random.seed(1)
  np.random.seed(1)
  rng_seq = hk.PRNGSequence(1)
  
  max_test_length = eval_params.max_test_length
    
  print(f"Running evaluation on pre-generated data")
  print(f"Data path: {eval_params.data_file_path}")
  print(f"Evaluating lengths 1 to {max_test_length}")
  
  apply_fn = jax.jit(model.apply)

  results = []
  lengths = range(1, max_test_length + 1)
  if use_tqdm:
    lengths = tqdm.tqdm(lengths, desc="Evaluating")

  eval_data = load_evaluation_data(eval_params.data_file_path)
    
  for length in lengths:
    if length % 32 == 0:
        jax.clear_caches()

    try:
      batches = eval_data.get(length, [])
    except (FileNotFoundError, ValueError) as e:
      logging.warning(f"No data found for length {length}, skipping: {e}")
      continue

    sub_accuracies = []
    # Process each pre-generated batch
    for batch in batches:
      outputs = apply_fn(params, next(rng_seq), batch['input'])

      batch_accuracy = float(np.mean(eval_params.accuracy_fn(outputs, batch['output'])))
      sub_accuracies.append(batch_accuracy)
      
      # tensorboard writer 
      if writer != None:
        writer.add_scalar("range_eval/accuracy", np.mean(sub_accuracies), length)
    log_data = {
        'length': length,
        'accuracy': np.mean(sub_accuracies),
    }
    logging.info(log_data)
    results.append(log_data)
  print("Time taken for evaluation:", time() - init_time)
    
  return results
