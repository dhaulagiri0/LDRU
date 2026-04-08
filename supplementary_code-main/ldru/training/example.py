"""Example script to train and evaluate a network."""

import argparse

import haiku as hk
import jax.numpy as jnp
import numpy as np
import warnings

# Suppress the JAX complex casting warning (S5 model)
warnings.filterwarnings("ignore", category=jnp.ComplexWarning)

from ldru.training import constants
from ldru.training import curriculum as curriculum_lib
from ldru.training import training

import logging
logging.basicConfig(
  format='%(asctime)s %(levelname)-8s %(message)s',
  level=logging.INFO,
  datefmt='%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True, type=str)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--mod", default=5, type=int)
parser.add_argument("--n", default=2, type=int)
parser.add_argument("--num_symbols", default=2, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--sequence_length", default=40, type=int)
parser.add_argument("--embedding_dim", default=64, type=int)
parser.add_argument("--num_heads", default=4, type=int)
parser.add_argument("--num_t_matrices", default=8, type=int)
parser.add_argument("--num_layers", default=1, type=int)
parser.add_argument("--chunk_size", default=2, type=int)
parser.add_argument("--thickness", default=1, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--dropout_prob", default=0.25, type=float)
parser.add_argument("--steps", default=1_000_000, type=int)
parser.add_argument("--architecture", default="ldru", type=str)
parser.add_argument("--optimizer", default="amsgrad", type=str)
parser.add_argument("--max_range_test_length", default=500, type=int)
parser.add_argument("--share_weight", default=False, type=bool)
parser.add_argument("--operator", default="MLP", type=str, choices=constants.BINARY_OPERATORS.keys())
parser.add_argument("--eval_data_path", default=None,
                    help="Path to pre-generated evaluation data. If None, will generate on the fly.")
parser.add_argument("--use_tensorboard", default=False, type=bool)
parser.add_argument("--validate", default=False, type=bool)
args = parser.parse_args()
logging.info(args)

def main() -> None:
  
  architecture_params = {
    'embedding_dim': args.embedding_dim,
    'dropout_prob': args.dropout_prob,
    'positional_encodings': None,
    'positional_encodings_params': None,
    'num_heads': args.num_heads,
    'share_weight': args.share_weight,
    'use_front_rear_pos': False,
    'num_layers': None if args.num_layers == 0 else args.num_layers,
    'emb_init_scale': 0.02,
    'chunk_size': args.chunk_size,
    'thickness': args.thickness,
  }
  if args.architecture in ['ldru']:
    architecture_params['binary_operator'] = constants.BINARY_OPERATORS[args.operator]
  
  # Create the task.
  curriculum = curriculum_lib.UniformCurriculum(
      values=list(range(1, args.sequence_length + 1)))
  
  # Instantiate task (some depend on different parameters)
  if args.task in ['modular_arithmetic']:
    task = constants.TASK_BUILDERS[args.task](args.mod)
  elif args.task in ['d_n']:
    task = constants.TASK_BUILDERS[args.task](args.n, args.sequence_length)
  elif args.task in ['n_prefix_symbols']:
    task = constants.TASK_BUILDERS[args.task](args.num_symbols, args.n) # n is prefix length
  else:
    task = constants.TASK_BUILDERS[args.task]()

  # Instantiate model and evaluation model (i.e., no dropout)
  if args.architecture in ['rnn', 'lstm']:
    model = constants.MODEL_BUILDERS[args.architecture](
        output_size=task.output_size,
        hidden_size=architecture_params['embedding_dim'],)
    
    eval_params = architecture_params.copy()
    eval_params['dropout_prob'] = 0.0
    eval_model = constants.MODEL_BUILDERS[args.architecture](
        output_size=task.output_size,
        hidden_size=architecture_params['embedding_dim'],)
    
  elif args.architecture in ['sdssm']:
    model = constants.MODEL_BUILDERS[args.architecture](
        output_size=task.output_size,
        state_size=architecture_params['embedding_dim'],
        num_transition_matrices=args.num_t_matrices)
    
    eval_params = architecture_params.copy()
    eval_params['dropout_prob'] = 0.0 
    eval_model = constants.MODEL_BUILDERS[args.architecture](
        output_size=task.output_size,
        state_size=architecture_params['embedding_dim'],
        num_transition_matrices=args.num_t_matrices)
    
  else:
    model = constants.MODEL_BUILDERS[args.architecture](
        output_size=task.output_size,
        **architecture_params)
    
    eval_params = architecture_params.copy()
    eval_params['dropout_prob'] = 0.0  # for eval
    eval_model = constants.MODEL_BUILDERS[args.architecture](
        output_size=task.output_size,
        **eval_params)

  model = hk.transform(model)
  eval_model = hk.transform(eval_model)

  # Create the loss and accuracy based on the pointwise ones.
  def loss_fn(output, target):
    loss = jnp.mean(jnp.sum(task.pointwise_loss_fn(output, target), axis=-1))
    return loss, {}

  def accuracy_fn(output, target):
    mask = task.accuracy_mask(target)
    return jnp.sum(mask * task.accuracy_fn(output, target)) / jnp.sum(mask)

  # Create the final training parameters.
  training_params = training.ClassicTrainingParams(
      seed=0,
      model_init_seed=args.seed,
      training_steps=args.steps,
      log_frequency=1000,
      l2_lambda=1e-3,
      length_curriculum=curriculum,
      batch_size=args.batch_size,
      task=task,
      model=model,
      architecture=args.architecture,
      task_str=args.task,
      eval_model=eval_model,
      loss_fn=loss_fn,
      learning_rate=args.lr,
      accuracy_fn=accuracy_fn,
      compute_full_range_test=True,
      max_range_test_length=args.max_range_test_length,
      range_test_total_batch_size=512,
      range_test_sub_batch_size=64,
      architecture_params=architecture_params,
      optimizer=args.optimizer,
      eval_data_path=args.eval_data_path,
      use_tensorboard=args.use_tensorboard,
      validate=args.validate
  )
  
  training_worker = training.TrainingWorker(training_params, use_tqdm=False)
  _, eval_results, _ = training_worker.run()

  # Gather results and print final score.
  accuracies = [r['accuracy'] for r in eval_results]
  score = np.mean(accuracies[args.sequence_length + 1:])
  print(f'OOD accuracy: {score}')

if __name__ == '__main__':
  main()
