"""Script to train a model and save the embeddings (and their tsne projections) for visualization"""
import json
import argparse
import jax
import haiku as hk
import jax.numpy as jnp
import pickle

from ldru.training import constants
from ldru.training import curriculum as curriculum_lib
from ldru.training import training

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

from ldru.utils.monoid_processing import DFA, compute_transition_monoid

import logging
logging.basicConfig(
  format='%(asctime)s %(levelname)-8s %(message)s',
  level=logging.INFO,
  datefmt='%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True, type=str)
parser.add_argument("--seed", default=256, type=int)
parser.add_argument("--mod", default=5, type=int)
parser.add_argument("--n", default=2, type=int)
parser.add_argument("--num_symbols", default=2, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--sequence_length", default=40, type=int)
parser.add_argument("--embedding_dim", default=64, type=int)
parser.add_argument("--num_heads", default=4, type=int)
parser.add_argument("--num_layers", default=1, type=int)
parser.add_argument("--chunk_size", default=2, type=int)
parser.add_argument("--thickness", default=1, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--dropout_prob", default=0.25, type=float)
parser.add_argument("--steps", default=1_000_000, type=int)
parser.add_argument("--architecture", default="ldru", type=str)
parser.add_argument("--optimizer", default="amsgrad", type=str)
parser.add_argument("--max_range_test_length", default=500, type=int)
parser.add_argument("--share_weight", default=False, type=bool,)
parser.add_argument("--operator", default="MLP", type=str, choices=constants.BINARY_OPERATORS.keys())
parser.add_argument("--eval_data_path", default=None,
                    help="Path to pre-generated evaluation data. If None, will generate on the fly.")
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
    if args.task in ['d_n']:
        task = constants.TASK_BUILDERS[args.task](args.n, args.sequence_length)
    else:
        raise ValueError(f"Task {args.task} is not supported in this script.")

    # Create the model.
    if args.architecture in ['rnn', 'lstm']:
        model = constants.MODEL_BUILDERS[args.architecture](
            output_size=task.output_size,
            hidden_size=architecture_params['embedding_dim'],)
        eval_params = architecture_params.copy()
        eval_params['dropout_prob'] = 0.0  # for eval
        eval_model = constants.MODEL_BUILDERS[args.architecture](
            output_size=task.output_size,
            hidden_size=architecture_params['embedding_dim'],)
    else:
        model = constants.MODEL_BUILDERS[args.architecture](
            output_size=task.output_size,
            **architecture_params)
        eval_params = architecture_params.copy()
        eval_params['dropout_prob'] = 0.0  # for eval
        eval_model = constants.MODEL_BUILDERS[args.architecture](
            output_size=task.output_size,
            return_embeddings=True,  # Return embeddings for visualization
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
        log_frequency= 1000,
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
        compute_full_range_test=False, # No evaluation
        max_range_test_length=args.max_range_test_length,
        range_test_total_batch_size=512,
        range_test_sub_batch_size=64,
        architecture_params=architecture_params,
        optimizer=args.optimizer,
        eval_data_path=args.eval_data_path,
        use_tensorboard=False
        )

    training_worker = training.TrainingWorker(training_params, use_tqdm=False)
    _, _, params = training_worker.run()

    with open(f'seq{args.sequence_length}_D{args.n}_s{args.seed}_params.pkl', 'wb') as f:
        pickle.dump(params, f)

    # Visualize the embeddings if available.
    # double n length
    max_length = args.n * 2
    # jax_dfa = DFA.create_balanced_brackets_dfa(args.n)

    def generate_words(alphabet_size, max_length):
        words = []
        
        # Generate words of each length
        for length in tqdm(range(1, max_length + 1, 1)):
            if length == 0:
                # Empty word
                words.append(jnp.array([], dtype=jnp.int32))
            else:
                # Generate all words of this length
                indices = jnp.arange(alphabet_size ** length)
                new_words = []
                
                for idx in indices:
                    # Convert index to base-alphabet_size representation
                    word = []
                    temp_idx = idx
                    for _ in range(length):
                        word.append(temp_idx % alphabet_size)
                        temp_idx //= alphabet_size
                    # Check if the word is valid (balanced brackets)
                    # _, acc, _ = jax_dfa.run_sequence(proposed_word)
                    # if task.belongs_to_lang(proposed_word):
                    new_words.append(jnp.array(word[::-1], dtype=jnp.int32))

                words.extend(new_words)
        
        return words
    
    probe_words = generate_words(args.num_symbols, max_length)

    jax_dfa = DFA.create_balanced_brackets_dfa(args.n)

    equivalence_classes, function_to_class, word_to_class, hash_table = compute_transition_monoid(jax_dfa, max_word_length=args.n * 2, even_only=False)

    model_apply_fn = jax.jit(eval_model.apply)

    def embed_words(words, apply_fn):
        embeddings = []
        classes = []
        for word in tqdm(words):
            # encode word
            word = jnp.array(word, dtype=jnp.int32)
            state_mapping = jax_dfa.get_sequence_function(word).tolist()
            # Do not add annihiliation class
            if state_mapping == [jax_dfa.num_states - 1] * jax_dfa.num_states:
                # all states map to rejecting sink state, skip this word
                continue
            label = function_to_class[tuple(state_mapping)]
            one_hot_word = jax.nn.one_hot(word, num_classes=2, dtype=jnp.float32)
            one_hot_word = jnp.expand_dims(one_hot_word, axis=0)
            _, embedding = apply_fn(params, jax.random.PRNGKey(0), one_hot_word)
            embeddings.append(embedding)
            classes.append(label)
        return jnp.concatenate(embeddings, axis=0), jnp.array(classes)

    # embeddings = embed_words(eval_model, lengths, batch_size=128)
    embeddings, classes = embed_words(probe_words, model_apply_fn)
    # dump to json
    with open(f'seq{args.sequence_length}_D{args.n}_s{args.seed}_embeddings.json', 'w') as f:
        json.dump({'embeddings': embeddings.tolist(), 'classes': classes.tolist()}, f)

    tsne = TSNE(n_components=2, perplexity=15, random_state=42)
    X_embedded = tsne.fit_transform(embeddings)

    plt.figure(figsize=(5, 5))
    unique_classes = np.unique(classes)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_classes)))

    for i, cls in enumerate(unique_classes):
        mask = classes == cls
        plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                    s=2, alpha=0.85, c=[colors[i]], label=f'EC {cls}')

    plt.title('Embedding Visualization')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'seq{args.sequence_length}_D{args.n}_s{args.seed}_tsne.png', bbox_inches='tight')

if __name__ == '__main__':
  main()
