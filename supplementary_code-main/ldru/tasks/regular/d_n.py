"""Compute whether the input string is D_n."""

import argparse
import functools
from typing import Mapping

import jax
from jax import numpy as jnp
import jax.nn as jnn

from ldru.tasks import task


class D_n(task.GeneralizationTask):
    """A task which goal is to check whether the input string is D_n.

    The language D_n is defined as the set of strings with balanced parentheses up to depth n.
    Inductively defined as D_1 = (01)* and D_n = (0D_{n-1}1)*.
    
    D_n strings are the same as Dyck-1 strings up to depth n.
    """
    def __init__(self, n = 2, training_length = 40):
        self.sigma = jnp.array([0., 1.])
        self.n = n
        self.training_length = training_length

    def generate_balanced_sequences(self, key, num, length, max_depth):
        if length % 2 != 0:
            return None
        initial = jnp.zeros((num, length))
        keys = jax.random.split(key, num)
        
        depths = jnp.ones(num) * max_depth
        n = length // 2

        def generate_one_sequence(key, result, m_depth):
            def step_fn(carry, idx):
                key, result, r, opens_used, consecutive_same, remaining = carry

                # Calculate remaining positions and required closings
                remaining_positions = length - idx
                min_required_closes = r  # Must close all currently open parentheses
                remaining_opens_allowed = n - opens_used  # Can still open this many

                # Get previous action (or default to 0 for the first position)
                prev_action = jnp.where(idx > 0, result[idx-1], 0)
                
                # Calculate base probability of closing
                prob_close_val = jax.lax.cond(
                    r > 0,
                    lambda: (r * (r + remaining_positions + 2)) / (2 * remaining_positions * (r + 1)),
                    lambda: 0.0
                )

                key, subkey1, subkey2 = jax.random.split(key, 3)

                # Same essential constraints as before
                must_open = (r == 0) & (remaining_opens_allowed > 0)
                must_close = (r >= m_depth) | (remaining_opens_allowed == 0)
                must_open = must_open | ((remaining_positions > min_required_closes) & 
                                        (remaining_positions - min_required_closes == remaining_opens_allowed))
                can_open = (r < m_depth) & (remaining_opens_allowed > 0)
                can_close = (r > 0)
                would_need_after_open = r + 1
                must_close = must_close | ((remaining_positions - 1) < would_need_after_open)
                
                depth_ratio = r / m_depth
                depth_bias = 0.1 * depth_ratio  
                
                # Add some noise for exploration
                epsilon = 0.15 * jax.random.normal(subkey2)

                prob_close_val = jax.lax.cond(
                    length <= self.training_length, 
                    lambda: jnp.clip(prob_close_val + epsilon - depth_bias, 0.0, 1.0), 
                    lambda: jnp.clip(prob_close_val, 0.0, 1.0)
                )

                random_val = jax.random.uniform(subkey1)
                should_close = jnp.logical_and(
                    jnp.logical_and(~must_open, can_close),
                    (random_val < prob_close_val) | must_close
                )

                # Default action is to open if allowed
                action = jnp.where(
                    jnp.logical_or(must_close, should_close), 
                    1,  # Close
                    jnp.where(can_open, 0, 1)  # Open if can, otherwise close
                )

                # Update result array
                result = result.at[idx].set(action)

                # Update consecutive counter
                consecutive_same = jnp.where(action == prev_action, consecutive_same + 1, 1)

                # Update state
                r_new = r + (1 - 2 * action)  # +1 for open, -1 for close
                opens_used_new = opens_used + (1 - action)  # Increment if we opened

                return (key, result, r_new, opens_used_new, consecutive_same, remaining_positions - 1), None

            # Initialize state: (key, result, r, opens_used, consecutive_same, remaining_positions)
            initial_state = (key, result, 0, 0, 0, length)

            # Generate the string while maintaining validity constraints
            position_indices = jnp.arange(length)
            (_, result, final_r, finalopens, _, _), _ = jax.lax.scan(
                step_fn,
                initial_state,
                position_indices
            )

            return result

        return jax.vmap(generate_one_sequence)(keys, initial, depths)

    def belongs_to_lang(self, string):
        depth = self.find_depth(string)
        return (depth <= self.n) and (depth != -1)
    
    def find_depth(self, string: jnp.ndarray) -> jnp.ndarray:
        def scan_fn(carry, char):
            max_depth, current_depth, is_valid = carry
            
            # Update current depth based on character
            # 0 = open paren (+1), 1 = close paren (-1)
            delta = 1 - 2 * char
            new_current_depth = current_depth + delta
            
            # Check for invalid closing (negative depth)
            invalid_closing = new_current_depth < 0
            
            # Update maximum depth
            new_max_depth = jnp.maximum(max_depth, new_current_depth)
            
            # Update validity flag
            new_is_valid = is_valid & (~invalid_closing)
            
            return (new_max_depth, new_current_depth, new_is_valid), None
        
        init_carry = (0, 0, True)
        
        # Scan through the string
        (max_depth, final_depth, is_valid), _ = jax.lax.scan(
            scan_fn,
            init_carry,
            string
        )
        
        # String is valid if is_valid is True and final_depth is 0
        final_is_valid = is_valid & (final_depth == 0)
        
        return jnp.where(final_is_valid, max_depth, -1)
    
    def find_all_depths(self, string: jnp.ndarray) -> jnp.ndarray:
        def scan_fn(carry, char):
            max_depth, current_depth, is_valid = carry
            
            # If already invalid, propagate -1 state
            new_depth = jnp.where(is_valid, current_depth, -1)
            
            # Only update depth if still valid
            # 0 = open paren (+1), 1 = close paren (-1)
            delta = 1 - 2 * char
            new_current_depth = new_depth + delta * is_valid
            
            # Check for invalid closing (negative depth)
            invalid_closing = new_current_depth < 0
            invalid_opening = new_current_depth > self.n
            
            # Update maximum depth
            new_max_depth = jnp.maximum(max_depth, new_current_depth)
            
            # Update validity flag - once invalid, stays invalid
            new_is_valid = is_valid & (~jnp.logical_or(invalid_closing, invalid_opening))
            
            # Return -1 if invalid, otherwise return the current depth
            output_depth = jnp.where(new_is_valid, new_current_depth, -1)
            
            return (new_max_depth, new_current_depth, new_is_valid), output_depth
        
        init_carry = (self.n, 0, True)
        
        # Scan through the string and collect all depths
        (max_depth, final_depth, is_valid), all_depths = jax.lax.scan(
            scan_fn,
            init_carry,
            string
        )
        
        # Prepend initial state (0) to match expected output format
        return jnp.concatenate([jnp.array([0]), all_depths])

    def find_all_depths_batch(self, string_batch: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self.find_all_depths)(string_batch)
    
    def find_depth_batch(self, string_batch: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self.find_depth)(string_batch)
    
    def generate_positives(self, rng, num, length, max_depth):
        if length % 2 != 0:
            raise ValueError("Length must be even for balanced sequences.")
        # rng1, rng2, rng3 = jax.random.split(rng, 3)
        uniform = self.generate_balanced_sequences(rng, num, length, max_depth)
        return uniform
        
    def generate_negatives(self, rng, num, length):
        return jax.random.randint(rng, (num, length), 0, 2)

    
    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def sample_batch(self, rng: jnp.ndarray, batch_size: int,
                    length: int) -> Mapping[str, jnp.ndarray]:
        """Returns a batch of strings and the expected class."""
        rng, swap_rng = jax.random.split(rng)
        if length % 2 == 1:
            # Not possible to have a positive example of odd length
            inputs = self.generate_negatives(rng, batch_size, length)
            labels = jnp.zeros(batch_size)
        else:
            # For even lengths, generate positive and negative examples
            half_size = batch_size // 2
            pos_rng, neg_rng = jax.random.split(rng)
            positives = self.generate_positives(pos_rng, half_size, length, self.n)
            negatives = self.generate_negatives(neg_rng, batch_size - half_size, length)
            inputs = jnp.concatenate([positives, negatives])
            labels = self.find_depth_batch(inputs)
            labels = jnp.where(jnp.logical_and(labels <= self.n, labels > 0), 1, 0)
            
        # Convert to one-hot representation
        one_hot_strings = jnn.one_hot(inputs, 2)
        ans = jnn.one_hot(labels, 2)
        
        indices = jax.random.permutation(swap_rng, batch_size)
        one_hot_strings = one_hot_strings[indices]
        ans = ans[indices]
        
        return {
            'input': one_hot_strings,
            'output': ans,
        }
            
    @property
    def input_size(self) -> int:
        """Returns the input size for the models."""
        return 2
        
    @property
    def output_size(self) -> int:
        """Returns the output size for the models."""
        return 2

if __name__ == "__main__":
    # Create the task
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(f"Task: D_{args.max_depth}")
    d_n = D_n(n = args.max_depth)
    rng = jax.random.PRNGKey(args.seed)
    
    print("\n=== Testing positive example generation ===")
    pos_examples = d_n.generate_positives(rng, 32, 20, d_n.n)
    print(pos_examples)
    for i, ex in enumerate(pos_examples):
        print(f"Positive example {i+1}: {''.join([str(int(c)) for c in ex])}")
    
    # Test negative example generation
    print("\n=== Testing negative example generation ===")
    neg_examples = d_n.generate_negatives(rng, 32, 20)
    print(neg_examples)
    for i, ex in enumerate(neg_examples):
        print(f"Negative example {i+1}: {''.join(str(int(c)) for c in ex)}")