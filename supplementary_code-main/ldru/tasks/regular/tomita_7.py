"""Compute whether the input string is Tomita 7."""

import functools

import jax
from jax import numpy as jnp
import jax.nn as jnn

from ldru.tasks import task

class DFA:
    def __init__(self, num_states, alphabet_size, transition_matrix, initial_state, accepting_states):
        """
        Initialize a DFA with JAX arrays for efficient computation.
        
        Args:
            num_states: Number of states in the DFA
            alphabet_size: Size of the alphabet (e.g., 2 for binary)
            transition_matrix: JAX array of shape (num_states, alphabet_size) where
                              transition_matrix[state, symbol] gives the next state
            initial_state: The starting state (integer)
            accepting_states: JAX array of booleans indicating which states are accepting
        """
        self.num_states = num_states
        self.alphabet_size = alphabet_size
        self.transition_matrix = transition_matrix
        self.initial_state = initial_state
        self.accepting_states = accepting_states
    
    def transition_step(self, state, symbol):
        """
        Single step transition function.
        
        Args:
            state: Current state
            symbol: Input symbol (0 or 1)
            
        Returns:
            Next state
        """
        return self.transition_matrix[state, symbol]
    
    def run_sequence(self, sequence):
        """
        Run the DFA on a sequence.
        
        Args:
            sequence: JAX array of symbols (integers)
            
        Returns:
            Tuple of (final_state, is_accepted, states_trace)
        """
        # Define the scan function
        def scan_step(state, symbol):
            next_state = self.transition_step(state, symbol)
            return next_state, next_state
        
        # Run the scan
        final_state, states_trace = jax.lax.scan(scan_step, self.initial_state, sequence)
        
        # Check if accepted
        is_accepted = self.accepting_states[final_state]
        
        return final_state, is_accepted, states_trace
    
    def run_batch(self, sequences, lengths=None):
        """
        Run the DFA on a batch of sequences.
        
        Args:
            sequences: JAX array of shape (batch_size, max_seq_length)
            lengths: JAX array of shape (batch_size,) with the actual length of each sequence,
                    or None if all sequences are the same length
            
        Returns:
            Tuple of (final_states, is_accepted)
        """
        # Define the scan function for a single sequence
        def process_sequence(sequence):
            if lengths is not None:
                # Create a mask for valid positions
                mask = jnp.arange(sequence.shape[0]) < lengths
                # Apply the mask (replace invalid positions with a safe value, e.g., 0)
                sequence = jnp.where(mask, sequence, jnp.zeros_like(sequence))
            
            final_state, is_accepted, _ = self.run_sequence(sequence)
            return final_state, is_accepted
        
        # Vectorize over the batch
        return jax.vmap(process_sequence)(sequences)

class Tomita7(task.GeneralizationTask):
    """A task to generate binary strings with Tomita 5 labels. A tomita 5 string is one with an even number of 0s and 1s.
    No length 1 string will be accepted. `

    """
    def __init__(self):
          super().__init__()
          # Define the alphabet and number of states
          self.sigma = ['0', '1']
          self.n_letters = len(self.sigma)
          self.num_states = 5  # q0 through q3
          
          # Map state names to indices for the JAX implementation
          self.state_to_idx = {
              'q0': 0,
              'q1': 1,
              'q2': 2,
              'q3': 3,
              'q4': 4,
          }
          
          # Create transition matrix
          transition_matrix = jnp.zeros((self.num_states, self.n_letters), dtype=jnp.int32)
          
          # Define transitions based on the original definition
          # q0 transitions
          transition_matrix = transition_matrix.at[0, 0].set(0)  # q0 --0--> q0
          transition_matrix = transition_matrix.at[0, 1].set(1)  # q0 --1--> q1
          
          # q1 transitions
          transition_matrix = transition_matrix.at[1, 0].set(2)  # q1 --0--> q2
          transition_matrix = transition_matrix.at[1, 1].set(1)  # q1 --1--> q1
          
          # q2 transitions
          transition_matrix = transition_matrix.at[2, 0].set(2)  # q2 --0--> q2
          transition_matrix = transition_matrix.at[2, 1].set(3)  # q2 --1--> q3

          transition_matrix = transition_matrix.at[3, 0].set(4)  # q3 --0--> q4
          transition_matrix = transition_matrix.at[3, 1].set(3)  # q3 --1--> q3

          transition_matrix = transition_matrix.at[4, 0].set(4)  # q4 --0--> q4
          transition_matrix = transition_matrix.at[4, 1].set(4)  # q4 --1--> q4
          
          # Define accepting states (q0, q1, q2 are accepting)
          accepting_states = jnp.array([True, True, True, True, False])
          
          # Create the DFA
          self.dfa = DFA(
              num_states=self.num_states,
              alphabet_size=self.n_letters,
              transition_matrix=transition_matrix,
              initial_state=0,  # q0
              accepting_states=accepting_states
          )


    def belongs_to_lang(self, seq):
        """
        Check if the sequence belongs to the language defined by the DFA.
        
        Args:
            seq: JAX array of shape (batch_size, max_seq_length)
            
        Returns:
            JAX array of booleans indicating acceptance
        """
        _, is_accepted = self.dfa.run_batch(seq)
        # jax.debug.print("{}", final_states)
        return is_accepted
    
    def generate_positives(self, rng, batch_size, length):
        """
        Generate positive samples from the language.
        These are sequences that are accepted by the DFA
        """
        # Oversample by a factor to account for filtering
        oversample_factor = 2  # Adjust based on expected rejection rate
        oversampled_size = int(batch_size * oversample_factor)
        
        rng, sample_key = jax.random.split(rng, 2)
        rng_keys = jax.random.split(sample_key, num=oversampled_size)
        
        def generate_sequence(sub_rng, length):
            """
            Generate a sequence by performing a random walk through the DFA that never visits state q4.
            
            Args:
                sub_rng: JAX random number generator key
                length: Desired length of the sequence
                
            Returns:
                Tuple of (sequence, final_state) where sequence is a binary array and final_state is the DFA state
            """
            # Initialize sequence and state
            sequence = jnp.zeros(length, dtype=jnp.int32)
            current_state = 0  # Start at q0

            def get_valid_options(state, sub_rng):
                return jax.lax.switch(state, [
                    lambda key: jax.random.choice(key, jnp.array([0, 1]), p=jnp.array([(1 - 4/jnp.maximum(length, 16)), 4/jnp.maximum(length, 16)])),  # State 0: can use 0 or 1
                    lambda key: jax.random.choice(key, jnp.array([0, 1]), p=jnp.array([4/jnp.maximum(length, 16), (1 - 4/jnp.maximum(length, 16))])),  # State 1: can use 0 or 1
                    lambda key: jax.random.choice(key, jnp.array([0, 1]), p=jnp.array([(1 - 4/jnp.maximum(length, 16)), 4/jnp.maximum(length, 16)])),     # State 2: can only use 1
                    lambda _: 1,     # State 3: can only use 1 otherwise enters rejecting sink state
                    lambda _: -1,     # State 4: no valid options (shouldn't reach here)
                ], sub_rng)
            
            def body_fn(i, vals):
                key, curr_state, seq = vals
                new_key, key = jax.random.split(key)
                symbol = get_valid_options(curr_state, key)
                
                # Perform the transition
                next_state = self.dfa.transition_matrix[curr_state, symbol]
                
                # Update sequence
                new_seq = seq.at[i].set(symbol)
                
                # Generate new key for next iteration
                
                return new_key, next_state, new_seq
            
            # Use scan for better JIT compatibility
            def scan_fn(carry, i):
                key, state, seq = carry
                # i = jnp.sum(jnp.ones_like(seq)) - jnp.sum(jnp.ones_like(seq))  # Always 0, but jittable
                key, next_state, new_seq = body_fn(i, (key, state, seq))
                return (key, next_state, new_seq), next_state
            
            init_carry = (sub_rng, current_state, sequence)
            indices = jnp.arange(length)
            (_, final_state, final_sequence), _ = jax.lax.scan(scan_fn, init_carry, indices)
            
            return final_sequence, final_state
        
        # Generate sequences and their final states
        sequences_and_states = jax.vmap(generate_sequence, in_axes=(0, None))(rng_keys, length)
        sequences, final_states = sequences_and_states
        
        is_accepting = jnp.logical_or(jnp.logical_or(jnp.logical_or(final_states == 0, final_states == 1), final_states == 2), final_states == 3)
        
        # Filter sequences based on mask
        def filter_sequences(sequences):
            """
            Filter sequences to keep only those not ending in state 3 (i.e., accepted states 0, 1, 2).
            
            Args:
                rng: JAX random key
                sequences: Batch of generated sequences
                final_states: Final DFA states for each sequence
                target_size: Number of sequences to return
            
            Returns:
                Filtered sequences of size target_size
            """
            # Get indices of accepted sequences
            accepted_indices = jnp.where(is_accepting, size=batch_size)[0]
            
            # Return the selected sequences
            return jax.lax.index_take(sequences, (accepted_indices,), axes=(0,))
        
        # Filter to get exactly batch_size valid sequences
        filtered_sequences = filter_sequences(sequences)
        return filtered_sequences
    
    def generate_negatives(self, rng, batch_size, length):
        """
        Generate positive samples from the language.
        These are sequences that are accepted by the DFA, i.e. those that have no odd 0-sequence after an odd 1-sequence.
        Oversamples and filters out sequences that end in state 3.
        """
        # Oversample by a factor to account for filtering
        oversample_factor = 5  # Adjust based on expected rejection rate
        oversampled_size = int(batch_size * oversample_factor)
        
        rng, sample_key = jax.random.split(rng, 2)
        rng_keys = jax.random.split(sample_key, num=oversampled_size)
        
        def generate_sequence(sub_rng, length):
            """
            Generate a sequence by performing a random walk through the DFA that never visits state q4.
            
            Args:
                sub_rng: JAX random number generator key
                length: Desired length of the sequence
                
            Returns:
                Tuple of (sequence, final_state) where sequence is a binary array and final_state is the DFA state
            """
            # Initialize sequence and state
            sequence = jnp.zeros(length, dtype=jnp.int32)
            current_state = 0  # Start at q0
            
            def body_fn(i, vals):
                key, curr_state, seq = vals
                new_key, key = jax.random.split(key)
                symbol = jax.random.choice(key, jnp.array([0, 1]))
                
                # Perform the transition
                next_state = self.dfa.transition_matrix[curr_state, symbol]
                
                # Update sequence
                new_seq = seq.at[i].set(symbol)
                
                # Generate new key for next iteration
                new_key, _ = jax.random.split(key)
                
                return new_key, next_state, new_seq
            
            # Use scan for better JIT compatibility
            def scan_fn(carry, i):
                key, state, seq = carry
                # i = jnp.sum(jnp.ones_like(seq)) - jnp.sum(jnp.ones_like(seq))  # Always 0, but jittable
                key, next_state, new_seq = body_fn(i, (key, state, seq))
                return (key, next_state, new_seq), next_state
            
            init_carry = (sub_rng, current_state, sequence)
            indices = jnp.arange(length)
            (_, final_state, final_sequence), _ = jax.lax.scan(scan_fn, init_carry, indices)
            
            return final_sequence, final_state
        
        # Generate sequences and their final states
        sequences_and_states = jax.vmap(generate_sequence, in_axes=(0, None))(rng_keys, length)
        sequences, final_states = sequences_and_states
        
        # Create a mask for sequences not ending in state 3 (i.e., ending in 0, 1, or 2)
        is_rejected = final_states == 4
        
        # Filter sequences based on mask
        def filter_sequences(sequences):
            """
            Filter sequences
            
            Args:
                rng: JAX random key
                sequences: Batch of generated sequences
                final_states: Final DFA states for each sequence
                target_size: Number of sequences to return
            
            Returns:
                Filtered sequences of size target_size
            """

            rejected_indices = jnp.where(is_rejected, size=batch_size)[0]
            
            # Return the selected sequences
            return jax.lax.index_take(sequences, (rejected_indices,), axes=(0,))
        
        # Filter to get exactly batch_size valid sequences
        filtered_sequences = filter_sequences(sequences)
        return filtered_sequences
    
    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def sample_batch(self, rng, batch_size, length):
        """Returns a batch of strings and the expected class."""
        pos_rng, neg_rng, swap_rng = jax.random.split(rng, 3)
        half_size = batch_size // 2
        positives = self.generate_positives(pos_rng, half_size, length)
        negatives = self.generate_negatives(neg_rng, batch_size - half_size, length)
        inputs = jnp.concatenate([positives, negatives])
        labels = self.belongs_to_lang(inputs).astype(int)
            
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
    
if __name__ == '__main__':
  tomita7 = Tomita7()
  for i in range(1, 41, 1):
    sample = tomita7.sample_batch(jax.random.PRNGKey(i), 16, i)
    test_sequences = sample["input"].argmax(-1)
    labels = sample["output"].argmax(-1)
    print(f"Length: {i}")
    print("Input:", test_sequences)
    print("Output:", labels)