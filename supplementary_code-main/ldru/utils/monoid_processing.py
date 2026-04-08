import matplotlib.pyplot as plt
from ldru.tasks.regular.d_n import D_n
import jax
import jax.numpy as jnp
import numpy as np
from collections import Counter
from functools import partial

from tqdm import tqdm

import pandas as pd
import seaborn as sns

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
    
    @staticmethod
    def create_balanced_brackets_dfa(max_depth):
        """
        Create a DFA for balanced brackets with a maximum nesting depth.
        
        Args:
            max_depth: Maximum allowed nesting depth
            
        Returns:
            A JaxDFA object that accepts balanced brackets with depth <= max_depth
        """
        # States represent the current nesting depth (0 to max_depth)
        num_states = max_depth + 2  # Include reject state
        alphabet_size = 2  # 0 for open bracket, 1 for close bracket
        
        # Initialize transition matrix with reject state (max_depth + 1)
        transition_matrix = jnp.full((num_states, alphabet_size), max_depth + 1, dtype=jnp.int32)
        
        # Define transitions for valid states
        for depth in range(max_depth + 1):
            # Opening bracket increases depth
            if depth < max_depth:
                transition_matrix = transition_matrix.at[depth, 0].set(depth + 1)
            
            # Closing bracket decreases depth
            if depth > 0:
                transition_matrix = transition_matrix.at[depth, 1].set(depth - 1)
        
        # Set accepting states (only depth 0 is accepting)
        accepting_states = jnp.zeros(num_states, dtype=jnp.bool_)
        accepting_states = accepting_states.at[0].set(True)
        
        return DFA(num_states, alphabet_size, transition_matrix, 0, accepting_states)
    
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
            Tuple of (final_state, is_accepted)
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
    
    def get_sequence_function(self, sequence):
        """
        Get the function representation of a sequence for the transition monoid.
        
        Args:
            sequence: JAX array of symbols
            
        Returns:
            JAX array representing the function (how it maps each state)
        """
        # For each starting state, compute where we end up
        def compute_ending_state(start_state):
            def scan_step(state, symbol):
                next_state = self.transition_step(state, symbol)
                return next_state, None
            
            final_state, _ = jax.lax.scan(scan_step, start_state, sequence)
            return final_state
        
        # Vectorize over all possible starting states
        all_states = jnp.arange(self.num_states)
        function_representation = jax.vmap(compute_ending_state)(all_states)
        
        return function_representation
    
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
    
    def get_batch_functions(self, sequences, lengths=None):
        """
        Get the function representations for a batch of sequences.
        
        Args:
            sequences: JAX array of shape (batch_size, max_seq_length)
            lengths: JAX array of shape (batch_size,) with the actual length of each sequence,
                    or None if all sequences are the same length
            
        Returns:
            JAX array of shape (batch_size, num_states) representing the function for each sequence
        """
        # Define the function to compute a single function representation
        def get_function(sequence):
            if lengths is not None:
                # Create a mask for valid positions
                mask = jnp.arange(sequence.shape[0]) < lengths
                # Apply the mask (replace invalid positions with a safe value, e.g., 0)
                sequence = jnp.where(mask, sequence, jnp.zeros_like(sequence))
            
            return self.get_sequence_function(sequence)
        
        # Vectorize over the batch
        return jax.vmap(get_function)(sequences)
    
    def get_masked_sequence_function(self, sequence, mask):
        """
        Get the function representation of a masked sequence for the transition monoid.
        
        Args:
            sequence: JAX array of symbols (with potential padding)
            mask: Boolean mask indicating which positions in the sequence are valid
            
        Returns:
            JAX array representing the function (how it maps each state)
        """
        # For each starting state, compute where we end up
        def compute_ending_state(start_state):
            def scan_step(state, inputs):
                symbol, is_valid = inputs
                # Only transition if the position is valid according to the mask
                next_state = jnp.where(
                    is_valid,
                    self.transition_step(state, symbol),
                    state  # No change if masked
                )
                return next_state, None
            
            # Combine sequence and mask for the scan
            scan_inputs = (sequence, mask)
            final_state, _ = jax.lax.scan(scan_step, start_state, scan_inputs)
            return final_state
        
        # Vectorize over all possible starting states
        all_states = jnp.arange(self.num_states)
        function_representation = jax.vmap(compute_ending_state)(all_states)
        
        return function_representation


def create_balanced_brackets_dfa(max_depth):
    """
    Create a DFA for balanced brackets with a maximum nesting depth.
    
    Args:
        max_depth: Maximum allowed nesting depth
        
    Returns:
        A DFA object that accepts balanced brackets with depth <= max_depth
    """
    # States represent the current nesting depth (0 to max_depth)
    states = list(range(max_depth + 1))
    
    # We'll use a special state to represent rejection (when depth becomes invalid)
    reject_state = max_depth + 1
    states.append(reject_state)
    
    alphabet = [0, 1] # 0 for open bracket, 1 for close bracket
    
    # Define the transition function
    transition_function = {}
    
    for depth in range(max_depth + 1):
        # Opening bracket increases depth
        if depth < max_depth:
            transition_function[(depth, 0)] = depth + 1
        else:
            # Can't open bracket if at max depth
            transition_function[(depth, 0)] = reject_state
        
        # Closing bracket decreases depth
        if depth > 0:
            transition_function[(depth, 1)] = depth - 1
        else:
            # Can't close bracket if at depth 0
            transition_function[(depth, 1)] = reject_state
    
    # All transitions from reject state lead to reject state
    transition_function[(reject_state, 0)] = reject_state
    transition_function[(reject_state, 1)] = reject_state
    
    initial_state = 0
    accepting_states = {0}  # Only depth 0 is accepting
    
    return DFA(states, alphabet, transition_function, initial_state, accepting_states)

def compute_transition_monoid(dfa, max_word_length=12, hash_mod_size=100000, even_only=True):
    """
    Compute the transition monoid of a DFA using JAX with hash-based function lookup.
    
    Args:
        dfa: A JaxDFA object
        max_word_length: Maximum word length to consider (for efficiency)
        hash_mod_size: Size of hash table (larger means fewer collisions)
        even_only: If True, only consider words with positive, even length
        
    Returns:
        Tuple of:
        - List of equivalence classes (each represented by its function)
        - Dictionary mapping function tuple to class index (for compatibility)
        - Dictionary mapping representative words to their class index (for compatibility)
        - JAX-compatible hash table structures for function lookups
    """
    # Generate all words up to max_word_length (keeping your existing code)
    def generate_words(alphabet_size, max_length):
        words = []
        
        # Generate words of each length
        for length in tqdm(range(0, max_length + 1, 2 if even_only else 1)):
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
                    new_words.append(jnp.array(word[::-1], dtype=jnp.int32))
                
                words.extend(new_words)
        
        return words
    
    all_words = generate_words(dfa.alphabet_size, max_word_length)
    print("generated words")
    
    # Filter words if even_only is True
    # if even_only:
    #     all_words = [word for word in all_words if word.size > 0 and word.size % 2 == 0]
    
    # Compute function representation for each word
    word_functions = []
    for word in tqdm(all_words):
        word_function = dfa.get_sequence_function(word)
        word_functions.append((word, word_function))
    
    # Group words by their function representation
    equivalence_classes = {}
    for word, function in word_functions:
        function_tuple = tuple(function.tolist())
        if function_tuple not in equivalence_classes:
            equivalence_classes[function_tuple] = []
        equivalence_classes[function_tuple].append(word)
    
    # Create a mapping from function to class index
    function_to_class = {func: idx for idx, func in enumerate(equivalence_classes.keys())}
    
    # Create a mapping from words to their class
    word_to_class = {}
    for function, words in equivalence_classes.items():
        class_idx = function_to_class[function]
        representative = min(words, key=lambda w: w.size)
        for word in words:
            word_tuple = tuple(word.tolist())
            word_to_class[word_tuple] = (class_idx, representative)
    
    # ---- Add hash table creation ----
    
    # Determine function tuple length (assuming all functions have the same length)
    sample_func = next(iter(function_to_class.keys()))
    func_length = len(sample_func)
    
    # Define hash function
    def compute_hash(function_tuple, prime=31, mod_size=hash_mod_size):
        """Convert tuple to a hash value."""
        hash_val = 0
        for val in function_tuple:
            hash_val = (hash_val * prime + val) % mod_size
        return hash_val
    
    # Count collisions to determine array sizes
    collision_counts = np.zeros(hash_mod_size, dtype=np.int32)
    for func_tuple in function_to_class:
        hash_val = compute_hash(func_tuple)
        collision_counts[hash_val] += 1
    
    max_collisions = np.max(collision_counts)
    print(f"Maximum collisions at any bucket: {max_collisions}")
    
    # Create hash table arrays
    keys_array = np.full((hash_mod_size, max_collisions, func_length), -1, dtype=np.int32)
    values_array = np.full((hash_mod_size, max_collisions), -1, dtype=np.int32)
    collision_indices = np.zeros(hash_mod_size, dtype=np.int32)
    
    # Fill hash table arrays
    for func_tuple, class_idx in function_to_class.items():
        hash_val = compute_hash(func_tuple)
        idx = collision_indices[hash_val]
        keys_array[hash_val, idx] = func_tuple
        values_array[hash_val, idx] = class_idx
        collision_indices[hash_val] += 1
    
    # Convert to JAX arrays
    keys_array_jax = jnp.array(keys_array)
    values_array_jax = jnp.array(values_array)
    
    # Create JAX hash function
    def compute_hash_jax(function_array, prime=31):
        """JAX version of hash function."""
        def hash_step(i, h):
            return (h * prime + function_array[i]) % hash_mod_size
        
        return jax.lax.fori_loop(0, function_array.shape[0], hash_step, 0)
    
    # Create lookup function for use in JAX context
    def lookup_class_index(function_array):
        hash_val = compute_hash_jax(function_array)
        
        def check_match(i, found_and_value):
            found, value = found_and_value
            key_match = jnp.all(keys_array_jax[hash_val, i] == function_array)
            new_found = found | key_match
            new_value = jnp.where(key_match & ~found, 
                                 values_array_jax[hash_val, i], 
                                 value)
            return (new_found, new_value)
        
        init_state = (False, jnp.array(-1, dtype=jnp.int32))
        found, value = jax.lax.fori_loop(0, max_collisions, check_match, init_state)
        
        return value
    
    # Create hash table structure with necessary components
    hash_table = {
        'keys': keys_array_jax,
        'values': values_array_jax,
        'compute_hash': compute_hash_jax,
        'lookup': lookup_class_index,
        'mod_size': hash_mod_size,
        'max_collisions': max_collisions
    }
    
    return list(equivalence_classes.values()), function_to_class, word_to_class, hash_table

def get_sequence_equivalence_classes_matrix(sequence, dfa, hash_table):
    seq_length = sequence.shape[0]
    max_level = int(jnp.ceil(jnp.log2(seq_length)))
    
    # Initialize output matrix
    matrix = jnp.full((max_level, seq_length), -1, dtype=jnp.int32)
    
    def process_level(level_idx, mat):
        window_size = 2 ** (level_idx + 1)
        indices = jnp.arange(seq_length)
        
        # Process windows at this level
        def process_window(window_idx):
            # Calculate start and end indices
            start_idx = window_idx * window_size
            end_idx = jnp.minimum(start_idx + window_size, seq_length)
            
            # Check if window is valid (at least half the target size)
            # is_valid = (end_idx - start_idx) >= window_size / 2
            
            # Create mask for positions within this window
            window_mask = (indices >= start_idx) & (indices < end_idx)
            
            # Function to process a valid window
            def handle_valid_window():
                # Extract the window using a mask-based approach
                padded_window = jnp.zeros(window_size, dtype=sequence.dtype) - 1

                # Fill the padded window with actual values
                def fill_window(i, acc):
                    src_idx = start_idx + i
                    valid = src_idx < seq_length
                    val = jnp.where(valid, sequence[src_idx], jnp.array(-1, dtype=sequence.dtype))
                    return acc.at[i].set(val)
                
                window_data = jax.lax.fori_loop(0, window_size, fill_window, padded_window)
                
                # Create validity mask for the window
                pos_mask = jnp.arange(window_size) < (end_idx - start_idx)
                
                # Get sequence function and class index
                window_function = dfa.get_masked_sequence_function(window_data, pos_mask)
                class_index = hash_table['lookup'](window_function)
                
                # Return the class index
                return class_index
            
            class_idx = handle_valid_window()
            
            # Create a matrix update for this window
            mask = jnp.zeros(seq_length, dtype=jnp.int32)
            # Update only the end position of the window
            end_pos = jnp.minimum(end_idx - 1, seq_length - 1)
            mask = mask.at[end_pos].set(1)
            
            # Return (mask, class_index) tuple for scatter update
            return mask * class_idx + (-1) * (1 - mask)
        
        # Process all windows at this level and combine results
        max_windows = (seq_length + window_size - 1) // window_size
        window_indices = jnp.arange(max_windows)
        
        # Map the window processing function over all windows
        window_results = jax.vmap(process_window)(window_indices)
        # Combine window results (take maximum value at each position since -1 is invalid)
        combined_results = jnp.maximum(jnp.max(window_results, axis=0), -1)
        # Update the matrix for this level
        return mat.at[level_idx].set(combined_results)
    
    for i in range(max_level):
        matrix = process_level(i, matrix)

    return matrix

def pretty_print_matrix(matrix):
    """
    Pretty prints a JAX NumPy matrix, showing only non-(-1) elements
    with an indication of powers of 2 for each row.
    
    Args:
        matrix: JAX NumPy array to be printed
    """
    for row_idx, row in enumerate(matrix):
        power = 2 ** (row_idx + 1)
        # Get indices of elements that are not -1
        non_neg_indices = jnp.where(row != -1)[0]
        
        # If there are non-(-1) elements in this row
        if len(non_neg_indices) > 0:
            print(f"Window {power}): {row[non_neg_indices]}")
        else:
            print(f"Window {power}): All elements are -1")
        # print(row)

def compute_monoid_products(dfa, equivalence_classes, hash_table):
    """
    Compute the products of all pairs of equivalence classes using the hash table lookup.
    
    This version uses the hash table for faster lookup of function class indices.
    
    Args:
        dfa: A JaxDFA object
        equivalence_classes: List of equivalence classes (each containing words)
        function_to_class: Dictionary mapping function tuple to class index
        word_to_class: Dictionary mapping representative words to their class index
        hash_table: JAX-compatible hash table structures for function lookups
        
    Returns:
        product_table: 2D array where product_table[i, j] is the class index of the product
                      of class i and class j
    """
    num_classes = len(equivalence_classes)
    product_table = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    # Select a representative word from each class
    representatives = []
    for class_words in equivalence_classes:
        # Find the shortest word in this equivalence class
        rep_word = min(class_words, key=lambda w: w.size)
        representatives.append(rep_word)
    
    # Compute the product of each pair of classes using JAX for batch processing
    @jax.jit
    def get_product_class(word_i, word_j):
        # Concatenate the words to get their product
        product_word = jnp.concatenate([word_i, word_j])
        # Compute the function for the product word
        product_function = dfa.get_sequence_function(product_word)
        # jax.debug.print("{}", product_function)
        # Look up the class index using the hash table
        return hash_table['lookup'](product_function)
    
    for i in range(num_classes):
        for j in range(num_classes):
            # Take representative words from each class
            word_i = representatives[i]
            word_j = representatives[j]
            # print("Left", i, "right", j)
            # Get the product class using the hash table lookup
            product_class = get_product_class(word_i, word_j)
            product_table[i, j] = product_class.item()  # Convert from JAX array to scalar
    
    return product_table

def pretty_print_monoid_table(product_table, class_labels=None):
    """
    Pretty print the monoid product table.
    
    Args:
        product_table: 2D array where product_table[i, j] is the class index of the product
                      of class i and class j
        class_labels: Optional labels for the equivalence classes (defaults to indices)
    """
    n = product_table.shape[0]
    
    if class_labels is None:
        class_labels = [str(i) for i in range(n)]
    
    # Create a pandas DataFrame for nice formatting
    df = pd.DataFrame(product_table, 
                     index=class_labels,
                     columns=class_labels)
    
    print("Monoid Product Table:")
    print(df)
    return df

def visualize_monoid_heatmap(product_table, class_labels=None, figsize=(14, 12), 
                         show_values=True, colormap=None, save_path=None, annihilator_ec = -1):
    """
    Visualize the monoid product table as a binary heatmap.
    
    Args:
        product_table: 2D array where product_table[i, j] is the class index of the product
                      of class i and class j
        class_labels: Optional labels for the equivalence classes (defaults to indices)
        figsize: Figure size tuple (width, height)
        show_values: Whether to show the actual values in each cell
        colormap: Optional custom colormap (defaults to green for valid, black for unknown)
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib figure and axis objects
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    n = product_table.shape[0]
    
    if class_labels is None:
        class_labels = [str(i) for i in range(n)]
    
    # Create a binary mask: 1 for valid products, 0 for unknown (-1)
    binary_mask = (product_table != annihilator_ec).astype(int)
    
    # Create a custom colormap if not provided
    if colormap is None:
        colormap = ListedColormap(['black', '#21918c'])  
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the heatmap
    im = ax.imshow(binary_mask, cmap=colormap, aspect='equal')
    
    # Set the ticks and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    
    # Rotate the x tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    
    # Add title and labels
    ax.set_title("Monoid Product Heatmap")
    ax.set_xlabel("Right Factor")
    ax.set_ylabel("Left Factor")
    
    # Add text annotations with the actual class indices if requested
    if show_values:
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, product_table[i, j],
                               ha="center", va="center", 
                               color="white" if product_table[i, j] != annihilator_ec else "gray",
                               fontsize=7)
    
    # Add a colorbar legend
    cbar = ax.figure.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Annihilation', 'Valid'])
    
    fig.tight_layout()
    
    # Save the figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

@jax.jit
def find_classes_mapping_state(hash_table, source_state, target_state, num_states):
    """
    Find all equivalence classes whose functions map a specific source state to a specific target state.
    
    Args:
        keys_array: JAX array of function tuples [hash_mod_size, max_collisions, func_length]
        values_array: JAX array of class indices [hash_mod_size, max_collisions]
        source_state: The state to map from (index i)
        target_state: The state to map to (value j)
        hash_mod_size: Size of the hash table
        max_collisions: Maximum number of collisions per hash bucket
        num_states: Number of states in the DFA
        
    Returns:
        JAX array: Mask of which class indices match the condition
    """
    values_array = hash_table['values']
    keys_array = hash_table['keys']
    hash_mod_size = hash_table['mod_size']
    max_collisions = hash_table['max_collisions']

    # Create a mask array for all possible class indices
    # Assuming class indices are 0...n-1 where n is the number of classes
    max_class_idx = jnp.max(values_array)
    result_mask = jnp.zeros(max_class_idx + 1, dtype=jnp.bool_)
    
    # Function to check a single hash bucket
    def process_bucket(bucket_idx, mask):
        # Function to check each collision entry
        def process_collision(collision_idx, inner_mask):
            # Get function and class
            function = keys_array[bucket_idx, collision_idx]
            class_idx = values_array[bucket_idx, collision_idx]
            
            # Check if this is a valid entry
            is_valid = class_idx >= 0
            
            # Check if function maps source_state to target_state
            maps_correctly = function[source_state] == target_state
            
            # Update mask if conditions are met
            should_set = is_valid & maps_correctly
            new_mask = jax.lax.cond(
                should_set,
                lambda m: m.at[class_idx].set(True),
                lambda m: m,
                inner_mask
            )
            
            return new_mask
        
        # Process all collisions in this bucket
        new_mask = jax.lax.fori_loop(
            0, max_collisions,
            lambda i, m: process_collision(i, m),
            mask
        )
        
        return new_mask
    
    # Process all buckets
    final_mask = jax.lax.fori_loop(
        0, hash_mod_size,
        lambda i, m: process_bucket(i, m),
        result_mask
    )
    
    return final_mask

def find_products_resulting_in_class(target_class_idx, product_table, equivalence_classes, prev_window_size=None):
    """
    Find all pairs of equivalence classes whose product results in the target class.
    
    Args:
        target_class_idx: The index of the target equivalence class
        product_table: 2D array where product_table[i, j] is the class index of the product
                      of class i and class j
                      
    Returns:
        list: Tuples of (left_class_idx, right_class_idx) whose product equals target_class_idx
    """
    result_pairs = []
    n = product_table.shape[0]
    
    for i in range(n):
        for j in range(n):
            if product_table[i, j] == target_class_idx:
                # potentially valid IF possible to combine to make a string of prev_window_size
                if prev_window_size is not None:
                    skip = False
                    for possible_i in equivalence_classes[i]:
                        for possible_j in equivalence_classes[j]:
                            if possible_i.shape[0] + possible_j.shape[0] == prev_window_size:
                                result_pairs.append((i, j))
                                skip = True
                                break
                        if skip:
                            break
                else:
                    result_pairs.append((i, j))
    
    return result_pairs
   
def get_batch_equivalence_classes_matrix(sequences, dfa, hash_table,):
    """
    Generate equivalence class matrices for a batch of sequences.
    
    Args:
        sequences: JAX array of shape (batch_size, max_seq_length)
        dfa: JaxDFA object
        function_to_class: Function that maps function representation to class index
        num_classes: Total number of equivalence classes
        
    Returns:
        Batch of matrices, where each matrix is the result of get_sequence_equivalence_classes_matrix_scan
    """
    return jax.vmap(partial(get_sequence_equivalence_classes_matrix, dfa=dfa, hash_table=hash_table))(sequences)



def visualize_counter_heatmap(counters_list, lengths, use_proportions=True, title="Element Count Heatmap", 
                              figsize=(10, 8), cmap="viridis", annot=False, fmt=".3f"):
    """
    Create a heatmap visualization from a list of Counter objects.
    
    Args:
        counters_list: List of Counter objects or dictionaries with count values
        use_proportions: If True, convert counts to proportions (default True)
        title: Title for the heatmap plot
        figsize: Figure size as tuple (width, height)
        cmap: Colormap for the heatmap
        annot: If True, write the data value in each cell
        fmt: String formatting code to use when adding annotations
        
    Returns:
        matplotlib figure and axes objects
    """
    # Get all unique keys across all counters
    all_keys = set()
    for counter in counters_list:
        all_keys.update(counter.keys())
    all_keys = sorted(all_keys)
    
    # Create a DataFrame with counts or proportions
    data = []
    for i, counter in enumerate(counters_list):
        # Get raw counts for this counter
        row = [counter.get(key, 0) for key in all_keys]
        
        # Convert to proportions if requested
        if use_proportions:
            total = sum(counter.values())
            if total > 0:  # Avoid division by zero
                row = [count / total for count in row]
            else:
                row = [0.0 for _ in row]
        
        data.append(row)
    
    df = pd.DataFrame(data, columns=[str(k) for k in all_keys])
    
    # For row labels, use "Counter 0", "Counter 1", etc.
    row_labels = [f"{i}" for i in lengths]
    df.index = row_labels
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    heatmap = sns.heatmap(df, annot=annot, fmt=fmt, cmap=cmap, ax=ax)
    
    plt.title(title)
    plt.tight_layout()
    
    return fig, ax

from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

def pretty_print_transition_monoid(equivalence_classes, function_to_class, dfa=None, max_display=10, 
                                  visualize=False, compact=False, show_function=True):
    """
    Pretty print the equivalence classes of a transition monoid.
    
    Args:
        equivalence_classes: List of lists, where each inner list contains words in the same class
        function_to_class: Dictionary mapping function tuples to class indices
        dfa: Optional JaxDFA object (for state names if available)
        max_display: Maximum number of words to display per class
        visualize: Whether to visualize the function representations as matrices
        compact: If True, use a more compact representation for large monoids
        show_function: If True, show the function representation for each class
        
    Returns:
        None (prints output to console)
    """
    print(f"Transition Monoid: {len(equivalence_classes)} classes")
    print("=" * 50)
    
    # Extract state names if DFA is provided
    state_names = None
    if dfa is not None and hasattr(dfa, 'state_names'):
        state_names = dfa.state_names
    
    # Reverse mapping from class index to function
    class_to_function = {idx: func for func, idx in function_to_class.items()}
    
    # Sort classes by size (smaller classes first)
    # sorted_indices = sorted(range(len(equivalence_classes)), 
    #                        key=lambda i: len(equivalence_classes[i]))
    sorted_indices = range(len(equivalence_classes))
    
    # Print each class
    for idx in sorted_indices:
        words = equivalence_classes[idx]
        
        # Get the function representation
        function = class_to_function[idx]
        
        # Determine the class identity (use the shortest word as representative)
        representative = min(words, key=lambda w: w.size)
        rep_str = word_to_string(representative)
        
        # Simplified printing for compact mode
        if compact and len(equivalence_classes) > 20:
            if len(words) > 1:
                print(f"Class {idx} (rep: '{rep_str}'): {len(words)} words")
            else:
                print(f"Class {idx}: '{rep_str}'")
            continue
            
        # Full printing mode
        print(f"\nClass {idx} - Representative: '{rep_str}'")
        print(f"Size: {len(words)} word{'s' if len(words) > 1 else ''}")
        
        if show_function:
            print("Function: ", end="")
            if state_names:
                # Print with state names
                func_with_names = [f"{state_names[i]}→{state_names[val]}" for i, val in enumerate(function)]
                print(", ".join(func_with_names))
            else:
                # Print raw function values
                print(function)
        
        # Display words (limit by max_display)
        if len(words) > max_display:
            display_words = words[:max_display]
            print(f"Words (showing {max_display} of {len(words)}):")
        else:
            display_words = words
            print("Words:")
            
        # Format and print words
        word_strs = [word_to_string(w) for w in display_words]
        if len(word_strs) <= 5:
            print("  " + ", ".join(word_strs))
        else:
            # Format in multiple columns for better readability
            table_data = []
            for i in range(0, len(word_strs), 3):
                row = word_strs[i:i+3]
                while len(row) < 3:
                    row.append("")
                table_data.append(row)
            print(tabulate(table_data, tablefmt="plain"))
            
        # Visualize function as a matrix
        if visualize and not compact:
            visualize_function(function, idx, state_names)
    
    print("\n" + "=" * 50)

def word_to_string(word):
    """Convert a JAX array word to a string representation."""
    if word.size == 0:
        return "ε"  # Empty word (epsilon)
    else:
        return "".join(str(int(i)) for i in word)

def visualize_function(function, class_idx, state_names=None):
    """Visualize a function as a directed graph matrix."""
    if len(function) > 15:  # Skip visualization for very large functions
        print("  (Function matrix too large to visualize)")
        return
        
    n = len(function)
    
    # Create a matrix representation
    matrix = np.zeros((n, n), dtype=int)
    for i, j in enumerate(function):
        matrix[i, j] = 1
    
    # Skip visualization if running in non-interactive environment
    try:
        plt.figure(figsize=(min(8, n+2), min(6, n+1)))
        plt.imshow(matrix, cmap='Blues', interpolation='nearest')
        
        # Add grid lines
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
        
        # Set axis labels
        if state_names:
            plt.xticks(range(n), state_names, rotation=45)
            plt.yticks(range(n), state_names)
        else:
            plt.xticks(range(n))
            plt.yticks(range(n))
        
        plt.title(f"Transition Function for Class {class_idx}")
        plt.xlabel("Target State")
        plt.ylabel("Source State")
        
        # Annotate the cells
        for i in range(n):
            for j in range(n):
                if matrix[i, j] != 0:
                    plt.text(j, i, '1', ha='center', va='center', 
                           color='white' if matrix[i, j] > 0.5 else 'black')
        
        plt.colorbar(ticks=[0, 1], label="Transition Present")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"  (Could not visualize: {e})")

def print_monoid_summary(equivalence_classes, function_to_class, word_to_class):
    """Print a summary of the transition monoid structure."""
    print("\nTransition Monoid Summary")
    print("=" * 50)
    
    # Count class sizes
    class_sizes = [len(cls) for cls in equivalence_classes]
    
    # Create a size distribution table
    size_counts = {}
    for size in class_sizes:
        if size not in size_counts:
            size_counts[size] = 0
        size_counts[size] += 1
    
    print("Class Size Distribution:")
    sizes = sorted(size_counts.keys())
    if len(sizes) > 10:
        # Summarize if there are too many different sizes
        print(f"  Classes of size 1: {size_counts.get(1, 0)}")
        print(f"  Classes of size 2-10: {sum(size_counts.get(i, 0) for i in range(2, 11))}")
        print(f"  Classes of size >10: {sum(size_counts.get(i, 0) for i in sizes if i > 10)}")
    else:
        # Print full distribution
        for size in sizes:
            print(f"  Classes of size {size}: {size_counts[size]}")
    
    # Find the largest classes
    largest_size = max(class_sizes) if class_sizes else 0
    largest_classes = [i for i, cls in enumerate(equivalence_classes) if len(cls) == largest_size]
    
    print(f"\nLargest class size: {largest_size}")
    print(f"Number of classes with this size: {len(largest_classes)}")
    
    # Print the identity element if it exists (usually the empty word)
    empty_word = tuple(jnp.array([], dtype=jnp.int32).tolist())
    if empty_word in word_to_class:
        id_class, id_rep = word_to_class[empty_word]
        print(f"\nIdentity element is in Class {id_class}")

def get_num_ecs(length: int) -> int:
    layers = jnp.ceil(jnp.log2(length)).astype(jnp.int32) - 1
    return sum((length - (2**(layer+1)-1) + 2**(layer+2) - 1) // 2**(layer+2) for layer in range(layers))

def count_ec_pairs(ec_matrices):
    pair_counter = Counter()

    batchsize, layers, seq_len = ec_matrices.shape

    num_ecs = get_num_ecs(seq_len)

    def process_sequence(ec_matrix):
        pairs = jnp.zeros((num_ecs, 2), dtype=jnp.int32) + -1
        get_layer = jnp.zeros(num_ecs, dtype=jnp.int32)
        ec_idx = 0
        
        # First pass: determine which layer each EC belongs to
        for l in range(layers):
            stride = 2**(l+2)
            base_pos = 2**(l+1) - 1
            ecs_in_layer = (seq_len - base_pos + stride - 1) // stride
            
            # Set layer values for this range of ECs
            get_layer = get_layer.at[ec_idx:ec_idx+ecs_in_layer].set(l)
            
            # Move to next layer's starting index
            ec_idx += ecs_in_layer
        
        # Second pass: calculate positions and set pairs
        for ec_idx in range(num_ecs):
            # Use pre-calculated layer for this EC
            layer = get_layer[ec_idx]
            
            # Calculate positions for this EC
            stride = 2**(layer+2)
            base_pos = 2**(layer+1) - 1
            
            # Calculate relative position within the layer
            layer_start_idx = 0
            for l in range(layer):
                prev_stride = 2**(l+2)
                prev_base_pos = 2**(l+1) - 1
                layer_start_idx += (seq_len - prev_base_pos + prev_stride - 1) // prev_stride
            
            remaining_idx = ec_idx - layer_start_idx
            
            pos1 = base_pos + remaining_idx * stride
            pos2 = jnp.minimum(pos1 + 2**(layer+1), seq_len - 1)
            
            # Set pair
            # print(layer, pos1, pos2)
            pairs = pairs.at[ec_idx, 0].set(ec_matrix[layer, pos1])
            pairs = pairs.at[ec_idx, 1].set(ec_matrix[layer, pos2])
        
        return pairs
    
    result = jax.vmap(process_sequence)(ec_matrices)

    # turn result into a counter
    for batch_idx in range(batchsize):
        for ec_idx in range(num_ecs):
            pair = (int(result[batch_idx, ec_idx, 0]), int(result[batch_idx, ec_idx, 1]))
            pair_counter[pair] += 1

    return pair_counter

def test_sequences(jax_dfa, hash_table, equivalence_classes, task):
    # strings = [
    #     "00101000110100111001001010101010110100101010111000101010110011100101101000110100101011101100001100111000110100101011011000101010101011110100001010111000101010101010101010110010101010101010101011010010110100101110010011011100011010001010111000110010110101100100101010101010101101101101001000110011001110001100101110001011110000101100101010110010101010111100001100101101101100001010101010101100101100101010101010111000110100101100101010101100101010101011001010110011100011100011100011101100101100101101",
    #     "00011000101111010100001111000011010010101010101010101010111100101001010011011001100100110100110010101110001010111000101101010010101010111000101100101010101010101110110100110101001100011001001011001110110001001010101110100010101010101010101110001111010001010100101010111010010100110101001010101010110010101010101010101010110100101010110011001110100010101100110100101011100011001010101011001100111100001011100011010010111010001100110010101101010100101110001101100011001100110101100011100111001011001011",
    #     "00010011011000101011001011001111000100101011010011100011001011100101001011001011001011011000101011101100100011100010111000101011001101101010010110001010101011100011101001010010101010110100101010101010101011001010101100101100101101100011100011001010101010110011100010101010101011001010110011001110001010110010101010101110001010101010101011110000101010101100110010101011100101010010101010111010001100101101100100101100110010101110100011101000101100101010101111001000101011110000110011010101100111010101",
    #     "00010010101010111001101000111001010010101010110100111001011001001010101011100010110010110010101011110100100010101100101100110010110110100010111000101011001011101100001110101100100010110010110010101010111010010011001111010010101001100011001101010100101010110010101010111001101001001010101010101010101011001010111000101110010011101001100011010010101010110011001010101010110010111100100011011000101010101011110000111000101011001011001010110100101010101011110100101101001101000010111101001100001011110011"
    # ]
    strings = [
        "00100110001011010010101111000011100011010011101100011001001010101011001011010011100010101011010101010101110010010101100110001011001010101010101101110011000011001110001011001010111010101010010010111011001100010010101011001110001011001010101011001101100110100100101101010010101101001100101010110011110100001011001101101100001101100010101010101100110010110010101101001100101100110110011100001010101010101010101010101010101011001111001000101010101110010110100011011000101011100111010100011001001011010111",
        "00101101000011010110001100101011100100110100101010101010110010110011001010101010110011010011110000101101001011001110010011001011010011010010110110001101001011001010101101110100010010111000101010101011001011110010001010101110110000110010101100101011001010101010101101010011010010110100101010101100101100101010110011010011110001010100101011001010110011101010110000111000101011011010100010111000101011001011010010110010101101001010101010101100101010101010101100111001101001001011011011010010010111001101",
        "01000101100101011000111101000110101001110100100010110011001110100111010100101001001111001010110001001100111000101100110100111000111001010011101000110010110011011000101011001110100011011101000111000010101111010000101010101100110111000110010010101100110101011010001011001010101100101101100011001010101010110101100111010000110010101010101101001100110110001101100010110011101010010010101100101010101100101010110011001011010011001011001100101010110010110010101011001010101100110100110011010101100010110111"
        ]
    sequences = [[int(c) for c in s] for s in strings]
    sequences = jnp.array(sequences)

    print(task.find_all_depths_batch(sequences))
    seen_classes = get_batch_equivalence_classes_matrix(sequences, jax_dfa, hash_table)
    print(seen_classes)
    pair_counts = count_ec_pairs(seen_classes)
    print(pair_counts) 
    exit()

    num_classes = len(equivalence_classes)
    train_matrix = np.zeros((num_classes, num_classes))
    
    for (ec1, ec2), count in pair_counts.items():
        if 0 <= ec1 < num_classes and 0 <= ec2 < num_classes:
            train_matrix[ec1, ec2] = count

    # Normalize matrices
    train_matrix_norm = train_matrix / train_matrix.sum() if train_matrix.sum() > 0 else train_matrix

    plt.figure(figsize=(14, 12))
    train_log = np.log10(train_matrix_norm + 1e-10)
    sns.heatmap(train_log, 
            cmap="viridis",
            mask=(train_matrix == 0))
    plt.title("Log10 of Training EC Pair Frequencies")
    plt.xlabel("To EC")
    plt.ylabel("From EC")
    plt.tight_layout()
    plt.savefig("probe_ec_frequencies.png")

# Example usage
def dfa_example(max_depth=6, max_word_length=12):
    # Create a balanced brackets DFA
    jnp.set_printoptions(threshold=10_000)
    jax_dfa = DFA.create_balanced_brackets_dfa(max_depth)

    max_training_length = 40
    task = D_n(max_depth, max_training_length)
    
    # Compute the transition monoid
    equivalence_classes, function_to_class, word_to_class, hash_table = compute_transition_monoid(jax_dfa, max_word_length=max_word_length, even_only=False)

    print(f"D_{max_depth} --- transition monoid has {len(equivalence_classes)} elements")
    # test_sequences(jax_dfa, hash_table, equivalence_classes, task)
    # return
    # pretty_print_transition_monoid(equivalence_classes, function_to_class, dfa=jax_dfa, max_display=5, visualize=False, compact=False, show_function=True)
    return
    monoid_products = compute_monoid_products(jax_dfa, equivalence_classes, hash_table)
    pretty_print_monoid_table(monoid_products, class_labels=range(len(equivalence_classes)))
    
    # monoid_products = compute_monoid_products(jax_dfa, equivalence_classes, hash_table)
    # pretty_print_monoid_table(monoid_products, class_labels=range(len(equivalence_classes)))
    visualize_monoid_heatmap(monoid_products, class_labels=range(len(equivalence_classes)), figsize=(10, 8), show_values=True, save_path="monoid_heatmap.png",
                             annihilator_ec=hash_table["lookup"](jnp.array([(max_depth + 1)] * (max_depth + 2))))
    counters = []
    # need to work it out so both sets witness equal numbers of equivalence classes so we need to vary the batch size
    training_lengths = range(2, max_training_length + 2, 2) # does not include simple lengths ith little possible equivalence classes
    test_lengths = range(500, 502, 2)
    num_training_ecs = 0
    num_test_ecs = 0
    for length in training_lengths:
        num_training_ecs += get_num_ecs(length)
    for length in test_lengths:
        num_test_ecs += get_num_ecs(length)
    mean_training_ecs = num_training_ecs / len(training_lengths)    
    mean_test_ecs = num_test_ecs / len(test_lengths)

    print(f"Training ECs: {num_training_ecs}")
    print(f"Mean training ECs: {mean_training_ecs}")
    print(f"Test ECs: {num_test_ecs}")
    print(f"Mean test ECs: {mean_test_ecs}")
    # exit()

    batch_size = 256
    max_batch_size = 4096
    # work out correction factor for the batch size
    multiplier = mean_test_ecs / mean_training_ecs
    # multiplier = 1
    print(f"Multiplier: {multiplier}")

    lengths = list(training_lengths) + list(test_lengths)

    training_ecs = dict()
    test_ecs = dict()

    training_pair_counts = {}
    test_pair_counts = {}

    rng = jax.random.PRNGKey(0)

    for length in tqdm(lengths):
        print("LEN:", length)
        if length in training_lengths:
            total_samples = int(batch_size * multiplier)
        else:
            total_samples = batch_size
        # Process in batches if total_samples exceeds max_batch_size
        if total_samples > max_batch_size:
            num_batches = (total_samples + max_batch_size - 1) // max_batch_size  # Ceiling division
            
            # Initialize counts for this length
            length_pair_counts = {}
            length_counts = Counter()
            rngs = jax.random.split(rng, num_batches)
            for batch_idx in range(num_batches):
                # For the last batch, we might need fewer samples
                current_batch_size = min(max_batch_size, total_samples - batch_idx * max_batch_size)
                
                # Generate batch with a unique PRNGKey for each batch
                batch = task.generate_positives(rngs[batch_idx], current_batch_size, length, task.n).astype(int)
                # Process this batch
                print("generated batch")
                seen_classes = get_batch_equivalence_classes_matrix(batch, jax_dfa, hash_table)
                print("got seen classes")
                batch_pair_counts = count_ec_pairs(seen_classes)
                print("counted pairs")
                # Aggregate pair counts for this batch
                for pair, count in batch_pair_counts.items():
                    if pair in length_pair_counts:
                        length_pair_counts[pair] += count
                    else:
                        length_pair_counts[pair] = count
                
                # Count equivalence classes for this batch
                flat_tensor = seen_classes.flatten()
                np_array = np.asarray(flat_tensor)
                np_array = np_array[np_array != -1]
                batch_counts = Counter(np_array)
                
                # Combine with counts for this length
                length_counts.update(batch_counts)
                print("done")
            
            # Now use the aggregated counts as if they came from one large batch
            pair_counts = length_pair_counts
            counts = length_counts
            
        else:
            # Original code path for small batches
            batch = task.generate_positives(rng, total_samples, length, task.n).astype(int)
            seen_classes = get_batch_equivalence_classes_matrix(batch, jax_dfa, hash_table)
            pair_counts = count_ec_pairs(seen_classes)
            flat_tensor = seen_classes.flatten()
            np_array = np.asarray(flat_tensor)
            np_array = np_array[np_array != -1]
            counts = Counter(np_array)
        
        # Aggregate counts across lengths - same as original code
        if length in training_lengths:
            for pair, count in pair_counts.items():
                if pair in training_pair_counts:
                    training_pair_counts[pair] += count
                else:
                    training_pair_counts[pair] = count
            
            for key, value in counts.items():
                if key in training_ecs:
                    training_ecs[key] += value
                else:
                    training_ecs[key] = value
        else:
            for pair, count in pair_counts.items():
                if pair in test_pair_counts:
                    test_pair_counts[pair] += count
                else:
                    test_pair_counts[pair] = count
            
            for key, value in counts.items():
                if key in test_ecs:
                    test_ecs[key] += value
                else:
                    test_ecs[key] = value
        
        counters.append(counts)


    # Find pairs that appear significantly more in test than training
    significant_differences = {}
    all_pairs = set(training_pair_counts.keys()) | set(test_pair_counts.keys())

    # Normalize counts to frequencies
    total_train_count = sum(training_pair_counts.values())
    total_test_count = sum(test_pair_counts.values())

    for pair in all_pairs:
        ec1, ec2 = pair
        train_count = training_pair_counts.get(pair, 0)
        test_count = test_pair_counts.get(pair, 0)
        
        # Calculate frequencies
        train_freq = train_count / total_train_count if total_train_count > 0 else 0
        test_freq = test_count / total_test_count if total_test_count > 0 else 0
        
        # Calculate relative difference (only include pairs that are not extremely rare)
        min_frequency_threshold = 0.0001  # Adjust as needed
        if train_freq > min_frequency_threshold:
            rel_diff = (test_freq / train_freq)
            abs_diff = test_freq - train_freq
            significant_differences[pair] = (train_freq, test_freq, rel_diff, abs_diff)

    # Sort by relative difference for pairs with meaningful presence
    sorted_differences = sorted(significant_differences.items(), 
                            key=lambda x: x[1][2], 
                            reverse=True)

    # Print the top 20 pairs with the largest frequency differences
    print("Top 20 EC pairs with largest frequency differences (test vs training):")
    print("EC Pair | Training Freq | Test Freq | Ratio (Test/Train) | Abs Diff")
    print("-" * 75)
    for (ec1, ec2), (train_freq, test_freq, ratio, abs_diff) in sorted_differences[:20]:
        print(f"({ec1},{ec2}): {train_freq:.6f} | {test_freq:.6f} | {ratio:.2f}x | {abs_diff:.6f}")

    # Create transition matrix visualization
    num_classes = max(
        max([p[0] for p in all_pairs]),
        max([p[1] for p in all_pairs])
    ) + 1

    # Create transition matrices
    train_matrix = np.zeros((num_classes, num_classes))
    test_matrix = np.zeros((num_classes, num_classes))

    for (ec1, ec2), count in training_pair_counts.items():
        if 0 <= ec1 < num_classes and 0 <= ec2 < num_classes:
            train_matrix[ec1, ec2] = count
            
    for (ec1, ec2), count in test_pair_counts.items():
        if 0 <= ec1 < num_classes and 0 <= ec2 < num_classes:
            test_matrix[ec1, ec2] = count

    # Normalize matrices
    train_matrix_norm = train_matrix / train_matrix.sum() if train_matrix.sum() > 0 else train_matrix
    test_matrix_norm = test_matrix / test_matrix.sum() if test_matrix.sum() > 0 else test_matrix

    # Calculate ratio with handling for zeros
    ratio_matrix = np.zeros_like(train_matrix_norm)
    ratio_matrix = test_matrix_norm / (train_matrix_norm + 1e-12)
    ratio_matrix = np.clip(ratio_matrix, 0, 1000)  # Limit to a reasonable range

    # Create heatmap for the ratio
    plt.figure(figsize=(14, 12))
    sns.heatmap(ratio_matrix, 
            cmap="coolwarm", center=0,  # Only show meaningful transitions
            vmin=-20, vmax=100)  # Limit color scale
    plt.title("Ratio of Test/Train EC Pair Frequencies")
    plt.xlabel("To EC")
    plt.ylabel("From EC")
    plt.tight_layout()
    plt.savefig("ec_pair_ratio.png")


    # Additionally, create a visualization of the most imbalanced pairs
    top_pairs = sorted_differences[:30]  # Adjust number as needed

    plt.figure(figsize=(14, 10))
    pair_labels = [f"({e1},{e2})" for (e1, e2), _ in top_pairs]
    ratios = [ratio for _, (_, _, ratio, _) in top_pairs]

    y_pos = np.arange(len(pair_labels))
    plt.barh(y_pos, ratios, align='center')
    plt.yticks(y_pos, pair_labels)
    plt.gca().invert_yaxis()  # Labels read top-to-bottom
    plt.xlabel('Test/Train Frequency Ratio')
    plt.title('Top Imbalanced EC Pairs (Test vs Training)')
    plt.tight_layout()
    plt.savefig("top_imbalanced_pairs.png")

    # Create separate visualizations for training and test frequencies
    plt.figure(figsize=(14, 12))
    train_log = np.log10(train_matrix_norm + 1e-10)
    sns.heatmap(train_log, 
            cmap="viridis",
            mask=(train_matrix == 0))
    plt.title("Log10 of Training EC Pair Frequencies")
    plt.xlabel("To EC")
    plt.ylabel("From EC")
    plt.tight_layout()
    plt.savefig("training_ec_frequencies.png")


    plt.figure(figsize=(14, 12))
    test_log = np.log10(test_matrix_norm + 1e-10)
    sns.heatmap(test_log, 
            cmap="viridis",
            mask=(test_matrix == 0))
    plt.title("Log10 of Test EC Pair Frequencies")
    plt.xlabel("To EC")
    plt.ylabel("From EC")
    plt.tight_layout()
    plt.savefig("test_ec_frequencies.png")

    missing = []
    for key in test_ecs:
        if key not in training_ecs:
            missing.append(key)

    print("Missing Equivalence Classes:", missing)

    total_training = sum(training_ecs.values())
    total_test = sum(test_ecs.values())

    training_ecs_normalized = {k: v / total_training for k, v in training_ecs.items()}
    test_ecs_normalized = {k: v / total_test for k, v in test_ecs.items()}

    # Convert dictionaries to pandas Series
    s1 = pd.Series(training_ecs_normalized)
    s2 = pd.Series(test_ecs_normalized)

    # Get all unique keys
    all_keys = sorted(set(s1.index) | set(s2.index))

    ratios = {}
    for key in all_keys:
        if key not in s1:
            training_ecs_normalized[key] = 0
        if key not in s2:
            test_ecs_normalized[key] = 0
        # get ratio
        ratios[key] = test_ecs_normalized[key] / training_ecs_normalized[key] if training_ecs_normalized[key] > 0 else 1000
    
    # sort ratios
    ratios = {k: v for k, v in sorted(ratios.items(), key=lambda item: item[1])}
    for key in ratios:
        print(key, ratios[key])

    # Create a DataFrame with all keys and fill missing values with 0
    df = pd.DataFrame({
        'Dict1': [s1.get(k, 0) for k in all_keys],
        'Dict2': [s2.get(k, 0) for k in all_keys]
    }, index=all_keys)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(15, 8))

    # Define the width of the bars
    bar_width = 0.35

    # Define the positions of the bars
    x = np.arange(len(all_keys))

    # Create the bars
    ax.bar(x - bar_width/2, df['Dict1'], bar_width, color='cyan', alpha=0.8, label='Training')
    ax.bar(x + bar_width/2, df['Dict2'], bar_width, color='blue', alpha=0.8, label='Test')

    # Add some text for labels, title and custom x-axis tick labels
    ax.set_xlabel('ECs')
    ax.set_ylabel('Counts')
    ax.set_title('Distribution shift')
    ax.set_xticks(x)
    ax.set_xticklabels(all_keys, rotation=45)
    ax.legend()

    # Add grid lines for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.set_yscale('log')
    # Adjust layout to make sure everything fits
    plt.tight_layout()
    plt.savefig("distribution_shift.png")


    # fig, ax = visualize_counter_heatmap(counters, lengths, title="Equivalence Class Heatmap", figsize=(10, 8), cmap="viridis")
    # fig.savefig("equivalence_class_heatmap_tt.png")

    # lengths = test_lengths
    # counters = []
    # for length in tqdm(lengths):
    #     batch = task.generate_positives(jax.random.PRNGKey(0), 128, length).astype(int)
    #     seen_classes = get_batch_equivalence_classes_matrix(batch, jax_dfa, hash_table)
    #     # print(seen_classes)
    #     flat_tensor =  seen_classes.flatten()
    #     np_array = np.asarray(flat_tensor)
    #     np_array = np_array[np_array != -1]
    #     counts = Counter(np_array)
    #     counters.append(counts)

    # fig, ax = visualize_counter_heatmap(counters, lengths, title="Equivalence Class Heatmap", figsize=(10, 8), cmap="viridis")
    # fig.savefig("equivalence_class_heatmap_test.png")

if __name__ == "__main__":
    # ns = [2, 3, 4, 6, 8, 12]
    # max_word_lens = [6, 8, 10, 12, 16, 24]
    ns = [6]
    max_word_lens = [12]
    for n, max_word_length in zip(ns, max_word_lens):
        dfa_example(max_depth=n, max_word_length=max_word_length)