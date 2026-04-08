"""Compute the answer to a modular arithmetic expression."""
# Original source: Delétang et al. (2023)
# We made some edits to ensure that the task evaluates the expression sequentially from left to right, following the Moore machine given in the appendix.

import functools
from typing import Optional, Sequence

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

from ldru.tasks import task

# Public as this may be used to encode/decode strings of numbers/symbols.
OP_BY_CHARACTER = {'+': 0, '-': 1, '*': 2, '_': 3}

def _replace_blanks(expression: jnp.ndarray, modulus: int) -> jnp.ndarray:
    """Replaces blank symbols in expression with either `+` or `0`."""
    mask = (expression == OP_BY_CHARACTER['_'] + modulus)
    operator_mask = mask.at[::2].set(False)
    residual_mask = mask.at[1::2].set(False)

    blanks_replaced = jnp.where(operator_mask, OP_BY_CHARACTER['+'] + modulus, expression)
    blanks_replaced = jnp.where(residual_mask, 0, blanks_replaced)
    return blanks_replaced

def _evaluate_single_operation(carry: jnp.ndarray, x: tuple[jnp.ndarray, jnp.ndarray], modulus: int) -> jnp.ndarray:
    """Evaluates a single operation between the carry and the next number under modulus.
    
    Args:
        carry: Current accumulated result
        x: Tuple of (operator, number) for the current operation
        modulus: Modulus for arithmetic
        
    Returns:
        New accumulated result after applying operation
    """
    op, num = x
    op = op - modulus  # Convert back to original operator encoding
    # jax.debug.print("carry {carry}, op {op}, num {num}", carry=carry, op=op, num=num)
    # Using where instead of if/else for jit compatibility
    result = jnp.where(
        op == OP_BY_CHARACTER['+'],
        (carry + num) % modulus,
        jnp.where(
            op == OP_BY_CHARACTER['-'],
            (carry - num) % modulus,
            (carry * num) % modulus  # multiplication case
        )
    )
    return result

def _evaluate_expression(expression: jnp.ndarray, modulus: int) -> jnp.ndarray:
    """Returns the result of evaluating a modular arithmetic expression from left to right."""
    expression = _replace_blanks(expression, modulus)
    
    # Initialize with first number
    init_carry = expression[0]
    
    # Prepare operators and numbers for scanning
    ops = expression[1::2]  # Odd indices are operators
    nums = expression[2::2]  # Even indices after first are numbers
    
    # Scan through the expression evaluating each operation
    return jax.lax.fori_loop(
        0,
        ops.shape[0],
        lambda i, acc: _evaluate_single_operation(
            acc,
            (ops[i], nums[i]),
            modulus
        ),
        init_carry
    )

class ModularArithmetic(task.GeneralizationTask):
    """A task with the goal of reducing a simple arithmetic expression.
    
    The input is a string, composed of numbers (in {0, ..., modulus-1}), and
    operators (in {+, -, *}). The output is the reduced value of this expression,
    which is also in {0, ..., modulus-1}. Operations are performed strictly left
    to right, with modulus applied after each operation.

    Examples (modulo 5):
        1 + 2 * 3 = ((1 + 2) % 5 * 3) % 5 = (3 * 3) % 5 = 4
        1 - 1 - 1 = ((1 - 1) % 5 - 1) % 5 = (0 - 1) % 5 = 4
        0 * 1 + 4 * 3 = (((0 * 1) % 5 + 4) % 5 * 3) % 5 = ((0 + 4) % 5 * 3) % 5 = (4 * 3) % 5 = 2
    """

    def __init__(
        self,
        modulus: int = 5,
        operators: Optional[Sequence[str]] = None,
    ) -> None:
        """Initializes the modular arithmetic task.

        Args:
            modulus: The modulus used for the computation. We use 5 in the paper.
            operators: Operators to be used in the sequences. By default it's None,
                meaning all operators available are used.
        """
        self._modulus = modulus
        if operators is None:
            operators = ('+', '-', '*')
        self._operators = (OP_BY_CHARACTER[op] for op in operators)
        self.ops = self._modulus + jnp.array(list(self._operators))

    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def sample_batch(
        self,
        rng: jnp.ndarray,
        batch_size: int,
        length: int,
    ) -> task.Batch:
        """Returns a batch of modular arithmetic expressions and their labels.

        Args:
            rng: The jax random number generator.
            batch_size: The size of the batch returned.
            length: The length of the sequence. As this length must be odd for the
                modular arithmetic dataset, if it's not, we force it to be by
                subtracting one to the length passed.
        """
        # Subtracting one to the length if it's not odd already.
        if length % 2 != 1:
            length -= 1

        batch = jnp.empty((batch_size, length), dtype=int)
        rng1, rng2 = jax.random.split(rng)
        remainders = jax.random.randint(
            rng1, (batch_size, length // 2 + 1), 0, self._modulus
        )

        operations = jrandom.choice(rng2, self.ops, (batch_size, length // 2))
        batch = batch.at[:, ::2].set(remainders)
        expressions = batch.at[:, 1::2].set(operations)

        evaluate = functools.partial(_evaluate_expression, modulus=self._modulus)
        if length == 1:
            labels = expressions[:, 0]
        else:
            labels = jax.vmap(evaluate)(expressions)
            
        labels = jnn.one_hot(labels, self._modulus)
        one_hot_expressions = jnn.one_hot(
            expressions, self._modulus + len(OP_BY_CHARACTER)
        )
        
        return {'input': one_hot_expressions, 'output': labels}

    @property
    def input_size(self) -> int:
        """Returns the input size for the models."""
        return self._modulus + len(OP_BY_CHARACTER)

    @property
    def output_size(self) -> int:
        """Returns the output size for the models."""
        return self._modulus
    
    def decode(self, encoded: jnp.ndarray) -> str:
        """Decodes a one-hot encoded expression into a human-readable string."""
        def decode_sample(sample: jnp.ndarray) -> str:
            return ''.join([
                str(i) if i < self._modulus 
                else list(OP_BY_CHARACTER.keys())[i - self._modulus] 
                for i in sample.argmax(axis=-1)
            ])
        return '\n'.join([decode_sample(encoded[i]) for i in range(encoded.shape[0])])

if __name__ == "__main__":
    # Test the implementation with some examples
    task = ModularArithmetic()
    batch = task.sample_batch(jax.random.PRNGKey(0), 10, 10)
    
    print("Generated Expressions:")
    print(task.decode(batch['input']))
    print("\nGenerated Labels:")
    print(batch['output'].argmax(axis=-1))
    def decode_sample(sample: jnp.ndarray) -> str:
        return ''.join([
            str(i) if i < task._modulus 
            else list(OP_BY_CHARACTER.keys())[i - task._modulus] 
            for i in sample.argmax(axis=-1)
        ])
    inpt = jnp.array([[2, 6, 3, 7, 2]])
    print(inpt.shape)
    print("Decoded", task.decode(jnn.one_hot(inpt, task._modulus + len(OP_BY_CHARACTER))))
    outpt = _evaluate_expression(inpt[0], 5)
    print("\nEvaluation Example:")
    print(f"Input: {inpt}")
    print(f"Output: {outpt}")