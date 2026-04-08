# Haiku version of SD-SSM model
# Source: Terzić et al. (2025)

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from typing import Optional, Callable


class SDSSMBlock(hk.Module):
    def __init__(self,
                 embed_size: int,
                 hidden_size: int,
                 num_A: int = 2,
                 Lp_norm: float = 1.2,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.Lp_norm = Lp_norm
        self.num_A = num_A
        self.embed_size = embed_size

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, L, D = x.shape
        
        # Linear layers
        A_selector = hk.Linear(self.num_A, name="A_selector")
        B_layer = hk.Linear(self.hidden_size, with_bias=False, name="B")
        
        # Initialize A_dict parameter
        initializer_bound = np.sqrt(6) / np.sqrt(2 * self.hidden_size)
        A_dict_init = hk.initializers.RandomUniform(
            minval=-initializer_bound, 
            maxval=initializer_bound
        )
        A_dict = hk.get_parameter(
            "A_dict", 
            shape=[self.hidden_size, self.hidden_size, self.num_A], 
            init=A_dict_init
        )
        
        # Initialize output and hidden state
        output = jnp.zeros((B, L, self.hidden_size))
        hidden_state = jnp.zeros((B, self.hidden_size))
        
        # Get selections and inputs
        selections = jax.nn.softmax(A_selector(x), axis=-1)  # B x L x K
        inputs_ = B_layer(x)  # B x L x N
        
        def scan_fn(hidden_state, inputs):
            i, selection, input_i = inputs
            
            # Compute weighted A matrix: einsum('n1 n2 k, b k -> b n1 n2')
            A = jnp.einsum('ijk,bk->bij', A_dict, selection)
            
            # Lp normalization of A
            A_norms = jnp.linalg.norm(A, ord=self.Lp_norm, axis=2, keepdims=True)
            transition_matrix = A / A_norms
            
            # Update hidden state: einsum('b n1 n2, b n2 -> b n1')
            hidden_state = jnp.einsum('bij,bj->bi', transition_matrix, hidden_state) + input_i
            
            return hidden_state, hidden_state
        
        # Prepare scan inputs
        indices = jnp.arange(L)
        scan_inputs = (indices, selections.transpose(1, 0, 2), inputs_.transpose(1, 0, 2))
        
        # Run scan
        _, output_states = jax.lax.scan(scan_fn, hidden_state, scan_inputs)
        
        # Transpose back to (B, L, hidden_size)
        output = output_states.transpose(1, 0, 2)
        
        return output


class SDSSM(hk.Module):
    def __init__(self,
                 output_size: int,
                 input_size: int,
                 state_size: int,
                 num_transition_matrices: int = 2,
                 return_all_outputs: bool = False,
                 Lp_norm: float = 1.2,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.return_all_outputs = return_all_outputs
        self.output_size = output_size
        self.input_size = input_size
        self.state_size = state_size
        self.num_transition_matrices = num_transition_matrices
        self.Lp_norm = Lp_norm

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # SDSSM block
        block = SDSSMBlock(
            embed_size=self.input_size,
            hidden_size=self.state_size,
            num_A=self.num_transition_matrices,
            Lp_norm=self.Lp_norm,
            name="sdssm_block"
        )
        
        rnn_out = block(x)
        
        # Layer normalization
        rnn_out_norm = hk.LayerNorm(
            axis=-1, 
            create_scale=True, 
            create_offset=True,
            name="layer_norm"
        )(rnn_out)
        
        # Readout layer
        readout = hk.Linear(self.output_size, name="readout")
        output = readout(rnn_out_norm)
        
        if not self.return_all_outputs:
            output = output[:, -1, :]
        
        return output


def make_sdssm(
    output_size: int,
    state_size: int,
    return_all_outputs: bool = False,
    num_transition_matrices: int = 2,
    Lp_norm: float = 1.2) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Returns an SDSSM model, not haiku transformed.

    Args:
        output_size: The output size of the model.
        state_size: The hidden state size of the SDSSM.
        return_all_outputs: Whether to return the whole sequence of outputs.
        num_transition_matrices: Number of learnable transition matrices.
        Lp_norm: The Lp norm used for normalizing transition matrices.
    """
    
    def sdssm_model(x: jnp.ndarray) -> jnp.ndarray:
        model = SDSSM(
            output_size=output_size,
            input_size=x.shape[-1],
            state_size=state_size,
            num_transition_matrices=num_transition_matrices,
            return_all_outputs=return_all_outputs,
            Lp_norm=Lp_norm
        )
        return model(x)
    
    return sdssm_model


# Example of how to use the model:
if __name__ == "__main__":

    batch_size = 32
    seq_length = 100
    input_size = 64
    state_size = 128
    output_size = 10
    
    model_fn = make_sdssm(
        output_size=output_size,
        state_size=state_size,
        return_all_outputs=False,
        num_transition_matrices=2,
        Lp_norm=1.2
    )
    
    model = hk.transform(model_fn)
    
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((batch_size, seq_length, input_size))
    params = model.init(rng, dummy_input)
    
    output = model.apply(params, rng, dummy_input)
    print(f"Output shape: {output.shape}") 
    
    # Example with return_all_outputs=True
    model_fn_all = make_sdssm(
        output_size=output_size,
        state_size=state_size,
        return_all_outputs=True,
        num_transition_matrices=2,
        Lp_norm=1.2
    )
    
    model_all = hk.transform(model_fn_all)
    params_all = model_all.init(rng, dummy_input)
    output_all = model_all.apply(params_all, rng, dummy_input)
    print(f"Output shape (all outputs): {output_all.shape}")  # Should be (32, 100, 10)