# Transformation of the S5 model in Flax to Haiku comptatible format
# Source: Smith, Warrington, and Linderman (2023)
# We try to keep the structure as similar as possible to this repo, but it uses an older version of jax and gives an error regarding `jax.numpy.DeviceArray` which should all be replaced with `jax.array` in the versions we use

import jax.numpy as jnp
from jax.scipy.linalg import block_diag

import haiku as hk
from functools import partial

from s5.seq_model import BatchClassificationModel
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO

from haiku._src.flax.transform_flax import lift 


def make_s5_haiku(
    output_size: int,
    return_all_outputs: bool = False,
    embedding_dim: int = 256,
    num_layers: int = 1,
    ssm_size_base: int = 256,
    blocks: int = 8,
    activation: str = 'gelu',
    dropout_prob: float = 0.1,
    mode: str = 'pool',
    prenorm: bool = False,
    batchnorm: bool = False,
    bn_momentum: float = 0.9,
    conj_sym: bool = True,
    clip_eigs: bool = True,
    bidirectional: bool = False,
    discretization: str = 'zoh',
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    C_init: str = 'trunc_standard_normal',
    padded: bool = False, 
    integration_timesteps_value: float = 1.0,
    **kwargs
) -> callable:
    """Returns a Haiku-compatible S5 model function.

    Args:
        output_size: The output size of the model.
        return_all_outputs: Whether to return the whole sequence of outputs or just the last one.
        input_window: The number of tokens that are fed at once (for compatibility, currently unused).
        embedding_dim: Model dimension (feature size).
        num_layers: Number of S5 layers to stack.
        ssm_size_base: Base size of the state space model.
        blocks: Number of blocks for block-diagonal structure.
        activation: Activation function to use.
        dropout_prob: Dropout rate.
        mode: 'pool' for mean pooling, 'last' for last state.
        prenorm: Whether to use prenorm or postnorm.
        batchnorm: Whether to use batchnorm or layernorm.
        bn_momentum: Batchnorm momentum if batchnorm is used.
        conj_sym: Whether to enforce conjugate symmetry.
        clip_eigs: Whether to clip eigenvalues.
        bidirectional: Whether to use bidirectional model.
        discretization: Discretization method ('zoh' or 'bilinear').
        dt_min: Minimum timescale value.
        dt_max: Maximum timescale value.
        C_init: C matrix initialization method.
        padded: Whether input uses padding.
        integration_timesteps_value: Value to use for integration timesteps.
        **kwargs: Additional arguments.
    """
    
    block_size = int(ssm_size_base / blocks)
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)
    
    if conj_sym:
        block_size = block_size // 2
        ssm_size = ssm_size_base // 2
    else:
        ssm_size = ssm_size_base
    
    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T
    
    Lambda = (Lambda * jnp.ones((blocks, block_size))).ravel()
    V = block_diag(*([V] * blocks))
    Vinv = block_diag(*([Vc] * blocks))
    
    ssm_init_fn = init_S5SSM(
        H=embedding_dim,
        P=ssm_size,
        Lambda_re_init=Lambda.real,
        Lambda_im_init=Lambda.imag,
        V=V,
        Vinv=Vinv,
        C_init=C_init,
        discretization=discretization,
        dt_min=dt_min,
        dt_max=dt_max,
        conj_sym=conj_sym,
        clip_eigs=clip_eigs,
        bidirectional=bidirectional
    )
    
    actual_mode = 'last' if not return_all_outputs else mode
    
    # Create the Flax model class
    model_cls = partial(
        BatchClassificationModel,
        ssm=ssm_init_fn,
        d_output=output_size,
        d_model=embedding_dim,
        n_layers=num_layers,
        padded=padded,
        activation=activation,
        dropout=dropout_prob,
        mode=actual_mode,
        prenorm=prenorm,
        batchnorm=batchnorm,
        bn_momentum=bn_momentum,
    )

    def haiku_s5_model(x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Haiku-compatible S5 model function.
        
        Args:
            x: Input array of shape (batch_size, seq_length, input_dim)
            
        Returns:
            Output array of shape (batch_size, output_size) if return_all_outputs=False,
            or (batch_size, seq_length, output_size) if return_all_outputs=True
        """
        batch_size, seq_length, input_dim = x.shape
        
        # Create integration_timesteps
        integration_timesteps = jnp.full((batch_size, seq_length), integration_timesteps_value)
        
        # Create and lift the Flax model
        flax_model = model_cls()
        lifted_model = lift(flax_model, name="s5_model")
        
        if padded:
            lengths = jnp.full((batch_size,), seq_length, dtype=jnp.int32)
            model_input = (x, lengths)
        else:
            model_input = x
        
        # Create RNG keys for flax dropout
        dropout_rng = hk.next_rng_key() if hk.running_init() or dropout_prob > 0 else None
        rngs = {'dropout': dropout_rng} if dropout_rng is not None else {}
        
        # Apply the lifted model
        output = lifted_model(model_input, integration_timesteps, rngs=rngs)
        
        return output
    
    return haiku_s5_model