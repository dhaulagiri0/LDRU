import datetime
import os
import sys
from typing import Optional
import jax
import jax.numpy as jnp
import optuna
from causal_ldru_v2 import create_causal_ldru_model
from train_causal_ldru import (
    LDRUExperimenstConfig,
    create_lstm_model,
    create_transformer_model,
    resolve_binary_operator,
    train_model,
)


def compute_model_param_count(
    model_creation_fn,
    config: LDRUExperimenstConfig,
    seq_length: int,
    seed: int,
    use_lstm: bool,
) -> int:
    """Initialize model parameters on a dummy batch and count total parameters."""
    model = model_creation_fn(config)
    input_length = max(1, seq_length - 1) if use_lstm else max(1, seq_length)
    dummy_batch = jnp.zeros((1, input_length), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(seed), dummy_batch)
    return int(sum(x.size for x in jax.tree.leaves(params)))


def _parse_candidates(raw_values: str, cast, arg_name: str):
    values = []
    for token in raw_values.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(cast(token))
    if not values:
        raise ValueError(f"--{arg_name} must contain at least one value.")
    return values


def _parse_bool_candidates(raw_values: str, arg_name: str):
    true_tokens = {"1", "true", "t", "yes", "y", "on"}
    false_tokens = {"0", "false", "f", "no", "n", "off"}
    values = []
    for token in raw_values.split(","):
        normalized = token.strip().lower()
        if not normalized:
            continue
        if normalized in true_tokens:
            values.append(True)
        elif normalized in false_tokens:
            values.append(False)
        else:
            raise ValueError(
                f"Unsupported boolean token '{token}' in --{arg_name}. "
                "Use comma-separated true/false style values."
            )
    if not values:
        raise ValueError(f"--{arg_name} must contain at least one value.")
    return values


def configure_output(file_path: Optional[str]):
    """Redirect stdout/stderr to the given file path (append, line-buffered).

    Provide a path via environment variable LDRU_PRINT_FILE to enable redirection
    without changing function signatures. If file_path is None or empty, do nothing.
    """
    if not file_path:
        return

    # Ensure directory exists
    print(f"directory provided: {file_path}")
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # Open file in append mode with line buffering
    f = open(file_path, "a", buffering=1)

    # Redirect stdout and stderr
    sys.stdout = f
    sys.stderr = f

    # Log the redirection timestamp
    print(
        f"[{datetime.datetime.now().isoformat()}] Redirecting stdout/stderr to: {file_path}"
    )


def make_objective(
    model_creation_fn,
    args,
    use_lstm=False,
    use_transformer=False,
    seq2seq=True,
):
    hidden_dim_candidates = _parse_candidates(
        args.hidden_dim_candidates, int, "hidden_dim_candidates"
    )
    embedding_dim_candidates = _parse_candidates(
        args.embedding_dim_candidates, int, "embedding_dim_candidates"
    )
    initial_learning_rate_candidates = _parse_candidates(
        args.initial_learning_rate_candidates, float, "initial_learning_rate_candidates"
    )
    warmup_steps_candidates = _parse_candidates(
        args.warmup_steps_candidates, int, "warmup_steps_candidates"
    )
    warmup_enabled_candidates = _parse_bool_candidates(
        args.warmup_enabled_candidates, "warmup_enabled_candidates"
    )
    prenorm_gelu_candidates = _parse_bool_candidates(
        args.ldru_prenorm_gelu_block_candidates,
        "ldru_prenorm_gelu_block_candidates",
    )
    tie_embeddings_candidates = _parse_bool_candidates(
        args.tie_embeddings_ldru_candidates, "tie_embeddings_ldru_candidates"
    )
    use_multi_operator_candidates = _parse_bool_candidates(
        args.use_multi_operator_ldru_candidates, "use_multi_operator_ldru_candidates"
    )
    num_operators_candidates = _parse_candidates(
        args.num_operators_candidates, int, "num_operators_candidates"
    )
    operator_min_weight_candidates = _parse_candidates(
        args.operator_min_weight_candidates, float, "operator_min_weight_candidates"
    )
    l2_lambda_candidates = _parse_candidates(
        args.l2_lambda_candidates, float, "l2_lambda_candidates"
    )
    num_transformer_heads_candidates = _parse_candidates(
        args.num_transformer_heads_candidates, int, "num_transformer_heads_candidates"
    )
    if any(v <= 0 for v in embedding_dim_candidates):
        raise ValueError("--embedding_dim_candidates values must be > 0.")
    if any(v <= 0 for v in hidden_dim_candidates):
        raise ValueError("--hidden_dim_candidates values must be > 0.")
    if any(v <= 0 for v in initial_learning_rate_candidates):
        raise ValueError("--initial_learning_rate_candidates values must be > 0.")
    if any(v < 0 for v in warmup_steps_candidates):
        raise ValueError("--warmup_steps_candidates values must be >= 0.")
    if any(v <= 0 for v in num_operators_candidates):
        raise ValueError("--num_operators_candidates values must be > 0.")
    if any(v < 0 for v in operator_min_weight_candidates):
        raise ValueError("--operator_min_weight_candidates values must be >= 0.")

    def objective(trial):
        # Define the hyperparameter search space
        num_layers = trial.suggest_int("num_layers", args.num_layers_min, args.num_layers_max)
        if use_transformer:
            num_transformer_heads = trial.suggest_categorical(
                "num_transformer_heads", num_transformer_heads_candidates
            )
            num_transformer_layers = trial.suggest_int(
                "num_transformer_layers",
                args.num_transformer_layers_min,
                args.num_transformer_layers_max,
            )
        hidden_dim = trial.suggest_categorical("hidden_dim", hidden_dim_candidates)
        embedding_dim = trial.suggest_categorical(
            "embedding_dim", embedding_dim_candidates
        )
        dropout_prob = trial.suggest_float(
            "dropout_prob",
            args.dropout_min,
            args.dropout_max,
            step=args.dropout_step,
        )
        initial_learning_rate = trial.suggest_categorical(
            "initial_learning_rate", initial_learning_rate_candidates
        )
        use_warmup = trial.suggest_categorical("use_warmup", warmup_enabled_candidates)
        warmup_steps = (
            trial.suggest_categorical("warmup_steps", warmup_steps_candidates)
            if use_warmup
            else 0
        )
        tie_embeddings_ldru = trial.suggest_categorical(
            "tie_embeddings_ldru", tie_embeddings_candidates
        )
        ldru_prenorm_gelu_block = trial.suggest_categorical(
            "ldru_prenorm_gelu_block", prenorm_gelu_candidates
        )
        use_multi_operator_ldru = trial.suggest_categorical(
            "use_multi_operator_ldru", use_multi_operator_candidates
        )
        if use_multi_operator_ldru:
            num_operators = trial.suggest_categorical(
                "num_operators", num_operators_candidates
            )
            operator_min_weight = trial.suggest_categorical(
                "operator_min_weight", operator_min_weight_candidates
            )
        else:
            # Keep these inert when multi-operator is disabled so candidate-list
            # changes do not affect non-multi-operator trials.
            num_operators = 1
            operator_min_weight = 0.0
        l2_lambda = trial.suggest_categorical("l2_lambda", l2_lambda_candidates)

        if use_multi_operator_ldru and hidden_dim % num_operators != 0:
            raise optuna.TrialPruned(
                "Pruned trial: hidden_dim must be divisible by sampled num_operators "
                f"when use_multi_operator_ldru is enabled (hidden_dim={hidden_dim}, "
                f"num_operators={num_operators})."
            )
        if num_operators <= 0:
            raise optuna.TrialPruned(
                f"Pruned trial: num_operators must be > 0 (got {num_operators})."
            )
        if operator_min_weight < 0:
            raise optuna.TrialPruned(
                "Pruned trial: operator_min_weight must be >= 0 "
                f"(got {operator_min_weight})."
            )
        if operator_min_weight >= 1.0 / num_operators:
            raise optuna.TrialPruned(
                "Pruned trial: operator_min_weight must be < 1/num_operators "
                f"(operator_min_weight={operator_min_weight}, "
                f"num_operators={num_operators})."
            )

        # Create the model configuration
        operator_name = "grc" if args.use_grc else args.binary_operator
        config = LDRUExperimenstConfig(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            max_sequence_length=max(args.max_sequence_length, args.seq_length),
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob,
            initial_learning_rate=initial_learning_rate,
            l2_lambda=l2_lambda,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            vocab_size=args.vocab_size,
            num_epochs=args.epochs_per_trial,  # Fixed number of epochs for faster search
            num_transformer_heads=num_transformer_heads if use_transformer else None,
            num_transformer_layers=num_transformer_layers if use_transformer else None,
            min_learning_rate=initial_learning_rate
            / 1000,  # Set a minimum learning rate for the scheduler
            use_positional_encoding=(True if use_transformer else False),
            use_alibi=(args.use_alibi if use_transformer else False),
            operator=resolve_binary_operator(operator_name),
            binop_expansion_factor=args.binop_expansion_factor,
            ablation_expansion_mode=args.ablation_expansion_mode,
            ablation_combine_mode=args.ablation_combine_mode,
            tie_embeddings_ldru=tie_embeddings_ldru,
            ldru_prenorm_gelu_block=ldru_prenorm_gelu_block,
            tie_embeddings=tie_embeddings_ldru,
            prenorm_gelu_block=ldru_prenorm_gelu_block,
            use_multi_operator_ldru=use_multi_operator_ldru,
            num_operators=num_operators,
            operator_min_weight=operator_min_weight,
        )

        tolerance = (
            int(args.param_count_tolerance_abs)
            if args.param_count_tolerance_abs is not None
            else int(args.target_param_count * args.param_count_tolerance_ratio)
        )
        lower_bound = args.target_param_count - tolerance
        upper_bound = args.target_param_count + tolerance
        param_count = compute_model_param_count(
            model_creation_fn=model_creation_fn,
            config=config,
            seq_length=args.seq_length,
            seed=args.param_count_seed,
            use_lstm=use_lstm,
        )
        trial.set_user_attr("param_count", param_count)
        trial.set_user_attr("target_param_count", args.target_param_count)
        trial.set_user_attr("param_count_tolerance", tolerance)
        trial.set_user_attr("param_count_lower_bound", lower_bound)
        trial.set_user_attr("param_count_upper_bound", upper_bound)
        trial.set_user_attr("param_count_delta", param_count - args.target_param_count)
        if not (lower_bound <= param_count <= upper_bound):
            raise optuna.TrialPruned(
                "Pruned trial: parameter count out of bounds "
                f"(count={param_count:,}, target={args.target_param_count:,}, "
                f"allowed=[{lower_bound:,}, {upper_bound:,}])."
            )

        trial_model_name_prefix = (
            f"{args.model_name_prefix}_trial{trial.number}"
            if args.model_name_prefix
            else f"trial{trial.number}"
        )
        train_kwargs = dict(
            log_dir=args.tensorboard_log_dir,
            config=config,
            enable_logging=args.enable_logging,
            model_creation_fn=model_creation_fn,
            use_lstm=use_lstm,
            use_transformer=use_transformer,
            use_transformer_ldru=False,
            seq2seq=seq2seq,
            checkpoint_dir=args.checkpoint_dir,
            resume_from_checkpoint=None,
            tokenizer_path=args.tokenizer_path,
            model_prefix=trial_model_name_prefix,
            generate_samples=False,  # Disable sample generation during hyperparameter search for faster trials
            streaming_shuffle_buffer_size=args.streaming_shuffle_buffer_size,
            streaming_chunk_line_buffer=args.streaming_chunk_line_buffer,
            optimizer_name=args.optimizer,
            train_steps_per_epoch=args.train_steps_per_epoch,
            validation_steps_per_epoch=args.validation_steps_per_epoch,
            test_steps_per_epoch=args.test_steps_per_epoch,
            compute_dtype=args.compute_dtype,
            train_stride=args.train_stride,
            train_seq_bin_path=args.train_seq_bin,
            val_seq_bin_path=args.val_seq_bin,
            test_seq_bin_path=args.test_seq_bin,
            seq_bin_dtype=args.seq_bin_dtype,
            seq_bin_format=args.seq_bin_format,
            seq_meta_json=args.seq_meta_json,
            nanogpt_ppl_metric=args.nanogpt_ppl_metric,
            warmup_steps=warmup_steps,
        )
        if args.dataset_mode == "text":
            train_kwargs["text_file_path"] = args.train_text_file
            train_kwargs["validation_text_file_path"] = args.validation_text_file
            train_kwargs["test_text_file_path"] = args.test_text_file
        else:
            train_kwargs["text_file_path"] = None
            train_kwargs["validation_text_file_path"] = None
            train_kwargs["test_text_file_path"] = None

        # Call train_model and capture the best validation perplexity
        _, _, _, _, best_val_perplexity = train_model(**train_kwargs)

        # Return the best validation perplexity as the objective value
        return best_val_perplexity

    return objective


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter search for LDRU model")
    parser.add_argument(
        "--num_trials",
        type=int,
        default=10,
        help="Number of hyperparameter trials to run",
    )
    parser.add_argument(
        "--enable_logging", action="store_true", help="Enable logging to TensorBoard"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="causal_ldru",
        choices=["causal_ldru", "transformer", "lstm"],
        help="Type of model to train (LDRU, lstm or transformer)",
    )
    parser.add_argument(
        "--seq_length",
        "--max_seq_len",
        type=int,
        default=256,
        help="Sequence length to use for training and evaluation",
    )
    parser.add_argument(
        "--epochs_per_trial",
        type=int,
        default=20,
        help="Number of epochs to train for each trial (default: 20)",
    )
    parser.add_argument(
        "--vocab_size",
        "--max_vocab_size",
        type=int,
        default=8000,
        help="Vocabulary size to use for training (default: 16000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size to use for training (default: 512)",
    )
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="text",
        choices=["text", "pretokenized_bin"],
        help="Dataset input mode: raw text files or pretokenized sequence binaries.",
    )
    parser.add_argument(
        "--train_text_file",
        type=str,
        default="wikitext-2-raw-train.txt",
        help="Path to training text file when --dataset_mode text.",
    )
    parser.add_argument(
        "--validation_text_file",
        type=str,
        default="wikitext-2-raw-validation.txt",
        help="Path to validation text file when --dataset_mode text.",
    )
    parser.add_argument(
        "--test_text_file",
        type=str,
        default=None,
        help="Optional path to test text file.",
    )
    parser.add_argument(
        "--train_seq_bin",
        type=str,
        default=None,
        help="Path to pretokenized train sequence binary when --dataset_mode pretokenized_bin.",
    )
    parser.add_argument(
        "--val_seq_bin",
        type=str,
        default=None,
        help="Optional path to pretokenized validation sequence binary.",
    )
    parser.add_argument(
        "--test_seq_bin",
        type=str,
        default=None,
        help="Optional path to pretokenized test sequence binary.",
    )
    parser.add_argument(
        "--seq_meta_json",
        type=str,
        default=None,
        help="Optional metadata JSON emitted by pretokenization.",
    )
    parser.add_argument(
        "--seq_bin_format",
        type=str,
        default="auto",
        choices=["auto", "sequence", "token_stream"],
        help="Binary format for pretokenized sequence files.",
    )
    parser.add_argument(
        "--seq_bin_dtype",
        type=str,
        default="uint16",
        choices=["uint16", "uint32", "int32"],
        help="Dtype used by pretokenized sequence binaries.",
    )
    parser.add_argument(
        "--train_stride",
        type=int,
        default=None,
        help="Stride for sequence windows; defaults to max_seq_len//2 when omitted.",
    )
    parser.add_argument(
        "--streaming_shuffle_buffer_size",
        type=int,
        default=8192,
        help="Streaming shuffle buffer size.",
    )
    parser.add_argument(
        "--streaming_chunk_line_buffer",
        type=int,
        default=4096,
        help="Streaming chunk line buffer size.",
    )
    parser.add_argument(
        "--train_steps_per_epoch",
        type=int,
        default=None,
        help="Optional cap on train steps per epoch.",
    )
    parser.add_argument(
        "--validation_steps_per_epoch",
        type=int,
        default=None,
        help="Optional cap on validation steps per epoch.",
    )
    parser.add_argument(
        "--test_steps_per_epoch",
        type=int,
        default=None,
        help="Optional cap on test steps per epoch.",
    )
    parser.add_argument(
        "--compute_dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16"],
        help="Compute dtype passed to train_model.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "amsgrad", "muon"],
        help="Optimizer name passed to train_model.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=512,
        help="Token embedding dimension.",
    )
    parser.add_argument(
        "--operator_min_weight",
        type=float,
        default=0.01,
        help="Minimum operator mixture weight floor for multi-operator LDRU.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=3072,
        help="Model max sequence length in config.",
    )
    parser.add_argument(
        "--model_name_prefix",
        type=str,
        default="",
        help="Prefix prepended to trial model names.",
    )
    parser.add_argument(
        "--tensorboard_log_dir",
        type=str,
        default="optuna_logs",
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="optuna_checkpoints",
        help="Checkpoint directory for trials.",
    )
    parser.add_argument(
        "--binary_operator",
        type=str,
        default="default",
        choices=["default", "binary", "convex_gated", "grc", "ablation"],
        help="Binary operator to use for LDRU composition.",
    )
    parser.add_argument(
        "--binop_expansion_factor",
        type=int,
        default=4,
        help="Expansion factor for compatible binary operators.",
    )
    parser.add_argument(
        "--ablation_expansion_mode",
        type=str,
        default="grc",
        choices=["binary", "grc"],
        help="Expansion mode for ablation operator.",
    )
    parser.add_argument(
        "--ablation_combine_mode",
        type=str,
        default="grc",
        choices=["binary", "grc"],
        help="Combine mode for ablation operator.",
    )
    parser.add_argument(
        "--tie_embeddings_ldru",
        action="store_true",
        default=False,
        help="Tie LDRU token embedding and output projection weights.",
    )
    parser.add_argument(
        "--ldru_prenorm_gelu_block",
        action="store_true",
        default=False,
        help="Enable LDRU pre-norm + GELU block.",
    )
    parser.add_argument(
        "--use_multi_operator_ldru",
        action="store_true",
        default=False,
        help="Use multi-operator LDRU composition.",
    )
    parser.add_argument(
        "--num_operators",
        type=int,
        default=4,
        help="Number of operators used when --use_multi_operator_ldru is set.",
    )
    parser.add_argument(
        "--nanogpt_ppl_metric",
        action="store_true",
        default=False,
        help="Report NanoGPT-style perplexity metric.",
    )
    parser.add_argument(
        "--embedding_dim_candidates",
        type=str,
        default=None,
        help=(
            "Comma-separated embedding-dim candidates. "
            "Defaults to the singleton value from --embedding_dim."
        ),
    )
    parser.add_argument(
        "--hidden_dim_candidates",
        type=str,
        default="128,256,512,1024",
        help="Comma-separated hidden-dim search candidates.",
    )
    parser.add_argument(
        "--initial_learning_rate_candidates",
        type=str,
        default="1e-6,5e-6,1e-5,5e-5,1e-4,5e-4,1e-3,5e-3",
        help="Comma-separated initial-learning-rate search candidates.",
    )
    parser.add_argument(
        "--l2_lambda_candidates",
        type=str,
        default="0.0,1e-6,5e-6,1e-5,5e-5,1e-4,1e-3,5e-3,1e-2,5e-2,1e-1,5e-1",
        help="Comma-separated l2-lambda search candidates.",
    )
    parser.add_argument(
        "--warmup_enabled_candidates",
        type=str,
        default=None,
        help=(
            "Comma-separated warmup toggle candidates (true/false). "
            "Defaults to 'false'."
        ),
    )
    parser.add_argument(
        "--warmup_steps_candidates",
        type=str,
        default=None,
        help=(
            "Comma-separated warmup-step candidates used when warmup is enabled. "
            "Defaults to singleton from --warmup_steps_default."
        ),
    )
    parser.add_argument(
        "--warmup_steps_default",
        type=int,
        default=0,
        help="Fallback warmup steps when --warmup_steps_candidates is omitted.",
    )
    parser.add_argument(
        "--ldru_prenorm_gelu_block_candidates",
        type=str,
        default=None,
        help=(
            "Comma-separated prenorm+GELU toggle candidates (true/false). "
            "Defaults to singleton from --ldru_prenorm_gelu_block."
        ),
    )
    parser.add_argument(
        "--tie_embeddings_ldru_candidates",
        type=str,
        default=None,
        help=(
            "Comma-separated weight-tying toggle candidates (true/false). "
            "Defaults to singleton from --tie_embeddings_ldru."
        ),
    )
    parser.add_argument(
        "--use_multi_operator_ldru_candidates",
        type=str,
        default=None,
        help=(
            "Comma-separated multi-operator toggle candidates (true/false). "
            "Defaults to singleton from --use_multi_operator_ldru."
        ),
    )
    parser.add_argument(
        "--num_operators_candidates",
        type=str,
        default=None,
        help=(
            "Comma-separated num_operators candidates. "
            "Defaults to singleton from --num_operators."
        ),
    )
    parser.add_argument(
        "--operator_min_weight_candidates",
        type=str,
        default=None,
        help=(
            "Comma-separated operator_min_weight candidates. "
            "Defaults to singleton from --operator_min_weight."
        ),
    )
    parser.add_argument(
        "--num_layers_min",
        type=int,
        default=1,
        help="Minimum num_layers for search.",
    )
    parser.add_argument(
        "--num_layers_max",
        type=int,
        default=4,
        help="Maximum num_layers for search.",
    )
    parser.add_argument(
        "--dropout_min",
        type=float,
        default=0.0,
        help="Minimum dropout probability for search.",
    )
    parser.add_argument(
        "--dropout_max",
        type=float,
        default=0.6,
        help="Maximum dropout probability for search.",
    )
    parser.add_argument(
        "--dropout_step",
        type=float,
        default=0.1,
        help="Dropout probability step size for search.",
    )
    parser.add_argument(
        "--num_transformer_heads_candidates",
        type=str,
        default="4,8,16",
        help="Comma-separated transformer-head candidates.",
    )
    parser.add_argument(
        "--num_transformer_layers_min",
        type=int,
        default=2,
        help="Minimum transformer-layer count for search.",
    )
    parser.add_argument(
        "--num_transformer_layers_max",
        type=int,
        default=8,
        help="Maximum transformer-layer count for search.",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="ldru_hyperparam_search",
        help="Name of the Optuna study (default: ldru_hyperparam_search)",
    )
    parser.add_argument(
        "--storage_url",
        type=str,
        default="sqlite:///optuna_study.db",
        help="URL for Optuna storage (e.g., sqlite:///optuna_study.db) to enable distributed optimization across multiple runs",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to a pre-trained tokenizer (optional, if not provided, a new tokenizer will be trained on the dataset)",
    )
    parser.add_argument(
        "--print_log_file",
        type=str,
        default=None,
        help="Path to a file to redirect stdout/stderr (optional, if not provided, output will go to console)",
    )
    parser.add_argument(
        "--use_alibi",
        action="store_false",
        default=True,
        help="Whether to use ALiBi positional bias in the transformer model (only applicable if --model_type is transformer)",
    )
    parser.add_argument(
        "--use_grc",
        action="store_true",
        default=False,
        help="Whether to use GRC operator in the LDRU model (only applicable if --model_type is causal_ldru)",
    )
    parser.add_argument(
        "--target_param_count",
        type=int,
        default=100_000_000,
        help="Target parameter count used to gate trials before training (default: 100000000).",
    )
    parser.add_argument(
        "--param_count_tolerance_ratio",
        type=float,
        default=0.10,
        help=(
            "Relative tolerance around --target_param_count when "
            "--param_count_tolerance_abs is not set (default: 0.10 for ±10%%)."
        ),
    )
    parser.add_argument(
        "--param_count_tolerance_abs",
        type=int,
        default=None,
        help=(
            "Absolute tolerance around --target_param_count. "
            "If provided, overrides --param_count_tolerance_ratio."
        ),
    )
    parser.add_argument(
        "--param_count_seed",
        type=int,
        default=42,
        help="Seed used for deterministic model-init parameter counting.",
    )
    args = parser.parse_args()
    if args.embedding_dim_candidates is None:
        args.embedding_dim_candidates = str(args.embedding_dim)
    if args.warmup_enabled_candidates is None:
        args.warmup_enabled_candidates = "false"
    if args.warmup_steps_candidates is None:
        args.warmup_steps_candidates = str(args.warmup_steps_default)
    if args.ldru_prenorm_gelu_block_candidates is None:
        args.ldru_prenorm_gelu_block_candidates = (
            "true" if args.ldru_prenorm_gelu_block else "false"
        )
    if args.tie_embeddings_ldru_candidates is None:
        args.tie_embeddings_ldru_candidates = (
            "true" if args.tie_embeddings_ldru else "false"
        )
    if args.use_multi_operator_ldru_candidates is None:
        args.use_multi_operator_ldru_candidates = (
            "true" if args.use_multi_operator_ldru else "false"
        )
    if args.num_operators_candidates is None:
        args.num_operators_candidates = str(args.num_operators)
    if args.operator_min_weight_candidates is None:
        args.operator_min_weight_candidates = str(args.operator_min_weight)

    configure_output(args.print_log_file)
    if args.num_trials <= 0:
        raise ValueError("--num_trials must be > 0.")
    if args.seq_length <= 0:
        raise ValueError("--seq_length/--max_seq_len must be > 0.")
    if args.vocab_size <= 0:
        raise ValueError("--vocab_size/--max_vocab_size must be > 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0.")
    if args.epochs_per_trial <= 0:
        raise ValueError("--epochs_per_trial must be > 0.")
    if args.embedding_dim <= 0:
        raise ValueError("--embedding_dim must be > 0.")
    if args.max_sequence_length <= 0:
        raise ValueError("--max_sequence_length must be > 0.")
    if args.num_layers_min <= 0 or args.num_layers_max <= 0:
        raise ValueError("--num_layers_min/--num_layers_max must be > 0.")
    if args.num_layers_min > args.num_layers_max:
        raise ValueError("--num_layers_min must be <= --num_layers_max.")
    if args.num_transformer_layers_min <= 0 or args.num_transformer_layers_max <= 0:
        raise ValueError(
            "--num_transformer_layers_min/--num_transformer_layers_max must be > 0."
        )
    if args.num_transformer_layers_min > args.num_transformer_layers_max:
        raise ValueError(
            "--num_transformer_layers_min must be <= --num_transformer_layers_max."
        )
    if args.dropout_min < 0.0 or args.dropout_max > 1.0 or args.dropout_min > args.dropout_max:
        raise ValueError("--dropout_min/--dropout_max must satisfy 0 <= min <= max <= 1.")
    if args.dropout_step <= 0:
        raise ValueError("--dropout_step must be > 0.")
    if args.train_stride is not None and args.train_stride <= 0:
        raise ValueError("--train_stride must be > 0 when provided.")
    if args.streaming_shuffle_buffer_size <= 0:
        raise ValueError("--streaming_shuffle_buffer_size must be > 0.")
    if args.streaming_chunk_line_buffer <= 0:
        raise ValueError("--streaming_chunk_line_buffer must be > 0.")
    if args.train_steps_per_epoch is not None and args.train_steps_per_epoch <= 0:
        raise ValueError("--train_steps_per_epoch must be > 0 when provided.")
    if args.validation_steps_per_epoch is not None and args.validation_steps_per_epoch <= 0:
        raise ValueError("--validation_steps_per_epoch must be > 0 when provided.")
    if args.test_steps_per_epoch is not None and args.test_steps_per_epoch <= 0:
        raise ValueError("--test_steps_per_epoch must be > 0 when provided.")
    if args.num_operators <= 0:
        raise ValueError("--num_operators must be > 0.")
    if args.operator_min_weight < 0:
        raise ValueError("--operator_min_weight must be >= 0.")
    if args.warmup_steps_default < 0:
        raise ValueError("--warmup_steps_default must be >= 0.")
    if args.dataset_mode == "text":
        if not args.train_text_file or not args.validation_text_file:
            raise ValueError(
                "--train_text_file and --validation_text_file are required for --dataset_mode text."
            )
    else:
        if not args.train_seq_bin:
            raise ValueError(
                "--train_seq_bin is required for --dataset_mode pretokenized_bin."
            )
    if args.target_param_count <= 0:
        raise ValueError("--target_param_count must be > 0.")
    if args.param_count_tolerance_abs is not None and args.param_count_tolerance_abs < 0:
        raise ValueError("--param_count_tolerance_abs must be >= 0 when provided.")
    if args.param_count_tolerance_abs is None and args.param_count_tolerance_ratio < 0:
        raise ValueError("--param_count_tolerance_ratio must be >= 0.")
    effective_tolerance = (
        args.param_count_tolerance_abs
        if args.param_count_tolerance_abs is not None
        else int(args.target_param_count * args.param_count_tolerance_ratio)
    )
    print(
        "Parameter-count gate: "
        f"target={args.target_param_count:,}, "
        f"tolerance=±{effective_tolerance:,}, "
        f"range=[{args.target_param_count - effective_tolerance:,}, "
        f"{args.target_param_count + effective_tolerance:,}]"
    )

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        storage=args.storage_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        make_objective(
            model_creation_fn=(
                create_causal_ldru_model
                if args.model_type == "causal_ldru"
                else (
                    create_transformer_model
                    if args.model_type == "transformer"
                    else create_lstm_model
                )
            ),
            args=args,
            use_lstm=(args.model_type == "lstm"),
            use_transformer=(args.model_type == "transformer"),
            seq2seq=True,
        ),
        n_trials=args.num_trials,
    )

    completed_trials = study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
    )
    if not completed_trials:
        print("No completed trials found. All trials may have been pruned or failed.")
    else:
        best_trial = study.best_trial
        best_param_count = best_trial.user_attrs.get("param_count")
        if best_param_count is not None:
            print(
                f"Best trial: #{best_trial.number} value={best_trial.value} "
                f"params={best_param_count:,}"
            )
        else:
            print(f"Best trial: #{best_trial.number} value={best_trial.value}")
        print(f"Best hyperparameters: {best_trial.params}")
