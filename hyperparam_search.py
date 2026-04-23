import datetime
import os
import sys
from typing import Optional
import optuna
from causal_ldru_v2 import create_causal_ldru_model
from train_causal_ldru import (
    LDRUExperimenstConfig,
    create_lstm_model,
    create_transformer_model,
    train_model,
)


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
    use_lstm=False,
    use_transformer=False,
    seq2seq=True,
    batch_size=128,
    vocab_size=1500,
    num_epochs=20,
    seq_length=256,
    tokenizer_path=None,
    use_alibi=True,
):
    def objective(trial):
        # Define the hyperparameter search space
        num_layers = trial.suggest_int("num_layers", 1, 4)
        if use_transformer:
            num_transformer_heads = trial.suggest_categorical(
                "num_transformer_heads", [4, 8, 16]
            )
            num_transformer_layers = trial.suggest_int("num_transformer_layers", 2, 8)
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512, 1024])
        dropout_prob = trial.suggest_float("dropout_prob", 0.0, 0.6, step=0.1)
        initial_learning_rate = trial.suggest_categorical(
            "initial_learning_rate",
            [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        )
        l2_lambda = trial.suggest_categorical(
            "l2_lambda",
            [
                0.0,
                1e-6,
                5e-6,
                1e-5,
                5e-5,
                1e-4,
                5e-5,
                1e-3,
                5e-3,
                1e-2,
                5e-2,
                1e-1,
                5e-1,
            ],
        )

        # Create the model configuration
        config = LDRUExperimenstConfig(
            embedding_dim=512,
            num_layers=num_layers,
            max_sequence_length=3072,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob,
            initial_learning_rate=initial_learning_rate,
            l2_lambda=l2_lambda,
            batch_size=batch_size,
            seq_length=seq_length,
            vocab_size=vocab_size,
            num_epochs=num_epochs,  # Fixed number of epochs for faster search
            num_transformer_heads=num_transformer_heads if use_transformer else None,
            num_transformer_layers=num_transformer_layers if use_transformer else None,
            min_learning_rate=initial_learning_rate
            / 1000,  # Set a minimum learning rate for the scheduler
            use_positional_encoding=(True if use_transformer else False),
            use_alibi=(use_alibi if use_transformer else False),
        )

        # Call train_model and capture the best validation perplexity
        _, _, _, _, best_val_perplexity = train_model(
            log_dir="optuna_logs",
            config=config,
            enable_logging=False,
            text_file_path="wikitext-2-raw-train.txt",
            validation_text_file_path="wikitext-2-raw-validation.txt",
            model_creation_fn=model_creation_fn,  # Replace with the appropriate model creation function
            use_lstm=use_lstm,
            use_transformer=use_transformer,
            use_transformer_ldru=False,
            seq2seq=seq2seq,
            checkpoint_dir="optuna_checkpoints",
            resume_from_checkpoint=None,
            tokenizer_path=tokenizer_path,
            generate_samples=False,  # Disable sample generation during hyperparameter search for faster trials
        )

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
    args = parser.parse_args()

    num_trials = args.num_trials
    enable_logging = args.enable_logging
    model_type = args.model_type
    seq_length = args.seq_length
    epochs_per_trial = args.epochs_per_trial
    vocab_size = args.vocab_size
    batch_size = args.batch_size
    study_name = args.study_name
    print_log_file = args.print_log_file
    configure_output(print_log_file)

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=args.storage_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        make_objective(
            model_creation_fn=(
                create_causal_ldru_model
                if model_type == "causal_ldru"
                else (
                    create_transformer_model
                    if model_type == "transformer"
                    else create_lstm_model
                )
            ),
            use_lstm=(model_type == "lstm"),
            use_transformer=(model_type == "transformer"),
            seq2seq=True,
            batch_size=batch_size,
            vocab_size=vocab_size,
            num_epochs=epochs_per_trial,
            seq_length=seq_length,
            tokenizer_path=args.tokenizer_path,
            use_alibi=args.use_alibi,
        ),
        n_trials=num_trials,
    )

    # Run the hyperparameter search
    best_val_perplexity = float("inf")
    best_hyperparams = None
