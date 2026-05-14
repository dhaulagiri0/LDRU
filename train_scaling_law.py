import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import jax
import jax.numpy as jnp

from train_causal_ldru import (
    LDRUExperimenstConfig,
    create_causal_ldru_model,
    resolve_binary_operator,
)


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run one scaling-law training job for a single hidden_dim by computing a "
            "token budget from parameter count, then calling train_causal_ldru.py."
        )
    )
    parser.add_argument("--hidden_dim", type=int, required=True, help="Model hidden dim.")
    parser.add_argument(
        "--embedding_dim", type=int, required=True, help="Embedding dimension."
    )
    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size.")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of LDRU layers.")
    parser.add_argument("--max_seq_len", type=int, default=32, help="Training sequence length.")
    parser.add_argument(
        "--binary_operator",
        type=str,
        default="default",
        choices=["default", "binary", "convex_gated", "grc", "ablation"],
        help="Binary operator for LDRU composition.",
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
        help="Expansion stage mode when binary_operator=ablation.",
    )
    parser.add_argument(
        "--ablation_combine_mode",
        type=str,
        default="grc",
        choices=["binary", "grc"],
        help="Combine stage mode when binary_operator=ablation.",
    )
    parser.add_argument(
        "--token_scale_factor",
        type=float,
        required=True,
        help="Token scaling coefficient in: target_tokens = a * params^b",
    )
    parser.add_argument(
        "--token_scale_exponent",
        type=float,
        default=1.0,
        help="Token scaling exponent b in: target_tokens = a * params^b",
    )
    parser.add_argument(
        "--param_count_basis",
        type=str,
        default="non_embedding",
        choices=["non_embedding", "total"],
        help="Whether scaling uses total params or non-embedding params.",
    )
    parser.add_argument(
        "--embedding_count_multiplier",
        type=int,
        default=1,
        help="Embedding matrices to subtract (1 tied, 2 untied).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for parameter-count init only.",
    )
    parser.add_argument(
        "--train_script",
        type=str,
        default="train_causal_ldru.py",
        help="Path to the trainer script to execute.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional JSON path to save scaling metadata for this run.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print computed command and exit without launching training.",
    )
    return parser.parse_known_args()


def compute_total_params(
    hidden_dim: int,
    embedding_dim: int,
    vocab_size: int,
    num_layers: int,
    max_seq_len: int,
    binary_operator: str,
    binop_expansion_factor: int,
    ablation_expansion_mode: str,
    ablation_combine_mode: str,
    seed: int,
) -> int:
    config = LDRUExperimenstConfig(
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        seq_length=max_seq_len,
        max_sequence_length=max(3072, max_seq_len),
        operator=resolve_binary_operator(binary_operator),
        binop_expansion_factor=binop_expansion_factor,
        ablation_expansion_mode=ablation_expansion_mode,
        ablation_combine_mode=ablation_combine_mode,
    )
    model = create_causal_ldru_model(config)
    dummy_batch = jnp.zeros((1, max_seq_len), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(seed), dummy_batch)
    return int(sum(x.size for x in jax.tree.leaves(params)))


def main():
    args, passthrough = parse_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    if args.hidden_dim <= 0:
        raise ValueError("--hidden_dim must be > 0")
    if args.embedding_dim <= 0:
        raise ValueError("--embedding_dim must be > 0")
    if args.vocab_size <= 0:
        raise ValueError("--vocab_size must be > 0")
    if args.num_layers <= 0:
        raise ValueError("--num_layers must be > 0")
    if args.max_seq_len <= 1:
        raise ValueError("--max_seq_len must be > 1")
    if args.binop_expansion_factor <= 0:
        raise ValueError("--binop_expansion_factor must be > 0")
    if args.token_scale_factor <= 0:
        raise ValueError("--token_scale_factor must be > 0")
    if args.embedding_count_multiplier <= 0:
        raise ValueError("--embedding_count_multiplier must be > 0")

    train_script_path = Path(args.train_script)
    if not train_script_path.exists():
        raise FileNotFoundError(f"Training script not found: {train_script_path}")

    total_params = compute_total_params(
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        binary_operator=args.binary_operator,
        binop_expansion_factor=args.binop_expansion_factor,
        ablation_expansion_mode=args.ablation_expansion_mode,
        ablation_combine_mode=args.ablation_combine_mode,
        seed=args.seed,
    )

    embedding_params = (
        args.embedding_dim * args.vocab_size * args.embedding_count_multiplier
    )
    non_embedding_params = max(1, total_params - embedding_params)
    basis_params = (
        non_embedding_params if args.param_count_basis == "non_embedding" else total_params
    )

    target_tokens = int(
        max(1, round(args.token_scale_factor * (basis_params ** args.token_scale_exponent)))
    )

    cmd = [
        sys.executable,
        str(train_script_path),
        "--hidden_dim",
        str(args.hidden_dim),
        "--embedding_dim",
        str(args.embedding_dim),
        "--max_vocab_size",
        str(args.vocab_size),
        "--num_layers",
        str(args.num_layers),
        "--max_seq_len",
        str(args.max_seq_len),
        "--binary_operator",
        args.binary_operator,
        "--binop_expansion_factor",
        str(args.binop_expansion_factor),
        "--ablation_expansion_mode",
        args.ablation_expansion_mode,
        "--ablation_combine_mode",
        args.ablation_combine_mode,
        "--target_tokens",
        str(target_tokens),
        *passthrough,
    ]

    metadata = {
        "hidden_dim": args.hidden_dim,
        "embedding_dim": args.embedding_dim,
        "vocab_size": args.vocab_size,
        "num_layers": args.num_layers,
        "max_seq_len": args.max_seq_len,
        "binary_operator": args.binary_operator,
        "binop_expansion_factor": args.binop_expansion_factor,
        "ablation_expansion_mode": args.ablation_expansion_mode,
        "ablation_combine_mode": args.ablation_combine_mode,
        "token_scale_factor": args.token_scale_factor,
        "token_scale_exponent": args.token_scale_exponent,
        "param_count_basis": args.param_count_basis,
        "embedding_count_multiplier": args.embedding_count_multiplier,
        "total_params": total_params,
        "embedding_params_estimate": embedding_params,
        "non_embedding_params_estimate": non_embedding_params,
        "basis_params": basis_params,
        "target_tokens": target_tokens,
        "trainer_command": cmd,
    }

    print("Scaling-law run setup:")
    print(json.dumps(metadata, indent=2))

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved scaling metadata: {output_path}")

    if args.dry_run:
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
