#!/usr/bin/env python3
"""
Pre-tokenize text corpora into contiguous token-stream binary files.

Outputs:
- <out_dir>/<basename>_<split>.bin         (flat token stream)
- <out_dir>/<basename>_meta.json           (dtype/tokenizer/split metadata)

Sequence windowing is intentionally deferred to the training script.

Supported tokenizers:
- tiktoken (GPT-2 by default)
- sentencepiece (load existing model or train new one)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from argparse import SUPPRESS
from typing import Optional

import numpy as np


@dataclass
class TokenizerInfo:
    tokenizer_type: str
    tokenizer_name_or_path: str
    vocab_size: int
    eos_token_id: Optional[int]


class _BaseTokenizer:
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def vocab_size(self) -> int:
        raise NotImplementedError

    def eos_token_id(self) -> Optional[int]:
        return None


class _TiktokenTokenizer(_BaseTokenizer):
    def __init__(self, encoding_name: str):
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'tiktoken'. Install with: pip install tiktoken"
            ) from exc
        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return self.enc.encode_ordinary(text)

    def vocab_size(self) -> int:
        return int(self.enc.n_vocab)

    def eos_token_id(self) -> Optional[int]:
        return int(self.enc.eot_token)


class _SentencePieceTokenizer(_BaseTokenizer):
    def __init__(self, model_path: str):
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'sentencepiece'. Install with: pip install sentencepiece"
            ) from exc
        self.model_path = model_path
        self.spm = spm.SentencePieceProcessor()
        self.spm.load(model_path)

    def encode(self, text: str) -> list[int]:
        return self.spm.encode(text)

    def vocab_size(self) -> int:
        return int(self.spm.get_piece_size())

    def eos_token_id(self) -> Optional[int]:
        # sentencepiece defaults to </s> id if configured, else -1
        eos = int(self.spm.eos_id())
        return None if eos < 0 else eos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize text files into contiguous token-stream binaries."
    )
    parser.add_argument("--train_text", type=str, required=True, help="Train text path.")
    parser.add_argument("--val_text", type=str, default=None, help="Val text path.")
    parser.add_argument("--test_text", type=str, default=None, help="Test text path.")
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="tiktoken_gpt2",
        choices=["tiktoken_gpt2", "sentencepiece"],
        help="Tokenizer backend.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help=(
            "Tokenizer identifier: tiktoken encoding name (e.g. gpt2), "
            "or sentencepiece .model path."
        ),
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="SentencePiece vocab size when training a new SP model.",
    )
    parser.add_argument(
        "--train_sentencepiece",
        action="store_true",
        help="Train a new sentencepiece model from --train_text.",
    )
    parser.add_argument(
        "--sp_model_prefix",
        type=str,
        default=None,
        help="SentencePiece model prefix when training.",
    )
    parser.add_argument(
        "--append_eos",
        action="store_true",
        help="Append EOS/EOT token after each input line/document if available.",
    )
    # Deprecated flags kept for backward compatibility with older launch scripts.
    parser.add_argument("--seq_length", type=int, default=None, help=SUPPRESS)
    parser.add_argument("--stride", type=int, default=None, help=SUPPRESS)
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "uint16", "uint32", "int32"],
        help="Binary token dtype for output sequences.",
    )
    parser.add_argument(
        "--line_limit",
        type=int,
        default=0,
        help="Optional cap on lines per split (0 = no cap).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/pretokenized",
        help="Output directory for binary files and metadata.",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default="dataset",
        help="Base name for output files.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=50000,
        help="Print progress every N input lines (default: 50000).",
    )
    return parser.parse_args()


def _pick_dtype(dtype_arg: str, vocab_size: int) -> np.dtype:
    if dtype_arg == "auto":
        return np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32
    if dtype_arg == "uint16":
        return np.uint16
    if dtype_arg == "uint32":
        return np.uint32
    return np.int32


def _train_sentencepiece_if_needed(
    train_text: str,
    sp_model_prefix: str,
    vocab_size: int,
) -> str:
    import sentencepiece as spm

    Path(sp_model_prefix).parent.mkdir(parents=True, exist_ok=True)
    model_path = f"{sp_model_prefix}.model"
    if Path(model_path).exists():
        return model_path

    spm.SentencePieceTrainer.train(
        input=train_text,
        model_prefix=sp_model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
    )
    return model_path


def _build_tokenizer(args: argparse.Namespace, out_dir: Path) -> tuple[_BaseTokenizer, TokenizerInfo]:
    if args.tokenizer_type == "tiktoken_gpt2":
        encoding_name = args.tokenizer_path if args.tokenizer_path else "gpt2"
        tok = _TiktokenTokenizer(encoding_name)
        info = TokenizerInfo(
            tokenizer_type="tiktoken_gpt2",
            tokenizer_name_or_path=encoding_name,
            vocab_size=tok.vocab_size(),
            eos_token_id=tok.eos_token_id(),
        )
        return tok, info

    # sentencepiece path
    model_path = args.tokenizer_path
    if args.train_sentencepiece:
        prefix = args.sp_model_prefix
        if prefix is None:
            prefix = str(out_dir / f"{args.basename}_spm_vocab{args.vocab_size}")
        model_path = _train_sentencepiece_if_needed(args.train_text, prefix, args.vocab_size)
    if not model_path:
        raise ValueError(
            "For sentencepiece, provide --tokenizer_path or set --train_sentencepiece."
        )

    tok = _SentencePieceTokenizer(model_path)
    info = TokenizerInfo(
        tokenizer_type="sentencepiece",
        tokenizer_name_or_path=model_path,
        vocab_size=tok.vocab_size(),
        eos_token_id=tok.eos_token_id(),
    )
    return tok, info


def _write_split_token_stream(
    text_path: Path,
    out_bin: Path,
    tokenizer: _BaseTokenizer,
    out_dtype: np.dtype,
    append_eos: bool,
    eos_token_id: Optional[int],
    line_limit: int,
    progress_every: int,
) -> dict:
    if not text_path.exists():
        raise FileNotFoundError(f"Missing input text file: {text_path}")

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    count_lines = 0
    count_tokens = 0
    max_token_id = -1

    with open(text_path, "r", encoding="utf-8", errors="replace") as f_in, open(
        out_bin, "wb"
    ) as f_out:
        for i, line in enumerate(f_in, start=1):
            if line_limit > 0 and i > line_limit:
                break
            ids = tokenizer.encode(line)
            if append_eos and eos_token_id is not None:
                ids.append(eos_token_id)
            if ids:
                arr = np.asarray(ids, dtype=out_dtype)
                arr.tofile(f_out)
                local_max = int(np.max(arr))
                if local_max > max_token_id:
                    max_token_id = local_max
            count_tokens += len(ids)
            count_lines += 1
            if progress_every > 0 and (i % progress_every == 0):
                print(
                    f"  lines={i:,} tokens={count_tokens:,}",
                    flush=True,
                )

    if count_tokens == 0:
        print(f"[warn] no tokens written for split file: {text_path}")

    return {
        "source_text": str(text_path),
        "out_bin": str(out_bin),
        "num_lines": int(count_lines),
        "num_tokens": int(count_tokens),
        "dtype": np.dtype(out_dtype).name,
        "max_token_id_seen": int(max_token_id),
    }


def main():
    args = parse_args()
    if args.seq_length is not None or args.stride is not None:
        print(
            "[warn] --seq_length/--stride are ignored: this script now writes "
            "flat token streams and defers sequence windowing to training."
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, tok_info = _build_tokenizer(args, out_dir)
    out_dtype = _pick_dtype(args.dtype, tok_info.vocab_size)
    if np.issubdtype(out_dtype, np.unsignedinteger):
        max_allowed = int(np.iinfo(out_dtype).max)
        if tok_info.vocab_size - 1 > max_allowed:
            raise ValueError(
                f"Tokenizer vocab size {tok_info.vocab_size} does not fit {out_dtype}."
            )

    splits = {"train": args.train_text, "val": args.val_text, "test": args.test_text}
    split_meta = {}
    for split_name, split_path in splits.items():
        if not split_path:
            continue
        out_bin = out_dir / f"{args.basename}_{split_name}.bin"
        print(f"[{split_name}] writing token stream -> {out_bin}")
        split_meta[split_name] = _write_split_token_stream(
            text_path=Path(split_path),
            out_bin=out_bin,
            tokenizer=tokenizer,
            out_dtype=out_dtype,
            append_eos=args.append_eos,
            eos_token_id=tok_info.eos_token_id,
            line_limit=args.line_limit,
            progress_every=args.progress_every,
        )
        print(
            f"[{split_name}] tokens={split_meta[split_name]['num_tokens']:,} "
            f"dtype={split_meta[split_name]['dtype']}"
        )

    meta = {
        "format": "token_stream",
        "tokenizer": {
            "type": tok_info.tokenizer_type,
            "name_or_path": tok_info.tokenizer_name_or_path,
            "vocab_size": tok_info.vocab_size,
            "eos_token_id": tok_info.eos_token_id,
            "append_eos": bool(args.append_eos),
        },
        "token_stream_config": {
            "dtype": np.dtype(out_dtype).name,
            "append_eos_per_line": bool(args.append_eos),
        },
        "splits": split_meta,
    }

    meta_path = out_dir / f"{args.basename}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
