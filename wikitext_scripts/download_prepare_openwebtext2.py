#!/usr/bin/env python3
"""
Download OpenWebText2 shards and prepare train/val/test plain-text files.

This script supports two stages:
1) Download *.jsonl.zst shards from a URL list (optional).
2) Stream-process shards into deterministic split text files.
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare OpenWebText2 into train/val/test .txt files."
    )
    parser.add_argument(
        "--urls_file",
        type=str,
        default=None,
        help="Path to text file with shard URLs (one per line).",
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default="data/openwebtext2/raw",
        help="Directory where *.jsonl.zst shards will be stored.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/openwebtext2/prepared",
        help="Directory for output split files.",
    )
    parser.add_argument(
        "--train_filename", type=str, default="openwebtext2_train.txt"
    )
    parser.add_argument("--val_filename", type=str, default="openwebtext2_val.txt")
    parser.add_argument("--test_filename", type=str, default="openwebtext2_test.txt")
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.005,
        help="Validation split ratio (deterministic hash split).",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.005,
        help="Test split ratio (deterministic hash split).",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=32,
        help="Minimum cleaned text length to keep a document.",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=0,
        help="Optional cap on processed docs (0 = no cap).",
    )
    parser.add_argument(
        "--source_glob",
        type=str,
        default="*.jsonl.zst",
        help="Glob pattern to find source shard files in download_dir.",
    )
    parser.add_argument(
        "--overwrite_outputs",
        action="store_true",
        help="Overwrite output split files if they already exist.",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip downloading and only run preparation on local shards.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=100_000,
        help="Progress print frequency in number of documents.",
    )
    return parser.parse_args()


def read_urls(urls_file: Path) -> list[str]:
    if not urls_file.exists():
        raise FileNotFoundError(f"URLs file not found: {urls_file}")
    urls = []
    for line in urls_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    if not urls:
        raise ValueError(f"No URLs found in: {urls_file}")
    return urls


def download_shards(urls: list[str], download_dir: Path):
    download_dir.mkdir(parents=True, exist_ok=True)
    for idx, url in enumerate(urls, start=1):
        filename = url.rsplit("/", 1)[-1]
        if not filename:
            raise ValueError(f"Could not infer filename from URL: {url}")
        out_path = download_dir / filename
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[download {idx}/{len(urls)}] exists, skipping: {out_path.name}")
            continue
        print(f"[download {idx}/{len(urls)}] {url}")
        cmd = [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "-o",
            str(out_path),
            url,
        ]
        subprocess.run(cmd, check=True)


def iter_jsonl_zst_docs(path: Path) -> Iterable[dict]:
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'zstandard'. Install with: pip install zstandard"
        ) from exc

    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            import io

            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def extract_text(record: dict) -> Optional[str]:
    for key in ("text", "content", "body", "document", "raw_content"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return None


_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = _WS_RE.sub(" ", text)
    return text.strip()


def split_bucket(text: str, val_ratio: float, test_ratio: float) -> str:
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    v = int(digest[:8], 16) / 0xFFFFFFFF
    if v < val_ratio:
        return "val"
    if v < val_ratio + test_ratio:
        return "test"
    return "train"


def prepare_dataset(args: argparse.Namespace):
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("Require: val_ratio >= 0, test_ratio >= 0, val_ratio + test_ratio < 1")
    if args.min_chars <= 0:
        raise ValueError("--min_chars must be > 0")
    if args.progress_every <= 0:
        raise ValueError("--progress_every must be > 0")
    if args.max_docs < 0:
        raise ValueError("--max_docs must be >= 0")

    download_dir = Path(args.download_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / args.train_filename
    val_path = output_dir / args.val_filename
    test_path = output_dir / args.test_filename

    for p in (train_path, val_path, test_path):
        if p.exists() and not args.overwrite_outputs:
            raise FileExistsError(
                f"Output exists: {p}. Use --overwrite_outputs to replace."
            )

    shard_paths = sorted(download_dir.glob(args.source_glob))
    if not shard_paths:
        raise FileNotFoundError(
            f"No shard files found in {download_dir} matching {args.source_glob}"
        )

    print(f"Found {len(shard_paths)} shard files.")
    print(f"Writing outputs to: {output_dir}")

    counts = {"train": 0, "val": 0, "test": 0, "skipped": 0}
    chars = {"train": 0, "val": 0, "test": 0}

    processed = 0
    with open(train_path, "w", encoding="utf-8") as train_f, open(
        val_path, "w", encoding="utf-8"
    ) as val_f, open(test_path, "w", encoding="utf-8") as test_f:
        out = {"train": train_f, "val": val_f, "test": test_f}

        for shard_idx, shard_path in enumerate(shard_paths, start=1):
            print(f"[prepare {shard_idx}/{len(shard_paths)}] {shard_path.name}")
            for rec in iter_jsonl_zst_docs(shard_path):
                text = extract_text(rec)
                if not text:
                    counts["skipped"] += 1
                    continue
                text = clean_text(text)
                if len(text) < args.min_chars:
                    counts["skipped"] += 1
                    continue

                bucket = split_bucket(text, args.val_ratio, args.test_ratio)
                out[bucket].write(text + "\n")
                counts[bucket] += 1
                chars[bucket] += len(text)
                processed += 1

                if args.max_docs and processed >= args.max_docs:
                    print(f"Reached --max_docs={args.max_docs}. Stopping early.")
                    break

                if processed % args.progress_every == 0:
                    print(
                        f"Processed {processed:,} docs "
                        f"(train={counts['train']:,}, val={counts['val']:,}, test={counts['test']:,}, skipped={counts['skipped']:,})"
                    )
            if args.max_docs and processed >= args.max_docs:
                break

    print("Preparation complete.")
    print(
        f"Docs: train={counts['train']:,}, val={counts['val']:,}, test={counts['test']:,}, skipped={counts['skipped']:,}"
    )
    print(
        f"Chars: train={chars['train']:,}, val={chars['val']:,}, test={chars['test']:,}"
    )
    print(f"Train file: {train_path}")
    print(f"Val file:   {val_path}")
    print(f"Test file:  {test_path}")


def main():
    args = parse_args()

    if not args.skip_download:
        if not args.urls_file:
            raise ValueError("--urls_file is required unless --skip_download is set.")
        urls = read_urls(Path(args.urls_file))
        print(f"Downloading {len(urls)} shards...")
        download_shards(urls, Path(args.download_dir))
    else:
        print("Skipping download step (--skip_download set).")

    prepare_dataset(args)


if __name__ == "__main__":
    main()
