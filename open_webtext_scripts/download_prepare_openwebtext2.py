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
import os
import re
import subprocess
import tarfile
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
        "--tar_url",
        type=str,
        default="https://huggingface.co/datasets/segyges/OpenWebText2/resolve/main/openwebtext2.jsonl.zst.tar",
        help="Single tarball URL containing OpenWebText2 shard files.",
    )
    parser.add_argument(
        "--tar_filename",
        type=str,
        default="openwebtext2.jsonl.zst.tar",
        help="Local filename for downloaded tarball.",
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
        "--source_backend",
        type=str,
        default="tar",
        choices=["tar", "urls", "hf"],
        help="Data source backend: tar/urls local shards or huggingface dataset stream.",
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="segyges/OpenWebText2",
        help="Hugging Face dataset name when --source_backend hf.",
    )
    parser.add_argument(
        "--hf_config",
        type=str,
        default=None,
        help="Optional Hugging Face dataset config/subset name.",
    )
    parser.add_argument(
        "--hf_split",
        type=str,
        default="train",
        help="Hugging Face split to read from.",
    )
    parser.add_argument(
        "--hf_text_field",
        type=str,
        default="text",
        help="Text field name in HF dataset records (fallbacks still apply).",
    )
    parser.add_argument(
        "--hf_no_streaming",
        action="store_true",
        help="Disable HF streaming and materialize dataset locally.",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip downloading and only run preparation on local shards.",
    )
    parser.add_argument(
        "--skip_extract",
        action="store_true",
        help="Skip tar extraction step (only relevant with --tar_url).",
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


def download_file(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[download] exists, skipping: {out_path}")
        return
    print(f"[download] {url}")
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


def extract_tarball(tar_path: Path, dest_dir: Path):
    if not tar_path.exists():
        raise FileNotFoundError(f"Tarball not found: {tar_path}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {tar_path} -> {dest_dir}")
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(path=dest_dir)


def iter_jsonl_zst_docs(path: Path) -> Iterable[dict]:
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'zstandard'. Install with: pip install zstandard"
        ) from exc

    dctx = zstd.ZstdDecompressor()
    try:
        with open(path, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                import io

                text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
                for line in text_stream:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    except Exception as exc:
        print(f"[warn] failed reading shard {path}: {exc}. Skipping.")


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


def iter_hf_docs(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    streaming: bool,
) -> Iterable[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'datasets'. Install with: pip install datasets"
        ) from exc

    def _iter_chain_messages(exc: BaseException) -> list[str]:
        msgs: list[str] = []
        seen: set[int] = set()
        cur: Optional[BaseException] = exc
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            msgs.append(f"{type(cur).__name__}: {cur}")
            cur = cur.__cause__ if cur.__cause__ is not None else cur.__context__
        return msgs

    def _looks_like_compressed_jsonl_error(exc: BaseException) -> bool:
        chain_msgs = _iter_chain_messages(exc)
        msg = "\n".join(chain_msgs).lower()
        chain_types = {m.split(":", 1)[0].lower() for m in chain_msgs}
        has_json_decode_signal = (
            "json parse error" in msg
            or "arrowinvalid" in msg
            or "unicodedecodeerror" in msg
        )
        # In some datasets versions, the filename context is only logged (not in exception text),
        # so also treat DatasetGenerationError + decode/parse signals as this failure mode.
        has_dataset_generation_wrapper = "datasetgenerationerror" in chain_types
        return (".jsonl.zst" in msg and has_json_decode_signal) or (
            has_dataset_generation_wrapper and has_json_decode_signal
        )

    def _extract_cache_shard_paths_from_exception(exc: BaseException) -> list[Path]:
        paths: set[Path] = set()
        chain_msg = "\n".join(_iter_chain_messages(exc))
        for match in re.findall(r"(/[^'\"\s]+\.jsonl\.zst)", chain_msg):
            p = Path(match)
            if p.exists():
                paths.add(p)
                parent = p.parent
                if parent.exists():
                    for shard_path in parent.glob("*.jsonl.zst"):
                        if shard_path.exists():
                            paths.add(shard_path)
        return sorted(paths)

    def _find_dataset_cache_shards() -> list[Path]:
        cache_roots: list[Path] = []
        env_datasets_cache = os.environ.get("HF_DATASETS_CACHE")
        if env_datasets_cache:
            cache_roots.append(Path(env_datasets_cache).expanduser())
        else:
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                cache_roots.append(Path(hf_home).expanduser() / "datasets")
            cache_roots.append(Path("~/.cache/huggingface/datasets").expanduser())

        paths: set[Path] = set()
        for root in cache_roots:
            extracted_root = root / "downloads" / "extracted"
            if not extracted_root.exists():
                continue
            for child in extracted_root.iterdir():
                if not child.is_dir():
                    continue
                for shard_path in child.glob("*.jsonl.zst"):
                    if shard_path.exists():
                        paths.add(shard_path)

        return sorted(paths)

    def _iter_hf_snapshot_jsonl_zst(hint_exc: Optional[BaseException] = None) -> Iterable[dict]:
        if hint_exc is not None:
            cache_shards = _extract_cache_shard_paths_from_exception(hint_exc)
            if cache_shards:
                print(f"HF fallback using {len(cache_shards)} cached *.jsonl.zst shards.")
                for shard_idx, shard_path in enumerate(cache_shards, start=1):
                    print(
                        f"[hf-fallback-cache {shard_idx}/{len(cache_shards)}] {shard_path.name}"
                    )
                    yield from iter_jsonl_zst_docs(shard_path)
                return

        cache_shards = _find_dataset_cache_shards()
        if cache_shards:
            print(f"HF fallback found {len(cache_shards)} *.jsonl.zst shards in datasets cache.")
            for shard_idx, shard_path in enumerate(cache_shards, start=1):
                print(f"[hf-fallback-cache-scan {shard_idx}/{len(cache_shards)}] {shard_path.name}")
                yield from iter_jsonl_zst_docs(shard_path)
            return

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "Need huggingface_hub for fallback shard download. "
                "Install with: pip install huggingface_hub"
            ) from exc

        snapshot_path = Path(
            snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                allow_patterns=["*.jsonl.zst"],
            )
        )
        shard_paths = sorted(snapshot_path.rglob("*.jsonl.zst"))
        if not shard_paths:
            raise FileNotFoundError(
                f"No *.jsonl.zst files found in HF snapshot for {dataset_name}."
            )
        if split != "train":
            print(
                "HF compressed-shard fallback does not expose named splits; "
                f"reading all shards (requested split={split})."
            )
        print(f"HF fallback found {len(shard_paths)} *.jsonl.zst shards.")
        for shard_idx, shard_path in enumerate(shard_paths, start=1):
            print(f"[hf-fallback {shard_idx}/{len(shard_paths)}] {shard_path.name}")
            yield from iter_jsonl_zst_docs(shard_path)

    attempts = [streaming] + ([] if not streaming else [False])
    last_exc: Optional[BaseException] = None
    for mode in attempts:
        try:
            ds = load_dataset(
                path=dataset_name,
                name=dataset_config,
                split=split,
                streaming=mode,
            )
            for rec in ds:
                yield rec
            return
        except NotImplementedError as exc:
            last_exc = exc
            # Some HF datasets expose TAR archives where streaming extraction is unsupported.
            if mode and "tar archives" in str(exc).lower():
                print(
                    "HF streaming for TAR archives is not supported for this dataset; "
                    "retrying with non-streaming mode."
                )
                continue
            raise
        except Exception as exc:
            last_exc = exc
            if mode and _looks_like_compressed_jsonl_error(exc):
                print(
                    "HF builder failed to parse compressed JSONL shards in streaming mode; "
                    "retrying with non-streaming mode."
                )
                continue
            if (not mode) and _looks_like_compressed_jsonl_error(exc):
                print(
                    "HF builder cannot parse compressed *.jsonl.zst shards for this dataset; "
                    "falling back to manual shard iteration."
                )
                yield from _iter_hf_snapshot_jsonl_zst(hint_exc=exc)
                return
            raise

    if last_exc is not None:
        raise last_exc


def prepare_dataset_from_records(args: argparse.Namespace, records: Iterable[dict]):
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

    print(f"Writing outputs to: {output_dir}")

    counts = {"train": 0, "val": 0, "test": 0, "skipped": 0}
    chars = {"train": 0, "val": 0, "test": 0}

    processed = 0
    with open(train_path, "w", encoding="utf-8") as train_f, open(
        val_path, "w", encoding="utf-8"
    ) as val_f, open(test_path, "w", encoding="utf-8") as test_f:
        out = {"train": train_f, "val": val_f, "test": test_f}

        for rec in records:
            text = None
            if args.hf_text_field and isinstance(rec, dict):
                field_val = rec.get(args.hf_text_field)
                if isinstance(field_val, str):
                    text = field_val
            if text is None:
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


def prepare_dataset(args: argparse.Namespace):
    download_dir = Path(args.download_dir)
    shard_paths = sorted(download_dir.glob(args.source_glob))
    if not shard_paths:
        raise FileNotFoundError(
            f"No shard files found in {download_dir} matching {args.source_glob}"
        )
    print(f"Found {len(shard_paths)} shard files.")

    def _records():
        for shard_idx, shard_path in enumerate(shard_paths, start=1):
            print(f"[prepare {shard_idx}/{len(shard_paths)}] {shard_path.name}")
            yield from iter_jsonl_zst_docs(shard_path)

    prepare_dataset_from_records(args, _records())


def main():
    args = parse_args()
    download_dir = Path(args.download_dir)

    if args.source_backend == "hf":
        print(
            f"Using Hugging Face dataset backend: {args.hf_dataset}"
            + (f" ({args.hf_config})" if args.hf_config else "")
            + f", split={args.hf_split}, streaming={not args.hf_no_streaming}"
        )
        records = iter_hf_docs(
            dataset_name=args.hf_dataset,
            dataset_config=args.hf_config,
            split=args.hf_split,
            streaming=not args.hf_no_streaming,
        )
        prepare_dataset_from_records(args, records)
        return

    if not args.skip_download:
        if args.source_backend == "urls":
            if not args.urls_file:
                raise ValueError("--urls_file is required when --source_backend urls.")
            urls = read_urls(Path(args.urls_file))
            print(f"Downloading {len(urls)} shards from --urls_file...")
            download_shards(urls, download_dir)
        else:  # tar
            if not args.tar_url:
                raise ValueError("--tar_url is required when --source_backend tar.")
            tar_path = download_dir / args.tar_filename
            download_file(args.tar_url, tar_path)
            if not args.skip_extract:
                extract_tarball(tar_path, download_dir)
    else:
        print("Skipping download step (--skip_download set).")

    prepare_dataset(args)


if __name__ == "__main__":
    main()
