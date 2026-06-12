import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import matplotlib
import numpy as np
import sentencepiece as spm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class SequenceTokenizer:
    name: str
    encode: callable
    decode: callable


WIKITEXT_HEADING_RE = re.compile(r"^\s*=+\s.*\s=+\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project train/val text sequences into 2D with pretrained embeddings."
        )
    )
    parser.add_argument("--train_file", type=str, required=True, help="Train .txt file.")
    parser.add_argument("--val_file", type=str, required=True, help="Validation .txt file.")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="distilbert-base-uncased",
        help="HuggingFace model name/path for sequence embeddings.",
    )
    parser.add_argument(
        "--sequence_tokenizer_path",
        type=str,
        default=None,
        help=(
            "Optional tokenizer for sequence chunking. "
            "Use a SentencePiece .model path (e.g., tokenizers/...model). "
            "If omitted, uses the embedding model tokenizer."
        ),
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=64,
        help="Chunk length in tokenizer tokens for each sequence.",
    )
    parser.add_argument(
        "--sequence_length_min",
        type=int,
        default=None,
        help="Minimum sequence length for geometric sweep (e.g., 64).",
    )
    parser.add_argument(
        "--sequence_length_max",
        type=int,
        default=None,
        help="Maximum sequence length for geometric sweep (e.g., 512).",
    )
    parser.add_argument(
        "--sequence_length_multiplier",
        type=int,
        default=2,
        help="Multiplier for geometric sweep (default: 2).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=32,
        help="Stride in tokenizer tokens between windows.",
    )
    parser.add_argument(
        "--chunk_line_buffer",
        type=int,
        default=4096,
        help="Number of lines buffered per tokenization chunk when streaming text files.",
    )
    parser.add_argument(
        "--max_sequences_per_split",
        type=int,
        default=1000,
        help="Max sampled sequences from each split (train and val).",
    )
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default="random_starts",
        choices=["random_starts", "reservoir_windows", "first_windows"],
        help=(
            "Sampling strategy: random_starts (global random start positions, "
            "very low locality), reservoir_windows (random over stride windows), "
            "first_windows (sequential baseline)."
        ),
    )
    parser.add_argument(
        "--mix_splits_before_sampling",
        action="store_true",
        help="Artificially mix sampled train/val sequences before plotting.",
    )
    parser.add_argument(
        "--mixed_train_ratio",
        type=float,
        default=0.5,
        help="When mixing splits, fraction re-labeled as train (default: 0.5).",
    )
    parser.add_argument(
        "--color_by",
        type=str,
        default="split",
        choices=["split", "article"],
        help="Color scatter points by split label or by article ID.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "cls"],
        help="Embedding pooling strategy for transformer outputs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=128,
        help="Max token length fed into embedding model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device selection.",
    )
    parser.add_argument(
        "--projection_method",
        type=str,
        default="umap",
        choices=["umap", "tsne"],
        help="2D projection method (default: umap).",
    )
    parser.add_argument(
        "--umap_n_neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors.",
    )
    parser.add_argument(
        "--umap_min_dist",
        type=float,
        default=0.1,
        help="UMAP min_dist.",
    )
    parser.add_argument(
        "--umap_metric",
        type=str,
        default="cosine",
        help="UMAP metric.",
    )
    parser.add_argument(
        "--tsne_perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity.",
    )
    parser.add_argument(
        "--tsne_learning_rate",
        type=float,
        default=200.0,
        help="t-SNE learning rate.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output_plot",
        type=str,
        default="projection_train_val.png",
        help="Output PNG path for the 2D scatter plot.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional CSV output for 2D coordinates and source split.",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of stratified CV folds for logistic separation score.",
    )
    parser.add_argument(
        "--output_scores_json",
        type=str,
        default=None,
        help="Optional JSON output path for embedding separation scores.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for auto-saving per-sequence-length artifacts.",
    )
    parser.add_argument(
        "--highd_enabled",
        action="store_true",
        help="Enable high-dimensional train/validation separation analysis.",
    )
    parser.add_argument(
        "--highd_cluster_method",
        type=str,
        default="kmeans",
        choices=["kmeans"],
        help="Clustering method for high-dimensional analysis.",
    )
    parser.add_argument(
        "--highd_num_clusters",
        type=int,
        default=8,
        help="Target number of clusters for high-dimensional clustering metrics.",
    )
    parser.add_argument(
        "--highd_runs",
        type=int,
        default=5,
        help="Number of random restarts for high-dimensional clustering analysis.",
    )
    parser.add_argument(
        "--highd_mmd_sample_size",
        type=int,
        default=500,
        help="Sample size for MMD gamma estimation and MMD computation (0=all points).",
    )
    return parser.parse_args()


def load_hf_modules():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "This script needs 'transformers' and 'torch'. Install with:\n"
            "  pip install transformers torch"
        ) from exc
    return torch, AutoModel, AutoTokenizer


def resolve_device(torch_module, device: str) -> str:
    has_cuda = torch_module.cuda.is_available()
    has_mps = (
        hasattr(torch_module.backends, "mps")
        and torch_module.backends.mps.is_built()
        and torch_module.backends.mps.is_available()
    )

    if device == "auto":
        # Prefer Metal on Apple Silicon/macOS, otherwise CUDA if available.
        if has_mps:
            return "mps"
        if has_cuda:
            return "cuda"
        return "cpu"

    if device == "mps" and not has_mps:
        print("Requested device 'mps' is unavailable; falling back to CPU.")
        return "cpu"
    if device == "cuda" and not has_cuda:
        print("Requested device 'cuda' is unavailable; falling back to CPU.")
        return "cpu"
    return device


def build_sequence_tokenizer(
    tokenizer_path: str, embedding_model: str, auto_tokenizer_cls
) -> SequenceTokenizer:
    if tokenizer_path is None:
        tok = auto_tokenizer_cls.from_pretrained(embedding_model)
        return SequenceTokenizer(
            name=f"hf:{embedding_model}",
            encode=lambda text: tok.encode(text, add_special_tokens=False),
            decode=lambda ids: tok.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            ),
        )

    tokenizer_path_obj = Path(tokenizer_path)
    if tokenizer_path_obj.suffix == ".model":
        sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path_obj))
        return SequenceTokenizer(
            name=f"sentencepiece:{tokenizer_path_obj.name}",
            encode=lambda text: sp.encode(text, out_type=int),
            decode=lambda ids: sp.decode(ids),
        )

    tok = auto_tokenizer_cls.from_pretrained(tokenizer_path)
    return SequenceTokenizer(
        name=f"hf:{tokenizer_path}",
        encode=lambda text: tok.encode(text, add_special_tokens=False),
        decode=lambda ids: tok.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        ),
    )


def iter_token_windows_from_text_file(
    file_path: str,
    seq_tokenizer: SequenceTokenizer,
    seq_length: int,
    stride: int,
    chunk_line_buffer: int,
    split_name: str,
) -> Iterator[Tuple[List[int], int]]:
    token_buffer: List[int] = []
    window_start = 0
    article_id = 0

    def _flush_available_windows(current_article_id: int) -> Iterator[Tuple[List[int], int]]:
        nonlocal token_buffer, window_start
        while window_start + seq_length <= len(token_buffer):
            yield token_buffer[window_start : window_start + seq_length], current_article_id
            window_start += stride

        if window_start >= max(seq_length, stride * 8):
            token_buffer = token_buffer[window_start:]
            window_start = 0

    def _process_buffered_lines(
        buffered_lines: List[str], current_article_id: int
    ) -> Iterator[Tuple[List[int], int, int]]:
        nonlocal token_buffer, window_start
        for buffered_line in buffered_lines:
            if WIKITEXT_HEADING_RE.match(buffered_line):
                yield from (
                    (window, window_article_id, current_article_id)
                    for window, window_article_id in _flush_available_windows(
                        current_article_id
                    )
                )
                token_buffer = []
                window_start = 0
                current_article_id += 1
                continue
            token_buffer.extend(seq_tokenizer.encode(buffered_line))
            yield from (
                (window, window_article_id, current_article_id)
                for window, window_article_id in _flush_available_windows(current_article_id)
            )
        yield [], -1, current_article_id

    lines: List[str] = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc=f"{split_name} lines", unit="line"):
            lines.append(line)
            if len(lines) >= chunk_line_buffer:
                updated_article_id = article_id
                for maybe_window, maybe_article_id, updated_article_id in _process_buffered_lines(
                    lines, article_id
                ):
                    if maybe_article_id >= 0:
                        yield maybe_window, maybe_article_id
                article_id = updated_article_id
                lines.clear()

    if lines:
        updated_article_id = article_id
        for maybe_window, maybe_article_id, updated_article_id in _process_buffered_lines(
            lines, article_id
        ):
            if maybe_article_id >= 0:
                yield maybe_window, maybe_article_id
        article_id = updated_article_id
    yield from _flush_available_windows(article_id)


def collect_sequence_texts(
    split_name: str,
    file_path: str,
    seq_tokenizer: SequenceTokenizer,
    seq_length: int,
    stride: int,
    chunk_line_buffer: int,
    max_sequences: int,
    sampling_mode: str,
    rng: random.Random,
) -> Tuple[List[str], List[int]]:
    print(f"[{split_name}] Streaming token windows from file...")
    window_stride = 1 if sampling_mode == "random_starts" else stride
    use_reservoir = sampling_mode in {"random_starts", "reservoir_windows"}
    windows = iter_token_windows_from_text_file(
        file_path=file_path,
        seq_tokenizer=seq_tokenizer,
        seq_length=seq_length,
        stride=window_stride,
        chunk_line_buffer=chunk_line_buffer,
        split_name=split_name,
    )

    if max_sequences <= 0 and use_reservoir:
        raise ValueError(
            f"{sampling_mode} requires --max_sequences_per_split > 0 to avoid exhausting memory."
        )

    if max_sequences <= 0:
        all_windows = [(w, a) for w, a in tqdm(windows, desc=f"{split_name} windows", unit="window")]
        texts = [seq_tokenizer.decode(w) for w, _ in all_windows]
        article_ids = [a for _, a in all_windows]
        return texts, article_ids

    if not use_reservoir:
        out_windows: List[List[int]] = []
        out_article_ids: List[int] = []
        for window, article_id in tqdm(
            windows, total=max_sequences, desc=f"{split_name} windows"
        ):
            out_windows.append(window)
            out_article_ids.append(article_id)
            if len(out_windows) >= max_sequences:
                break
        return [seq_tokenizer.decode(w) for w in out_windows], out_article_ids

    reservoir: List[Tuple[List[int], int]] = []
    seen = 0
    for window, article_id in tqdm(windows, desc=f"{split_name} windows", unit="window"):
        seen += 1
        if len(reservoir) < max_sequences:
            reservoir.append((window, article_id))
            continue
        replace_at = rng.randint(0, seen - 1)
        if replace_at < max_sequences:
            reservoir[replace_at] = (window, article_id)
    return (
        [seq_tokenizer.decode(w) for w, _ in reservoir],
        [a for _, a in reservoir],
    )


def pool_hidden_state(last_hidden_state, attention_mask, torch_module, pooling: str):
    if pooling == "cls":
        return last_hidden_state[:, 0, :]
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def embed_texts(
    texts: List[str],
    tokenizer,
    model,
    torch_module,
    device: str,
    batch_size: int,
    model_max_length: int,
    pooling: str,
) -> np.ndarray:
    vectors: List[np.ndarray] = []
    model.eval()
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=model_max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch_module.inference_mode():
            outputs = model(**encoded)
            pooled = pool_hidden_state(
                outputs.last_hidden_state,
                encoded["attention_mask"],
                torch_module,
                pooling,
            )
            pooled = torch_module.nn.functional.normalize(pooled, p=2, dim=1)
        vectors.append(pooled.detach().cpu().numpy())
    return np.concatenate(vectors, axis=0)


def run_tsne(
    embeddings: np.ndarray, perplexity: float, learning_rate: float, seed: int
) -> np.ndarray:
    n = embeddings.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 total sequences for t-SNE.")
    effective_perplexity = min(perplexity, n - 1)
    effective_perplexity = max(2.0, effective_perplexity)
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate=learning_rate,
        random_state=seed,
        init="pca",
        metric="cosine",
    )
    return tsne.fit_transform(embeddings)


def run_umap(
    embeddings: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    seed: int,
) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP selected but 'umap-learn' is not installed. Install with:\n"
            "  pip install umap-learn"
        ) from exc
    if embeddings.shape[0] < 2:
        raise ValueError("Need at least 2 sequences for UMAP.")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


def run_projection(
    embeddings: np.ndarray,
    method: str,
    tsne_perplexity: float,
    tsne_learning_rate: float,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    seed: int,
) -> np.ndarray:
    if method == "umap":
        return run_umap(
            embeddings=embeddings,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            seed=seed,
        )
    return run_tsne(
        embeddings=embeddings,
        perplexity=tsne_perplexity,
        learning_rate=tsne_learning_rate,
        seed=seed,
    )


def compute_logistic_separation_scores(
    embeddings: np.ndarray, n_train: int, cv_folds: int, seed: int
) -> dict:
    n_total = int(embeddings.shape[0])
    n_val = n_total - n_train
    labels = np.concatenate(
        [np.zeros(n_train, dtype=np.int32), np.ones(n_val, dtype=np.int32)]
    )
    splits = int(max(2, min(cv_folds, n_train, n_val)))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=1000, random_state=seed)

    acc_scores = cross_val_score(clf, embeddings, labels, cv=cv, scoring="accuracy")
    auc_scores = cross_val_score(clf, embeddings, labels, cv=cv, scoring="roc_auc")

    return {
        "n_total": n_total,
        "n_train": int(n_train),
        "n_validation": int(n_val),
        "cv_folds_used": splits,
        "logreg_cv_accuracy_mean": float(acc_scores.mean()),
        "logreg_cv_accuracy_std": float(acc_scores.std()),
        "logreg_cv_roc_auc_mean": float(auc_scores.mean()),
        "logreg_cv_roc_auc_std": float(auc_scores.std()),
    }


def _safe_cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    x_norm = float(np.linalg.norm(x))
    y_norm = float(np.linalg.norm(y))
    if x_norm == 0.0 or y_norm == 0.0:
        return 0.0
    return float(np.dot(x, y) / (x_norm * y_norm))


def _estimate_rbf_gamma(
    x: np.ndarray, y: np.ndarray, sample_size: int, rng: np.random.Generator
) -> float:
    pooled = np.concatenate([x, y], axis=0)
    n = pooled.shape[0]
    if sample_size > 0 and n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        pooled = pooled[idx]
    d2 = pairwise_distances(pooled, metric="sqeuclidean")
    upper = d2[np.triu_indices_from(d2, k=1)]
    upper = upper[upper > 0.0]
    if upper.size == 0:
        return 1.0 / max(1, pooled.shape[1])
    median_d2 = float(np.median(upper))
    return 1.0 / max(1e-12, 2.0 * median_d2)


def _compute_mmd_rbf(
    x: np.ndarray, y: np.ndarray, gamma: float, sample_size: int, rng: np.random.Generator
) -> float:
    if sample_size > 0 and x.shape[0] > sample_size:
        x = x[rng.choice(x.shape[0], size=sample_size, replace=False)]
    if sample_size > 0 and y.shape[0] > sample_size:
        y = y[rng.choice(y.shape[0], size=sample_size, replace=False)]

    n = x.shape[0]
    m = y.shape[0]
    if n < 2 or m < 2:
        return float("nan")

    k_xx = rbf_kernel(x, x, gamma=gamma)
    k_yy = rbf_kernel(y, y, gamma=gamma)
    k_xy = rbf_kernel(x, y, gamma=gamma)

    sum_xx = float((k_xx.sum() - np.trace(k_xx)) / (n * (n - 1)))
    sum_yy = float((k_yy.sum() - np.trace(k_yy)) / (m * (m - 1)))
    sum_xy = float(k_xy.mean())
    mmd2 = sum_xx + sum_yy - 2.0 * sum_xy
    return float(max(0.0, mmd2))


def _mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def compute_highd_separation_scores(
    embeddings: np.ndarray,
    n_train: int,
    cluster_method: str,
    num_clusters: int,
    runs: int,
    mmd_sample_size: int,
    seed: int,
) -> dict:
    if cluster_method != "kmeans":
        raise ValueError(f"Unsupported high-dimensional clustering method: {cluster_method}")
    if runs <= 0:
        raise ValueError("--highd_runs must be > 0")

    x_train = embeddings[:n_train]
    x_val = embeddings[n_train:]
    n_val = x_val.shape[0]

    if n_train < 2 or n_val < 2:
        raise ValueError("Need at least 2 train and 2 validation samples for high-D analysis.")

    k = int(max(2, min(num_clusters, n_train, n_val)))
    if k < 2:
        raise ValueError("Effective high-D cluster count is < 2; increase samples.")

    labels_split = np.concatenate(
        [np.zeros(n_train, dtype=np.int32), np.ones(n_val, dtype=np.int32)]
    )
    silhouette = float(silhouette_score(embeddings, labels_split, metric="cosine"))

    train_centroid = x_train.mean(axis=0)
    val_centroid = x_val.mean(axis=0)
    centroid_l2 = float(np.linalg.norm(train_centroid - val_centroid))
    centroid_cosine_similarity = _safe_cosine_similarity(train_centroid, val_centroid)
    centroid_cosine_distance = float(1.0 - centroid_cosine_similarity)

    rng = np.random.default_rng(seed)
    gamma = _estimate_rbf_gamma(x_train, x_val, sample_size=mmd_sample_size, rng=rng)
    mmd_rbf = _compute_mmd_rbf(
        x_train, x_val, gamma=gamma, sample_size=mmd_sample_size, rng=rng
    )

    train_ari_vals: List[float] = []
    val_ari_vals: List[float] = []
    train_nmi_vals: List[float] = []
    val_nmi_vals: List[float] = []
    train_ami_vals: List[float] = []
    val_ami_vals: List[float] = []

    for run_idx in range(runs):
        train_kmeans = KMeans(n_clusters=k, n_init=10, random_state=seed + run_idx)
        val_kmeans = KMeans(n_clusters=k, n_init=10, random_state=seed + 10_000 + run_idx)

        train_self_labels = train_kmeans.fit_predict(x_train)
        val_self_labels = val_kmeans.fit_predict(x_val)

        val_from_train_labels = train_kmeans.predict(x_val)
        train_from_val_labels = val_kmeans.predict(x_train)

        train_ari_vals.append(adjusted_rand_score(train_self_labels, train_from_val_labels))
        val_ari_vals.append(adjusted_rand_score(val_self_labels, val_from_train_labels))

        train_nmi_vals.append(
            normalized_mutual_info_score(train_self_labels, train_from_val_labels)
        )
        val_nmi_vals.append(normalized_mutual_info_score(val_self_labels, val_from_train_labels))

        train_ami_vals.append(adjusted_mutual_info_score(train_self_labels, train_from_val_labels))
        val_ami_vals.append(adjusted_mutual_info_score(val_self_labels, val_from_train_labels))

    train_ari_mean, train_ari_std = _mean_std(train_ari_vals)
    val_ari_mean, val_ari_std = _mean_std(val_ari_vals)
    train_nmi_mean, train_nmi_std = _mean_std(train_nmi_vals)
    val_nmi_mean, val_nmi_std = _mean_std(val_nmi_vals)
    train_ami_mean, train_ami_std = _mean_std(train_ami_vals)
    val_ami_mean, val_ami_std = _mean_std(val_ami_vals)

    return {
        "cluster_method": cluster_method,
        "highd_runs": int(runs),
        "highd_num_clusters_requested": int(num_clusters),
        "highd_num_clusters_used": int(k),
        "highd_split_silhouette_cosine": silhouette,
        "highd_centroid_l2_distance": centroid_l2,
        "highd_centroid_cosine_similarity": centroid_cosine_similarity,
        "highd_centroid_cosine_distance": centroid_cosine_distance,
        "highd_mmd_rbf_gamma": float(gamma),
        "highd_mmd_rbf_mmd2": mmd_rbf,
        "highd_train_cross_ari_mean": train_ari_mean,
        "highd_train_cross_ari_std": train_ari_std,
        "highd_val_cross_ari_mean": val_ari_mean,
        "highd_val_cross_ari_std": val_ari_std,
        "highd_train_cross_nmi_mean": train_nmi_mean,
        "highd_train_cross_nmi_std": train_nmi_std,
        "highd_val_cross_nmi_mean": val_nmi_mean,
        "highd_val_cross_nmi_std": val_nmi_std,
        "highd_train_cross_ami_mean": train_ami_mean,
        "highd_train_cross_ami_std": train_ami_std,
        "highd_val_cross_ami_mean": val_ami_mean,
        "highd_val_cross_ami_std": val_ami_std,
    }


def plot_tsne(
    coords: np.ndarray,
    n_train: int,
    output_path: str,
    title: str,
    color_by: str,
    article_ids: List[int],
):
    train_coords = coords[:n_train]
    val_coords = coords[n_train:]
    plt.figure(figsize=(10, 8))
    if color_by == "article":
        article_arr = np.asarray(article_ids, dtype=np.int32)
        train_article = article_arr[:n_train]
        val_article = article_arr[n_train:]
        scatter_train = plt.scatter(
            train_coords[:, 0],
            train_coords[:, 1],
            c=train_article,
            cmap="tab20",
            s=18,
            alpha=0.75,
            marker="o",
            label="train",
        )
        plt.scatter(
            val_coords[:, 0],
            val_coords[:, 1],
            c=val_article,
            cmap="tab20",
            s=18,
            alpha=0.75,
            marker="^",
            label="test",
        )
        cbar = plt.colorbar(scatter_train)
        cbar.set_label("article_id")
    else:
        plt.scatter(
            train_coords[:, 0],
            train_coords[:, 1],
            s=18,
            alpha=0.75,
            label="train",
            color="#1f77b4",
        )
        plt.scatter(
            val_coords[:, 0],
            val_coords[:, 1],
            s=18,
            alpha=0.75,
            label="test",
            color="#ff7f0e",
        )
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def maybe_write_csv(
    output_csv: str,
    coords: np.ndarray,
    n_train: int,
    train_texts: List[str],
    val_texts: List[str],
    train_article_ids: List[int],
    val_article_ids: List[int],
):
    if output_csv is None:
        return
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "x", "y", "article_id", "sequence"])
        for i, seq in enumerate(train_texts):
            writer.writerow(
                [
                    "train",
                    float(coords[i, 0]),
                    float(coords[i, 1]),
                    int(train_article_ids[i]),
                    seq,
                ]
            )
        for j, seq in enumerate(val_texts):
            idx = n_train + j
            writer.writerow(
                [
                    "test",
                    float(coords[idx, 0]),
                    float(coords[idx, 1]),
                    int(val_article_ids[j]),
                    seq,
                ]
            )


def build_sequence_lengths(args: argparse.Namespace) -> List[int]:
    if args.sequence_length_min is None and args.sequence_length_max is None:
        if args.sequence_length <= 1:
            raise ValueError("--sequence_length must be > 1")
        return [args.sequence_length]
    if args.sequence_length_min is None or args.sequence_length_max is None:
        raise ValueError(
            "Both --sequence_length_min and --sequence_length_max are required for a sweep."
        )
    if args.sequence_length_min <= 1:
        raise ValueError("--sequence_length_min must be > 1")
    if args.sequence_length_max < args.sequence_length_min:
        raise ValueError("--sequence_length_max must be >= --sequence_length_min")
    if args.sequence_length_multiplier <= 1:
        raise ValueError("--sequence_length_multiplier must be > 1")

    lengths: List[int] = []
    current = args.sequence_length_min
    while current <= args.sequence_length_max:
        lengths.append(current)
        current *= args.sequence_length_multiplier
    return lengths


def resolve_output_paths(
    args: argparse.Namespace, seq_length: int, sweep_mode: bool
) -> Tuple[str, str, str]:
    if args.output_dir is None and not sweep_mode:
        return args.output_plot, args.output_csv, args.output_scores_json

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("tsne_sweep_outputs")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    base = f"seq_{seq_length}"
    plot_path = str(output_dir / f"{base}.png")
    csv_path = str(output_dir / f"{base}.csv")
    score_path = str(output_dir / f"{base}_scores.json")
    return plot_path, csv_path, score_path


def resolve_effective_model_max_length(
    requested_max_length: int, embedding_tokenizer, embedding_model
) -> int:
    if requested_max_length <= 0:
        raise ValueError("--model_max_length must be > 0")

    def _usable_limit(v) -> int:
        if v is None:
            return 0
        try:
            x = int(v)
        except (TypeError, ValueError):
            return 0
        # HuggingFace tokenizers often use very large sentinels for "no explicit limit".
        if x <= 0 or x > 1_000_000:
            return 0
        return x

    limits = [requested_max_length]
    limits.append(_usable_limit(getattr(embedding_tokenizer, "model_max_length", None)))
    limits.append(
        _usable_limit(getattr(getattr(embedding_model, "config", None), "max_position_embeddings", None))
    )
    limits.append(_usable_limit(getattr(getattr(embedding_model, "config", None), "n_positions", None)))

    valid_limits = [x for x in limits if x > 0]
    if not valid_limits:
        return int(requested_max_length)
    return int(min(valid_limits))


def main():
    args = parse_args()

    torch, AutoModel, AutoTokenizer = load_hf_modules()
    device = resolve_device(torch, args.device)

    for path in [args.train_file, args.val_file]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Input file not found: {path}")
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")
    if args.chunk_line_buffer <= 0:
        raise ValueError("--chunk_line_buffer must be > 0")
    if not (0.0 < args.mixed_train_ratio < 1.0):
        raise ValueError("--mixed_train_ratio must be in (0, 1)")
    if args.cv_folds < 2:
        raise ValueError("--cv_folds must be >= 2")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if args.highd_num_clusters < 2:
        raise ValueError("--highd_num_clusters must be >= 2")
    if args.highd_runs <= 0:
        raise ValueError("--highd_runs must be > 0")
    if args.highd_mmd_sample_size < 0:
        raise ValueError("--highd_mmd_sample_size must be >= 0")

    embedding_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
    embedding_model = AutoModel.from_pretrained(args.embedding_model).to(device)
    effective_model_max_length = resolve_effective_model_max_length(
        requested_max_length=args.model_max_length,
        embedding_tokenizer=embedding_tokenizer,
        embedding_model=embedding_model,
    )
    seq_tokenizer = build_sequence_tokenizer(
        args.sequence_tokenizer_path, args.embedding_model, AutoTokenizer
    )

    print(f"Embedding model: {args.embedding_model}")
    print(f"Sequence tokenizer: {seq_tokenizer.name}")
    print(f"Device: {device}")
    print(f"Projection method: {args.projection_method}")
    print(f"Sampling mode: {args.sampling_mode}")
    print(f"Pooling: {args.pooling}")
    print(
        f"Embedding max length (requested/effective): "
        f"{args.model_max_length}/{effective_model_max_length}"
    )
    print(f"Streaming chunk_line_buffer: {args.chunk_line_buffer}")
    seq_lengths = build_sequence_lengths(args)
    sweep_mode = len(seq_lengths) > 1
    if sweep_mode:
        print(f"Sequence length sweep: {seq_lengths}")

    summary_rows = []
    for seq_length in seq_lengths:
        print(f"\n=== Running sequence_length={seq_length} ===")
        rng = random.Random(args.seed + seq_length)

        train_sequences, train_article_ids = collect_sequence_texts(
            split_name="train",
            file_path=args.train_file,
            seq_tokenizer=seq_tokenizer,
            seq_length=seq_length,
            stride=args.stride,
            chunk_line_buffer=args.chunk_line_buffer,
            max_sequences=args.max_sequences_per_split,
            sampling_mode=args.sampling_mode,
            rng=rng,
        )
        val_sequences, val_article_ids = collect_sequence_texts(
            split_name="validation",
            file_path=args.val_file,
            seq_tokenizer=seq_tokenizer,
            seq_length=seq_length,
            stride=args.stride,
            chunk_line_buffer=args.chunk_line_buffer,
            max_sequences=args.max_sequences_per_split,
            sampling_mode=args.sampling_mode,
            rng=rng,
        )

        if not train_sequences:
            raise ValueError(
                f"No train sequences created for seq_length={seq_length}. Try smaller length."
            )
        if not val_sequences:
            raise ValueError(
                f"No validation sequences created for seq_length={seq_length}. Try smaller length."
            )

        if args.mix_splits_before_sampling:
            combined = list(
                zip(
                    train_sequences + val_sequences,
                    train_article_ids + val_article_ids,
                )
            )
            rng.shuffle(combined)
            mixed_train_count = int(round(len(combined) * args.mixed_train_ratio))
            mixed_train = combined[:mixed_train_count]
            mixed_val = combined[mixed_train_count:]
            train_sequences = [seq for seq, _ in mixed_train]
            train_article_ids = [article_id for _, article_id in mixed_train]
            val_sequences = [seq for seq, _ in mixed_val]
            val_article_ids = [article_id for _, article_id in mixed_val]
            print(
                f"Applied split mixing: train={len(train_sequences)}, validation={len(val_sequences)}"
            )

        print(f"Train sequences: {len(train_sequences)}")
        print(f"Test sequences: {len(val_sequences)}")

        combined_sequences = train_sequences + val_sequences
        n_train = len(train_sequences)
        embeddings = embed_texts(
            texts=combined_sequences,
            tokenizer=embedding_tokenizer,
            model=embedding_model,
            torch_module=torch,
            device=device,
            batch_size=args.batch_size,
            model_max_length=effective_model_max_length,
            pooling=args.pooling,
        )
        separation_scores = compute_logistic_separation_scores(
            embeddings=embeddings,
            n_train=n_train,
            cv_folds=args.cv_folds,
            seed=args.seed,
        )
        if args.highd_enabled:
            highd_scores = compute_highd_separation_scores(
                embeddings=embeddings,
                n_train=n_train,
                cluster_method=args.highd_cluster_method,
                num_clusters=args.highd_num_clusters,
                runs=args.highd_runs,
                mmd_sample_size=args.highd_mmd_sample_size,
                seed=args.seed,
            )
            separation_scores.update(highd_scores)
        print("Embedding-space split separation (LogReg + Stratified CV):")
        print(f"  accuracy_mean: {separation_scores['logreg_cv_accuracy_mean']:.6f}")
        print(f"  accuracy_std: {separation_scores['logreg_cv_accuracy_std']:.6f}")
        print(f"  roc_auc_mean: {separation_scores['logreg_cv_roc_auc_mean']:.6f}")
        print(f"  roc_auc_std: {separation_scores['logreg_cv_roc_auc_std']:.6f}")
        if args.highd_enabled:
            print("High-D split separation (clustering + distribution metrics):")
            print(
                f"  val_cross_ari_mean: {separation_scores['highd_val_cross_ari_mean']:.6f}"
            )
            print(
                f"  train_cross_ari_mean: {separation_scores['highd_train_cross_ari_mean']:.6f}"
            )
            print(
                "  split_silhouette_cosine: "
                f"{separation_scores['highd_split_silhouette_cosine']:.6f}"
            )
            print(f"  mmd_rbf_mmd2: {separation_scores['highd_mmd_rbf_mmd2']:.6f}")

        output_plot, output_csv, output_scores_json = resolve_output_paths(
            args=args, seq_length=seq_length, sweep_mode=sweep_mode
        )
        separation_scores["sequence_length"] = int(seq_length)
        if output_scores_json:
            with open(output_scores_json, "w", encoding="utf-8") as f:
                json.dump(separation_scores, f, indent=2)
            print(f"Saved separation scores: {output_scores_json}")

        coords = run_projection(
            embeddings=embeddings,
            method=args.projection_method,
            tsne_perplexity=args.tsne_perplexity,
            tsne_learning_rate=args.tsne_learning_rate,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
            umap_metric=args.umap_metric,
            seed=args.seed,
        )

        title = (
            f"{args.projection_method.upper()} of sequence embeddings\n"
            f"model={args.embedding_model}, seq_len={seq_length}, stride={args.stride}"
        )
        plot_tsne(
            coords,
            n_train,
            output_plot,
            title,
            args.color_by,
            train_article_ids + val_article_ids,
        )
        maybe_write_csv(
            output_csv,
            coords,
            n_train,
            train_sequences,
            val_sequences,
            train_article_ids,
            val_article_ids,
        )

        print(f"Saved plot: {output_plot}")
        if output_csv:
            print(f"Saved CSV: {output_csv}")
        summary_rows.append(
            {
                "sequence_length": seq_length,
                "plot": output_plot,
                "csv": output_csv,
                "scores_json": output_scores_json,
                "accuracy_mean": separation_scores["logreg_cv_accuracy_mean"],
                "roc_auc_mean": separation_scores["logreg_cv_roc_auc_mean"],
                "highd_val_cross_ari_mean": separation_scores.get(
                    "highd_val_cross_ari_mean", float("nan")
                ),
                "highd_split_silhouette_cosine": separation_scores.get(
                    "highd_split_silhouette_cosine", float("nan")
                ),
                "highd_mmd_rbf_mmd2": separation_scores.get(
                    "highd_mmd_rbf_mmd2", float("nan")
                ),
            }
        )

    if sweep_mode:
        summary_path = Path(resolve_output_paths(args, seq_lengths[0], True)[0]).parent / "summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sequence_length",
                    "accuracy_mean",
                    "roc_auc_mean",
                    "highd_val_cross_ari_mean",
                    "highd_split_silhouette_cosine",
                    "highd_mmd_rbf_mmd2",
                    "plot",
                    "csv",
                    "scores_json",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Sweep summary: {summary_path}")


if __name__ == "__main__":
    main()
