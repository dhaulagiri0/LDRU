import chex
import jax
import jax.numpy as jnp
import optax
from optax.contrib import muon as optax_muon
import numpy as np
from typing import Callable, Tuple, Dict, List, Optional, Iterator
import tqdm
import re
import os
import json
import math
from collections import Counter
import matplotlib
import os
import json
import orbax.checkpoint as ocp
import datetime
import shutil

matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from causal_ldru_v2 import CausalLDRUConfig, create_causal_ldru_model, BinaryOperator
from causal_ldru_v2 import (
    CausalLDRULayer,
    CausalLDRUEncoder,
    CausalLDRULanguageModel,
)
from improved_binary_operator import (
    ConvexGatedBinaryOperator,
    GRCOperator,
    AblationBinaryOperator,
)
from tensorboardX import SummaryWriter

# Import transformer from supplementary_code
import sys

sys.path.append("supplementary_code-main")
from ldru.models.transformer import (
    make_transformer,
    make_transformer_encoder,
    TransformerConfig,
    TransformerEncoder,
)
from ldru.models import positional_encodings as pos_encs_lib
import sentencepiece as spm

DEFAULT_TOKENIZER_FOLDER = "tokenizers"


from enum import Enum


class TokenizerType(str, Enum):
    SENTENCEPIECE = "sentencepiece"
    TEXT = "text"
    TIKTOKEN_GPT2 = "tiktoken_gpt2"


class ComputeDType(str, Enum):
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"


BINARY_OPERATOR_REGISTRY = {
    "default": None,
    "binary": BinaryOperator,
    "convex_gated": ConvexGatedBinaryOperator,
    "grc": GRCOperator,
    "ablation": AblationBinaryOperator,
}


def _resolve_compute_dtype(compute_dtype: str) -> jnp.dtype:
    if isinstance(compute_dtype, ComputeDType):
        compute_dtype = compute_dtype.value
    if compute_dtype == ComputeDType.FLOAT32.value:
        return jnp.float32
    if compute_dtype == ComputeDType.BFLOAT16.value:
        return jnp.bfloat16
    raise ValueError(
        f"Unsupported compute dtype '{compute_dtype}'. "
        f"Expected one of: {[d.value for d in ComputeDType]}"
    )


def configure_mixed_precision(compute_dtype: str) -> None:
    """
    Configure Haiku module policies.

    Params stay fp32 for optimizer stability. Compute can be fp32/bfloat16.
    Output remains fp32 for numerically stable loss computation.
    """
    import haiku as hk

    resolved_dtype = _resolve_compute_dtype(compute_dtype)
    policy_ctor = getattr(hk.mixed_precision, "Policy", None)
    if policy_ctor is not None:
        policy = policy_ctor(
            param_dtype=jnp.float32,
            compute_dtype=resolved_dtype,
            output_dtype=jnp.float32,
        )
    else:
        # Compatibility path for older Haiku versions.
        import jmp

        policy = jmp.Policy(
            param_dtype=jnp.float32,
            compute_dtype=resolved_dtype,
            output_dtype=jnp.float32,
        )

    module_classes = [
        hk.Linear,
        hk.Embed,
        hk.LayerNorm,
        hk.LSTM,
        BinaryOperator,
        ConvexGatedBinaryOperator,
        GRCOperator,
        AblationBinaryOperator,
        CausalLDRULayer,
        CausalLDRUEncoder,
        CausalLDRULanguageModel,
        TransformerEncoder,
    ]
    if not hasattr(hk.mixed_precision, "set_policy"):
        raise AttributeError(
            "This Haiku version does not expose mixed_precision.set_policy."
        )
    for module_cls in module_classes:
        hk.mixed_precision.set_policy(module_cls, policy)


def resolve_binary_operator(operator_name: Optional[str]):
    if operator_name is None:
        return None
    if operator_name not in BINARY_OPERATOR_REGISTRY:
        raise ValueError(
            f"Unknown binary operator '{operator_name}'. "
            f"Available: {', '.join(BINARY_OPERATOR_REGISTRY.keys())}"
        )
    return BINARY_OPERATOR_REGISTRY[operator_name]


def binary_operator_to_name(operator_cls) -> str:
    if operator_cls is None:
        return "default"
    for name, cls in BINARY_OPERATOR_REGISTRY.items():
        if cls is operator_cls:
            return name
    return getattr(operator_cls, "__name__", "custom")


@chex.dataclass
class LDRUExperimenstConfig:
    """Configuration for causal LDRU language model."""

    # Model architecture
    embedding_dim: int
    vocab_size: int
    num_layers: int = 1
    hidden_dim: int = 512

    # LDRU specific
    widening_factor: int = 4
    emb_init_scale: float = 0.02

    # Causal modeling
    causal_masking: bool = True
    max_sequence_length: int = 1024
    use_positional_encoding: bool = False

    # Binary operator
    operator: Optional[Callable] = None
    binop_expansion_factor: int = 4
    ablation_expansion_mode: str = "grc"
    ablation_combine_mode: str = "grc"

    # Scan method: 'default' (assoc_scan), 'simple', or 'pairwise'
    scan_method: str = "default"

    # Whether to expand sequence to power of 2 with random zero insertion
    expand_to_power_of_2: bool = False

    # Whether to apply attention at each scan step in custom associative scan
    attention_per_scan_step: bool = False

    # specifies transformer encoding type
    use_alibi: bool = False  # defaults to sin cos

    num_transformer_layers: int = 5
    num_transformer_heads: int = 8
    use_embeddings: bool = True
    share_embeddings: bool = False
    chunk_size: Optional[int] = None  # Use full attention
    causal_masking: bool = True  # Critical for causal language modeling
    tie_embeddings_transformer: bool = False
    tie_embeddings_ldru: bool = False
    transformer_prenorm_gelu_block: bool = False
    ldru_prenorm_gelu_block: bool = False
    # Aliases consumed directly by causal_ldru_v2 components.
    tie_embeddings: bool = False
    prenorm_gelu_block: bool = False

    # General training hyperparameters
    initial_learning_rate: float = (1e-3,)
    l2_lambda: float = (0.2,)
    seq_length: int = (32,)
    batch_size: int = (128,)
    min_learning_rate: float = (1e-6,)  # Set min LR for cosine decay
    num_epochs: int = (100,)
    dropout_prob: float = 0.3

    blelloch_random: bool = False  # Whether to use random pairing in Blelloch scan


# base tokenizer class to define the interface for different tokenizers (SPTokenizer, TextTokenizer, etc.)
class BaseTokenizer:
    def get_piece_size(self) -> int:
        """Get vocabulary size from tokenizer data."""
        return None

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs using tokenizer data."""
        return None

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text using tokenizer data."""
        return None

    @classmethod
    def do_cleaning(cls) -> bool:
        return False

    def get_tokenizer_path(self) -> str:
        """Get the path to save/load the tokenizer model."""
        return None

    def get_tokenizer_type(self) -> str:
        """Get the type of tokenizer (e.g., 'sentencepiece', 'text')."""
        return None


# Wrapper for SentencePiece tokenizer to fit our BaseTokenizer interface.
class SPTokenizer(BaseTokenizer):
    """Wrapper for SentencePiece tokenizer to fit our BaseTokenizer interface."""

    def __init__(
        self,
        text_file_path: str = None,
        vocab_size: int = None,
        model_prefix: str = "ptb_tokenizer",
        model_path=None,
        use_whitespace_tokenization=True,
    ):
        super().__init__()
        self.tokenizer = None
        if model_path is not None:
            self.tokenizer = self._load_sentencepiece_tokenizer(model_path)
            self.model_path = model_path
        else:
            self.tokenizer = self._create_sentencepiece_tokenizer(
                text_file_path, vocab_size, model_prefix, use_whitespace_tokenization
            )
            self.model_path = f"{model_prefix}.model"
        self.type = TokenizerType.SENTENCEPIECE

    def _load_sentencepiece_tokenizer(
        self, model_path: str
    ) -> spm.SentencePieceProcessor:
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(model_path)
        return tokenizer

    def _create_sentencepiece_tokenizer(
        self,
        text_file_path: str,
        vocab_size: int,
        model_prefix: str,
        use_whitespace_tokenization: bool,
    ) -> spm.SentencePieceProcessor:
        spm.SentencePieceTrainer.train(
            input=text_file_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            split_by_whitespace=use_whitespace_tokenization,
        )
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(f"{model_prefix}.model")
        return tokenizer

    def get_piece_size(self) -> int:
        return self.tokenizer.get_piece_size()

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)

    @classmethod
    def do_cleaning(cls) -> bool:
        """SentencePiece can handle raw text, so no cleaning needed."""
        return False

    def get_tokenizer_path(self):
        return self.model_path

    def get_tokenizer_type(self):
        return self.type


# Simple word level tokenizer
class TextTokenizer(BaseTokenizer):
    """Simple word-level tokenizer for text data."""

    def __init__(self, text: str, vocab_size: int = None, tokenizer_data=None):
        """
        Args:
            text: Raw text data
            vocab_size: Maximum vocabulary size (None for no limit)
        """
        if tokenizer_data is not None:
            self.vocab_size = tokenizer_data["vocab_size"]
            self.word_to_id = tokenizer_data["word_to_id"]
            self.id_to_word = tokenizer_data["id_to_word"]
            self.special_tokens = tokenizer_data["special_tokens"]
            print(f"Restored tokenizer with vocabulary size: {self.vocab_size}")
            print(f"Sample words: {list(self.word_to_id.keys())[:20]}...")
            return

        # Tokenize text into words (simple whitespace + punctuation splitting)
        words = self._tokenize_words(text)

        # Get word frequencies
        word_counts = Counter(words)

        # Sort by frequency and take top vocab_size
        if vocab_size:
            most_common = word_counts.most_common(
                vocab_size - 3
            )  # Reserve space for special tokens
        else:
            most_common = word_counts.most_common()

        # Create vocabulary with special tokens
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,  # Beginning of sequence
        }

        # Add words to vocabulary
        self.word_to_id = self.special_tokens.copy()
        self.id_to_word = {v: k for k, v in self.special_tokens.items()}

        for word, _ in most_common:
            if word not in self.word_to_id:
                idx = len(self.word_to_id)
                self.word_to_id[word] = idx
                self.id_to_word[idx] = word

        self.vocab_size = len(self.word_to_id)
        print(f"Created tokenizer with vocabulary size: {self.vocab_size}")
        print(f"Sample words: {list(self.word_to_id.keys())[:20]}...")
        self.type = TokenizerType.TEXT

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning: normalize whitespace and remove excessive punctuation."""
        # Basic text cleaning for word-level tokenization
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove excessive punctuation (keep basic punctuation)
        text = re.sub(r"[^\w\s\.\,\!\?\;\:\-\'\"]", "", text)
        # Remove extra spaces around punctuation
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = re.sub(r"([,.!?;:])\s+", r"\1 ", text)

        return text

    def _tokenize_words(self, text: str) -> List[str]:
        """
        Split text into words using simple regex.
        Keeps punctuation as separate tokens.
        """
        # Split on whitespace and punctuation, but keep punctuation
        import re

        text = self._clean_text(text)

        # Pattern to split on whitespace and capture punctuation as separate tokens
        pattern = r"(\w+|[^\w\s])"
        tokens = re.findall(pattern, text.lower())

        # Filter out empty strings
        tokens = [token for token in tokens if token.strip()]

        return tokens

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        # clean text
        text = self._clean_text(text)
        words = self._tokenize_words(text)
        return [self.word_to_id.get(word, self.word_to_id["<UNK>"]) for word in words]

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        words = []
        for idx in token_ids:
            word = self.id_to_word.get(idx, "<UNK>")
            if word not in ["<PAD>", "<UNK>", "<BOS>"]:
                words.append(word)

        # Simple reconstruction: join with spaces
        # More sophisticated tokenizers would handle punctuation differently
        return " ".join(words)

    def get_piece_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size

    @classmethod
    def do_cleaning(cls) -> bool:
        return True

    def get_tokenizer_path(self):
        return None

    def get_tokenizer_type(self):
        return self.type


class TiktokenGPT2Tokenizer(BaseTokenizer):
    """Wrapper for tiktoken GPT-2 encoding."""

    def __init__(self, encoding_name: str = "gpt2"):
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'tiktoken'. Install with: pip install tiktoken"
            ) from exc

        self.encoding_name = encoding_name
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.type = TokenizerType.TIKTOKEN_GPT2

    def get_piece_size(self) -> int:
        return int(self.tokenizer.n_vocab)

    def encode(self, text: str) -> List[int]:
        # Keep parity with existing pipeline expectations: no implicit EOT insertion.
        return self.tokenizer.encode_ordinary(text)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)

    @classmethod
    def do_cleaning(cls) -> bool:
        return False

    def get_tokenizer_path(self):
        return self.encoding_name

    def get_tokenizer_type(self):
        return self.type


TOKENIZER_TYPE_TO_CLASS = {
    TokenizerType.SENTENCEPIECE: SPTokenizer,
    TokenizerType.TEXT: TextTokenizer,
    TokenizerType.TIKTOKEN_GPT2: TiktokenGPT2Tokenizer,
}


def load_text(file_path: str) -> str:
    """
    Load text from a file.
    """
    print(f"Loading text from: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        # Try with different encodings
        with open(file_path, "r", encoding="latin1") as f:
            text = f.read()

    return text


def create_text_dataset(
    text: str, tokenizer: BaseTokenizer, seq_length: int, stride: int = None
) -> jnp.ndarray:
    """
    Create dataset from text by splitting into sequences.
    For LDRU: we create sequences where we predict the last token from the previous ones.

    Args:
        text: Raw text
        tokenizer: Tokenizer instance
        seq_length: Length of each sequence (including the target token)
        stride: Step size between sequences (default: seq_length for no overlap)

    Returns:
        Array of token sequences [num_sequences, seq_length]
    """
    if stride is None:
        stride = seq_length

    # Tokenize text
    print("Tokenizing text...")
    token_ids = tokenizer.encode(text)
    print(f"Total tokens: {len(token_ids):,}")

    # Create sequences where each sequence is [context..., target]
    sequences = []
    for i in range(0, len(token_ids) - seq_length + 1, stride):
        sequence = token_ids[i : i + seq_length]
        sequences.append(sequence)

    sequences = np.array(sequences)
    print(f"Created {len(sequences):,} sequences of length {seq_length}")
    print(f"Each sequence: [context tokens (length {seq_length-1})] -> [target token]")

    return jnp.array(sequences)


def create_dataset_from_text_file(
    file_path: str,
    seq_length: int = 128,
    stride: int = None,
    train_split: float = 0.9,
    tokenizer: BaseTokenizer = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, BaseTokenizer]:
    """
    Complete pipeline to create train/val datasets from text file.

    Args:
        file_path: Path to text file
        seq_length: Length of each sequence
        max_vocab_size: Maximum vocabulary size
        stride: Step size between sequences
        train_split: Fraction of data for training

    Returns:
        Tuple of (train_sequences, val_sequences, tokenizer)
    """
    text = load_text(file_path)

    # Create sequences
    sequences = create_text_dataset(text, tokenizer, seq_length, stride)

    # Split into train/val
    num_train = int(len(sequences) * train_split)

    # Shuffle before splitting
    rng_key = jax.random.PRNGKey(42)
    indices = jax.random.permutation(rng_key, len(sequences))
    shuffled_sequences = sequences[indices]

    train_sequences = shuffled_sequences[:num_train]
    val_sequences = shuffled_sequences[num_train:]

    if train_split < 1.0:
        print(f"Training sequences: {len(train_sequences):,}")
        print(f"Validation sequences: {len(val_sequences):,}")
    else:
        print(f"All sequences used for dataset: {len(train_sequences):,}")

    return train_sequences, val_sequences


def next_token_loss(params, model, rng_key, token_ids):
    """
    Compute next token prediction loss for LDRU (only position 0 is meaningful).

    Since LDRU only outputs meaningful information at position 0 after tree reduction,
    we need to modify our approach. We'll train the model to predict the LAST token
    in the sequence using the output at position 0.
    """
    batch_size, seq_length = token_ids.shape

    # Mask out the last token for input (LDRU will predict it)
    input_token_ids = token_ids[:, :-1]  # [B, L-1]
    # Forward pass
    logits = model.apply(params, rng_key, input_token_ids)  # [B, L-1, V]

    # LDRU's meaningful output is at position 0 after processing the full sequence
    # We use this to predict the last token of the sequence
    pred_logits = logits[:, 0, :]  # [B, V] - position 0 contains the summary
    targets = token_ids[:, -1]  # [B] - predict the last token

    # Compute cross-entropy loss
    log_probs = jax.nn.log_softmax(pred_logits, axis=-1)
    target_log_probs = jnp.take_along_axis(
        log_probs, targets[..., None], axis=-1
    ).squeeze(-1)

    # Mean loss over batch
    loss = -jnp.mean(target_log_probs)

    # Compute accuracy
    predictions = jnp.argmax(pred_logits, axis=-1)
    accuracy = jnp.mean(predictions == targets)

    return loss, {"accuracy": accuracy, "perplexity": jnp.exp(loss)}


def lstm_next_token_loss(params, model, rng_key, token_ids):
    """
    Compute next token prediction loss for LSTM models.

    LSTM outputs logits at every position, so we can use standard
    sequence-to-sequence prediction across all positions.
    """
    batch_size, seq_length = token_ids.shape

    # For LSTM, we use the full sequence and predict next tokens
    input_token_ids = token_ids[:, :-1]  # [B, L-1] - input context
    targets = token_ids[:, 1:]  # [B, L-1] - targets (shifted by 1)

    # Forward pass
    logits = model.apply(params, rng_key, input_token_ids)  # [B, L-1, V]

    # Compute loss across all positions (full sequence-to-sequence)
    # This is more standard for language modeling with LSTMs
    log_probs = jax.nn.log_softmax(logits, axis=-1)  # [B, L-1, V]

    # Gather target probabilities for each position
    target_log_probs = jnp.take_along_axis(
        log_probs, targets[..., None], axis=-1
    ).squeeze(
        -1
    )  # [B, L-1]

    # Mean loss over all positions and batch
    loss = -jnp.mean(target_log_probs)

    # Compute accuracy (using last position for consistency with reporting)
    last_position_logits = logits[:, -1, :]  # [B, V]
    last_position_targets = targets[:, -1]  # [B]
    predictions = jnp.argmax(last_position_logits, axis=-1)
    accuracy = jnp.mean(predictions == last_position_targets)

    # Compute per-position perplexity for seq2seq models
    per_position_loss = -jnp.mean(
        target_log_probs, axis=0
    )  # [L-1] - average over batch
    per_position_perplexity = jnp.exp(per_position_loss)  # [L-1]

    return loss, {
        "accuracy": accuracy,
        "perplexity": jnp.exp(loss),
        "per_position_perplexity": per_position_perplexity,
        "per_position_loss": per_position_loss,
    }


def lstm_last_position_loss(params, model, rng_key, token_ids):
    """
    Compute next token prediction loss for LSTM models using only the last position.
    This is similar to how LDRU works but using the last position instead of position 0.
    """
    batch_size, seq_length = token_ids.shape

    # For LSTM, we use the full sequence and predict next tokens
    input_token_ids = token_ids[:, :-1]  # [B, L-1] - input context
    targets = token_ids[:, 1:]  # [B, L-1] - targets (shifted by 1)

    # Forward pass
    logits = model.apply(params, rng_key, input_token_ids)  # [B, L-1, V]

    # Use the last position's output to predict the next token
    pred_logits = logits[:, -1, :]  # [B, V] - use last position
    target = token_ids[:, -1]  # [B] - predict the last token

    # Compute cross-entropy loss
    log_probs = jax.nn.log_softmax(pred_logits, axis=-1)
    target_log_probs = jnp.take_along_axis(
        log_probs, target[..., None], axis=-1
    ).squeeze(-1)

    # Mean loss over batch
    loss = -jnp.mean(target_log_probs)

    # Compute accuracy
    predictions = jnp.argmax(pred_logits, axis=-1)
    accuracy = jnp.mean(predictions == target)

    return loss, {"accuracy": accuracy, "perplexity": jnp.exp(loss)}


def ldru_seq2seq_loss(params, model, rng_key, token_ids, lambda_l2=0.0):
    """
    Sequence-to-sequence autoregressive LM loss for LDRU.
    This mirrors the standard nanoGPT-style objective (no clipping/fallback shaping).
    """
    input_token_ids = token_ids[:, :-1]  # [B, L-1]
    targets = token_ids[:, 1:]  # [B, L-1]

    logits = model.apply(params, rng_key, input_token_ids)  # [B, L-1, V]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(
        log_probs, targets[..., None], axis=-1
    ).squeeze(-1)  # [B, L-1]

    loss = -jnp.mean(target_log_probs)
    loss += lambda_l2 * compute_l2_loss(params)

    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == targets)

    per_position_loss = -jnp.mean(target_log_probs, axis=0)
    per_position_perplexity = jnp.exp(per_position_loss)

    return loss, {
        "accuracy": accuracy,
        "perplexity": jnp.exp(loss),
        "per_position_perplexity": per_position_perplexity,
        "per_position_loss": per_position_loss,
    }


def transformer_seq2seq_loss(params, model, rng_key, token_ids, lambda_l2=0.0):
    """Standard autoregressive seq2seq LM loss (nanoGPT-style, no clipping)."""
    input_token_ids = token_ids[:, :-1]  # [B, L-1]
    targets = token_ids[:, 1:]  # [B, L-1]

    logits = model.apply(params, rng_key, input_token_ids)  # [B, L-1, V]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(
        log_probs, targets[..., None], axis=-1
    ).squeeze(-1)  # [B, L-1]

    loss = -jnp.mean(target_log_probs)
    loss += lambda_l2 * compute_l2_loss(params)

    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == targets)

    per_position_loss = -jnp.mean(target_log_probs, axis=0)
    per_position_perplexity = jnp.exp(per_position_loss)

    return loss, {
        "accuracy": accuracy,
        "perplexity": jnp.exp(loss),
        "per_position_perplexity": per_position_perplexity,
        "per_position_loss": per_position_loss,
    }


def compute_l2_loss(params, module_name=None):
    l2_sum = 0.0
    if module_name is not None:
        for path, param in params.items():
            # Skip if "layer_norm" is in the parameter path
            if "layer_norm" in path.lower():
                continue
            if module_name in path and isinstance(param, jnp.ndarray):
                l2_sum += jnp.sum(jnp.square(param))
    else:
        for path, param in params.items():
            # Skip if "layer_norm" is in the parameter path
            if "layer_norm" in path.lower():
                continue
            if isinstance(param, jnp.ndarray):
                l2_sum += jnp.sum(jnp.square(param))
    return l2_sum


def create_data_loader(sequences, batch_size, rng_key):
    """Simple data loader that yields batches."""

    num_samples = sequences.shape[0]
    num_batches = num_samples // batch_size

    if num_samples == 0 or num_batches == 0:
        return

    # For very large datasets, avoid materializing a full permutation.
    if num_samples > 2_000_000:
        seed = int(jax.random.randint(rng_key, shape=(), minval=0, maxval=2**31 - 1))
        np_rng = np.random.default_rng(seed)
        for _ in range(num_batches):
            batch_indices = np_rng.integers(0, num_samples, size=batch_size)
            yield np.asarray(sequences[batch_indices])
        return

    indices = np.asarray(jax.random.permutation(rng_key, num_samples))
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_indices = indices[start_idx:end_idx]
        yield np.asarray(sequences[batch_indices])


def iter_token_sequences_from_text_file(
    file_path: str,
    tokenizer: BaseTokenizer,
    seq_length: int,
    stride: int,
    chunk_line_buffer: int = 4096,
) -> Iterator[np.ndarray]:
    """Stream token sequences from a text file without materializing full datasets."""
    if seq_length <= 0:
        raise ValueError("seq_length must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    token_buffer: List[int] = []
    window_start = 0

    def _flush_available_sequences() -> Iterator[np.ndarray]:
        nonlocal token_buffer, window_start
        while window_start + seq_length <= len(token_buffer):
            yield np.asarray(
                token_buffer[window_start : window_start + seq_length], dtype=np.int32
            )
            window_start += stride

        if window_start >= max(seq_length, stride * 8):
            token_buffer = token_buffer[window_start:]
            window_start = 0

    lines: List[str] = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            lines.append(line)
            if len(lines) >= chunk_line_buffer:
                token_buffer.extend(tokenizer.encode("".join(lines)))
                lines.clear()
                yield from _flush_available_sequences()

    if lines:
        token_buffer.extend(tokenizer.encode("".join(lines)))
        yield from _flush_available_sequences()


def estimate_num_sequences_from_text_file(
    file_path: str,
    tokenizer: BaseTokenizer,
    seq_length: int,
    stride: int,
    chunk_line_buffer: int = 4096,
) -> int:
    """Count sequence windows produced by streaming tokenization."""
    if seq_length <= 0:
        raise ValueError("seq_length must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    token_buffer: List[int] = []
    window_start = 0
    count = 0
    lines: List[str] = []

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            lines.append(line)
            if len(lines) >= chunk_line_buffer:
                token_buffer.extend(tokenizer.encode("".join(lines)))
                lines.clear()
                while window_start + seq_length <= len(token_buffer):
                    count += 1
                    window_start += stride
                if window_start >= max(seq_length, stride * 8):
                    token_buffer = token_buffer[window_start:]
                    window_start = 0

    if lines:
        token_buffer.extend(tokenizer.encode("".join(lines)))
        while window_start + seq_length <= len(token_buffer):
            count += 1
            window_start += stride

    return count


def estimate_num_sequences_from_file_size(
    file_path: str,
    seq_length: int,
    stride: int,
    bytes_per_token: float = 4.0,
) -> int:
    """Fast rough estimate of sequence count from file size."""
    if seq_length <= 0:
        raise ValueError("seq_length must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if bytes_per_token <= 0:
        raise ValueError("bytes_per_token must be > 0")

    total_bytes = os.path.getsize(file_path)
    approx_tokens = max(0, int(total_bytes / bytes_per_token))
    if approx_tokens < seq_length:
        return 0
    return 1 + (approx_tokens - seq_length) // stride


def resolve_sequence_bin_dtype(dtype_name: str) -> np.dtype:
    dtype_map = {
        "uint16": np.uint16,
        "uint32": np.uint32,
        "int32": np.int32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(
            f"Unsupported --seq_bin_dtype '{dtype_name}'. "
            f"Expected one of: {list(dtype_map.keys())}"
        )
    return dtype_map[dtype_name]


def load_pretokenized_sequences(
    file_path: str, seq_length: int, dtype_name: str
) -> np.memmap:
    if seq_length <= 0:
        raise ValueError("seq_length must be > 0")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pretokenized sequence file not found: {file_path}")

    dtype = resolve_sequence_bin_dtype(dtype_name)
    itemsize = np.dtype(dtype).itemsize
    file_size = os.path.getsize(file_path)

    if file_size == 0:
        raise ValueError(f"Pretokenized sequence file is empty: {file_path}")
    if file_size % itemsize != 0:
        raise ValueError(
            f"File size ({file_size}) is not divisible by dtype itemsize ({itemsize}) "
            f"for {file_path}."
        )

    total_tokens = file_size // itemsize
    if total_tokens % seq_length != 0:
        raise ValueError(
            f"Token count ({total_tokens}) in {file_path} is not divisible by "
            f"seq_length ({seq_length})."
        )

    num_sequences = total_tokens // seq_length
    if num_sequences == 0:
        raise ValueError(
            f"No full sequences of length {seq_length} available in {file_path}."
        )

    return np.memmap(
        file_path, dtype=dtype, mode="r", shape=(num_sequences, seq_length)
    )


def load_pretokenized_token_stream(file_path: str, dtype_name: str) -> np.memmap:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pretokenized token file not found: {file_path}")

    dtype = resolve_sequence_bin_dtype(dtype_name)
    itemsize = np.dtype(dtype).itemsize
    file_size = os.path.getsize(file_path)

    if file_size == 0:
        raise ValueError(f"Pretokenized token file is empty: {file_path}")
    if file_size % itemsize != 0:
        raise ValueError(
            f"File size ({file_size}) is not divisible by dtype itemsize ({itemsize}) "
            f"for {file_path}."
        )

    total_tokens = file_size // itemsize
    if total_tokens == 0:
        raise ValueError(f"No tokens found in {file_path}.")

    return np.memmap(file_path, dtype=dtype, mode="r", shape=(total_tokens,))


def token_stream_to_sequence_view(
    token_stream: np.ndarray, seq_length: int, stride: int
) -> np.ndarray:
    if seq_length <= 0:
        raise ValueError("seq_length must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    total_tokens = int(token_stream.shape[0])
    if total_tokens < seq_length:
        raise ValueError(
            f"Token stream has only {total_tokens} tokens, fewer than seq_length={seq_length}."
        )

    num_sequences = 1 + (total_tokens - seq_length) // stride
    if num_sequences <= 0:
        raise ValueError(
            f"No full windows can be formed with seq_length={seq_length}, stride={stride}."
        )

    itemsize = token_stream.dtype.itemsize
    return np.lib.stride_tricks.as_strided(
        token_stream,
        shape=(num_sequences, seq_length),
        strides=(stride * itemsize, itemsize),
        writeable=False,
    )


def create_nanogpt_token_stream_loader(
    token_stream: np.ndarray,
    seq_length: int,
    batch_size: int,
    rng_key,
    num_batches: int,
) -> Iterator[np.ndarray]:
    """nanoGPT-style random contiguous token batches sampled with replacement."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if seq_length <= 0:
        raise ValueError("seq_length must be > 0")
    if num_batches <= 0:
        raise ValueError("num_batches must be > 0")

    total_tokens = int(token_stream.shape[0])
    max_start = total_tokens - seq_length
    if max_start <= 0:
        raise ValueError(
            f"Token stream has {total_tokens} tokens, cannot sample seq_length={seq_length}."
        )

    seed = int(jax.random.randint(rng_key, shape=(), minval=0, maxval=2**31 - 1))
    np_rng = np.random.default_rng(seed)
    for _ in range(num_batches):
        starts = np_rng.integers(0, max_start, size=batch_size)
        batch = np.stack(
            [np.asarray(token_stream[s : s + seq_length], dtype=np.int32) for s in starts],
            axis=0,
        )
        yield batch


def _pop_random_batch(
    shuffle_buffer: List[np.ndarray], batch_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Pop a random batch from a list-backed shuffle buffer."""
    picked_indices = np.sort(
        rng.choice(len(shuffle_buffer), size=batch_size, replace=False)
    )
    batch = np.stack([shuffle_buffer[int(idx)] for idx in picked_indices], axis=0)

    for idx in reversed(picked_indices):
        shuffle_buffer.pop(int(idx))

    return batch


def create_streaming_data_loader(
    file_path: str,
    tokenizer: BaseTokenizer,
    seq_length: int,
    stride: int,
    batch_size: int,
    rng_key,
    shuffle_buffer_size: int = 8192,
    chunk_line_buffer: int = 4096,
) -> Iterator[np.ndarray]:
    """Yield shuffled batches from streamed token windows."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    effective_buffer = max(batch_size, shuffle_buffer_size)
    seed = int(jax.random.randint(rng_key, shape=(), minval=0, maxval=2**31 - 1))
    np_rng = np.random.default_rng(seed)

    shuffle_buffer: List[np.ndarray] = []
    sequence_iter = iter_token_sequences_from_text_file(
        file_path=file_path,
        tokenizer=tokenizer,
        seq_length=seq_length,
        stride=stride,
        chunk_line_buffer=chunk_line_buffer,
    )

    target_watermark = effective_buffer + batch_size

    for sequence in sequence_iter:
        shuffle_buffer.append(sequence)

        # Keep a high-watermark buffer for randomness, but avoid draining it in
        # a burst (which causes long apparent stalls while refilling).
        while len(shuffle_buffer) >= target_watermark:
            yield _pop_random_batch(shuffle_buffer, batch_size, np_rng)

    while len(shuffle_buffer) >= batch_size:
        yield _pop_random_batch(shuffle_buffer, batch_size, np_rng)


def make_train_step(
    model,
    optimizer,
    use_lstm=False,
    use_transformer=False,
    use_transformer_ldru=False,
    use_ldru_transformer=False,
    seq2seq=True,
    lambda_l2=0.0,
):
    """Create a JIT-compiled training step function with model and optimizer captured."""

    def _train_step(params, opt_state, rng_key, batch):
        """Internal training step with model and optimizer captured."""
        # Choose appropriate loss function based on model type
        if use_transformer:
            # Use dedicated transformer loss function for proper autoregressive LM
            if seq2seq:
                loss_fn = lambda p: transformer_seq2seq_loss(
                    p, model, rng_key, batch, lambda_l2
                )
            else:
                loss_fn = lambda p: next_token_loss(p, model, rng_key, batch)
        elif use_transformer_ldru:
            # Transformer+LDRU hybrid: use LDRU loss functions since LDRU provides causal modeling
            if seq2seq:
                loss_fn = lambda p: transformer_seq2seq_loss(
                    p, model, rng_key, batch, lambda_l2
                )
            else:
                loss_fn = lambda p: next_token_loss(p, model, rng_key, batch)
        elif use_ldru_transformer:
            # LDRU+Transformer hybrid: use transformer loss functions since transformer provides final output
            if seq2seq:
                loss_fn = lambda p: transformer_seq2seq_loss(
                    p, model, rng_key, batch, lambda_l2
                )
            else:
                loss_fn = lambda p: next_token_loss(p, model, rng_key, batch)
        elif use_lstm:
            if seq2seq:
                loss_fn = lambda p: lstm_next_token_loss(p, model, rng_key, batch)
            else:
                loss_fn = lambda p: lstm_last_position_loss(p, model, rng_key, batch)
        else:
            if seq2seq:
                loss_fn = lambda p: ldru_seq2seq_loss(
                    p, model, rng_key, batch, lambda_l2
                )
            else:
                loss_fn = lambda p: next_token_loss(p, model, rng_key, batch)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # Update parameters
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, loss, metrics

    return jax.jit(_train_step)


def make_grad_step(
    model,
    use_lstm=False,
    use_transformer=False,
    use_transformer_ldru=False,
    use_ldru_transformer=False,
    seq2seq=True,
    lambda_l2=0.0,
):
    """Create a JIT-compiled gradient step that returns grads without updating."""

    def _grad_step(params, rng_key, batch):
        if use_transformer:
            if seq2seq:
                loss_fn = lambda p: transformer_seq2seq_loss(
                    p, model, rng_key, batch, lambda_l2
                )
            else:
                loss_fn = lambda p: next_token_loss(p, model, rng_key, batch)
        elif use_transformer_ldru:
            if seq2seq:
                loss_fn = lambda p: transformer_seq2seq_loss(
                    p, model, rng_key, batch, lambda_l2
                )
            else:
                loss_fn = lambda p: next_token_loss(p, model, rng_key, batch)
        elif use_ldru_transformer:
            if seq2seq:
                loss_fn = lambda p: transformer_seq2seq_loss(
                    p, model, rng_key, batch, lambda_l2
                )
            else:
                loss_fn = lambda p: next_token_loss(p, model, rng_key, batch)
        elif use_lstm:
            if seq2seq:
                loss_fn = lambda p: lstm_next_token_loss(p, model, rng_key, batch)
            else:
                loss_fn = lambda p: lstm_last_position_loss(p, model, rng_key, batch)
        else:
            if seq2seq:
                loss_fn = lambda p: ldru_seq2seq_loss(
                    p, model, rng_key, batch, lambda_l2
                )
            else:
                loss_fn = lambda p: next_token_loss(p, model, rng_key, batch)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        return loss, metrics, grads

    return jax.jit(_grad_step)


def create_lstm_model(config: LDRUExperimenstConfig):
    """Create a simple LSTM model for comparison."""
    import haiku as hk

    def lstm_forward(token_ids):
        batch_size, seq_length = token_ids.shape

        # Embedding layer
        embeddings = hk.Embed(config.vocab_size, config.embedding_dim)(token_ids)

        # LSTM layers
        x = embeddings
        for layer_idx in range(config.num_layers):
            lstm_layer = hk.LSTM(config.embedding_dim, name=f"lstm_{layer_idx}")

            # Use static_unroll for better performance
            def lstm_step(x_t, state):
                return lstm_layer(x_t, state)

            initial_state = lstm_layer.initial_state(batch_size)
            x, _ = hk.static_unroll(lstm_step, x, initial_state, time_major=False)

        # Output projection to vocabulary
        logits = hk.Linear(config.vocab_size)(x)
        return logits

    return hk.transform(lstm_forward)


def create_transformer_model(config: LDRUExperimenstConfig):
    """Create a transformer decoder model for causal language modeling."""
    import haiku as hk

    if config.use_alibi:
        print("Using ALiBi positional encodings for transformer model.")
        pos_encs = pos_encs_lib.PositionalEncodings.ALIBI
    else:
        pos_encs = pos_encs_lib.PositionalEncodings.SIN_COS

    # Create transformer config with causal masking enabled
    transformer_config = TransformerConfig(
        output_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_transformer_layers,
        num_heads=config.num_transformer_heads,
        dropout_prob=config.dropout_prob,
        emb_init_scale=config.emb_init_scale,
        use_embeddings=(
            False if config.tie_embeddings_transformer else config.use_embeddings
        ),
        share_embeddings=config.share_embeddings,
        chunk_size=config.chunk_size,
        positional_encodings=pos_encs,
        positional_encodings_params=(
            pos_encs_lib.SinCosParams(max_time=config.max_sequence_length)
            if not config.use_alibi
            else None
        ),
        widening_factor=config.widening_factor,
        causal_masking=config.causal_masking,
        share_weight=False,
        pre_norm_gelu_block=config.transformer_prenorm_gelu_block,
    )
    # Print config
    print("Transformer Config:")
    print(transformer_config)

    def transformer_forward(token_ids):
        transformer_encoder = TransformerEncoder(transformer_config)
        if config.tie_embeddings_transformer:
            emb_init = hk.initializers.TruncatedNormal(stddev=config.emb_init_scale)
            token_embedding = hk.get_parameter(
                "transformer_tied_token_embedding",
                shape=(config.vocab_size, config.embedding_dim),
                init=emb_init,
            )
            embedded_inputs = jnp.take(token_embedding, token_ids, axis=0)
            # Match TransformerEncoder embedding path scale when use_embeddings=True.
            embedded_inputs = embedded_inputs * jnp.sqrt(config.embedding_dim)
            encoded_output = transformer_encoder(embedded_inputs)  # [B, L, emb]
        else:
            # Convert token_ids to one-hot encoding as expected by the transformer
            one_hot_inputs = jax.nn.one_hot(token_ids, config.vocab_size)
            # Use transformer encoder in decoder-only mode with causal masking.
            encoded_output = transformer_encoder(one_hot_inputs)  # [B, L, emb]

        # Project to vocabulary size for next-token prediction
        # Return all positions for sequence-to-sequence loss
        if config.tie_embeddings_transformer:
            logits = jnp.einsum("ble,ve->blv", encoded_output, token_embedding)
        else:
            output_projection = hk.Linear(config.vocab_size)
            logits = output_projection(encoded_output)  # [B, L, vocab_size]

        return logits

    return hk.transform(transformer_forward)


def create_ldru_transformer_model(config: LDRUExperimenstConfig):
    """Create a hybrid model: LDRU encoder -> transformer decoder for causal language modeling."""
    import haiku as hk

    def ldru_transformer_forward(token_ids):
        # First pass through LDRU encoder
        from causal_ldru_v2 import CausalLDRUEncoder

        # Create LDRU config matching our embedding dimension
        ldru_config = CausalLDRUConfig(
            embedding_dim=config.embedding_dim,
            vocab_size=config.vocab_size,
            num_layers=1,  # Single LDRU layer after transformer
            hidden_dim=config.hidden_dim,
            widening_factor=config.widening_factor,
            dropout_prob=config.dropout_prob,
            causal_masking=True,  # LDRU provides causal modeling
            max_sequence_length=config.max_sequence_length,
            use_positional_encoding=False,  # Transformer already added positional encodings
            operator=config.operator,
            binop_expansion_factor=config.binop_expansion_factor,
            ablation_expansion_mode=config.ablation_expansion_mode,
            ablation_combine_mode=config.ablation_combine_mode,
            scan_method=config.scan_method,
            expand_to_power_of_2=config.expand_to_power_of_2,
            attention_per_scan_step=config.attention_per_scan_step,
            prenorm_gelu_block=config.ldru_prenorm_gelu_block,
            tie_embeddings=False,
        )

        ldru_encoder = CausalLDRUEncoder(ldru_config)
        embedded = hk.Embed(config.vocab_size, config.embedding_dim)(token_ids)
        ldru_output = ldru_encoder(embedded)  # [B, L, embedding_dim]

        # Now pass through transformer decoder for causal modeling
        transformer_config = TransformerConfig(
            output_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            num_layers=5,
            num_heads=8,
            dropout_prob=config.dropout_prob,
            emb_init_scale=0.02,
            use_embeddings=False,  # No embeddings since LDRU output is used
            share_embeddings=False,
            chunk_size=None,  # Use full attention
            positional_encodings=None,  # No positional encodings here, we expect the LDRU to have captured the positional info
            positional_encodings_params=pos_encs_lib.SinCosParams(
                max_time=config.max_sequence_length
            ),
            widening_factor=4,
            causal_masking=True,  # Critical for causal language modeling
            share_weight=False,
            pre_norm_gelu_block=config.transformer_prenorm_gelu_block,
        )

        transformer_encoder = TransformerEncoder(transformer_config)
        transformer_output = transformer_encoder(ldru_output)  # [B, L, embedding_dim]

        # Project to vocabulary size for next-token prediction
        output_projection = hk.Linear(config.vocab_size)
        logits = output_projection(transformer_output)  # [B, L, vocab_size]

        return logits

    return hk.transform(ldru_transformer_forward)


def create_transformer_ldru_model(config: LDRUExperimenstConfig):
    """Create a hybrid model: transformer encoder -> LDRU for causal language modeling."""
    import haiku as hk

    def transformer_ldru_forward(token_ids):
        # Create transformer config for the encoder (with causal masking for language modeling)
        transformer_config = TransformerConfig(
            output_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            num_layers=4,  # Fewer layers since we have LDRU after
            num_heads=8,  # Fewer heads to balance computation
            dropout_prob=config.dropout_prob,
            emb_init_scale=0.02,
            use_embeddings=True,
            share_embeddings=False,
            chunk_size=None,  # Use full attention
            positional_encodings=pos_encs_lib.PositionalEncodings.SIN_COS,
            positional_encodings_params=pos_encs_lib.SinCosParams(
                max_time=config.max_sequence_length
            ),
            widening_factor=4,  # Reduced widening factor
            causal_masking=True,  # Enable causal masking for proper language modeling
            share_weight=False,
            pre_norm_gelu_block=config.transformer_prenorm_gelu_block,
        )

        # Convert token_ids to one-hot encoding as expected by the transformer
        one_hot_inputs = jax.nn.one_hot(token_ids, config.vocab_size)

        # First pass through transformer encoder (with causal masking)
        transformer_encoder = TransformerEncoder(transformer_config)
        encoded_output = transformer_encoder(one_hot_inputs)  # [B, L, embedding_dim]

        # Apply dropout to encoder output
        encoded_output = hk.dropout(
            hk.next_rng_key(), config.dropout_prob, encoded_output
        )

        # Now pass through LDRU encoder for causal modeling
        # Import LDRU components
        from causal_ldru_v2 import CausalLDRUEncoder

        # Create LDRU config matching our embedding dimension
        ldru_config = CausalLDRUConfig(
            embedding_dim=config.embedding_dim,
            vocab_size=config.vocab_size,
            num_layers=1,  # Single LDRU layer after transformer
            hidden_dim=config.hidden_dim,
            widening_factor=config.widening_factor,
            dropout_prob=config.dropout_prob,
            causal_masking=True,  # LDRU provides causal modeling
            max_sequence_length=config.max_sequence_length,
            use_positional_encoding=False,  # Transformer already added positional encodings
            operator=config.operator,
            binop_expansion_factor=config.binop_expansion_factor,
            ablation_expansion_mode=config.ablation_expansion_mode,
            ablation_combine_mode=config.ablation_combine_mode,
            scan_method=config.scan_method,
            expand_to_power_of_2=config.expand_to_power_of_2,
            attention_per_scan_step=config.attention_per_scan_step,
            prenorm_gelu_block=config.ldru_prenorm_gelu_block,
            tie_embeddings=False,
        )

        # Pass encoded transformer output through LDRU encoder
        # The encoder does not have an embedding layer since we are treating transformer output as embeddings
        ldru_encoder = CausalLDRUEncoder(ldru_config)
        ldru_output = ldru_encoder(encoded_output)  # [B, L, embedding_dim]

        # Final projection to vocabulary size for next-token prediction
        output_projection = hk.Linear(config.vocab_size)
        logits = output_projection(ldru_output)  # [B, L, vocab_size]

        return logits

    return hk.transform(transformer_ldru_forward)


def prepare_tokenizer(
    tokenizer_type: TokenizerType,
    text_file_path: str,
    tokenizer_path: str,
    max_vocab_size: int,
    seq_length: int,
    model_name: str,
    tokenizer_folder: str = DEFAULT_TOKENIZER_FOLDER,
    dataset_name: str = "ptb",
):
    # verify tokenizer folder
    if not os.path.exists(tokenizer_folder):
        os.makedirs(tokenizer_folder)

    if tokenizer_type == TokenizerType.TEXT and tokenizer_path is None:
        print("Using TextTokenizer for word-level tokenization.")
        tokenizer = TextTokenizer(text_file_path, vocab_size=max_vocab_size)
    elif tokenizer_type == TokenizerType.TIKTOKEN_GPT2:
        encoding_name = tokenizer_path if tokenizer_path else "gpt2"
        print(f"Using tiktoken tokenizer with encoding: {encoding_name}")
        tokenizer = TiktokenGPT2Tokenizer(encoding_name=encoding_name)
    else:
        print("Using SPTokenizer for subword tokenization.")
        model_prefix = f"{tokenizer_folder}/{dataset_name}_tokenizer_vocab{max_vocab_size}_seq{seq_length}_for_{model_name}"
        print(f"Tokenizer model will be saved to: {model_prefix}.model")
        # No cleaning for text is necessary
        tokenizer = SPTokenizer(
            text_file_path=text_file_path,
            vocab_size=max_vocab_size,
            model_prefix=model_prefix,
            model_path=tokenizer_path,
        )

    return tokenizer


def create_datasets_for_training(
    training_text_file_path: str,
    validation_text_file_path: str = None,
    test_text_file_path: str = None,
    tokenizer: BaseTokenizer = None,
    seq_length: int = 128,
    stride: Optional[int] = None,
):
    train_data, val_data, test_data = None, None, None
    if stride is None:
        stride = max(1, seq_length // 2)
    if stride <= 0:
        raise ValueError("stride must be > 0")
    # Create dataset
    print("Creating training dataset from text file...")

    train_data, val_data = create_dataset_from_text_file(
        training_text_file_path,
        seq_length=seq_length,
        stride=stride,
        train_split=1.0 if validation_text_file_path else 0.9,
        tokenizer=tokenizer,
    )

    # Show sample text
    sample_sequence = train_data[0]
    sample_text = tokenizer.decode(sample_sequence.tolist())
    print(f"\\nSample sequence: '{sample_text[:100]}...'")

    if validation_text_file_path:
        print("Creating validation dataset from text file...")
        val_data, _ = create_dataset_from_text_file(
            validation_text_file_path,
            seq_length=seq_length,
            stride=stride,
            train_split=1.0,  # Use all data for validation
            tokenizer=tokenizer,  # Use same tokenizer to ensure consistent vocab
        )
        print(f"Validation sequences: {len(val_data):,}")

    if test_text_file_path:
        print("Creating test dataset from text file...")
        test_data, _ = create_dataset_from_text_file(
            test_text_file_path,
            seq_length=seq_length,
            stride=stride,
            train_split=1.0,  # Use all data for testing
            tokenizer=tokenizer,  # Use same tokenizer to ensure consistent vocab
        )
        print(f"Test sequences: {len(test_data):,}")

    return train_data, val_data, test_data


def make_eval_step(
    model,
    use_lstm=False,
    use_transformer=False,
    use_transformer_ldru=False,
    use_ldru_transformer=False,
    seq2seq=True,
):
    if use_transformer or use_transformer_ldru or use_ldru_transformer:
        loss_fn = transformer_seq2seq_loss if seq2seq else next_token_loss
    elif use_lstm:
        loss_fn = lstm_next_token_loss if seq2seq else lstm_last_position_loss
    else:
        loss_fn = ldru_seq2seq_loss if seq2seq else next_token_loss

    def _eval_step(params, key, batch):
        return loss_fn(params, model, key, batch)

    return jax.jit(_eval_step)


def train_model(
    log_dir: str,
    config: LDRUExperimenstConfig,
    rng_seed: int = 42,
    enable_logging: bool = True,
    text_file_path: str = None,
    validation_text_file_path: str = None,
    model_creation_fn=create_causal_ldru_model,
    use_lstm=False,
    use_transformer=False,
    use_transformer_ldru=False,  # New parameter for hybrid model
    use_ldru_transformer=False,  # New parameter for hybrid model
    seq2seq=True,  # Use seq2seq loss for LDRU by default
    save_checkpoints: bool = True,
    checkpoint_dir: str = "checkpoints",
    resume_from_checkpoint: str = None,
    tokenizer_path: str = None,  # Path to save/load tokenizer model
    test_text_file_path: str = None,  # Path to test text file
    tokenizer_type: TokenizerType = TokenizerType.SENTENCEPIECE,  # Type of tokenizer to use
    model_prefix: str = "",  # Prefix for model name in checkpoints and logs
    generate_samples: bool = True,  # Whether to generate sample text after training
    streaming_train: bool = True,
    streaming_shuffle_buffer_size: int = 8192,
    streaming_chunk_line_buffer: int = 4096,
    streaming_exact_sequence_estimate: bool = False,
    streaming_estimate_bytes_per_token: float = 4.0,
    train_seq_bin_path: str = None,
    val_seq_bin_path: str = None,
    test_seq_bin_path: str = None,
    seq_bin_dtype: str = "uint16",
    seq_bin_length: Optional[int] = None,
    seq_bin_format: str = "auto",
    seq_meta_json: str = None,
    optimizer_name: str = "adamw",
    target_tokens: Optional[int] = None,
    train_stride: Optional[int] = None,
    nanogpt_batching: bool = False,
    nanogpt_ppl_metric: bool = False,
    warmup_steps: int = 0,
    train_steps_per_epoch: Optional[int] = None,
    validation_steps_per_epoch: Optional[int] = None,
    test_steps_per_epoch: Optional[int] = None,
    compute_dtype: str = ComputeDType.FLOAT32.value,
):
    """Main training function."""

    # Training hyperparameters - adjusted for LDRU's single-output nature
    learning_rate = config.initial_learning_rate
    batch_size = config.batch_size  # Larger batch size for more stable gradients
    num_epochs = config.num_epochs
    min_learning_rate = config.min_learning_rate  # Minimum learning rate threshold
    max_vocab_size = config.vocab_size
    seq_length = config.seq_length

    model_type = (
        "lstm"
        if use_lstm
        else (
            "transformer"
            if use_transformer
            else (
                "transformer_ldru"
                if use_transformer_ldru
                else ("ldru_transformer" if use_ldru_transformer else "ldru")
            )
        )
    )
    loss_type = "seq2seq" if seq2seq else "lastpos"
    scan_type = config.scan_method
    if config.blelloch_random:
        scan_type = f"{scan_type}_blelloch_random"
    tokenizer_suffix = (
        "TKGPT2"
        if tokenizer_type == TokenizerType.TIKTOKEN_GPT2
        else "TXT" if tokenizer_type == TokenizerType.TEXT else "SP"
    )
    model_name = f"{model_prefix}_model_{model_type}_{loss_type}_silu_{scan_type}_{seq_length}_{tokenizer_suffix}"
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}")
    if enable_logging:
        if not os.path.exists(f"{log_dir}/{model_name}"):
            os.makedirs(f"{log_dir}/{model_name}")
        writer = SummaryWriter(logdir=f"{log_dir}/{model_name}")

    # Initialize RNG
    rng_key = jax.random.PRNGKey(rng_seed)
    rng_key, init_key, data_key = jax.random.split(rng_key, 3)

    use_pretokenized_bins = train_seq_bin_path is not None
    if not use_pretokenized_bins:
        assert (
            text_file_path is not None
        ), "Please provide a path to the training text file."

    configure_mixed_precision(compute_dtype)
    print(f"Compute dtype: {compute_dtype} (params/output kept in float32)")

    tokenizer = None
    if use_pretokenized_bins:
        print("Using pretokenized binary files for training data.")
        selected_seq_bin_format = seq_bin_format
        if seq_meta_json:
            if not os.path.exists(seq_meta_json):
                raise FileNotFoundError(f"--seq_meta_json not found: {seq_meta_json}")
            with open(seq_meta_json, "r", encoding="utf-8") as f:
                seq_meta = json.load(f)
            meta_format = seq_meta.get("format")
            if selected_seq_bin_format == "auto" and meta_format in (
                "sequence",
                "token_stream",
            ):
                selected_seq_bin_format = meta_format
            meta_seq_len = seq_meta.get("sequence_config", {}).get("seq_length")
            if (
                selected_seq_bin_format in ("auto", "sequence")
                and seq_bin_length is None
                and meta_seq_len is not None
            ):
                seq_bin_length = int(meta_seq_len)
            meta_tok = seq_meta.get("tokenizer", {})
            if tokenizer_path is None and meta_tok.get("name_or_path"):
                tokenizer_path = meta_tok.get("name_or_path")
            if (
                tokenizer_type == TokenizerType.SENTENCEPIECE
                and meta_tok.get("type") == "tiktoken_gpt2"
            ):
                tokenizer_type = TokenizerType.TIKTOKEN_GPT2
            meta_vocab = meta_tok.get("vocab_size")
            if meta_vocab is not None:
                config.vocab_size = int(meta_vocab)

        if selected_seq_bin_format == "auto":
            selected_seq_bin_format = "sequence"
        if selected_seq_bin_format not in ("sequence", "token_stream"):
            raise ValueError(
                f"Unsupported --seq_bin_format '{selected_seq_bin_format}'. "
                "Expected one of: auto, sequence, token_stream."
            )

        if selected_seq_bin_format == "sequence":
            seq_bin_length = seq_bin_length if seq_bin_length is not None else seq_length
            if seq_bin_length != seq_length:
                raise ValueError(
                    f"--seq_bin_length ({seq_bin_length}) must match training seq_length "
                    f"({seq_length})."
                )
        elif seq_bin_length is not None:
            print(
                "Ignoring --seq_bin_length because --seq_bin_format token_stream "
                "builds windows at runtime."
            )

        if tokenizer_type == TokenizerType.TIKTOKEN_GPT2:
            encoding_name = tokenizer_path if tokenizer_path else "gpt2"
            tokenizer = TiktokenGPT2Tokenizer(encoding_name=encoding_name)
        elif tokenizer_type == TokenizerType.SENTENCEPIECE and tokenizer_path:
            tokenizer = SPTokenizer(model_path=tokenizer_path)
        elif tokenizer_type == TokenizerType.TEXT:
            tokenizer = None

        if tokenizer is not None:
            vocab_size = tokenizer.get_piece_size()
            print(f"Tokenizer vocab size: {vocab_size}")
            config.vocab_size = vocab_size
    else:
        print("Creating dataset from text file...")
        # Prepare tokenizer
        tokenizer = prepare_tokenizer(
            tokenizer_type,
            text_file_path,
            tokenizer_path,
            max_vocab_size,
            seq_length,
            model_name,
            tokenizer_folder=DEFAULT_TOKENIZER_FOLDER,
            dataset_name=text_file_path.split("/")[-1].split(".")[
                0
            ],  # Use filename as dataset name
        )

        vocab_size = tokenizer.get_piece_size()
        print(f"Tokenizer vocab size: {vocab_size}")
        config.vocab_size = vocab_size

    train_data, val_data, test_data = None, None, None
    train_stride = max(1, seq_length // 2) if train_stride is None else train_stride
    if train_stride <= 0:
        raise ValueError("--train_stride must be > 0 when provided.")
    print(f"Training stride: {train_stride}")

    if use_pretokenized_bins and streaming_train:
        print(
            "Pretokenized binary mode does not use text streaming. "
            "Ignoring streaming_train and using indexed batch sampling."
        )
        streaming_train = False

    if nanogpt_batching and not (
        use_pretokenized_bins and selected_seq_bin_format == "token_stream"
    ):
        raise ValueError(
            "--nanogpt_batching currently requires token-stream pretokenized input "
            "(--train_seq_bin with --seq_bin_format token_stream)."
        )

    if streaming_train and validation_text_file_path is None:
        print(
            "Streaming training needs an explicit validation file. Falling back to in-memory train split."
        )
        streaming_train = False

    if use_pretokenized_bins:
        if selected_seq_bin_format == "token_stream":
            train_tokens = load_pretokenized_token_stream(
                train_seq_bin_path, dtype_name=seq_bin_dtype
            )
            if nanogpt_batching:
                train_data = train_tokens
                train_sequence_count = max(1, len(train_tokens) - seq_length)
                print(
                    f"Loaded train token stream: {len(train_tokens):,} tokens "
                    f"(nanoGPT-style random-offset batching, len={seq_length}) "
                    f"from {train_seq_bin_path}"
                )
            else:
                train_data = token_stream_to_sequence_view(
                    train_tokens, seq_length=seq_length, stride=train_stride
                )
                train_sequence_count = len(train_data)
                print(
                    f"Loaded train token stream: {len(train_tokens):,} tokens "
                    f"-> {len(train_data):,} windows (len={seq_length}, stride={train_stride}) "
                    f"from {train_seq_bin_path}"
                )
        else:
            train_data = load_pretokenized_sequences(
                train_seq_bin_path, seq_length=seq_length, dtype_name=seq_bin_dtype
            )
            train_sequence_count = len(train_data)
            print(f"Loaded train sequences: {len(train_data):,} from {train_seq_bin_path}")

        if val_seq_bin_path:
            if selected_seq_bin_format == "token_stream":
                val_tokens = load_pretokenized_token_stream(
                    val_seq_bin_path, dtype_name=seq_bin_dtype
                )
                if nanogpt_batching:
                    val_data = val_tokens
                    print(
                        f"Loaded val token stream: {len(val_tokens):,} tokens "
                        f"(nanoGPT-style random-offset batching) from {val_seq_bin_path}"
                    )
                else:
                    val_data = token_stream_to_sequence_view(
                        val_tokens, seq_length=seq_length, stride=train_stride
                    )
                    print(
                        f"Loaded val token stream: {len(val_tokens):,} tokens "
                        f"-> {len(val_data):,} windows from {val_seq_bin_path}"
                    )
            else:
                val_data = load_pretokenized_sequences(
                    val_seq_bin_path, seq_length=seq_length, dtype_name=seq_bin_dtype
                )
                print(f"Loaded val sequences: {len(val_data):,} from {val_seq_bin_path}")

        if test_seq_bin_path:
            if selected_seq_bin_format == "token_stream":
                test_tokens = load_pretokenized_token_stream(
                    test_seq_bin_path, dtype_name=seq_bin_dtype
                )
                if nanogpt_batching:
                    test_data = test_tokens
                    print(
                        f"Loaded test token stream: {len(test_tokens):,} tokens "
                        f"(nanoGPT-style random-offset batching) from {test_seq_bin_path}"
                    )
                else:
                    test_data = token_stream_to_sequence_view(
                        test_tokens, seq_length=seq_length, stride=train_stride
                    )
                    print(
                        f"Loaded test token stream: {len(test_tokens):,} tokens "
                        f"-> {len(test_data):,} windows from {test_seq_bin_path}"
                    )
            else:
                test_data = load_pretokenized_sequences(
                    test_seq_bin_path, seq_length=seq_length, dtype_name=seq_bin_dtype
                )
                print(f"Loaded test sequences: {len(test_data):,} from {test_seq_bin_path}")

        if nanogpt_batching:
            preview_loader = create_nanogpt_token_stream_loader(
                train_data,
                seq_length=seq_length,
                batch_size=batch_size,
                rng_key=data_key,
                num_batches=1,
            )
            preview_batch = next(preview_loader, None)
        else:
            preview_batch = np.asarray(train_data[:batch_size])
        if preview_batch.shape[0] == 0:
            raise ValueError(
                "No training batches could be created from pretokenized binaries."
            )
        if tokenizer is not None:
            sample_text = tokenizer.decode(preview_batch[0].tolist())
            print(f"\nSample sequence: '{sample_text[:100]}...'")

    elif streaming_train:
        print("Using streaming training dataset loader (low-memory mode).")
        if streaming_exact_sequence_estimate:
            train_sequence_count = estimate_num_sequences_from_text_file(
                file_path=text_file_path,
                tokenizer=tokenizer,
                seq_length=seq_length,
                stride=train_stride,
                chunk_line_buffer=streaming_chunk_line_buffer,
            )
            print(f"Estimated training sequences (exact pre-scan): {train_sequence_count:,}")
        else:
            train_sequence_count = estimate_num_sequences_from_file_size(
                file_path=text_file_path,
                seq_length=seq_length,
                stride=train_stride,
                bytes_per_token=streaming_estimate_bytes_per_token,
            )
            print(
                "Estimated training sequences (fast, size-based): "
                f"{train_sequence_count:,} "
                f"(bytes_per_token={streaming_estimate_bytes_per_token:.2f})"
            )

        preview_loader = create_streaming_data_loader(
            file_path=text_file_path,
            tokenizer=tokenizer,
            seq_length=seq_length,
            stride=train_stride,
            batch_size=batch_size,
            rng_key=data_key,
            shuffle_buffer_size=streaming_shuffle_buffer_size,
            chunk_line_buffer=streaming_chunk_line_buffer,
        )
        preview_batch = next(preview_loader, None)
        if preview_batch is None:
            raise ValueError(
                "No training batches could be created. Check seq_length/stride and input file size."
            )
        sample_text = tokenizer.decode(preview_batch[0].tolist())
        print(f"\nSample sequence: '{sample_text[:100]}...'")

        print("Creating validation dataset from text file...")
        val_data, _ = create_dataset_from_text_file(
            validation_text_file_path,
            seq_length=seq_length,
            stride=train_stride,
            train_split=1.0,
            tokenizer=tokenizer,
        )
        print(f"Validation sequences: {len(val_data):,}")

        if test_text_file_path:
            print("Creating test dataset from text file...")
            test_data, _ = create_dataset_from_text_file(
                test_text_file_path,
                seq_length=seq_length,
                stride=train_stride,
                train_split=1.0,
                tokenizer=tokenizer,
            )
            print(f"Test sequences: {len(test_data):,}")
    else:
        train_data, val_data, test_data = create_datasets_for_training(
            text_file_path,
            validation_text_file_path,
            test_text_file_path,
            tokenizer,
            seq_length,
            train_stride,
        )
        train_sequence_count = len(train_data)
        preview_batch = np.asarray(train_data[:batch_size])

    # Create model
    model = model_creation_fn(config)

    # Create evaluation model with dropout disabled
    eval_model = create_evaluation_model(config, model_creation_fn)
    print("Created evaluation model with dropout disabled")

    # Initialize model parameters
    print("Initializing model...")
    if model_creation_fn == create_transformer_model:
        # Transformer expects token IDs directly
        dummy_batch = jnp.asarray(
            preview_batch[:1]
        )  # [B, L-1] for next token prediction
    elif use_lstm:
        dummy_batch = jnp.asarray(
            preview_batch[:1, :-1]
        )  # LSTM needs input without last token
    else:
        dummy_batch = jnp.asarray(preview_batch[:1])  # LDRU uses full sequence
    params = model.init(init_key, dummy_batch)

    # Initialize optimizer with learning rate scheduling
    natural_train_steps_per_epoch = max(1, train_sequence_count // batch_size)
    requested_train_steps_per_epoch = train_steps_per_epoch
    if requested_train_steps_per_epoch is not None:
        if requested_train_steps_per_epoch <= 0:
            raise ValueError("--train_steps_per_epoch must be > 0 when provided.")
        train_steps_per_epoch = requested_train_steps_per_epoch
        print(
            "Using configured train steps per epoch: "
            f"{train_steps_per_epoch:,} "
            f"(natural estimate: {natural_train_steps_per_epoch:,})"
        )
    else:
        train_steps_per_epoch = natural_train_steps_per_epoch
    if validation_steps_per_epoch is not None and validation_steps_per_epoch <= 0:
        raise ValueError("--validation_steps_per_epoch must be > 0 when provided.")
    if test_steps_per_epoch is not None and test_steps_per_epoch <= 0:
        raise ValueError("--test_steps_per_epoch must be > 0 when provided.")
    if warmup_steps < 0:
        raise ValueError("--warmup_steps must be >= 0.")
    tokens_per_step = int(batch_size * seq_length)
    target_steps = None
    if target_tokens is not None:
        if target_tokens <= 0:
            raise ValueError("--target_tokens must be > 0 when provided.")
        target_steps = max(1, math.ceil(target_tokens / max(1, tokens_per_step)))
        print(
            "Token-budget mode enabled: "
            f"target_tokens={target_tokens:,}, "
            f"tokens_per_step={tokens_per_step:,}, "
            f"target_steps={target_steps:,}"
        )
    decay_steps = (
        max(1, target_steps)
        if target_steps is not None
        else max(1, num_epochs * train_steps_per_epoch)
    )
    warmup_steps_effective = min(int(warmup_steps), max(0, decay_steps - 1))
    if warmup_steps_effective > 0:
        warmup_schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=config.initial_learning_rate,
            transition_steps=warmup_steps_effective,
        )
        cosine_schedule = optax.schedules.cosine_decay_schedule(
            init_value=config.initial_learning_rate,
            decay_steps=max(1, decay_steps - warmup_steps_effective),
            alpha=min_learning_rate / config.initial_learning_rate,
        )
        learning_rate_schedule = optax.join_schedules(
            schedules=[warmup_schedule, cosine_schedule],
            boundaries=[warmup_steps_effective],
        )
    else:
        learning_rate_schedule = optax.schedules.cosine_decay_schedule(
            init_value=config.initial_learning_rate,
            decay_steps=decay_steps,
            alpha=min_learning_rate / config.initial_learning_rate,
        )
    print(
        "LR schedule: "
        f"decay_steps={decay_steps:,}, "
        f"warmup_steps={warmup_steps_effective:,}"
    )
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adamw":
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=learning_rate_schedule),
        )
    elif optimizer_name == "amsgrad":
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.amsgrad(
                learning_rate=learning_rate_schedule
            ),  # Use AMSGrad for better convergence
        )
    elif optimizer_name == "muon":
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax_muon(learning_rate=learning_rate_schedule),
        )
    else:
        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}'. Expected one of: adamw, amsgrad, muon."
        )
    opt_state = optimizer.init(params)

    # Learning rate tracking variables
    current_learning_rate = learning_rate
    epochs_without_improvement = 0
    best_validation_metric = float("inf")  # Track best validation loss or perplexity

    # Checkpoint handling
    start_epoch = 0
    best_val_perplexity = float("inf")
    training_metrics_history = []

    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        (
            loaded_params,
            loaded_optimizer_state,
            config,
            loaded_step,
            loaded_best_ppl,
            loaded_tokenizer,
        ) = load_checkpoint(resume_from_checkpoint)

        # Update training state
        params = loaded_params
        if loaded_optimizer_state is not None:
            opt_state = loaded_optimizer_state
        start_epoch = loaded_step + 1 if loaded_step is not None else 0
        best_val_perplexity = loaded_best_ppl

        # Use loaded tokenizer if we successfully loaded one from checkpoint
        if loaded_tokenizer is not None:
            tokenizer = loaded_tokenizer

        print(
            f"Resumed from epoch {start_epoch} with best validation perplexity: {best_val_perplexity:.4f}"
        )

    # Create checkpoint directory
    if save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {checkpoint_dir}")

    # Create JIT-compiled training step function
    compiled_train_step = make_train_step(
        model,
        optimizer,
        use_lstm=use_lstm,
        use_transformer=use_transformer,
        use_transformer_ldru=use_transformer_ldru,
        use_ldru_transformer=use_ldru_transformer,
        seq2seq=seq2seq,
        lambda_l2=config.l2_lambda,
    )

    compiled_eval_step = make_eval_step(
        eval_model,
        use_lstm=use_lstm,
        use_transformer=use_transformer,
        use_transformer_ldru=use_transformer_ldru,
        use_ldru_transformer=use_ldru_transformer,
        seq2seq=seq2seq,
    )

    print("Warming up JIT compilation...")

    dummy_batch = jnp.asarray(preview_batch)

    # warmup train
    params, opt_state, _, _ = compiled_train_step(
        params, opt_state, rng_key, dummy_batch
    )

    # warmup eval
    _ = compiled_eval_step(params, rng_key, dummy_batch)

    # warmup generation
    rng_key, gen_key = jax.random.split(rng_key)
    _ = test_generation(
        model,
        params,
        config,
        gen_key,
        tokenizer,
        max_length=8,  # small for compile
        verbose=False,
        seq2seq=seq2seq,
        eval_model=eval_model,
    )

    print("Warmup complete.")

    # Count parameters
    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"Model has {param_count:,} parameters")
    print(f"Optimizer: {optimizer_name}")
    print(f"Initial learning rate: {current_learning_rate:.2e}")

    # Training loop
    estimated_total_epochs = (
        max(1, math.ceil(target_steps / train_steps_per_epoch))
        if target_steps is not None
        else num_epochs
    )
    if target_steps is not None:
        estimated_remaining_epochs = max(0, estimated_total_epochs - start_epoch)
        print(
            "Estimated epochs to reach token budget: "
            f"total={estimated_total_epochs}, "
            f"remaining_from_current={estimated_remaining_epochs}"
        )
    epoch_print_total = estimated_total_epochs if target_steps is not None else num_epochs
    epoch_loop_limit = max(num_epochs, estimated_total_epochs)

    print("Starting training...")
    global_step = max(0, start_epoch * train_steps_per_epoch)
    stop_for_token_budget = False

    for epoch in range(start_epoch, epoch_loop_limit):
        if target_steps is not None and global_step >= target_steps:
            stop_for_token_budget = True
            break
        print(f"\nEpoch {epoch + 1}/{epoch_print_total}")

        # Create data loader for this epoch
        rng_key, epoch_key = jax.random.split(rng_key)
        if streaming_train:
            data_loader = create_streaming_data_loader(
                file_path=text_file_path,
                tokenizer=tokenizer,
                seq_length=seq_length,
                stride=train_stride,
                batch_size=batch_size,
                rng_key=epoch_key,
                shuffle_buffer_size=streaming_shuffle_buffer_size,
                chunk_line_buffer=streaming_chunk_line_buffer,
            )
        elif nanogpt_batching:
            data_loader = create_nanogpt_token_stream_loader(
                train_data,
                seq_length=seq_length,
                batch_size=batch_size,
                rng_key=epoch_key,
                num_batches=train_steps_per_epoch,
            )
        else:
            data_loader = create_data_loader(train_data, batch_size, epoch_key)

        epoch_losses = []
        epoch_accuracies = []
        epoch_perplexities = []
        new_best = False
        epoch_step_count = 0

        # Training batches
        pbar = tqdm.tqdm(data_loader, desc="Training", position=0, leave=True)
        for batch_np in pbar:
            rng_key, step_key = jax.random.split(rng_key)
            batch = jnp.asarray(batch_np)

            # Training step
            params, opt_state, loss, metrics = compiled_train_step(
                params, opt_state, step_key, batch
            )
            global_step += 1
            epoch_step_count += 1

            epoch_losses.append(float(loss))
            epoch_accuracies.append(float(metrics["accuracy"]))
            epoch_perplexities.append(float(metrics["perplexity"]))

            # Update progress bar with current metrics
            pbar.set_postfix(
                {
                    "Loss": f"{np.mean(epoch_losses):.4f}",
                    "Acc": f"{np.mean(epoch_accuracies):.4f}",
                    "PPL": f"{np.mean(epoch_perplexities):.1f}",
                    "LR": f"{current_learning_rate:.2e}",
                }
            )
            if target_steps is not None and global_step >= target_steps:
                stop_for_token_budget = True
                break
            if (
                requested_train_steps_per_epoch is not None
                and epoch_step_count >= requested_train_steps_per_epoch
            ):
                break

        if len(epoch_losses) == 0:
            raise ValueError(
                "No training batches were produced for this epoch. "
                "Check batch_size, dataset size, and data loader settings."
            )
        if (
            requested_train_steps_per_epoch is not None
            and epoch_step_count < requested_train_steps_per_epoch
            and not stop_for_token_budget
        ):
            print(
                "Warning: Data loader ended before configured steps-per-epoch were reached "
                f"({epoch_step_count:,}/{requested_train_steps_per_epoch:,})."
            )

        # Print epoch statistics
        avg_loss = np.mean(epoch_losses)
        avg_accuracy = np.mean(epoch_accuracies)
        avg_perplexity = np.mean(epoch_perplexities)

        print(f"Average loss: {avg_loss:.4f}")
        print(f"Average accuracy: {avg_accuracy:.4f}")
        print(f"Average perplexity: {avg_perplexity:.4f}")
        if nanogpt_ppl_metric:
            nanogpt_train_ppl = float(np.exp(avg_loss))
            print(f"Average perplexity (nanoGPT-style): {nanogpt_train_ppl:.4f}")
        print(f"Current learning rate: {current_learning_rate:.2e}")

        if enable_logging:
            writer.add_scalar("Loss/Train", avg_loss, epoch + 1)
            writer.add_scalar("Accuracy/Train", avg_accuracy, epoch + 1)
            writer.add_scalar("Perplexity/Train", avg_perplexity, epoch + 1)

        # Validation if available
        if val_data is not None and len(val_data) > 0:
            val_loss, val_metrics = evaluate_model_on_dataset(
                params,
                model,
                rng_key,
                val_data,
                epoch,
                batch_size,
                eval_model,  # Use dropout-free model for evaluation
                use_lstm=use_lstm,
                use_transformer=use_transformer,
                use_transformer_ldru=use_transformer_ldru,
                seq2seq=seq2seq,
                dataset_name="Validation",
                writer=writer if enable_logging else None,
                compiled_eval_step=compiled_eval_step,  # Pass the compiled evaluation step for efficiency
                max_eval_steps=validation_steps_per_epoch,
                nanogpt_batching=nanogpt_batching,
                seq_length=seq_length,
                nanogpt_ppl_metric=nanogpt_ppl_metric,
            )

            # Check for improvement and update learning rate if stagnant
            current_validation_metric = val_metrics[
                "perplexity"
            ]  # Use perplexity as primary metric

            if current_validation_metric < best_validation_metric:
                best_validation_metric = current_validation_metric
                epochs_without_improvement = 0
                print(f"✅ Validation improved! Reset patience counter.")
                new_best = True
            else:
                epochs_without_improvement += 1
                print(f"⚠️  No improvement for {epochs_without_improvement} epochs")

            # Save checkpoint if validation perplexity improved and saving is enabled
            if save_checkpoints and val_metrics["perplexity"] < best_val_perplexity:
                best_val_perplexity = val_metrics["perplexity"]

                # Create model type suffix for checkpoint name
                checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}")

                save_checkpoint(
                    checkpoint_dir=checkpoint_path,
                    step=epoch + 1,
                    params=params,
                    optimizer_state=opt_state,
                    config=config,
                    best_val_perplexity=best_val_perplexity,
                    metrics={
                        "train_loss": avg_loss,
                        "train_accuracy": avg_accuracy,
                        "train_perplexity": avg_perplexity,
                        "val_loss": val_loss,
                        "val_accuracy": val_metrics["accuracy"],
                        "val_perplexity": val_metrics["perplexity"],
                    },
                    tokenizer=tokenizer,  # Save tokenizer with checkpoint
                    save_best_only=True,  # Only save when we have a new best validation perplexity
                )
                print(f"New best validation perplexity: {best_val_perplexity:.4f}")
        else:
            # Without validation data, we do no checkpointing and just print training metrics
            print("No validation data provided, skipping validation and checkpointing.")

        # check if epoch has no improvement for a certain number of epochs and reduce learning rate if needed
        if target_steps is None and epochs_without_improvement >= 10:
            # terminate
            print(
                f"Early stopping triggered after {epochs_without_improvement} epochs without improvement."
            )
            break

        if test_data is not None and len(test_data) > 0 and new_best:
            print("\nEvaluating on test set...")
            _, _ = evaluate_model_on_dataset(
                params,
                model,
                rng_key,
                test_data,
                epoch,
                batch_size,
                eval_model,  # Use dropout-free model for evaluation
                use_lstm=use_lstm,
                use_transformer=use_transformer,
                use_transformer_ldru=use_transformer_ldru,
                seq2seq=seq2seq,
                dataset_name="Test",
                writer=writer if enable_logging else None,
                compiled_eval_step=compiled_eval_step,  # Pass the compiled evaluation step for efficiency
                max_eval_steps=test_steps_per_epoch,
                nanogpt_batching=nanogpt_batching,
                seq_length=seq_length,
                nanogpt_ppl_metric=nanogpt_ppl_metric,
            )

        # Generate text after each epoch to monitor progress
        print("\nSample generation:")
        if generate_samples is True:
            rng_key, gen_key = jax.random.split(rng_key)
            try:
                _ = test_generation(
                    model,
                    params,
                    config,
                    gen_key,
                    tokenizer,
                    max_length=32,
                    verbose=False,
                    seq2seq=seq2seq,
                    eval_model=eval_model,  # Use dropout-free model for generation
                )
                print()  # Add spacing after generation
            except Exception as e:
                print(f"Generation failed: {e}")
                print()

        if stop_for_token_budget:
            print(
                f"Reached target step budget ({global_step:,}/{target_steps:,} steps); "
                "stopping training."
            )
            break

    print("\nTraining completed")
    if target_steps is not None:
        actual_tokens_seen = int(global_step * tokens_per_step)
        print(
            "Token-budget summary: "
            f"steps={global_step:,}, "
            f"tokens_seen={actual_tokens_seen:,}, "
            f"target_tokens={target_tokens:,}"
        )

    # Save final checkpoint
    if save_checkpoints:
        # save to the same checkpoint path
        final_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}")

        save_checkpoint(
            checkpoint_dir=final_checkpoint_path,
            step=num_epochs,
            params=params,
            optimizer_state=opt_state,
            config=config,
            best_val_perplexity=best_val_perplexity,
            metrics={"final_training": True},
            save_best_only=False,  # Save final model regardless of validation performance
        )
        print(f"Saved final checkpoint: {final_checkpoint_path}")

        if best_val_perplexity < float("inf"):
            print(f"Best validation perplexity achieved: {best_val_perplexity:.4f}")

    return params, model, config, tokenizer, best_val_perplexity


def evaluate_model_on_dataset(
    params,
    model,
    rng_key,
    dataset,
    epoch,
    batch_size,
    eval_model=None,
    use_lstm=False,
    use_transformer=False,
    use_transformer_ldru=False,
    seq2seq=True,
    dataset_name="Validation",
    writer: SummaryWriter = None,
    compiled_eval_step=None,
    max_eval_steps: Optional[int] = None,
    nanogpt_batching: bool = False,
    seq_length: Optional[int] = None,
    nanogpt_ppl_metric: bool = False,
):
    rng_key, dataset_key = jax.random.split(rng_key)
    dataset_loss, dataset_metrics = evaluate_model(
        params,
        model,
        dataset_key,
        dataset,
        batch_size,
        use_lstm=use_lstm,
        use_transformer=use_transformer,
        use_transformer_ldru=use_transformer_ldru,
        seq2seq=seq2seq,
        eval_model=eval_model,  # Use dropout-free model for evaluation
        compiled_eval_step=compiled_eval_step,  # Pass the compiled evaluation step for efficiency
        max_eval_steps=max_eval_steps,
        nanogpt_batching=nanogpt_batching,
        seq_length=seq_length,
        nanogpt_ppl_metric=nanogpt_ppl_metric,
    )
    print(f"{dataset_name} loss: {dataset_loss:.4f}")
    print(f"{dataset_name} accuracy: {dataset_metrics['accuracy']:.4f}")
    print(f"{dataset_name} perplexity: {dataset_metrics['perplexity']:.4f}")
    if "nanogpt_perplexity" in dataset_metrics:
        print(
            f"{dataset_name} perplexity (nanoGPT-style): "
            f"{dataset_metrics['nanogpt_perplexity']:.4f}"
        )
    if "last_token_perplexity" in dataset_metrics:
        print(f"{dataset_name} last-token perplexity: {dataset_metrics['last_token_perplexity']:.4f}")
    if "last_token_perplexity_nanogpt" in dataset_metrics:
        print(
            f"{dataset_name} last-token perplexity (nanoGPT-style): "
            f"{dataset_metrics['last_token_perplexity_nanogpt']:.4f}"
        )

    # Report per-position perplexity for seq2seq models
    if seq2seq and "per_position_perplexity" in dataset_metrics:
        per_pos_ppl = dataset_metrics["per_position_perplexity"]

        # Create a nice formatted summary
        formatted_ppl = format_per_position_perplexity(per_pos_ppl)
        print(f"Per-position PPL: {formatted_ppl}")

        # Analyze trends
        trends = analyze_position_trends(per_pos_ppl)
        if trends["trend"] != "insufficient_data":
            print(f"Position trend: {trends['trend']} (R²={trends['r_squared']:.3f})")
            print(
                f"Best: pos {trends['best_position']} (PPL={trends['best_perplexity']:.2f}), "
                f"Worst: pos {trends['worst_position']} (PPL={trends['worst_perplexity']:.2f})"
            )

    if writer is not None:
        writer.add_scalar("Loss/{}".format(dataset_name), dataset_loss, epoch + 1)
        writer.add_scalar(
            "Accuracy/{}".format(dataset_name), dataset_metrics["accuracy"], epoch + 1
        )
        writer.add_scalar(
            "Perplexity/{}".format(dataset_name),
            dataset_metrics["perplexity"],
            epoch + 1,
        )
        if "nanogpt_perplexity" in dataset_metrics:
            writer.add_scalar(
                "Perplexity/{}_NanoGPTStyle".format(dataset_name),
                dataset_metrics["nanogpt_perplexity"],
                epoch + 1,
            )
        if "last_token_perplexity" in dataset_metrics:
            writer.add_scalar(
                "Perplexity/{}_LastToken".format(dataset_name),
                dataset_metrics["last_token_perplexity"],
                epoch + 1,
            )
        if "last_token_perplexity_nanogpt" in dataset_metrics:
            writer.add_scalar(
                "Perplexity/{}_LastToken_NanoGPTStyle".format(dataset_name),
                dataset_metrics["last_token_perplexity_nanogpt"],
                epoch + 1,
            )

        # Log per-position metrics if available
        if seq2seq and "per_position_perplexity" in dataset_metrics:
            per_pos_ppl = dataset_metrics["per_position_perplexity"]
            writer.add_scalar(
                "Perplexity/{}_Min".format(dataset_name),
                dataset_metrics["min_position_perplexity"],
                epoch + 1,
            )
            writer.add_scalar(
                "Perplexity/{}_Max".format(dataset_name),
                dataset_metrics["max_position_perplexity"],
                epoch + 1,
            )
            writer.add_scalar(
                "Perplexity/{}_Range".format(dataset_name),
                dataset_metrics["position_perplexity_range"],
                epoch + 1,
            )

            # Log individual position perplexities
            for pos, ppl in enumerate(per_pos_ppl):
                writer.add_scalar(
                    f"Perplexity/{dataset_name}_Position_{pos}", ppl, epoch + 1
                )

    return dataset_loss, dataset_metrics


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    params,
    optimizer_state,
    config: dict,
    best_val_perplexity: float,
    metrics: dict = None,
    tokenizer: Optional[BaseTokenizer] = None,
    save_best_only: bool = True,
):
    """
    Save JAX training state using Orbax.

    This implementation keeps only a single "step_*" checkpoint inside the
    provided checkpoint directory. When called during training to save an
    improved validation checkpoint, older step_* directories are removed so
    disk usage does not grow. Final models should be saved to a separate
    directory (the training code already does this) and will not be affected.
    """
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    # Ensure directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create checkpointer
    checkpointer = ocp.PyTreeCheckpointer()

    # Pack training state
    ckpt = {
        "params": params,
        "optimizer_state": optimizer_state,
    }

    # Save PyTree to a step-specific subdirectory
    if save_best_only:
        # check if metadata already exists to determine if this is the first checkpoint
        metadata_path = os.path.join(checkpoint_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            old_best_ppl = metadata.get("best_val_perplexity", float("inf"))
            if best_val_perplexity >= old_best_ppl:
                print(
                    f"Current validation perplexity ({best_val_perplexity:.4f}) is not better than the previous best ({old_best_ppl:.4f}). Skipping checkpoint save."
                )
                return
            else:
                print(
                    f"New best validation perplexity ({best_val_perplexity:.4f}) is better than the previous best ({old_best_ppl:.4f}). Saving checkpoint."
                )
        save_folder_name = f"best_model"
        step_path = os.path.join(checkpoint_dir, f"best_model")
    else:
        save_folder_name = f"step_{step}"
        step_path = os.path.join(checkpoint_dir, f"step_{step}")
    # Remove old checkpoints in the directory
    for filename in os.listdir(checkpoint_dir):
        # check if step_path already exists to avoid deleting the checkpoint we just saved
        if filename == save_folder_name:
            old_checkpoint_path = os.path.join(checkpoint_dir, filename)
            if os.path.isdir(old_checkpoint_path):
                shutil.rmtree(old_checkpoint_path)
                print(f"Removed old checkpoint: {old_checkpoint_path}")
    checkpointer.save(step_path, ckpt)

    # skip writing metadata if we are not saving this checkpoint
    if save_best_only:
        print(
            f"Checkpoint saved to {step_path} (best validation perplexity: {best_val_perplexity:.4f})"
        )
        # Save metadata separately (JSON is safer than pickle)
        # Convert config to dict for JSON serialization
        if hasattr(config, "__dict__"):
            config_dict = config.__dict__.copy()
        else:
            config_dict = dict(config) if isinstance(config, dict) else {}
        if "operator" in config_dict:
            config_dict["operator"] = binary_operator_to_name(config_dict["operator"])

        metadata = {
            "step": step,
            "best_val_perplexity": best_val_perplexity,
            "config": config_dict,
            "tokenizer_type": tokenizer.get_tokenizer_type() if tokenizer else None,
            "tokenizer_path": tokenizer.get_tokenizer_path() if tokenizer else None,
        }

        # Include metrics in metadata if provided
        if metrics is not None:
            for key, value in metrics.items():
                if isinstance(value, (float, int)):
                    metadata[key] = value
                else:
                    print(
                        f"Warning: Metric '{key}' has non-scalar value and will not be saved."
                    )

        with open(os.path.join(checkpoint_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"Saved checkpoint at step {step} -> {step_path}")


def load_checkpoint(checkpoint_dir: str, step: int, load_best_only: bool = True):
    """
    Load JAX training state using Orbax.
    """
    checkpoint_dir = os.path.abspath(checkpoint_dir)

    checkpointer = ocp.PyTreeCheckpointer()

    ckpt_path = (
        os.path.join(checkpoint_dir, f"step_{step}")
        if not load_best_only
        else os.path.join(checkpoint_dir, "best_model")
    )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    restored = checkpointer.restore(ckpt_path)

    # Load metadata
    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    metadata = {}
    config = None
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Reconstruct config from dict if available
        if "config" in metadata and isinstance(metadata["config"], dict):
            config_dict = metadata["config"].copy()
            if "operator" in config_dict:
                config_dict["operator"] = resolve_binary_operator(
                    config_dict["operator"]
                )
            config = LDRUExperimenstConfig(**config_dict)

    tokenizer_type = metadata.get("tokenizer_type")
    tokenizer_path = metadata.get("tokenizer_path")
    if tokenizer_type == TokenizerType.SENTENCEPIECE:
        tokenizer = SPTokenizer(model_path=tokenizer_path)
    elif tokenizer_type == TokenizerType.TIKTOKEN_GPT2:
        encoding_name = tokenizer_path if tokenizer_path else "gpt2"
        tokenizer = TiktokenGPT2Tokenizer(encoding_name=encoding_name)
    elif tokenizer_type == TokenizerType.TEXT:
        tokenizer = None
        print(
            "Warning: TextTokenizer cannot be reconstructed from checkpoint metadata. You will need to reinitialize it separately"
        )
    else:
        print(
            "No tokenizer information found in metadata. Tokenizer will not be loaded."
        )
        tokenizer = None

    print(f"Loaded checkpoint from step {step}")

    return (
        restored["params"],
        restored["optimizer_state"],
        config,
        metadata.get("step"),
        metadata.get("best_val_perplexity"),
        tokenizer,
    )


def format_per_position_perplexity(per_pos_ppl, max_display=16):
    """
    Create a nicely formatted string representation of per-position perplexity.

    Args:
        per_pos_ppl: Array of per-position perplexities
        max_display: Maximum number of positions to display

    Returns:
        Formatted string showing perplexity trends
    """
    if len(per_pos_ppl) == 0:
        return "No position data"

    # Convert to numpy if needed
    per_pos_ppl = np.array(per_pos_ppl)

    # Basic statistics
    min_val = np.min(per_pos_ppl)
    max_val = np.max(per_pos_ppl)
    mean_val = np.mean(per_pos_ppl)

    # Trend analysis
    if len(per_pos_ppl) > 1:
        # Linear regression to detect overall trend
        positions = np.arange(len(per_pos_ppl))
        slope = np.polyfit(positions, per_pos_ppl, 1)[0]
        trend = "↗" if slope > 0.1 else "↘" if slope < -0.1 else "→"
    else:
        trend = "→"

    # Format the display
    if len(per_pos_ppl) <= max_display:
        # Show all values
        values_str = " ".join([f"{val:.1f}" for val in per_pos_ppl])
        return f"[{values_str}] {trend} (range: {min_val:.1f}-{max_val:.1f})"
    else:
        # Show first few, last few, with summary
        n_show = max_display // 2 - 1
        start_vals = " ".join([f"{val:.1f}" for val in per_pos_ppl[:n_show]])
        end_vals = " ".join([f"{val:.1f}" for val in per_pos_ppl[-n_show:]])
        return f"[{start_vals} ... {end_vals}] {trend} (μ={mean_val:.1f}, range={min_val:.1f}-{max_val:.1f})"


def analyze_position_trends(per_pos_ppl):
    """
    Analyze trends in per-position perplexity data.

    Args:
        per_pos_ppl: Array of per-position perplexities

    Returns:
        Dictionary with trend analysis
    """
    per_pos_ppl = np.array(per_pos_ppl)

    if len(per_pos_ppl) < 2:
        return {"trend": "insufficient_data"}

    positions = np.arange(len(per_pos_ppl))

    # Linear regression
    slope, intercept = np.polyfit(positions, per_pos_ppl, 1)

    # R-squared for trend strength
    y_pred = slope * positions + intercept
    ss_res = np.sum((per_pos_ppl - y_pred) ** 2)
    ss_tot = np.sum((per_pos_ppl - np.mean(per_pos_ppl)) ** 2)
    r_squared = 1 - (ss_res / (ss_tot + 1e-8))

    # Categorize trend
    if abs(slope) < 0.05:
        trend_type = "stable"
    elif slope > 0:
        trend_type = "increasing"
    else:
        trend_type = "decreasing"

    # Find best and worst positions
    best_pos = np.argmin(per_pos_ppl)
    worst_pos = np.argmax(per_pos_ppl)

    return {
        "trend": trend_type,
        "slope": slope,
        "r_squared": r_squared,
        "best_position": int(best_pos),
        "best_perplexity": float(per_pos_ppl[best_pos]),
        "worst_position": int(worst_pos),
        "worst_perplexity": float(per_pos_ppl[worst_pos]),
        "range": float(np.max(per_pos_ppl) - np.min(per_pos_ppl)),
    }


def create_evaluation_model(config, model_creation_fn):
    """
    Create a model instance with dropout disabled for evaluation.
    This creates a new model with dropout_prob=0.0 while keeping all other config the same.
    """
    # Create a copy of the config with dropout disabled
    eval_config_params = config.__dict__.copy()
    eval_config_params["dropout_prob"] = 0.0
    eval_config = LDRUExperimenstConfig(**eval_config_params)

    # Create the evaluation model
    eval_model = model_creation_fn(eval_config)
    return eval_model


def evaluate_model(
    params,
    model,
    rng_key,
    val_data,
    batch_size,
    use_lstm=False,
    use_transformer=False,
    use_transformer_ldru=False,
    use_ldru_transformer=False,
    seq2seq=True,
    eval_model=None,  # Optional evaluation model with dropout disabled
    compiled_eval_step=None,  # Optional pre-compiled evaluation step for efficiency
    max_eval_steps: Optional[int] = None,
    nanogpt_batching: bool = False,
    seq_length: Optional[int] = None,
    nanogpt_ppl_metric: bool = False,
):
    """Evaluate model on validation data."""
    if nanogpt_batching:
        if seq_length is None:
            raise ValueError("seq_length is required when nanogpt_batching=True.")
        if not hasattr(val_data, "shape") or len(val_data.shape) != 1:
            raise ValueError(
                "nanogpt_batching expects 1D token-stream validation data."
            )
        total_possible = int(val_data.shape[0]) - int(seq_length)
        if total_possible <= 0:
            raise ValueError(
                f"Validation token stream too short ({int(val_data.shape[0])} tokens) "
                f"for seq_length={seq_length}."
            )
        eval_batches = (
            max_eval_steps
            if max_eval_steps is not None
            else max(1, total_possible // max(1, batch_size))
        )
        val_loader = create_nanogpt_token_stream_loader(
            val_data,
            seq_length=seq_length,
            batch_size=batch_size,
            rng_key=rng_key,
            num_batches=eval_batches,
        )
    else:
        val_loader = create_data_loader(val_data, batch_size, rng_key)

    # Use evaluation model if provided (should have dropout disabled)
    model_for_eval = eval_model if eval_model is not None else model

    losses = []
    accuracies = []
    perplexities = []
    per_position_perplexities = []  # For seq2seq models
    per_position_losses = []

    # ---- choose loss function ONCE ----
    if compiled_eval_step is None:
        if use_transformer or use_transformer_ldru or use_ldru_transformer:
            loss_fn = ldru_seq2seq_loss if seq2seq else next_token_loss
        elif use_lstm:
            loss_fn = lstm_next_token_loss if seq2seq else lstm_last_position_loss
        else:
            loss_fn = ldru_seq2seq_loss if seq2seq else next_token_loss

        # ---- jit ONCE ----
        @jax.jit
        def eval_step(params, key, batch):
            return loss_fn(params, model_for_eval, key, batch)

        compiled_eval_step = jax.jit(eval_step)

    # Wrap validation iterator with tqdm for progress reporting
    pbar = tqdm.tqdm(val_loader, desc="Validation", leave=False)
    eval_step_count = 0

    for batch in pbar:
        rng_key, step_key = jax.random.split(rng_key)

        # ---- fast compiled call ----
        loss, metrics = compiled_eval_step(params, step_key, batch)

        losses.append(float(loss))
        accuracies.append(float(metrics["accuracy"]))
        perplexities.append(float(metrics["perplexity"]))

        # collect per-position metrics
        if seq2seq and "per_position_perplexity" in metrics:
            per_position_perplexities.append(
                np.array(metrics["per_position_perplexity"])
            )
            per_position_losses.append(np.array(metrics["per_position_loss"]))

        pbar.set_postfix(
            {
                "Loss": f"{np.mean(losses):.4f}",
                "Acc": f"{np.mean(accuracies):.4f}",
                "PPL": f"{np.mean(perplexities):.1f}",
            }
        )
        eval_step_count += 1
        if max_eval_steps is not None and eval_step_count >= max_eval_steps:
            break

    if len(losses) == 0:
        raise ValueError(
            "No evaluation batches were produced. "
            "Check evaluation dataset size, batch_size, and step limits."
        )

    avg_loss = np.mean(losses)
    avg_metrics = {"accuracy": np.mean(accuracies), "perplexity": np.mean(perplexities)}
    if nanogpt_ppl_metric:
        avg_metrics["nanogpt_perplexity"] = float(np.exp(avg_loss))

    # Add per-position metrics for seq2seq models
    if seq2seq and len(per_position_perplexities) > 0:
        # Average per-position metrics across all batches
        avg_per_position_perplexity = np.mean(per_position_perplexities, axis=0)
        avg_per_position_loss = np.mean(per_position_losses, axis=0)

        avg_metrics["per_position_perplexity"] = avg_per_position_perplexity
        avg_metrics["per_position_loss"] = avg_per_position_loss
        avg_metrics["last_token_perplexity"] = float(avg_per_position_perplexity[-1])
        avg_metrics["last_token_loss"] = float(avg_per_position_loss[-1])
        if nanogpt_ppl_metric:
            avg_metrics["last_token_perplexity_nanogpt"] = float(
                np.exp(avg_per_position_loss[-1])
            )

        # Additional summary statistics
        avg_metrics["min_position_perplexity"] = np.min(avg_per_position_perplexity)
        avg_metrics["max_position_perplexity"] = np.max(avg_per_position_perplexity)
        avg_metrics["position_perplexity_range"] = np.max(
            avg_per_position_perplexity
        ) - np.min(avg_per_position_perplexity)

    return avg_loss, avg_metrics


def test_generation(
    model,
    params,
    config,
    rng_key,
    tokenizer=None,
    max_length=20,
    verbose=True,
    seq2seq=True,
    eval_model=None,  # Optional evaluation model with dropout disabled
):
    """
    Test text generation with the trained LDRU model.

    Since LDRU outputs meaningful predictions at position 0, we generate by:
    1. Starting with a context
    2. Using LDRU to predict the next token
    3. Appending the predicted token to context
    4. Repeating until we reach max_length
    """

    # Use evaluation model if provided (should have dropout disabled)
    model_for_generation = eval_model if eval_model is not None else model

    if tokenizer:
        # Use a meaningful prompt if we have a tokenizer
        prompt = "south korea 's economic boom which began in N stopped this year because of prolonged labor disputes trade conflicts and sluggish exports "
        start_tokens = tokenizer.encode(prompt)
        if len(start_tokens) == 0:
            start_tokens = [tokenizer.encode("The")]  # Use UNK token if prompt is empty

        if verbose:
            print(f"Generating text starting with: '{prompt}'")
            print(f"Start tokens: {start_tokens}")

        # Start with the prompt
        generated = start_tokens.copy()
    else:
        print("No tokenizer available, starting generation with a single token (0)")
        generated = [0]  # Start with a single token

    # Generate tokens one by one
    for i in range(max_length):
        rng_key, step_key = jax.random.split(rng_key)

        # Prepare current sequence as batch
        # Handle LSTM vs LDRU differently
        if seq2seq:
            # For LSTM, use the current sequence directly
            current_seq = generated[
                -min(len(generated), config.max_sequence_length - 1) :
            ]
            input_batch = jnp.array([current_seq])  # [1, seq_len]

            # Get logits - use the last position for LSTM
            logits = model_for_generation.apply(
                params, step_key, input_batch
            )  # [1, seq_len, vocab_size]
            next_token_logits = logits[0, -1, :]  # [vocab_size] - use last position
        else:
            # For LDRU, we need at least 2 tokens (context + position for prediction)
            current_seq = generated[
                -min(len(generated), config.max_sequence_length - 1) :
            ]

            # Add a dummy token at the end (LDRU will predict what this should be)
            dummy_token = 0  # Use PAD token as placeholder
            input_seq = current_seq + [dummy_token]
            input_batch = jnp.array([input_seq])  # [1, seq_len]

            # Get logits - position 0 contains the prediction for the last token
            logits = model_for_generation.apply(
                params, step_key, input_batch
            )  # [1, seq_length, vocab_size]
            next_token_logits = logits[
                0, 0, :
            ]  # [vocab_size] - prediction from position 0

        # Use temperature for more interesting generation
        temperature = 0.7
        scaled_logits = next_token_logits / temperature
        next_token = jax.random.categorical(step_key, scaled_logits, shape=(1,))[0]
        next_token = int(next_token)

        generated.append(next_token)

    if tokenizer:
        generated_text = tokenizer.decode(generated)
        print(f"Generated: '{generated_text}'")
        return generated_text
    else:
        if verbose:
            print(f"Generated sequence: {generated}")
        else:
            print(f"Generated: {generated}")
        return generated


def load_and_test_model(checkpoint_path: str, test_text: str = None):
    """Load a saved model and test text generation."""
    # Load checkpoint
    # throw unimplemented error
    raise NotImplementedError("load_and_test_model not implemented")

    params, config, tokenizer, epoch, best_ppl = load_checkpoint(checkpoint_path, 1)

    # Recreate model and determine model creation function
    if "lstm" in checkpoint_path.lower():
        model_creation_fn = create_lstm_model
        model = create_lstm_model(config)
        use_lstm = True
        use_transformer = False
        use_transformer_ldru = False
    elif "transformer_ldru" in checkpoint_path.lower():
        model_creation_fn = create_transformer_ldru_model
        model = create_transformer_ldru_model(config)
        use_lstm = False
        use_transformer = False
        use_transformer_ldru = True
    elif "transformer" in checkpoint_path.lower():
        model_creation_fn = create_transformer_model
        model = create_transformer_model(config)
        use_lstm = False
        use_transformer = True
        use_transformer_ldru = False
    else:
        model_creation_fn = create_causal_ldru_model
        model = create_causal_ldru_model(config)
        use_lstm = False
        use_transformer = False
        use_transformer_ldru = False

    # Create evaluation model with dropout disabled
    eval_model = create_evaluation_model(config, model_creation_fn)

    seq2seq = "seq2seq" in checkpoint_path.lower()

    print(
        f"\nLoaded model from epoch {epoch} (best validation perplexity: {best_ppl:.4f})"
    )
    model_type_str = (
        "LSTM"
        if use_lstm
        else (
            "Transformer"
            if use_transformer
            else ("Transformer+LDRU" if use_transformer_ldru else "LDRU")
        )
    )
    print(f"Model type: {model_type_str}")
    print(f"Loss type: {'Seq2Seq' if seq2seq else 'Last position'}")

    # Test generation
    rng_key = jax.random.PRNGKey(42)

    if test_text and tokenizer:
        # Test with custom text
        print(f"\nGenerating continuation for: '{test_text}'")
        start_tokens = tokenizer.encode(test_text)
        if len(start_tokens) == 0:
            start_tokens = [tokenizer.encode("The")]  # Use UNK token if input is empty

        generated = start_tokens.copy()

        for i in range(20):
            rng_key, step_key = jax.random.split(rng_key)

            if seq2seq:
                current_seq = generated[
                    -min(len(generated), config.max_sequence_length - 1) :
                ]
                input_batch = jnp.array([current_seq])
                logits = eval_model.apply(
                    params, step_key, input_batch
                )  # Use eval_model
                next_token_logits = logits[0, -1, :]
            else:
                current_seq = generated[
                    -min(len(generated), config.max_sequence_length - 1) :
                ]
                dummy_token = 0
                input_seq = current_seq + [dummy_token]
                input_batch = jnp.array([input_seq])
                logits = eval_model.apply(
                    params, step_key, input_batch
                )  # Use eval_model
                next_token_logits = logits[0, 0, :]

            temperature = 0.7
            scaled_logits = next_token_logits / temperature
            next_token = int(
                jax.random.categorical(step_key, scaled_logits, shape=(1,))[0]
            )
            generated.append(next_token)

        generated_text = tokenizer.decode(generated)
        print(f"Generated: '{generated_text}'")
    else:
        print("\nGenerating random sequence...")
        test_generation(
            model,
            params,
            config,
            rng_key,
            tokenizer,
            verbose=True,
            seq2seq=seq2seq,
            eval_model=eval_model,
        )

    return model, params, config, tokenizer


def _compute_per_token_metrics_single(params, model, rng_key, sequence, seq2seq=True):
    """
    Compute per-token perplexity and per-token accuracy for a single sequence.

    Args:
        params: Model parameters.
        model: Evaluation model (dropout disabled).
        rng_key: JAX PRNG key.
        sequence: 1-D array of token IDs [seq_length].
        seq2seq: Whether to use seq2seq (all-position) evaluation.

    Returns:
        Dict with:
            per_token_perplexity: np.array [L-1] – perplexity at each predicted position.
            per_token_accuracy:   np.array [L-1] – 1.0 / 0.0 per position (correct prediction).
            avg_accuracy:         float – average accuracy over all positions.
            avg_perplexity:       float – average perplexity over all positions.
            loss:                 float – mean cross-entropy loss.
    """
    # Add batch dimension: [1, seq_length]
    batch = jnp.expand_dims(sequence, axis=0)

    input_ids = batch[:, :-1]  # [1, L-1]
    targets = batch[:, 1:]  # [1, L-1]

    logits = model.apply(params, rng_key, input_ids)  # [1, L-1, V]

    # Match the same clipping as training loss functions (ldru_seq2seq_loss, etc.)
    # to ensure consistent perplexity numbers.
    logits = jnp.clip(logits, -10, 10)

    log_probs = jax.nn.log_softmax(logits, axis=-1)  # [1, L-1, V]

    # Per-token log prob of the correct target
    target_log_probs = jnp.take_along_axis(
        log_probs, targets[..., None], axis=-1
    ).squeeze(
        -1
    )  # [1, L-1]

    # Same clip as training: target_log_probs in [-10, 0]
    target_log_probs = jnp.clip(target_log_probs, -10, 0)

    # Per-token loss (negative log likelihood) – squeeze to [L-1]
    per_token_loss = -target_log_probs[0]  # [L-1], in [0, 10]
    per_token_perplexity = jnp.exp(per_token_loss)  # max exp(10) ≈ 22k

    # Per-token accuracy: did argmax prediction match the target?
    predictions = jnp.argmax(logits[0], axis=-1)  # [L-1]
    per_token_correct = (predictions == targets[0]).astype(jnp.float32)  # [L-1]

    # Predicted token confidence (softmax probability of the argmax token)
    probs = jax.nn.softmax(logits[0], axis=-1)  # [L-1, V]
    predicted_confidence = jnp.max(probs, axis=-1)  # [L-1]

    mean_loss = float(jnp.mean(per_token_loss))
    # Standard perplexity = exp(mean_loss), NOT mean(exp(loss)).
    # The arithmetic mean of per-token perplexities is heavily skewed by
    # outlier tokens.  exp(mean_loss) is the geometric mean of per-token
    # perplexities and matches how the training loop reports perplexity.
    avg_ppl = float(jnp.exp(jnp.clip(jnp.mean(per_token_loss), 0, 10)))
    avg_acc = float(jnp.mean(per_token_correct))

    return {
        "per_token_perplexity": np.array(per_token_perplexity),
        "per_token_accuracy": np.array(per_token_correct),
        "predicted_token_ids": np.array(predictions),
        "predicted_confidence": np.array(predicted_confidence),
        "avg_perplexity": avg_ppl,
        "avg_accuracy": avg_acc,
        "loss": mean_loss,
    }


def _plot_sequence_metrics(
    per_token_ppl,
    per_token_acc,
    seq_idx,
    token_labels=None,
    save_path=None,
    title_extra="",
    predicted_token_labels=None,
    predicted_confidence=None,
):
    """
    Plot per-token perplexity (line, log scale) and cumulative average accuracy
    on a single dual-axis figure.  Wrong predictions are annotated with the
    model's predicted token and its confidence.

    Args:
        per_token_ppl: np.array [L-1] – perplexity at each predicted position.
        per_token_acc: np.array [L-1] – 1/0 correct at each predicted position.
        seq_idx: Sequence index (for labelling).
        token_labels: Optional list of *target* token strings for the x-axis.
        save_path: If given, save figure to this path; otherwise show interactively.
        title_extra: Extra text appended to the title.
        predicted_token_labels: Optional list of *predicted* token strings (same length).
        predicted_confidence: Optional np.array [L-1] – model confidence (softmax
                              probability) for its top prediction at each position.
    """
    n_positions = len(per_token_ppl)
    positions = np.arange(n_positions)

    # Cumulative average accuracy
    cum_acc = np.cumsum(per_token_acc) / (positions + 1)

    fig, ax1 = plt.subplots(figsize=(max(8, n_positions * 0.35), 5))

    # --- Left axis: per-token perplexity (log scale) ---
    color_ppl = "#4C72B0"
    ax1.fill_between(positions, 1, per_token_ppl, alpha=0.15, color=color_ppl)
    ax1.plot(
        positions,
        per_token_ppl,
        color=color_ppl,
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="Token PPL",
    )
    ax1.set_yscale("log")
    ax1.set_xlabel(
        "Token Position (predicting token t from context 0..t-1)", fontsize=10
    )
    ax1.set_ylabel("Perplexity (log scale)", color=color_ppl, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_ppl)

    # --- Annotate wrong predictions with predicted token & confidence ---
    if predicted_confidence is not None and predicted_token_labels is not None:
        wrong_mask = per_token_acc < 0.5
        wrong_positions = positions[wrong_mask]
        for wp in wrong_positions:
            pred_tok = predicted_token_labels[wp]
            conf = predicted_confidence[wp]
            # Truncate long tokens for readability
            if len(pred_tok) > 12:
                pred_tok = pred_tok[:10] + "…"
            label_text = f'"{pred_tok}" {conf:.0%}'
            ax1.annotate(
                label_text,
                xy=(wp, per_token_ppl[wp]),
                xytext=(0, 8),
                textcoords="offset points",
                fontsize=6,
                color="#C44E52",
                ha="center",
                va="bottom",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="white",
                    ec="#C44E52",
                    alpha=0.7,
                    lw=0.5,
                ),
            )

    # --- Right axis: cumulative average accuracy ---
    ax2 = ax1.twinx()
    color_acc = "#DD8452"
    ax2.plot(
        positions,
        cum_acc,
        color=color_acc,
        marker="s",
        markersize=3,
        linewidth=2,
        label="Cumulative Avg Accuracy",
    )
    # Also scatter individual correct/incorrect
    ax2.scatter(
        positions,
        per_token_acc,
        color=color_acc,
        alpha=0.3,
        s=15,
        zorder=2,
        label="Token Correct (0/1)",
    )
    ax2.set_ylabel("Accuracy", color=color_acc, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax2.set_ylim(-0.05, 1.1)

    # Optional token labels on x-axis
    if token_labels is not None and n_positions <= 64:
        ax1.set_xticks(positions)
        ax1.set_xticklabels(token_labels, rotation=70, ha="right", fontsize=7)
    else:
        ax1.set_xticks(
            np.linspace(0, n_positions - 1, min(n_positions, 20)).astype(int)
        )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    # Use exp(mean(log(ppl))) = exp(mean_loss) for the title — consistent with
    # standard sequence perplexity (geometric mean, not arithmetic mean).
    avg_ppl = float(np.exp(np.clip(np.mean(np.log(per_token_ppl)), 0, 10)))
    avg_acc = float(np.mean(per_token_acc))
    title = f"Sequence {seq_idx}  —  PPL: {avg_ppl:.2f}  |  " f"Acc: {avg_acc:.2%}"
    if title_extra:
        title += f"  ({title_extra})"
    ax1.set_title(title, fontsize=11, fontweight="bold")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved plot → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _detect_model_type(checkpoint_path: str):
    """Detect model type and creation function from checkpoint filename."""
    cp_lower = checkpoint_path.lower()
    use_lstm = "lstm" in cp_lower
    use_transformer_ldru = (
        "transformer_ldru" in cp_lower and "ldru_transformer" not in cp_lower
    )
    use_ldru_transformer = "ldru_transformer" in cp_lower
    use_transformer = (
        "transformer" in cp_lower
        and not use_transformer_ldru
        and not use_ldru_transformer
    )
    seq2seq = "seq2seq" in cp_lower

    if use_lstm:
        model_creation_fn = create_lstm_model
        model_type_str = "LSTM"
    elif use_transformer_ldru:
        model_creation_fn = create_transformer_ldru_model
        model_type_str = "Transformer+LDRU"
    elif use_ldru_transformer:
        model_creation_fn = create_ldru_transformer_model
        model_type_str = "LDRU+Transformer"
    elif use_transformer:
        model_creation_fn = create_transformer_model
        model_type_str = "Transformer"
    else:
        model_creation_fn = create_causal_ldru_model
        model_type_str = "LDRU"

    return model_creation_fn, model_type_str, seq2seq


def _choose_sequence_indices(n_total: int, n_sequences: int) -> List[int]:
    """Pick n_sequences evenly-spaced indices from [0, n_total)."""
    n_sequences = min(n_sequences, n_total)
    if n_total <= n_sequences:
        return list(range(n_total))
    return np.linspace(0, n_total - 1, n_sequences, dtype=int).tolist()


def _evaluate_sequences(
    params,
    eval_model,
    tokenizer,
    sequences,
    chosen_indices,
    seq2seq,
    model_type_str,
    plot_dir,
    rng_seed=42,
):
    """
    Evaluate a model on the chosen sequences and save per-sequence plots.

    Returns a list of per-sequence result dicts.
    """
    os.makedirs(plot_dir, exist_ok=True)
    per_sequence_results = []
    rng_key = jax.random.PRNGKey(rng_seed)

    for plot_i, seq_idx in enumerate(chosen_indices):
        rng_key, step_key = jax.random.split(rng_key)
        seq = sequences[seq_idx]

        metrics = _compute_per_token_metrics_single(
            params, eval_model, step_key, seq, seq2seq=seq2seq
        )

        # Token labels for x-axis (the *target* tokens being predicted)
        target_ids = np.array(seq[1:])
        token_labels = [
            tokenizer.id_to_word.get(int(tid), f"[{int(tid)}]") for tid in target_ids
        ]

        # Predicted token labels (what the model actually guessed)
        predicted_token_labels = [
            tokenizer.id_to_word.get(int(pid), f"[{int(pid)}]")
            for pid in metrics["predicted_token_ids"]
        ]

        save_path = os.path.join(plot_dir, f"seq_{plot_i:03d}_idx{seq_idx}.png")
        _plot_sequence_metrics(
            per_token_ppl=metrics["per_token_perplexity"],
            per_token_acc=metrics["per_token_accuracy"],
            seq_idx=seq_idx,
            token_labels=token_labels,
            save_path=save_path,
            title_extra=model_type_str,
            predicted_token_labels=predicted_token_labels,
            predicted_confidence=metrics["predicted_confidence"],
        )

        per_sequence_results.append(
            {
                "seq_index": int(seq_idx),
                "avg_perplexity": metrics["avg_perplexity"],
                "avg_accuracy": metrics["avg_accuracy"],
                "loss": metrics["loss"],
                "per_token_perplexity": metrics["per_token_perplexity"].tolist(),
                "per_token_accuracy": metrics["per_token_accuracy"].tolist(),
                "token_labels": token_labels,
            }
        )

        print(
            f"  [{plot_i + 1}/{len(chosen_indices)}] seq {seq_idx:>5d}  "
            f"PPL={metrics['avg_perplexity']:.2f}  "
            f"Acc={metrics['avg_accuracy']:.2%}"
        )

    return per_sequence_results


def _save_summary_plot(per_sequence_results, plot_dir, model_type_str):
    """Save a bar-chart summary of per-sequence PPL and accuracy."""
    ppls = [r["avg_perplexity"] for r in per_sequence_results]
    accs = [r["avg_accuracy"] for r in per_sequence_results]
    n_sequences = len(per_sequence_results)

    fig, (ax_ppl, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_ppl.bar(range(n_sequences), ppls, color="#4C72B0", alpha=0.7)
    ax_ppl.set_xlabel("Sequence")
    ax_ppl.set_ylabel("Perplexity")
    ax_ppl.set_title("Per-Sequence Perplexity")

    ax_acc.bar(range(n_sequences), accs, color="#DD8452", alpha=0.7)
    ax_acc.set_xlabel("Sequence")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Per-Sequence Accuracy")
    ax_acc.set_ylim(0, 1.05)

    overall_ppl = float(np.mean(ppls))
    overall_acc = float(np.mean(accs))
    fig.suptitle(
        f"{model_type_str} — {n_sequences} Sequences  "
        f"(Mean PPL={overall_ppl:.2f}, Mean Acc={overall_acc:.2%})",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    summary_path = os.path.join(plot_dir, "summary.png")
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Summary plot → {summary_path}")


def evaluate_from_checkpoint(
    checkpoint_path: str,
    eval_text_file: str,
    step: int = 1,
    seq_length: int = None,
    stride: int = None,
    n_sequences: int = 10,
    plot_dir: str = None,
    batch_size: int = 32,
    tokenizer_model_path: str = None,
):
    """
    Load a model from checkpoint and evaluate it on a .txt file.

    - ``n_sequences > 0``: evaluate only those n evenly-spaced sequences with
      per-sequence plots (default behaviour).
    - ``n_sequences == 0``: run a fast aggregate evaluation over **all**
      sequences with no plots.  Use ``--n_sequences 0`` from the CLI.

    Args:
        checkpoint_path: Path to the .pkl checkpoint file.
        eval_text_file: Path to the .txt file to evaluate on.
        seq_length: Sequence length (default: 32).
        stride: Stride between sequences (default: seq_length // 2).
        n_sequences: Number of sequences to evaluate and plot (default: 10).
                     0 = aggregate-only over all sequences (no plots).
        plot_dir: Directory to save plots.  Defaults to
                  ``eval_plots/<checkpoint_stem>/``.
        batch_size: Batch size for aggregate evaluation (only used when
                    n_sequences == 0).

    Returns:
        Dict with evaluation results.
    """
    params, _, config, epoch, best_ppl, tokenizer = load_checkpoint(
        checkpoint_path, step=step
    )
    print(f"Model config: {config}")
    if tokenizer is None:
        raise ValueError(
            "Checkpoint does not contain a saved tokenizer. "
            "Cannot evaluate without the training vocabulary."
        )

    if seq_length is None:
        seq_length = 32
        print(f"  Using default seq_length={seq_length} (override with --seq_length)")
    if stride is None:
        stride = seq_length // 2

    # test tokenizer
    test_sentence = "This is a test sentence for the tokenizer."
    test_tokens = tokenizer.encode(test_sentence)
    print(f"Tokenizer test - input: '{test_sentence}'")
    print(f"Tokenizer test - output token IDs: {test_tokens}")

    model_creation_fn, model_type_str, seq2seq = _detect_model_type(checkpoint_path)

    mode_str = (
        "aggregate (all sequences, no plots)"
        if n_sequences == 0
        else f"{n_sequences} sequences with plots"
    )

    print(f"\n{'=' * 60}")
    print(f"  Evaluation")
    print(f"{'=' * 60}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Eval file  : {eval_text_file}")
    print(f"  Model type : {model_type_str}")
    print(f"  Loss type  : {'seq2seq' if seq2seq else 'last-position'}")
    print(f"  Epoch      : {epoch}")
    print(f"  Best val PPL (training): {best_ppl:.4f}")
    print(f"  Vocab size : {tokenizer.get_piece_size()}")
    print(f"  Seq length : {seq_length}  |  Stride : {stride}")
    print(f"  Mode       : {mode_str}")
    print(f"{'=' * 60}\n")

    # 2. Create evaluation model (dropout disabled)
    eval_model = create_evaluation_model(config, model_creation_fn)

    # 3. Load, clean, and tokenize evaluation text
    print(f"Loading evaluation text from: {eval_text_file}")
    # raw text is enough
    with open(eval_text_file, "r", encoding="utf-8") as f:
        text = f.read()
    sequences = create_text_dataset(text, tokenizer, seq_length, stride)
    print(f"  Total sequences available: {len(sequences):,}")

    if n_sequences == 0:
        # Detect boolean flags needed by evaluate_model
        cp_lower = checkpoint_path.lower()
        use_lstm = "lstm" in cp_lower
        use_transformer_ldru = (
            "transformer_ldru" in cp_lower and "ldru_transformer" not in cp_lower
        )
        use_ldru_transformer = "ldru_transformer" in cp_lower
        use_transformer = (
            "transformer" in cp_lower
            and not use_transformer_ldru
            and not use_ldru_transformer
        )

        rng_key = jax.random.PRNGKey(0)
        avg_loss, avg_metrics = evaluate_model(
            params=params,
            model=None,
            rng_key=rng_key,
            val_data=sequences,
            batch_size=batch_size,
            use_lstm=use_lstm,
            use_transformer=use_transformer,
            use_transformer_ldru=use_transformer_ldru,
            use_ldru_transformer=use_ldru_transformer,
            seq2seq=seq2seq,
            eval_model=eval_model,
        )

        print(f"\n{'=' * 60}")
        print(f"  Aggregate Results  ({len(sequences):,} sequences)")
        print(f"{'=' * 60}")
        print(f"  Loss       : {avg_loss:.4f}")
        print(f"  Perplexity : {avg_metrics['perplexity']:.4f}")
        print(f"  Accuracy   : {avg_metrics['accuracy']:.4f}")

        if best_ppl < float("inf"):
            ratio = avg_metrics["perplexity"] / best_ppl
            print(f"\n  Training best val PPL : {best_ppl:.4f}")
            print(f"  Ratio (eval/train)    : {ratio:.2f}")

        print(f"{'=' * 60}\n")

        return {
            "loss": float(avg_loss),
            "perplexity": float(avg_metrics["perplexity"]),
            "accuracy": float(avg_metrics["accuracy"]),
            "metrics": avg_metrics,
            "training_best_ppl": best_ppl,
            "epoch": epoch,
            "n_sequences": len(sequences),
        }

    chosen_indices = _choose_sequence_indices(len(sequences), n_sequences)

    if plot_dir is None:
        ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
        plot_dir = os.path.join("eval_plots", ckpt_stem)
    print(f"Saving plots to: {plot_dir}/")

    # 6. Evaluate only the selected sequences
    per_sequence_results = _evaluate_sequences(
        params,
        eval_model,
        tokenizer,
        sequences,
        chosen_indices,
        seq2seq,
        model_type_str,
        plot_dir,
    )

    _save_summary_plot(per_sequence_results, plot_dir, model_type_str)

    json_path = os.path.join(plot_dir, "per_sequence_results.json")
    with open(json_path, "w") as f:
        json.dump(per_sequence_results, f, indent=2)
    print(f"  Results JSON → {json_path}")

    print(f"\n{'=' * 60}")
    print(f"  Done – {len(per_sequence_results)} sequences evaluated and plotted.")
    print(f"{'=' * 60}\n")

    return {
        "training_best_ppl": best_ppl,
        "epoch": epoch,
        "per_sequence": per_sequence_results,
    }


def evaluate_sequence_length_range(
    checkpoint_path: str,
    eval_text_file: str,
    step: int = 1,
    min_seq_len: int = 4,
    max_seq_len: int = 64,
    plot_dir: str = None,
    batch_size: int = 512,
):
    """
    Evaluate a checkpoint across a range of sequence lengths (min_seq_len to
    max_seq_len inclusive, step 1).  For each length L the evaluation text is
    re-tokenised into windows of length L and the aggregate loss / perplexity /
    accuracy are computed.  A summary plot of perplexity vs sequence length is
    saved to *plot_dir*.

    Usage example::

        python train_causal_ldru.py \\
            --eval_seq_len_range checkpoints/my_model \\
            --eval_file ptb_test.txt \\
            --min_seq_len 4 --max_seq_len 64

    Args:
        checkpoint_path: Directory of the Orbax checkpoint to load.
        eval_text_file:  Path to the .txt evaluation file.
        step:            Checkpoint step to load (default: 1).
        min_seq_len:     Smallest sequence length to evaluate (default: 4).
        max_seq_len:     Largest  sequence length to evaluate (default: 64).
        plot_dir:        Where to save the summary plot and CSV.  Defaults to
                         ``eval_plots/<checkpoint_stem>_seqlen_range/``.
        batch_size:      Batch size for evaluation (default: 64).

    Returns:
        List of dicts, one per evaluated length, with keys
        ``seq_len``, ``n_sequences``, ``loss``, ``perplexity``, ``accuracy``.
    """
    if min_seq_len < 2:
        raise ValueError("min_seq_len must be >= 2 (need at least input + one target)")
    if max_seq_len < min_seq_len:
        raise ValueError("max_seq_len must be >= min_seq_len")

    # --- load checkpoint once ---
    params, _, config, epoch, best_ppl, tokenizer = load_checkpoint(
        checkpoint_path, step=step
    )
    if tokenizer is None:
        raise ValueError(
            "Checkpoint does not contain a saved tokenizer. "
            "Please pass --tokenizer_path so the tokenizer can be loaded."
        )

    model_creation_fn, model_type_str, seq2seq = _detect_model_type(checkpoint_path)
    eval_model = create_evaluation_model(config, model_creation_fn)

    cp_lower = checkpoint_path.lower()
    use_lstm = "lstm" in cp_lower
    use_transformer_ldru = (
        "transformer_ldru" in cp_lower and "ldru_transformer" not in cp_lower
    )
    use_ldru_transformer = "ldru_transformer" in cp_lower
    use_transformer = (
        "transformer" in cp_lower
        and not use_transformer_ldru
        and not use_ldru_transformer
    )

    # --- read the raw text once ---
    with open(eval_text_file, "r", encoding="utf-8") as fh:
        text = fh.read()

    seq_lengths = list(range(min_seq_len, max_seq_len + 1, 128))

    if plot_dir is None:
        ckpt_stem = os.path.basename(checkpoint_path.rstrip("/"))
        plot_dir = os.path.join("eval_plots", f"{ckpt_stem}_seqlen_range")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Sequence-Length Range Evaluation")
    print(f"{'=' * 60}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Eval file  : {eval_text_file}")
    print(f"  Model type : {model_type_str}")
    print(f"  Lengths    : {min_seq_len} → {max_seq_len} (step 1)")
    print(f"  Output dir : {plot_dir}")
    print(f"{'=' * 60}\n")

    results = []
    rng_key = jax.random.PRNGKey(0)

    for seq_len in tqdm.tqdm(seq_lengths, desc="Seq lengths"):
        stride = max(1, seq_len // 2)
        sequences = create_text_dataset(text, tokenizer, seq_len, stride)

        if len(sequences) == 0:
            print(f"  [seq_len={seq_len}] No sequences produced – skipping.")
            continue

        # change batch size dynamically based on sequence length to keep evaluation time reasonable
        max_batch_size_seq_len = 512 * 128
        batch_size = min(batch_size, max_batch_size_seq_len // seq_len)

        rng_key, eval_key = jax.random.split(rng_key)
        avg_loss, avg_metrics = evaluate_model(
            params=params,
            model=None,
            rng_key=eval_key,
            val_data=sequences,
            batch_size=batch_size,
            use_lstm=use_lstm,
            use_transformer=use_transformer,
            use_transformer_ldru=use_transformer_ldru,
            use_ldru_transformer=use_ldru_transformer,
            seq2seq=seq2seq,
            eval_model=eval_model,
        )

        row = {
            "seq_len": seq_len,
            "n_sequences": int(len(sequences)),
            "loss": float(avg_loss),
            "perplexity": float(avg_metrics["perplexity"]),
            "accuracy": float(avg_metrics["accuracy"]),
        }
        results.append(row)
        print(
            f"  seq_len={seq_len:4d} | n={row['n_sequences']:6,} | "
            f"loss={row['loss']:.4f} | ppl={row['perplexity']:.4f} | acc={row['accuracy']:.4f}"
        )

    if not results:
        print("No results collected – check that the eval file has enough text.")
        return results

    # --- save CSV ---
    csv_path = os.path.join(plot_dir, "seqlen_range_results.csv")
    with open(csv_path, "w") as fh:
        fh.write("seq_len,n_sequences,loss,perplexity,accuracy\n")
        for r in results:
            fh.write(
                f"{r['seq_len']},{r['n_sequences']},{r['loss']:.6f},{r['perplexity']:.6f},{r['accuracy']:.6f}\n"
            )
    print(f"\n  CSV  → {csv_path}")

    # --- save JSON ---
    json_path = os.path.join(plot_dir, "seqlen_range_results.json")
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  JSON → {json_path}")

    # --- plot ---
    seq_lens_arr = np.array([r["seq_len"] for r in results])
    perplexities = np.array([r["perplexity"] for r in results])
    accuracies = np.array([r["accuracy"] for r in results])

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_ppl = "#4C72B0"
    ax1.plot(
        seq_lens_arr,
        perplexities,
        color=color_ppl,
        marker="o",
        markersize=4,
        linewidth=1.5,
        label="Perplexity",
    )
    ax1.set_xlabel("Sequence Length", fontsize=12)
    ax1.set_ylabel("Perplexity", color=color_ppl, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_ppl)
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    color_acc = "#DD8452"
    ax2.plot(
        seq_lens_arr,
        accuracies,
        color=color_acc,
        marker="s",
        markersize=4,
        linewidth=1.5,
        linestyle="--",
        label="Accuracy",
    )
    ax2.set_ylabel("Accuracy", color=color_acc, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax2.set_ylim(0, 1.05)

    ckpt_stem = os.path.basename(checkpoint_path.rstrip("/"))
    ax1.set_title(
        f"Perplexity & Accuracy vs Sequence Length\n{ckpt_stem}  |  lengths {min_seq_len}–{max_seq_len}",
        fontsize=11,
        fontweight="bold",
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    fig.tight_layout()
    plot_path = os.path.join(plot_dir, "seqlen_range_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {plot_path}")

    print(
        f"\n  Best PPL  : {perplexities.min():.4f} at seq_len={seq_lens_arr[perplexities.argmin()]}"
    )
    print(
        f"  Best Acc  : {accuracies.max():.4f} at seq_len={seq_lens_arr[accuracies.argmax()]}"
    )
    print(f"{'=' * 60}\n")

    return results


def compare_models(
    checkpoint_paths: List[str],
    eval_text_file: str,
    seq_length: int = None,
    stride: int = None,
    n_sequences: int = 10,
    plot_dir: str = None,
):
    """
    Evaluate two or more models on the *same* n sequences and produce
    side-by-side comparison plots.

    The tokenizer from the **first** checkpoint is used to tokenize the
    evaluation text and select sequences.  All models receive identical
    input sequences so differences are attributable to the model alone.

    Args:
        checkpoint_paths: List of .pkl checkpoint paths (≥ 2).
        eval_text_file: Path to .txt evaluation file.
        seq_length: Sequence length (default: 32).
        stride: Stride between sequences (default: seq_length // 2).
        n_sequences: Number of sequences to compare (default: 10).
                     0 = aggregate-only over all sequences (no plots).
        plot_dir: Output directory.  Defaults to ``eval_plots/comparison/``.

    Returns:
        Dict mapping model name → list of per-sequence result dicts.
    """
    if len(checkpoint_paths) < 2:
        raise ValueError("compare_models requires at least 2 checkpoint paths.")

    if seq_length is None:
        seq_length = 32
    if stride is None:
        stride = seq_length // 2

    # ---- Load all models ----
    models_info = []  # list of (name, params, eval_model, seq2seq)
    first_tokenizer = None

    for cp_path in checkpoint_paths:
        params, config, tokenizer, _, _, epoch, best_ppl, _, _, _ = load_checkpoint(
            cp_path
        )
        if tokenizer is None:
            raise ValueError(f"Checkpoint {cp_path} has no saved tokenizer.")

        model_creation_fn, model_type_str, seq2seq = _detect_model_type(cp_path)
        eval_model = create_evaluation_model(config, model_creation_fn)

        # Friendly display name: <model_type>(<basename>)
        ckpt_stem = os.path.splitext(os.path.basename(cp_path))[0]
        display_name = f"{model_type_str} ({ckpt_stem})"

        models_info.append(
            {
                "name": display_name,
                "params": params,
                "eval_model": eval_model,
                "seq2seq": seq2seq,
                "model_type": model_type_str,
                "epoch": epoch,
                "best_ppl": best_ppl,
            }
        )

        if first_tokenizer is None:
            first_tokenizer = tokenizer

    tokenizer = first_tokenizer

    # ---- Prepare sequences (using first model's tokenizer) ----
    print(f"\nLoading evaluation text from: {eval_text_file}")
    text = load_text(eval_text_file)
    sequences = create_text_dataset(text, tokenizer, seq_length, stride)
    print(f"  Total sequences available: {len(sequences):,}")

    chosen_indices = _choose_sequence_indices(len(sequences), n_sequences)
    n_sequences = len(chosen_indices)

    # ---- Set up output directory ----
    if plot_dir is None:
        plot_dir = os.path.join("eval_plots", "comparison")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Model Comparison — {n_sequences} sequences")
    print(f"{'=' * 60}")
    for i, m in enumerate(models_info):
        print(
            f"  [{i + 1}] {m['name']}  (epoch {m['epoch']}, best PPL {m['best_ppl']:.2f})"
        )
    print(f"  Eval file  : {eval_text_file}")
    print(f"  Seq length : {seq_length}  |  Stride : {stride}")
    print(f"  Output dir : {plot_dir}/")
    print(f"{'=' * 60}\n")

    # ---- Evaluate each model on each sequence ----
    all_results = {m["name"]: [] for m in models_info}
    rng_key = jax.random.PRNGKey(42)

    for plot_i, seq_idx in enumerate(chosen_indices):
        seq = sequences[seq_idx]

        # Token labels (same for every model — same sequence)
        target_ids = np.array(seq[1:])
        token_labels = [
            tokenizer.id_to_word.get(int(tid), f"[{int(tid)}]") for tid in target_ids
        ]

        per_model_metrics = []
        model_names = []
        all_pred_labels = []
        for m in models_info:
            rng_key, step_key = jax.random.split(rng_key)
            metrics = _compute_per_token_metrics_single(
                m["params"], m["eval_model"], step_key, seq, seq2seq=m["seq2seq"]
            )
            per_model_metrics.append(metrics)
            model_names.append(m["name"])

            # Predicted token labels for this model
            pred_labels = [
                tokenizer.id_to_word.get(int(pid), f"[{int(pid)}]")
                for pid in metrics["predicted_token_ids"]
            ]
            all_pred_labels.append(pred_labels)

            all_results[m["name"]].append(
                {
                    "seq_index": int(seq_idx),
                    "avg_perplexity": metrics["avg_perplexity"],
                    "avg_accuracy": metrics["avg_accuracy"],
                    "loss": metrics["loss"],
                    "per_token_perplexity": metrics["per_token_perplexity"].tolist(),
                    "per_token_accuracy": metrics["per_token_accuracy"].tolist(),
                    "token_labels": token_labels,
                }
            )

        # Side-by-side comparison plot for this sequence
        save_path = os.path.join(plot_dir, f"cmp_{plot_i:03d}_idx{seq_idx}.png")
        _plot_comparison_metrics(
            per_model_metrics,
            model_names,
            seq_idx,
            token_labels=token_labels,
            save_path=save_path,
            all_predicted_token_labels=all_pred_labels,
        )

        summary_parts = "  |  ".join(
            f"{mn}: PPL={mm['avg_perplexity']:.2f} Acc={mm['avg_accuracy']:.2%}"
            for mn, mm in zip(model_names, per_model_metrics)
        )
        print(f"  [{plot_i + 1}/{n_sequences}] seq {seq_idx:>5d}  {summary_parts}")

    # ---- Summary comparison bar chart ----
    n_models = len(models_info)
    fig, (ax_ppl, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
    bar_width = 0.8 / n_models
    colors = plt.cm.tab10.colors

    for i, m in enumerate(models_info):
        results = all_results[m["name"]]
        ppls = [r["avg_perplexity"] for r in results]
        accs = [r["avg_accuracy"] for r in results]
        x = np.arange(n_sequences) + i * bar_width
        c = colors[i % len(colors)]
        ax_ppl.bar(x, ppls, width=bar_width, color=c, alpha=0.7, label=m["name"])
        ax_acc.bar(x, accs, width=bar_width, color=c, alpha=0.7, label=m["name"])

    ax_ppl.set_xlabel("Sequence")
    ax_ppl.set_ylabel("Perplexity")
    ax_ppl.set_title("Per-Sequence Perplexity")
    ax_ppl.legend(fontsize=7)
    ax_ppl.set_xticks(np.arange(n_sequences) + bar_width * (n_models - 1) / 2)
    ax_ppl.set_xticklabels([str(i) for i in range(n_sequences)])

    ax_acc.set_xlabel("Sequence")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Per-Sequence Accuracy")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.legend(fontsize=7, loc="lower right")
    ax_acc.set_xticks(np.arange(n_sequences) + bar_width * (n_models - 1) / 2)
    ax_acc.set_xticklabels([str(i) for i in range(n_sequences)])

    fig.suptitle(
        f"Model Comparison — {n_sequences} Sequences",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    summary_path = os.path.join(plot_dir, "comparison_summary.png")
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Summary comparison plot → {summary_path}")

    # Save JSON
    json_path = os.path.join(plot_dir, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results JSON → {json_path}")

    print(f"\n{'=' * 60}")
    print(f"  Done – {n_sequences} sequences compared across {n_models} models.")
    print(f"{'=' * 60}\n")

    return all_results


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a causal LDRU model")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer model file",
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default="ptb_train.txt",
        help="Path to text file for training",
    )
    parser.add_argument("--lstm", action="store_true", help="Use LSTM instead of LDRU")
    parser.add_argument(
        "--transformer", action="store_true", help="Use Transformer instead of LDRU"
    )
    parser.add_argument(
        "--transformer_ldru",
        action="store_true",
        help="Use Transformer encoder + LDRU hybrid model",
    )
    parser.add_argument(
        "--ldru_transformer",
        action="store_true",
        help="Use LDRU encoder + Transformer decoder hybrid model",
    )
    parser.add_argument(
        "--last_pos",
        action="store_true",
        help="Use only last position loss. Default is full sequence loss.",
    )
    parser.add_argument(
        "--no_checkpoint",
        action="store_true",
        help="Disable checkpoint saving (checkpoints are saved by default)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    parser.add_argument(
        "--test_model",
        type=str,
        default=None,
        help="Path to checkpoint file to load and test (no training)",
    )
    parser.add_argument(
        "--test_text",
        type=str,
        default=None,
        help="Text to use as prompt for model testing",
    )
    parser.add_argument(
        "--lr",
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "amsgrad", "muon"],
        help="Optimizer to use for training (default: adamw).",
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=42,
        help="Random seed for JAX PRNG (default: 42).",
    )
    parser.add_argument(
        "--binary_operator",
        type=str,
        default="default",
        choices=list(BINARY_OPERATOR_REGISTRY.keys()),
        help="Binary operator to use for LDRU composition (default: default).",
    )
    parser.add_argument(
        "--binop_expansion_factor",
        type=int,
        default=4,
        help="Hidden expansion factor for compatible binary operators like GRC (default: 4).",
    )
    parser.add_argument(
        "--ablation_expansion_mode",
        type=str,
        default="grc",
        choices=["binary", "grc"],
        help="Expansion stage mode for --binary_operator ablation.",
    )
    parser.add_argument(
        "--ablation_combine_mode",
        type=str,
        default="grc",
        choices=["binary", "grc"],
        help="Combine stage mode for --binary_operator ablation.",
    )
    parser.add_argument(
        "--blelloch_random",
        action="store_true",
        help="Use Blelloch random scan method for LDRU (default is deterministic)",
    )
    parser.add_argument(
        "--scan_method",
        type=str,
        default="default",
        choices=["default", "assoc", "sequential", "simple", "pairwise"],
        help=(
            "LDRU scan method: 'default'/'assoc' for associative tree scan, "
            "'sequential'/'simple' for naive left-to-right scan, "
            "'pairwise' for non-overlapping pair reductions + sequential scan."
        ),
    )
    parser.add_argument(
        "--no_logging",
        action="store_true",
        help="Disable TensorBoard logging (enabled by default)",
    )
    parser.add_argument(
        "--evaluate",
        type=str,
        default=None,
        help="Path to checkpoint file to evaluate (no training). Use with --eval_file.",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        help="Two or more checkpoint paths to compare on the same sequences. "
        "Use with --eval_file.  Example: --compare ckpt1.pkl ckpt2.pkl",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help="Path to .txt file to evaluate on (used with --evaluate or --compare)",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=None,
        help="Sequence length for evaluation (default: 32)",
    )
    parser.add_argument(
        "--n_sequences",
        type=int,
        default=10,
        help="Number of individual sequences to evaluate and plot (default: 10). "
        "Set to 0 for aggregate evaluation on ALL sequences with no plots.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=None,
        help="Directory to save evaluation plots (default: eval_plots/<checkpoint_name>/ or eval_plots/comparison/)",
    )
    parser.add_argument(
        "--val_text_file",
        type=str,
        default="ptb_val.txt",
        help="Path to validation text file (used for validation during training)",
    )
    parser.add_argument(
        "--test_text_file",
        type=str,
        default="ptb_test.txt",
        help="Path to text file to use as prompt for testing a saved model (used with --test_model)",
    )
    parser.add_argument(
        "--train_seq_bin",
        type=str,
        default=None,
        help="Optional path to pretokenized train sequence binary file.",
    )
    parser.add_argument(
        "--val_seq_bin",
        type=str,
        default=None,
        help="Optional path to pretokenized validation sequence binary file.",
    )
    parser.add_argument(
        "--test_seq_bin",
        type=str,
        default=None,
        help="Optional path to pretokenized test sequence binary file.",
    )
    parser.add_argument(
        "--seq_bin_dtype",
        type=str,
        default="uint16",
        choices=["uint16", "uint32", "int32"],
        help="Dtype used in pretokenized sequence binaries.",
    )
    parser.add_argument(
        "--seq_bin_length",
        type=int,
        default=None,
        help="Sequence length stored in pretokenized binaries (must match --max_seq_len).",
    )
    parser.add_argument(
        "--seq_bin_format",
        type=str,
        default="auto",
        choices=["auto", "sequence", "token_stream"],
        help=(
            "Binary format for --train_seq_bin/--val_seq_bin/--test_seq_bin. "
            "'sequence' expects pre-windowed rows; "
            "'token_stream' expects a flat token stream and windows it at runtime; "
            "'auto' infers from --seq_meta_json format when available."
        ),
    )
    parser.add_argument(
        "--seq_meta_json",
        type=str,
        default=None,
        help="Optional metadata JSON from pretokenization step.",
    )
    parser.add_argument(
        "--max_vocab_size",
        type=int,
        default=1500,
        help="Maximum vocabulary size (default: 1500)",
    )
    parser.add_argument(
        "--eval_step",
        type=int,
        default=1,
        help="Minimum frequency for a token to be included in the vocabulary (default: 5)",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=TokenizerType,
        default=TokenizerType.SENTENCEPIECE,
        help="Type of tokenizer to use (default: sentencepiece)",
    )
    parser.add_argument(
        "--model_name_prefix",
        type=str,
        default="",
        help="Suffix to append to model names (default: '')",
    )
    parser.add_argument(
        "--l2_lambda",
        type=float,
        default=1e-2,
        help="L2 regularization lambda for training (default: 1e-2)",
    )
    parser.add_argument(
        "--eval_seq_len_range",
        type=str,
        default=None,
        metavar="CHECKPOINT_DIR",
        help="Evaluate a checkpoint over a range of sequence lengths. "
        "Use with --eval_file, --min_seq_len, and --max_seq_len.",
    )
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=4,
        help="Minimum sequence length for --eval_seq_len_range (default: 4)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=64,
        help="Maximum sequence length for --eval_seq_len_range (default: 64)",
    )
    parser.add_argument(
        "--print_log_file",
        type=str,
        default=None,
        help="Path to the log file for printing (default: None)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for aggregate evaluation when --n_sequences 0 (default: 32)",
    )
    parser.add_argument(
        "--use_alibi",
        action="store_true",
        default=False,
        help="Whether to use ALiBi (Attention with Linear Biases) for attention scores.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--train_steps_per_epoch",
        type=int,
        default=None,
        help=(
            "Optional cap on training steps per epoch. "
            "If provided, each epoch stops after this many train steps "
            "(or earlier if the data loader is exhausted)."
        ),
    )
    parser.add_argument(
        "--validation_steps_per_epoch",
        type=int,
        default=None,
        help=(
            "Optional cap on validation steps per epoch. "
            "If provided, validation averages are computed over at most this many batches."
        ),
    )
    parser.add_argument(
        "--test_steps_per_epoch",
        type=int,
        default=None,
        help=(
            "Optional cap on test-evaluation steps per epoch (when test is run on new best)."
        ),
    )
    parser.add_argument(
        "--compute_dtype",
        type=str,
        default=ComputeDType.FLOAT32.value,
        choices=[d.value for d in ComputeDType],
        help=(
            "Compute dtype for model ops. "
            "Use 'float32' for current behavior or 'bfloat16' for bf16 compute."
        ),
    )
    parser.add_argument(
        "--target_tokens",
        type=int,
        default=None,
        help=(
            "Optional token budget. If set, training stops after approximately this many "
            "tokens (converted via batch_size * seq_length)."
        ),
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help=(
            "Linear warmup steps before cosine decay. "
            "Set 0 to disable warmup (default)."
        ),
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=300,
        help="Dimension of token embeddings (default: 300)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Number of layers in the model for ldru and lstm (default: 1)",
    )
    parser.add_argument(
        "--dropout_prob",
        type=float,
        default=0.3,
        help="Dropout rate for training (default: 0.3)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for LSTM and Transformer models (default: 256)",
    )
    parser.add_argument(
        "--num_transformer_heads",
        type=int,
        default=4,
        help="Number of attention heads for Transformer models (default: 4)",
    )
    parser.add_argument(
        "--num_transformer_layers",
        type=int,
        default=2,
        help="Number of transformer layers for Transformer models (default: 2)",
    )
    parser.add_argument(
        "--tie_embeddings_transformer",
        action="store_true",
        default=False,
        help="Tie transformer token-embedding and output-projection weights.",
    )
    parser.add_argument(
        "--tie_embeddings_ldru",
        action="store_true",
        default=False,
        help="Tie LDRU token-embedding and output-projection weights.",
    )
    parser.add_argument(
        "--transformer_prenorm_gelu_block",
        action="store_true",
        default=False,
        help="Use pre-norm + GELU transformer blocks (nanoGPT-style).",
    )
    parser.add_argument(
        "--ldru_prenorm_gelu_block",
        action="store_true",
        default=False,
        help="Enable an optional pre-norm + GELU FFN block inside each LDRU layer.",
    )
    parser.add_argument(
        "--nanogpt_ppl_metric",
        action="store_true",
        default=False,
        help="Also report perplexity as exp(mean loss) like nanoGPT.",
    )
    parser.add_argument(
        "--nanogpt_batching",
        action="store_true",
        default=False,
        help=(
            "Use nanoGPT-style random offset batching from token-stream bins "
            "(requires --seq_bin_format token_stream)."
        ),
    )
    parser.add_argument(
        "--tensorboard_log_dir",
        type=str,
        default="tensorboard_logs",
        help="Directory to save TensorBoard logs (default: tensorboard_logs)",
    )
    parser.add_argument(
        "--no_streaming_train",
        action="store_true",
        default=False,
        help="Disable streaming training loader and materialize full train dataset in memory.",
    )
    parser.add_argument(
        "--streaming_shuffle_buffer_size",
        type=int,
        default=8192,
        help="Shuffle buffer size (in sequences) for streaming training (default: 8192).",
    )
    parser.add_argument(
        "--streaming_chunk_line_buffer",
        type=int,
        default=4096,
        help="How many lines to tokenize per streaming chunk (default: 4096).",
    )
    parser.add_argument(
        "--train_stride",
        type=int,
        default=None,
        help=(
            "Stride for training/eval sequence windows. "
            "Default uses half-overlap: max_seq_len//2."
        ),
    )
    parser.add_argument(
        "--streaming_exact_sequence_estimate",
        action="store_true",
        default=False,
        help=(
            "Do an exact pre-scan to count streaming train windows. "
            "Disabled by default to avoid long startup on very large corpora."
        ),
    )
    parser.add_argument(
        "--streaming_estimate_bytes_per_token",
        type=float,
        default=4.0,
        help=(
            "Bytes/token used by the fast size-based streaming sequence estimator "
            "(default: 4.0)."
        ),
    )
    args = parser.parse_args()

    configure_output(args.print_log_file)

    # Sequence-length range evaluation mode
    if args.eval_seq_len_range:
        eval_file = args.eval_file if args.eval_file else args.text_file
        if not eval_file:
            print("Error: --eval_seq_len_range requires --eval_file (or --text_file).")
            exit(1)
        evaluate_sequence_length_range(
            checkpoint_path=args.eval_seq_len_range,
            eval_text_file=eval_file,
            step=args.eval_step,
            min_seq_len=args.min_seq_len,
            max_seq_len=args.max_seq_len,
            plot_dir=args.plot_dir,
        )
        exit(0)

    # Compare mode - evaluate multiple models on the same sequences
    if args.compare:
        eval_file = args.eval_file
        if eval_file is None:
            eval_file = args.text_file
        print(f"Comparing {len(args.compare)} models on: {eval_file}")
        compare_models(
            checkpoint_paths=args.compare,
            eval_text_file=eval_file,
            seq_length=args.seq_length,
            n_sequences=args.n_sequences,
            plot_dir=args.plot_dir,
        )
        exit(0)

    use_seq_bins = args.train_seq_bin is not None

    # check tokenizer path
    if args.tokenizer_path:
        tokenizer_path = args.tokenizer_path
        print(f"Using tokenizer from: {tokenizer_path}")
    else:
        tokenizer_path = None
        if not use_seq_bins:
            print(
                "No tokenizer path provided, training will create a new tokenizer from the training text"
            )
            print(
                "Usage: python train_causal_ldru.py --tokenizer_path <path_to_tokenizer_file>"
            )
            print(
                "Example: python train_causal_ldru.py --tokenizer_path /path/to/your/tokenizer.model"
            )

    # Evaluate mode - load checkpoint and evaluate on a text file
    if args.evaluate:
        eval_file = args.eval_file
        if eval_file is None:
            eval_file = args.text_file  # Fall back to --text_file
        print(f"Evaluating checkpoint: {args.evaluate}")
        print(f"Evaluation file: {eval_file}")
        evaluate_from_checkpoint(
            checkpoint_path=args.evaluate,
            eval_text_file=eval_file,
            seq_length=args.seq_length,
            n_sequences=args.n_sequences,
            plot_dir=args.plot_dir,
            tokenizer_model_path=tokenizer_path,
            step=args.eval_step,
        )
        exit(0)

    # Test mode - load and test a saved model without training
    if args.test_model:
        print(f"Testing saved model: {args.test_model}")
        load_and_test_model(args.test_model, args.test_text)
        exit(0)

    if use_seq_bins:
        text_file_path = None
        val_text_file_path = None
        test_text_file_path = None
        print(f"Training with pretokenized train bin: {args.train_seq_bin}")
        if args.val_seq_bin:
            print(f"Using pretokenized val bin: {args.val_seq_bin}")
        if args.test_seq_bin:
            print(f"Using pretokenized test bin: {args.test_seq_bin}")
    else:
        # Check if text file path is provided
        if args.text_file:
            text_file_path = args.text_file
            print(f"Training with text file: {text_file_path}")
        else:
            text_file_path = None
            print(
                "No training text file provided. Please provide a text file for training using --text_file."
            )
            exit(1)

        # check if validation text file is provided
        if args.val_text_file:
            val_text_file_path = args.val_text_file
            print(f"Using validation text file: {val_text_file_path}")
        else:
            val_text_file_path = None
            print(
                "No validation text file provided, splitting 10 percent of training data for validation"
            )
            print(
                "Usage: python train_causal_ldru.py --val_text_file <path_to_validation_text_file>"
            )
            print(
                "Example: python train_causal_ldru.py --val_text_file /path/to/your/val.txt"
            )

        if args.test_text_file:
            test_text_file_path = args.test_text_file
            print(f"Using test text file for model testing: {test_text_file_path}")
        else:
            test_text_file_path = None

    # Set model creation function
    model_creation_fn = create_causal_ldru_model
    use_lstm = False
    use_transformer = False
    use_transformer_ldru = False
    use_ldru_transformer = False
    model_type_name = "LDRU v2"
    seq2seq = True  # Default for LSTM
    if args.lstm:
        model_creation_fn = create_lstm_model
        use_lstm = True
        model_type_name = "LSTM"
    elif args.transformer:
        model_creation_fn = create_transformer_model
        use_transformer = True
        model_type_name = "Transformer"
    elif args.transformer_ldru:
        model_creation_fn = create_transformer_ldru_model
        use_transformer_ldru = True
        model_type_name = "Transformer+LDRU"
    elif args.ldru_transformer:
        model_creation_fn = create_ldru_transformer_model
        use_ldru_transformer = True
        model_type_name = "LDRU+Transformer"
    seq2seq = not args.last_pos  # Use full sequence unless --last_pos is specified

    if args.binop_expansion_factor <= 0:
        raise ValueError("--binop_expansion_factor must be > 0.")

    checkpoint_dir = args.checkpoint_dir
    resume_from_checkpoint = args.resume

    # Create checkpoint directory if it doesn't exist
    LOG_DIR = args.tensorboard_log_dir
    enable_logging = not args.no_logging
    if enable_logging:
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)

    config = LDRUExperimenstConfig(
        embedding_dim=args.embedding_dim,  # Larger for word-level
        num_layers=args.num_layers,
        max_sequence_length=3072,
        dropout_prob=args.dropout_prob,
        hidden_dim=args.hidden_dim,
        use_positional_encoding=(
            True if use_transformer or use_transformer_ldru else False
        ),
        expand_to_power_of_2=True if args.blelloch_random else False,
        use_alibi=args.use_alibi,
        vocab_size=args.max_vocab_size,  # Set vocab size based on tokenizer
        initial_learning_rate=args.lr,
        l2_lambda=args.l2_lambda,
        seq_length=args.max_seq_len,
        batch_size=args.batch_size,
        min_learning_rate=args.lr / 1000,  # Set min LR for cosine decay
        num_epochs=args.num_epochs,
        num_transformer_heads=args.num_transformer_heads,
        num_transformer_layers=args.num_transformer_layers,
        tie_embeddings_transformer=args.tie_embeddings_transformer,
        tie_embeddings_ldru=args.tie_embeddings_ldru,
        transformer_prenorm_gelu_block=args.transformer_prenorm_gelu_block,
        ldru_prenorm_gelu_block=args.ldru_prenorm_gelu_block,
        tie_embeddings=args.tie_embeddings_ldru,
        prenorm_gelu_block=args.ldru_prenorm_gelu_block,
        scan_method=args.scan_method,
        operator=resolve_binary_operator(args.binary_operator),
        binop_expansion_factor=args.binop_expansion_factor,
        ablation_expansion_mode=args.ablation_expansion_mode,
        ablation_combine_mode=args.ablation_combine_mode,
    )
    print(f"Selected binary operator: {binary_operator_to_name(config.operator)}")
    print(f"Selected scan method: {config.scan_method}")
    print(f"Binary operator expansion factor: {config.binop_expansion_factor}")
    print(
        "Feature toggles: "
        f"tie_embeddings_transformer={args.tie_embeddings_transformer}, "
        f"tie_embeddings_ldru={args.tie_embeddings_ldru}, "
        f"transformer_prenorm_gelu_block={args.transformer_prenorm_gelu_block}, "
        f"ldru_prenorm_gelu_block={args.ldru_prenorm_gelu_block}, "
        f"nanogpt_ppl_metric={args.nanogpt_ppl_metric}, "
        f"nanogpt_batching={args.nanogpt_batching}"
    )
    if args.binary_operator == "ablation":
        print(f"Ablation expansion mode: {config.ablation_expansion_mode}")
        print(f"Ablation combine mode: {config.ablation_combine_mode}")

    # Run training
    params, model, config, tokenizer, _ = train_model(
        log_dir=LOG_DIR,
        config=config,
        rng_seed=args.rng_seed,
        enable_logging=enable_logging,
        text_file_path=text_file_path,
        model_creation_fn=model_creation_fn,
        use_lstm=use_lstm,
        use_transformer=use_transformer,
        use_transformer_ldru=use_transformer_ldru,
        seq2seq=seq2seq,
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        tokenizer_path=tokenizer_path,  # Pass tokenizer path
        validation_text_file_path=val_text_file_path,  # Pass validation text file path
        test_text_file_path=test_text_file_path,
        tokenizer_type=args.tokenizer_type,
        model_prefix=args.model_name_prefix,
        streaming_train=not args.no_streaming_train,
        streaming_shuffle_buffer_size=args.streaming_shuffle_buffer_size,
        streaming_chunk_line_buffer=args.streaming_chunk_line_buffer,
        streaming_exact_sequence_estimate=args.streaming_exact_sequence_estimate,
        streaming_estimate_bytes_per_token=args.streaming_estimate_bytes_per_token,
        train_seq_bin_path=args.train_seq_bin,
        val_seq_bin_path=args.val_seq_bin,
        test_seq_bin_path=args.test_seq_bin,
        seq_bin_dtype=args.seq_bin_dtype,
        seq_bin_length=args.seq_bin_length,
        seq_bin_format=args.seq_bin_format,
        seq_meta_json=args.seq_meta_json,
        optimizer_name=args.optimizer,
        target_tokens=args.target_tokens,
        train_stride=args.train_stride,
        nanogpt_batching=args.nanogpt_batching,
        nanogpt_ppl_metric=args.nanogpt_ppl_metric,
        warmup_steps=args.warmup_steps,
        train_steps_per_epoch=args.train_steps_per_epoch,
        validation_steps_per_epoch=args.validation_steps_per_epoch,
        test_steps_per_epoch=args.test_steps_per_epoch,
        compute_dtype=args.compute_dtype,
    )

    print(f"\\nModel Summary:")
    print(f"- Model type: {model_type_name}")
    print(f"- Loss type: {'Full sequence' if seq2seq else 'Last position only'}")
    print(f"- Vocabulary size: {config.vocab_size}")
    print(f"- Embedding dimension: {config.embedding_dim}")
    print(f"- Number of layers: {config.num_layers}")
    print(f"- Supports sequences up to length: {config.max_sequence_length}")
    print(f"- Uses positional encoding: {config.use_positional_encoding}")

    if tokenizer:
        print(
            f"- Word-level tokenization with {tokenizer.get_piece_size()} unique words"
        )
