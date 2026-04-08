# Installation

```bash
conda create --name ldru python=3.10
conda activate ldru

# If pip wasn't already installed 
conda install pip

# Installing the package should install the required dependencies automatically
pip install -e .

# (Optional) Upgrade JAX version to GPU-ready version
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

# Usage

## Quick Examples

Here are example commands for training LDRU models on different tasks. These examples use smaller parameters for faster execution on PCs. For larger models (like those we evaluated), consider using a GPU:

### Parity Check Task
Train an LDRU model to determine if the number of 1s in a binary sequence is even or odd:

```bash
python ldru/training/example.py --architecture ldru --task parity_check --steps 10000 --embedding_dim 16 --batch_size 8 --sequence_length 20
```

### Modular Arithmetic Task  
Train an LDRU to compute the sum of digits modulo 5:

```bash
python ldru/training/example.py --architecture ldru --task modular_arithmetic --mod 5 --steps 1000000 --embedding_dim 64 --batch_size 16
```

### $D_4$ Task
Test length generalization on $D_4$ task with LDRU:

```bash
python ldru/training/example.py --architecture ldru --task d_n --n 4 --steps 20000 --embedding_dim 32 --batch_size 16 --max_range_test_length 100
```

### $P_{4,2}$ Task
Train an LDRU model on the $P_{4,2}$ task (prefix of length 4 with 2 symbols):

```bash
python ldru/training/example.py --architecture ldru --task n_prefix_symbols --n 4 --steps 10000 --embedding_dim 64 --batch_size 32 --num_symbols 2
```

## Available Tasks

The following regular language recognition tasks are available:

- `parity_check`: Count 1s modulo 2
- `modular_arithmetic`: Sum digits modulo N (specify with `--mod N`)  
- `even_pairs`: Even number of consecutive symbol pairs
- `cycle_navigation`: Navigate cycles in a graph
- `n_prefix_symbols`: Recognize N-prefix symbol patterns (specify with `--n N`)
- `tomita_1` through `tomita_7`: Tomita grammars
- `d_n`: $D_n$ regular language (specify with `--n N`)

See the appendix for more details on these tasks. Note: We did not evaluate `tomita_1` and `tomita_2` (equivalent to `d_1`) as they are both trivial tasks (accepting very specific sequences only, i.e., sequences in the regular expressions $0^\ast$ and $(01)^\ast$, respectively).

## Available Architectures

- `ldru`: Log-Depth Recurrent Unit (default, ours)
- `rnn`: Vanilla RNN (Elman 1990)
- `lstm`: LSTM (Hochreiter & Schmidhuber 1997)
- `transformer_encoder`: Transformer encoder (Vaswani et al. 2017)
- `sdssm`: Selective Dense State Space Model (Terzić et al. 2025)
- `s5_ssm`: S5 State Space Model (requires separate installation from authors' GitHub repository; Smith, Warrington, and Linderman 2023)

Note: When using `transformer_encoder` with `--num_layers 0`, this implements RegularGPT (Chi et al. 2023).

## Common Parameters

- `--task`: Task name (required)
- `--steps`: Number of training steps (default: 1,000,000)
- `--embedding_dim`: Model embedding dimension (default: 64)
- `--batch_size`: Training batch size (default: 256)  
- `--sequence_length`: Training sequence length (default: 40)
- `--architecture`: Model architecture (default: "ldru")
- `--lr`: Learning rate (default: 1e-3)
- `--seed`: Random seed (default: 0)
- `--max_range_test_length`: Maximum length for generalization testing (default: 500)
- `--num_layers`: Number of layers (default: 1)
- `--dropout_prob`: Dropout probability (default: 0.25)

## Architecture-Specific Parameters

### LDRU (Ours)
- `--operator`: Binary operator type - "MLP", "elementwise_sum", "gated_sum", "linear" (default: "MLP")

### Transformer Encoder / RegularGPT
- `--num_heads`: Number of attention heads (default: 4)
- `--num_layers`: Number of transformer layers (default: 1, use 0 to specify RegularGPT)

RegularGPT-specific parameters:
- `--thickness`: Thickness of adaptive layers in RegularGPT (default: 1)
- `--chunk_size`: Chunk size for RegularGPT (default: 1)
- `--share_weight`: Share weights in adaptive layers (default: False; recommended: True for RegularGPT)

### SD-SSM
- `--num_t_matrices`: Number of transition matrices (default: 8)

## Training Diagnostics

- `--use_tensorboard`: Enable TensorBoard logging for gradient information, loss, and training accuracy (default: False)
- `--validate`: Enable validation during training - samples 1024 sequences of length 500 and validates every 1000 steps (default: False)

## Output

The script will output training progress and final evaluation results. The OOD accuracy is printed at the end of training, which is the average accuracy on sequences longer than the training length (from `sequence_length`+1 to `max_range_test_length`).

# Visualization

For analyzing how LDRU learns to represent formal languages, the visualization script can be used to generate sequence embeddings and their t-SNE projections:

```bash
python ldru/training/visualization.py --task d_n --n 6 --steps 1000000 --lr 1e-4 --embedding_dim 64 --architecture ldru
```

This script:

1. Trains a model on the specified task (currently supports only `d_n` tasks)
2. Generates all sequences up to length $2n$ for the $D_n$ language (covers all equivalence classes)
3. Extracts embeddings from the trained model for each sequence
4. Computes transition monoid equivalence classes - the mathematical ground truth groupings
5. Creates t-SNE visualization showing how sequences cluster by equivalence class
6. Saves outputs:
   - `seq{length}_D{n}_s{seed}_params.pkl` - Trained model parameters
   - `seq{length}_D{n}_s{seed}_embeddings.json` - Sequence embeddings and class labels  
   - `seq{length}_D{n}_s{seed}_tsne.png` - t-SNE plot colored by equivalence class

The visualization helps analyze whether the model learns to group sequences according to their formal language equivalence classes, providing insight into the internal representations learned by LDRU.

## Visualization Parameters

Same parameters as the main training script, but note:
- Currently only supports `--task d_n` 
- Requires `--n` parameter to specify the $D_n$ language
- Suited only to the LDRU (possible to extend by adding embedding extraction for other architectures)

# Reproducing Paper Results

This section provides detailed instructions for reproducing the experimental results reported in the main paper and appendix.

## Experiment Organization

The experiments are organized by research questions in the `ldru/experiments/` directory:

- **`rq1/`**: Main architecture comparison (**RQ1**) - LDRU vs baselines across all task families
- **`rq2/`**: Training sequence length analysis (**RQ2**) - Impact of longer training sequences
- **`rq3/`**: LDRU operator ablation study (**RQ3**) - Comparison of different binary operators

## Main Paper Results

### **RQ1**: Architecture Comparison

Experiments for comparing the LDRU against baseline architectures (RNN, LSTM, Transformer, RegularGPT, S5, SD-SSM) across 21 regular tasks.

The main comparison scripts are organized by architecture in `ldru/experiments/rq1/`:

```bash
ldru/experiments/rq1/
├── ldru/          # LDRU results
├── lstm/          # LSTM baseline  
├── rnn/           # RNN baseline
├── transformer/   # Transformer baseline
├── regulargpt/    # RegularGPT baseline
├── s5/            # S5 baseline (SSM, requires separate installation)
└── sdssm/         # SD-SSM baseline
```

Each architecture directory contains scripts for different task families:

- **`d_n.sh`**: $D_n$ languages ($n=2,3,4,6,8,12$)
- **`deletang.sh`**: Delétang et al. tasks (parity, even pairs, cycle navigation)  
- **`prefix.sh`**: Prefix languages $P_{p,q}$ 
- **`tomita.sh`**: Tomita grammars (3,4,5,6,7)

### Running Main Comparison Experiments

To reproduce the main paper results, run all architecture experiments for each task family:

#### Example: D_n Languages Comparison
```bash
# You must execute from the top-level directory

# Run all architectures on D_n tasks
./ldru/experiments/rq1/ldru/d_n.sh --gpu 0
./ldru/experiments/rq1/lstm/d_n.sh --gpu 0
./ldru/experiments/rq1/rnn/d_n.sh --gpu 0
./ldru/experiments/rq1/transformer/d_n.sh --gpu 0
./ldru/experiments/rq1/regulargpt/d_n.sh --gpu 0
./ldru/experiments/rq1/s5/d_n.sh --gpu 0
./ldru/experiments/rq1/sdssm/d_n.sh --gpu 0
```

#### Example: All Task Families for LDRU
```bash
# You must execute from the top-level directory

# Run LDRU on all task families
./ldru/experiments/rq1/ldru/d_n.sh --gpu 0
./ldru/experiments/rq1/ldru/deletang.sh --gpu 0
./ldru/experiments/rq1/ldru/prefix.sh --gpu 0
./ldru/experiments/rq1/ldru/tomita.sh --gpu 0
```

### **RQ2**: Training Sequence Length Analysis

Analyze how longer training sequences influence length generalization performance.

```bash
# You must execute from the top-level directory

# LDRU with varying training sequence lengths (40, 60, 100, 150)
./ldru/experiments/rq2/ldru_d_n.sh --gpu 0

# RNN baseline with varying training sequence lengths
./ldru/experiments/rq2/rnn_d_n.sh --gpu 0
```

These experiments test $D_n$ tasks ($n=4,6,8,12$) with different training sequence lengths to demonstrate that longer training sequences improve generalization.

### **RQ3**: LDRU Operator Ablation

Compare different binary operators in the LDRU architecture.

```bash
# You must execute from the top-level directory

# Test different LDRU operators (elementwise_sum, gated_sum, linear, MLP)
# on tasks: even_pairs, modular_arithmetic, parity_check, cycle_navigation
./ldru/experiments/rq3/ldru_operators.sh --gpu 0
```

This experiment demonstrates that our MLP operator is necessary for complex tasks like modular arithmetic, while simpler operators suffice for basic tasks.

# Code Reference
This repository builds upon implementations from:
- Delétang et al. (2023) - Training framework
- Chi et al. (2023) - RegularGPT architecture
- Terzić et al. (2025) - SD-SSM architecture implementation