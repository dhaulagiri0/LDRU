#!/bin/bash
#!/bin/bash

# Parse command line arguments
GPU_ID=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--gpu GPU_ID]"
            echo "  --gpu GPU_ID    Set CUDA_VISIBLE_DEVICES to GPU_ID"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set GPU if specified
if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "Using GPU: $GPU_ID"
else
    echo "No GPU specified, using default GPU configuration"
fi

# Create output directory if it doesn't exist
mkdir -p D8

# LSTM additional seeds
for seed in {3..9}; do
    echo "Running LSTM D8 experiment with seed $seed..."
    # Run LSTM experiment on D8 task for 1M steps
    python -m ldru.training.example \
        --task d_n \
        --n 8 \
        --seed ${seed} \
        --architecture lstm \
        --embedding_dim 256 \
        --steps 1_000_000 \
        --lr 1e-4 \
        --dropout_prob 0.0 \
        > D8/lstm_d8_s${seed}.out 2> D8/lstm_d8_s${seed}.err
    echo "Completed LSTM D8 experiment with seed $seed."
done

# LDRU additional seeds
for seed in {3..9}; do
    echo "Running LDRU D8 experiment with seed $seed..."
    # Run LDRU experiment on D8 task for 1M steps
    python -m ldru.training.example \
        --task d_n \
        --n 8 \
        --seed ${seed} \
        --architecture ldru \
        --embedding_dim 64 \
        --steps 1_000_000 \
        --lr 1e-4 \
        --dropout_prob 0.25 \
        > D8/ldru_d8_s${seed}.out 2> D8/ldru_d8_s${seed}.err
    echo "Completed LDRU D8 experiment with seed $seed."
done

# RNN additional seeds (to test difference between RNN and LDRU)
for seed in {3..9}; do
    echo "Running RNN D8 experiment with seed $seed..."
    # Run RNN experiment on D8 task for 1M steps
    python -m ldru.training.example \
        --task d_n \
        --n 8 \
        --seed ${seed} \
        --architecture rnn \
        --embedding_dim 256 \
        --steps 1_000_000 \
        --lr 1e-4 \
        --dropout_prob 0.0 \
        > D8/rnn_d8_s${seed}.out 2> D8/rnn_d8_s${seed}.err
    echo "Completed RNN D8 experiment with seed $seed."
done

echo "D8 experiment completed. Check D8/ directory for output files."