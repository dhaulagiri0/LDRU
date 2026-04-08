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
mkdir -p even_pairs

for seed in {3..9}; do
    echo "Running RNN even_pairs experiment with seed $seed..."
    # Run RNN experiment on even_pairs task for 100k steps
    python -m ldru.training.example \
        --task even_pairs \
        --seed ${seed} \
        --architecture rnn \
        --embedding_dim 256 \
        --steps 100_000 \
        --lr 1e-3 \
        --dropout_prob 0.0 \
        --eval_data_path "ldru/test_data/EvenPairs" \
        > even_pairs/rnn_even_pairs_s${seed}.out 2> even_pairs/rnn_even_pairs_s${seed}.err
    echo "Completed RNN even_pairs experiment with seed $seed."
done

for seed in {3..9}; do
    echo "Running LDRU even_pairs experiment with seed $seed..."
    # Run LDRU experiment on even_pairs task for 100k steps
    python -m ldru.training.example \
        --task even_pairs \
        --seed ${seed} \
        --architecture ldru \
        --embedding_dim 64 \
        --steps 100_000 \
        --lr 1e-3 \
        --dropout_prob 0.1 \
        --eval_data_path "ldru/test_data/EvenPairs" \
        > even_pairs/ldru_even_pairs_s${seed}.out 2> even_pairs/ldru_even_pairs_s${seed}.err
    echo "Completed LDRU even_pairs experiment with seed $seed."
done    

echo "Even_pairs experiment completed. Check even_pairs/ directory for output files."