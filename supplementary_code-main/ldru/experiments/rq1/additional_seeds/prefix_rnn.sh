#!/bin/bash

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n)
            pref_length="$2"
            shift 2
            ;;
        --num_symbols)
            num_symbols="$2"
            shift 2
            ;;
        --gpu)
            gpu_id="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --n <prefix_length> --num_symbols <number_of_symbols> --gpu <gpu_id>"
            exit 1
            ;;
    esac
done

# Check if required arguments are provided
if [[ -z "$pref_length" || -z "$num_symbols" || -z "$gpu_id" ]]; then
    echo "Error: --n, --num_symbols, and --gpu arguments are required"
    echo "Usage: $0 --n <prefix_length> --num_symbols <number_of_symbols> --gpu <gpu_id>"
    exit 1
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=${gpu_id}



# Create task name based on parameters
task_name="P_${pref_length}_${num_symbols}"

# Create output directory if it doesn't exist
mkdir -p "${task_name}"

for seed in {3..9}; do
    echo "Running RNN ${task_name} experiment with seed $seed..."
    # Run RNN experiment on n_prefix_symbols task for 100k steps
    python -m ldru.training.example \
        --task n_prefix_symbols \
        --n ${pref_length} \
        --num_symbols ${num_symbols} \
        --seed ${seed} \
        --architecture rnn \
        --embedding_dim 256 \
        --steps 100_000 \
        --lr 1e-3 \
        --dropout_prob 0.0 \
        --eval_data_path "ldru/test_data/${task_name}" \
        > "${task_name}/rnn_${task_name}_s${seed}.out" 2> "${task_name}/rnn_${task_name}_s${seed}.err"
    echo "Completed RNN ${task_name} experiment with seed $seed."
done

for seed in {3..9}; do
    echo "Running LDRU ${task_name} experiment with seed $seed..."
    # Run LDRU experiment on n_prefix_symbols task for 100k steps
    python -m ldru.training.example \
        --task n_prefix_symbols \
        --n ${pref_length} \
        --num_symbols ${num_symbols} \
        --seed ${seed} \
        --architecture ldru \
        --embedding_dim 64 \
        --steps 100_000 \
        --lr 1e-3 \
        --dropout_prob 0.25 \
        --eval_data_path "ldru/test_data/${task_name}" \
        > "${task_name}/ldru_${task_name}_s${seed}.out" 2> "${task_name}/ldru_${task_name}_s${seed}.err"
    echo "Completed LDRU ${task_name} experiment with seed $seed."
done    

echo "${task_name} experiment completed. Check ${task_name}/ directory for output files."