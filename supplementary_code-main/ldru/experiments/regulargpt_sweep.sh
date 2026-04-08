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


# Experiment configuration for RegularGPT hyperparameter sweep
# Based on Table 7: Grid search over optimizer, learning rate, and dropout probability
SEED=0
TASK="modular_arithmetic"
OPTIMIZERS=("adam" "amsgrad")
LEARNING_RATES=(1e-4 3e-4 5e-4)
DROPOUT_PROBS=(0.0 0.1)

# Fixed architectural parameters for RegularGPT
ARCHITECTURE="transformer_encoder"
BATCH_SIZE=256
SEQUENCE_LENGTH=40
EMBEDDING_DIM=256
NUM_LAYERS=0  # RegularGPT is transformer_encoder with 0 layers
NUM_HEADS=8
CHUNK_SIZE=2
THICKNESS=1
SHARE_WEIGHT=true
TRAINING_STEPS=250000 # We did not run the full 1M steps for this sweep
MAX_RANGE_TEST_LENGTH=500

# Create results directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="regulargpt_sweep_${TASK}"
mkdir -p "$RESULTS_DIR"

# Log file for the entire experiment
LOG_FILE="$RESULTS_DIR/experiment_log.txt"

echo "Starting RegularGPT hyperparameter sweep at $(date)" | tee "$LOG_FILE"
echo "Results will be saved in: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "Task: $TASK" | tee -a "$LOG_FILE"
echo "Seed: $SEED" | tee -a "$LOG_FILE"
echo "Architecture: $ARCHITECTURE (RegularGPT)" | tee -a "$LOG_FILE"
echo "Optimizers: ${OPTIMIZERS[*]}" | tee -a "$LOG_FILE"
echo "Learning Rates: ${LEARNING_RATES[*]}" | tee -a "$LOG_FILE"
echo "Dropout Probabilities: ${DROPOUT_PROBS[*]}" | tee -a "$LOG_FILE"
if [ -n "$GPU_ID" ]; then
    echo "GPU: $GPU_ID" | tee -a "$LOG_FILE"
fi
echo "----------------------------------------" | tee -a "$LOG_FILE"

# Calculate total experiments
total_experiments=$((${#OPTIMIZERS[@]} * ${#LEARNING_RATES[@]} * ${#DROPOUT_PROBS[@]}))
current_experiment=0

echo "Total experiments: $total_experiments" | tee -a "$LOG_FILE"

# Main experiment loop
for optimizer in "${OPTIMIZERS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        for dropout in "${DROPOUT_PROBS[@]}"; do
            current_experiment=$((current_experiment + 1))
            
            echo "[$current_experiment/$total_experiments] Running: Optimizer=$optimizer, LR=$lr, Dropout=$dropout" | tee -a "$LOG_FILE"
            
            # Create output filenames
            output_file="$RESULTS_DIR/${TASK}_${optimizer}_lr${lr}_dropout${dropout}_seed${SEED}_regulargpt.out"
            error_file="$RESULTS_DIR/${TASK}_${optimizer}_lr${lr}_dropout${dropout}_seed${SEED}_regulargpt.err"
            
            # Run the experiment with separate stdout and stderr
            python -m ldru.training.example \
                --task "$TASK" \
                --seed "$SEED" \
                --lr "$lr" \
                --steps "$TRAINING_STEPS" \
                --architecture "$ARCHITECTURE" \
                --batch_size "$BATCH_SIZE" \
                --sequence_length "$SEQUENCE_LENGTH" \
                --embedding_dim "$EMBEDDING_DIM" \
                --num_layers "$NUM_LAYERS" \
                --num_heads "$NUM_HEADS" \
                --chunk_size "$CHUNK_SIZE" \
                --thickness "$THICKNESS" \
                --share_weight "$SHARE_WEIGHT" \
                --dropout_prob "$dropout" \
                --optimizer "$optimizer" \
                --max_range_test_length "$MAX_RANGE_TEST_LENGTH" \
                > "$output_file" 2> "$error_file"
            
            exit_code=$?
            
            # Check if the experiment completed successfully
            if [ $exit_code -eq 0 ]; then
                # Extract the final score from the output
                score=$(grep "OOD accuracy:" "$output_file" | tail -1 | awk '{print $3}')
                echo "  -> Completed successfully. OOD accuracy: $score" | tee -a "$LOG_FILE"
            else
                echo "  -> Failed with exit code $exit_code" | tee -a "$LOG_FILE"
                echo "  -> Check $error_file for error details" | tee -a "$LOG_FILE"
            fi
            
            echo "" | tee -a "$LOG_FILE"
        done
    done
done

echo "Experiment completed at $(date)" | tee -a "$LOG_FILE"

# Generate summary report
echo "Generating summary report..." | tee -a "$LOG_FILE"
summary_file="$RESULTS_DIR/summary_report.txt"

echo "REGULARGPT HYPERPARAMETER SWEEP SUMMARY" > "$summary_file"
echo "=======================================" >> "$summary_file"
echo "Timestamp: $TIMESTAMP" >> "$summary_file"
echo "Architecture: $ARCHITECTURE (RegularGPT)" >> "$summary_file"
echo "Task: $TASK" >> "$summary_file"
echo "Seed: $SEED" >> "$summary_file"
echo "Total experiments: $total_experiments" >> "$summary_file"
echo "Training Steps: $TRAINING_STEPS" >> "$summary_file"
echo "" >> "$summary_file"

echo "Grid Search Parameters:" >> "$summary_file"
echo "----------------------" >> "$summary_file"
echo "Optimizers: ${OPTIMIZERS[*]}" >> "$summary_file"
echo "Learning Rates: ${LEARNING_RATES[*]}" >> "$summary_file"
echo "Dropout Probabilities: ${DROPOUT_PROBS[*]}" >> "$summary_file"
echo "" >> "$summary_file"

echo "Results by Configuration:" >> "$summary_file"
echo "------------------------" >> "$summary_file"

# Create a table header
printf "%-10s %-8s %-8s %-10s\n" "Optimizer" "LR" "Dropout" "OOD_Accuracy" >> "$summary_file"
printf "%-10s %-8s %-8s %-10s\n" "---------" "--" "-------" "------------" >> "$summary_file"

best_score=0
best_config=""

for optimizer in "${OPTIMIZERS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        for dropout in "${DROPOUT_PROBS[@]}"; do
            output_file="$RESULTS_DIR/${TASK}_${optimizer}_lr${lr}_dropout${dropout}_seed${SEED}_regulargpt.out"
            error_file="$RESULTS_DIR/${TASK}_${optimizer}_lr${lr}_dropout${dropout}_seed${SEED}_regulargpt.err"
            
            if [ -f "$output_file" ]; then
                score=$(grep "OOD accuracy:" "$output_file" | tail -1 | awk '{print $3}')
                if [ -n "$score" ]; then
                    printf "%-10s %-8s %-8s %-10s\n" "$optimizer" "$lr" "$dropout" "$score" >> "$summary_file"
                    
                    # Track best score
                    if (( $(echo "$score > $best_score" | bc -l) )); then
                        best_score=$score
                        best_config="$optimizer, LR=$lr, Dropout=$dropout"
                    fi
                else
                    printf "%-10s %-8s %-8s %-10s\n" "$optimizer" "$lr" "$dropout" "FAILED" >> "$summary_file"
                fi
            else
                printf "%-10s %-8s %-8s %-10s\n" "$optimizer" "$lr" "$dropout" "NO_OUTPUT" >> "$summary_file"
            fi
        done
    done
done

echo "" >> "$summary_file"
echo "Best Configuration:" >> "$summary_file"
echo "------------------" >> "$summary_file"
echo "Score: $best_score" >> "$summary_file"
echo "Config: $best_config" >> "$summary_file"

echo "" | tee -a "$LOG_FILE"
echo "Summary report saved to: $summary_file" | tee -a "$LOG_FILE"
echo "All individual results saved in: $RESULTS_DIR" | tee -a "$LOG_FILE"