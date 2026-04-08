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


# Experiment configuration for LDRU optimizer ablation
SEEDS=(0 1 2 3 4 5 6 7 8 9)
TASK="modular_arithmetic"
OPTIMIZERS=("adam" "amsgrad")
DROPOUT_PROBS=(0.0 0.1)

# Fixed parameters (standard LDRU configuration)
ARCHITECTURE="ldru"
BATCH_SIZE=256
SEQUENCE_LENGTH=40
EMBEDDING_DIM=64
NUM_LAYERS=1
LEARNING_RATE=1e-3
TRAINING_STEPS=1000000
MAX_RANGE_TEST_LENGTH=500

# Create results directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="ldru_optimizer_ablation_${TASK}"
mkdir -p "$RESULTS_DIR"

# Log file for the entire experiment
LOG_FILE="$RESULTS_DIR/experiment_log.txt"

echo "Starting LDRU optimizer ablation experiment at $(date)" | tee "$LOG_FILE"
echo "Results will be saved in: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "Task: $TASK" | tee -a "$LOG_FILE"
echo "Seeds: ${SEEDS[*]}" | tee -a "$LOG_FILE"
echo "Architecture: $ARCHITECTURE" | tee -a "$LOG_FILE"
echo "Optimizers: ${OPTIMIZERS[*]}" | tee -a "$LOG_FILE"
echo "Dropout Probabilities: ${DROPOUT_PROBS[*]}" | tee -a "$LOG_FILE"
echo "Learning Rate: $LEARNING_RATE" | tee -a "$LOG_FILE"
echo "Training Steps: $TRAINING_STEPS" | tee -a "$LOG_FILE"
if [ -n "$GPU_ID" ]; then
    echo "GPU: $GPU_ID" | tee -a "$LOG_FILE"
fi
echo "----------------------------------------" | tee -a "$LOG_FILE"

# Calculate total experiments
total_experiments=$((${#OPTIMIZERS[@]} * ${#DROPOUT_PROBS[@]} * ${#SEEDS[@]}))
current_experiment=0

echo "Total experiments: $total_experiments" | tee -a "$LOG_FILE"

# Main experiment loop
for optimizer in "${OPTIMIZERS[@]}"; do
    for dropout in "${DROPOUT_PROBS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            current_experiment=$((current_experiment + 1))
            
            echo "[$current_experiment/$total_experiments] Running: Optimizer=$optimizer, Dropout=$dropout, Seed=$seed" | tee -a "$LOG_FILE"
            
            # Create output filenames
            output_file="$RESULTS_DIR/${TASK}_${optimizer}_dropout${dropout}_seed${seed}_${ARCHITECTURE}.out"
            error_file="$RESULTS_DIR/${TASK}_${optimizer}_dropout${dropout}_seed${seed}_${ARCHITECTURE}.err"
            
            # Run the experiment with separate stdout and stderr
            python -m ldru.training.example \
                --task "$TASK" \
                --seed "$seed" \
                --lr "$LEARNING_RATE" \
                --steps "$TRAINING_STEPS" \
                --architecture "$ARCHITECTURE" \
                --batch_size "$BATCH_SIZE" \
                --sequence_length "$SEQUENCE_LENGTH" \
                --embedding_dim "$EMBEDDING_DIM" \
                --num_layers "$NUM_LAYERS" \
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

echo "LDRU OPTIMIZER ABLATION SUMMARY" > "$summary_file"
echo "===============================" >> "$summary_file"
echo "Timestamp: $TIMESTAMP" >> "$summary_file"
echo "Architecture: $ARCHITECTURE" >> "$summary_file"
echo "Task: $TASK" >> "$summary_file"
echo "Learning Rate: $LEARNING_RATE" >> "$summary_file"
echo "Training Steps: $TRAINING_STEPS" >> "$summary_file"
echo "Total experiments: $total_experiments" >> "$summary_file"
echo "" >> "$summary_file"

echo "Ablation Parameters:" >> "$summary_file"
echo "-------------------" >> "$summary_file"
echo "Optimizers: ${OPTIMIZERS[*]}" >> "$summary_file"
echo "Dropout Probabilities: ${DROPOUT_PROBS[*]}" >> "$summary_file"
echo "Seeds: ${SEEDS[*]}" >> "$summary_file"
echo "" >> "$summary_file"

echo "Results by Configuration and Seed:" >> "$summary_file"
echo "----------------------------------" >> "$summary_file"

for optimizer in "${OPTIMIZERS[@]}"; do
    for dropout in "${DROPOUT_PROBS[@]}"; do
        echo "" >> "$summary_file"
        echo "Configuration: $optimizer, Dropout=$dropout" >> "$summary_file"
        
        for seed in "${SEEDS[@]}"; do
            output_file="$RESULTS_DIR/${TASK}_${optimizer}_dropout${dropout}_seed${seed}_${ARCHITECTURE}.out"
            error_file="$RESULTS_DIR/${TASK}_${optimizer}_dropout${dropout}_seed${seed}_${ARCHITECTURE}.err"
            
            if [ -f "$output_file" ]; then
                score=$(grep "OOD accuracy:" "$output_file" | tail -1 | awk '{print $3}')
                if [ -n "$score" ]; then
                    echo "  Seed $seed: $score" >> "$summary_file"
                else
                    echo "  Seed $seed: FAILED (check $error_file)" >> "$summary_file"
                fi
            else
                echo "  Seed $seed: NO OUTPUT FILE" >> "$summary_file"
            fi
        done
    done
done

# Generate statistics by configuration
echo "" >> "$summary_file"
echo "Statistics by Configuration:" >> "$summary_file"
echo "----------------------------" >> "$summary_file"

for optimizer in "${OPTIMIZERS[@]}"; do
    for dropout in "${DROPOUT_PROBS[@]}"; do
        scores=()
        for seed in "${SEEDS[@]}"; do
            output_file="$RESULTS_DIR/${TASK}_${optimizer}_dropout${dropout}_seed${seed}_${ARCHITECTURE}.out"
            if [ -f "$output_file" ]; then
                score=$(grep "OOD accuracy:" "$output_file" | tail -1 | awk '{print $3}')
                if [ -n "$score" ]; then
                    scores+=("$score")
                fi
            fi
        done
        
        if [ ${#scores[@]} -gt 0 ]; then
            # Calculate mean and std using awk
            stats=$(printf '%s\n' "${scores[@]}" | awk '
            {
                sum += $1
                sumsq += $1*$1
                count++
                values[count] = $1
            }
            END {
                if(count > 0) {
                    mean = sum/count
                    if(count > 1) {
                        variance = (sumsq - sum*sum/count)/(count-1)
                        std = sqrt(variance)
                    } else {
                        std = 0
                    }
                    printf "%.6f ± %.6f", mean, std
                } else {
                    print "N/A"
                }
            }')
            echo "  $optimizer, Dropout=$dropout: $stats (${#scores[@]}/${#SEEDS[@]} successful runs)" >> "$summary_file"
        else
            echo "  $optimizer, Dropout=$dropout: N/A (0/${#SEEDS[@]} successful runs)" >> "$summary_file"
        fi
    done
done

# Compare optimizers
echo "" >> "$summary_file"
echo "Comparison by Optimizer (averaged across dropout values):" >> "$summary_file"
echo "--------------------------------------------------------" >> "$summary_file"

for optimizer in "${OPTIMIZERS[@]}"; do
    all_scores=()
    for dropout in "${DROPOUT_PROBS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            output_file="$RESULTS_DIR/${TASK}_${optimizer}_dropout${dropout}_seed${seed}_${ARCHITECTURE}.out"
            if [ -f "$output_file" ]; then
                score=$(grep "OOD accuracy:" "$output_file" | tail -1 | awk '{print $3}')
                if [ -n "$score" ]; then
                    all_scores+=("$score")
                fi
            fi
        done
    done
    
    if [ ${#all_scores[@]} -gt 0 ]; then
        # Calculate mean and std using awk
        stats=$(printf '%s\n' "${all_scores[@]}" | awk '
        {
            sum += $1
            sumsq += $1*$1
            count++
        }
        END {
            if(count > 0) {
                mean = sum/count
                if(count > 1) {
                    variance = (sumsq - sum*sum/count)/(count-1)
                    std = sqrt(variance)
                } else {
                    std = 0
                }
                printf "%.6f ± %.6f", mean, std
            } else {
                print "N/A"
            }
        }')
        expected_runs=$((${#DROPOUT_PROBS[@]} * ${#SEEDS[@]}))
        echo "  $optimizer: $stats (${#all_scores[@]}/$expected_runs successful runs)" >> "$summary_file"
    else
        expected_runs=$((${#DROPOUT_PROBS[@]} * ${#SEEDS[@]}))
        echo "  $optimizer: N/A (0/$expected_runs successful runs)" >> "$summary_file"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "Summary report saved to: $summary_file" | tee -a "$LOG_FILE"
echo "All individual results saved in: $RESULTS_DIR" | tee -a "$LOG_FILE"