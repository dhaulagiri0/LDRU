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

# Experiment configuration
SEEDS=(0 1 2)
TASKS=(
    "tomita_3"
    "tomita_4" 
    "tomita_5"
    "tomita_6"
    "tomita_7"
)


# Default values 
DEFAULT_LR=1e-4
DEFAULT_STEPS=100000

# Other experiment parameters 
ARCHITECTURE="transformer_encoder"
BATCH_SIZE=256
SEQUENCE_LENGTH=40
EMBEDDING_DIM=64
NUM_LAYERS=5
NUM_HEADS=8
DROPOUT_PROB=0.1
OPTIMIZER="amsgrad"
MAX_RANGE_TEST_LENGTH=500

# Create results directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="${ARCHITECTURE}_tomita"
mkdir -p "$RESULTS_DIR"

# Log file for the entire experiment
LOG_FILE="$RESULTS_DIR/experiment_log.txt"

echo "Starting experiment at $(date)" | tee "$LOG_FILE"
echo "Results will be saved in: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "Tasks: ${TASKS[*]}" | tee -a "$LOG_FILE"
echo "Seeds: ${SEEDS[*]}" | tee -a "$LOG_FILE"
echo "Architecture: $ARCHITECTURE" | tee -a "$LOG_FILE"
if [ -n "$GPU_ID" ]; then
    echo "GPU: $GPU_ID" | tee -a "$LOG_FILE"
fi
echo "----------------------------------------" | tee -a "$LOG_FILE"

# Counter for progress tracking
total_experiments=$((${#SEEDS[@]} * ${#TASKS[@]}))
current_experiment=0

# Main experiment loop
for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        current_experiment=$((current_experiment + 1))
        
        # Get task-specific parameters or use defaults
        lr=${LEARNING_RATES[$task]:-$DEFAULT_LR}
        steps=${TRAINING_STEPS[$task]:-$DEFAULT_STEPS}
        
        echo "[$current_experiment/$total_experiments] Running: Task=$task, Seed=$seed, LR=$lr, Steps=$steps" | tee -a "$LOG_FILE"
        
        # Create output filenames
        output_file="$RESULTS_DIR/${task}_seed${seed}_${ARCHITECTURE}.out"
        error_file="$RESULTS_DIR/${task}_seed${seed}_${ARCHITECTURE}.err"
        
        # Run the experiment with separate stdout and stderr
        python -m ldru.training.example \
            --task "$task" \
            --seed "$seed" \
            --lr "$lr" \
            --steps "$steps" \
            --architecture "$ARCHITECTURE" \
            --batch_size "$BATCH_SIZE" \
            --sequence_length "$SEQUENCE_LENGTH" \
            --embedding_dim "$EMBEDDING_DIM" \
            --num_layers "$NUM_LAYERS" \
            --num_heads "$NUM_HEADS" \
            --dropout_prob "$DROPOUT_PROB" \
            --optimizer "$OPTIMIZER" \
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

echo "Experiment completed at $(date)" | tee -a "$LOG_FILE"

# Generate summary report
echo "Generating summary report..." | tee -a "$LOG_FILE"
summary_file="$RESULTS_DIR/summary_report.txt"

echo "EXPERIMENT SUMMARY" > "$summary_file"
echo "==================" >> "$summary_file"
echo "Timestamp: $TIMESTAMP" >> "$summary_file"
echo "Architecture: $ARCHITECTURE" >> "$summary_file"
echo "Total experiments: $total_experiments" >> "$summary_file"
echo "" >> "$summary_file"

echo "Results by Task and Seed:" >> "$summary_file"
echo "------------------------" >> "$summary_file"

for task in "${TASKS[@]}"; do
    echo "" >> "$summary_file"
    echo "Task: $task" >> "$summary_file"
    echo "Learning Rate: ${LEARNING_RATES[$task]:-$DEFAULT_LR}" >> "$summary_file"
    echo "Training Steps: ${TRAINING_STEPS[$task]:-$DEFAULT_STEPS}" >> "$summary_file"
    
    for seed in "${SEEDS[@]}"; do
        output_file="$RESULTS_DIR/${task}_seed${seed}_${ARCHITECTURE}.out"
        error_file="$RESULTS_DIR/${task}_seed${seed}_${ARCHITECTURE}.err"
        
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

echo "" | tee -a "$LOG_FILE"
echo "Summary report saved to: $summary_file" | tee -a "$LOG_FILE"
echo "All individual results saved in: $RESULTS_DIR" | tee -a "$LOG_FILE"