#PBS -l walltime=8:00:00
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -o logs/
#PBS -e logs/
#PBS -N opw-ldru-optuna

module purge
module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0
source ~/venv/ldru-venv/bin/activate

cd "$PBS_O_WORKDIR"
cd LDRU

MODE=${MODE:-train}
BATCH=${BATCH:-32}
VOCAB_SIZE=${VOCAB_SIZE:-50000}
RUN_NUM=${RUN_NUM:-1}
NUM_TRIALS=${NUM_TRIALS:-30}
MAX_LENS=${MAX_LENS:-512}
TARGET_PARAM_COUNT=${TARGET_PARAM_COUNT:-100000000}
PARAM_TOL_RATIO=${PARAM_TOL_RATIO:-0.10}
STORAGE_URL=${STORAGE_URL:-sqlite:///optuna_openwebtext_study.db}
BASE_STUDY_NAME=${BASE_STUDY_NAME:-openwebtext_ldru_scan_v${VOCAB_SIZE}_r${RUN_NUM}}
EXPERIMENT_NAME=openwebtext_optuna_tied_v${VOCAB_SIZE}_${RUN_NUM}_GRC
EMBEDDING_DIM_CANDIDATES=${EMBEDDING_DIM_CANDIDATES:-512,640,768,896,1024}
HIDDEN_DIM_CANDIDATES=${HIDDEN_DIM_CANDIDATES:-640,768,896,1024,1152}
LR_CANDIDATES=${LR_CANDIDATES:-1e-4,2e-4,3e-4,5e-4,6e-4}
WARMUP_ENABLED_CANDIDATES=${WARMUP_ENABLED_CANDIDATES:-false,true}
WARMUP_STEPS_CANDIDATES=${WARMUP_STEPS_CANDIDATES:-3000}
PRENORM_GELU_CANDIDATES=${PRENORM_GELU_CANDIDATES:-true,false}
TIE_EMBEDDINGS_CANDIDATES=${TIE_EMBEDDINGS_CANDIDATES:-true,false}
USE_MULTI_OPERATOR_CANDIDATES=${USE_MULTI_OPERATOR_CANDIDATES:-true,false}
NUM_OPERATORS_CANDIDATES=${NUM_OPERATORS_CANDIDATES:-16,8,4}
OPERATOR_MIN_WEIGHT_CANDIDATES=${OPERATOR_MIN_WEIGHT_CANDIDATES:-0.01}

TRAIN_SEQ_BIN=${TRAIN_SEQ_BIN:-data/pretokenized/openwebtext2_new/owt2_gpt2_new_train.bin}
VAL_SEQ_BIN=${VAL_SEQ_BIN:-data/pretokenized/openwebtext2_new/owt2_gpt2_new_val.bin}
TEST_SEQ_BIN=${TEST_SEQ_BIN:-data/pretokenized/openwebtext2_new/owt2_gpt2_new_test.bin}
SEQ_META_JSON=${SEQ_META_JSON:-data/pretokenized/openwebtext2_new/owt2_gpt2_new_meta.json}

for MAX_LEN in $MAX_LENS
do
    LOGDIR=logs/${EXPERIMENT_NAME}/seq${MAX_LEN}
    CKPT_DIR=optuna_checkpoints/${EXPERIMENT_NAME}/seq${MAX_LEN}
    TBOARD_DIR=tensorboard_logs/${EXPERIMENT_NAME}/seq${MAX_LEN}
    mkdir -p "$LOGDIR" "$CKPT_DIR" "$TBOARD_DIR"

    STUDY_NAME=${BASE_STUDY_NAME}_seq${MAX_LEN}

    ARGS=(
        --num_trials "$NUM_TRIALS"
        --study_name "$STUDY_NAME"
        --storage_url "$STORAGE_URL"
        --dataset_mode pretokenized_bin
        --model_type causal_ldru
        --max_vocab_size "$VOCAB_SIZE"
        --batch_size "$BATCH"
        --max_seq_len "$MAX_LEN"
        --epochs_per_trial 20
        --model_name_prefix "${EXPERIMENT_NAME}_${MODE}_seq${MAX_LEN}"
        --print_log_file "$LOGDIR/optuna_scan_${RUN_NUM}.txt"
        --tensorboard_log_dir "$TBOARD_DIR"
        --checkpoint_dir "$CKPT_DIR"
        --streaming_chunk_line_buffer 2048
        --streaming_shuffle_buffer_size 8192
        --optimizer adamw
        --compute_dtype bfloat16
        --train_steps_per_epoch 1000
        --validation_steps_per_epoch 100
        --test_steps_per_epoch 100
        --train_stride $((MAX_LEN / 2))
        --train_seq_bin "$TRAIN_SEQ_BIN"
        --val_seq_bin "$VAL_SEQ_BIN"
        --test_seq_bin "$TEST_SEQ_BIN"
        --seq_meta_json "$SEQ_META_JSON"
        --seq_bin_format token_stream
        --seq_bin_dtype uint16
        --binary_operator grc
        --nanogpt_ppl_metric
        --embedding_dim_candidates "$EMBEDDING_DIM_CANDIDATES"
        --hidden_dim_candidates "$HIDDEN_DIM_CANDIDATES"
        --initial_learning_rate_candidates "$LR_CANDIDATES"
        --warmup_enabled_candidates "$WARMUP_ENABLED_CANDIDATES"
        --warmup_steps_candidates "$WARMUP_STEPS_CANDIDATES"
        --ldru_prenorm_gelu_block_candidates "$PRENORM_GELU_CANDIDATES"
        --tie_embeddings_ldru_candidates "$TIE_EMBEDDINGS_CANDIDATES"
        --use_multi_operator_ldru_candidates "$USE_MULTI_OPERATOR_CANDIDATES"
        --num_operators_candidates "$NUM_OPERATORS_CANDIDATES"
        --operator_min_weight_candidates "$OPERATOR_MIN_WEIGHT_CANDIDATES"
        --target_param_count "$TARGET_PARAM_COUNT"
        --param_count_tolerance_ratio "$PARAM_TOL_RATIO"
    )

    echo "Running: python hyperparam_search.py ${ARGS[*]}"
    python hyperparam_search.py "${ARGS[@]}"
done
