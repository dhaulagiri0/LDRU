#PBS -l walltime=2:00:00
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -o logs/
#PBS -e logs/
#PBS -N cp-opw-eval

set -euo pipefail

module purge
module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0
source ~/venv/ldru-venv/bin/activate

cd "$PBS_O_WORKDIR"
cd LDRU

CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints/openwebtext_tied_v50000_10_GRC__seq512_model_ldru_seq2seq_silu_default_512_SP}

BATCH=${BATCH:-32}
MAX_LEN=${MAX_LEN:-512}
STRIDE=${STRIDE:-$((MAX_LEN / 2))}
RUN_NUM=${RUN_NUM:-1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-openwebtext_eval_ldru}
TEXT_LOGDIR=logs/${EXPERIMENT_NAME}/seq${MAX_LEN}
mkdir -p "$TEXT_LOGDIR"

TEST_SEQ_BIN=${TEST_SEQ_BIN:-data/pretokenized/openwebtext2_new/owt2_gpt2_new_test.bin}
SEQ_META_JSON=${SEQ_META_JSON:-data/pretokenized/openwebtext2_new/owt2_gpt2_new_meta.json}
SEQ_BIN_FORMAT=${SEQ_BIN_FORMAT:-token_stream}
SEQ_BIN_DTYPE=${SEQ_BIN_DTYPE:-uint16}
NANOGPT_PPL_METRIC=${NANOGPT_PPL_METRIC:-1}
NANOGPT_BATCHING=${NANOGPT_BATCHING:-0}

ARGS=(
    --evaluate_pretok "$CHECKPOINT_DIR"
    --test_seq_bin "$TEST_SEQ_BIN"
    --seq_meta_json "$SEQ_META_JSON"
    --seq_bin_format "$SEQ_BIN_FORMAT"
    --seq_bin_dtype "$SEQ_BIN_DTYPE"
    --seq_length "$MAX_LEN"
    --eval_stride "$STRIDE"
    --batch_size "$BATCH"
    --print_log_file "$TEXT_LOGDIR/opw_eval_${RUN_NUM}.txt"
)

if [ "$NANOGPT_PPL_METRIC" = "1" ]; then
    ARGS+=(--nanogpt_ppl_metric)
fi

if [ "$NANOGPT_BATCHING" = "1" ]; then
    ARGS+=(--nanogpt_batching)
fi

echo "Running: python train_causal_ldru.py ${ARGS[*]}"
python train_causal_ldru.py "${ARGS[@]}"
