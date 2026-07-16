#PBS -l walltime=5:30:00
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -o logs/
#PBS -e logs/
#PBS -N cp-opw-pretok

module purge
module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0
source ~/venv/ldru-venv/bin/activate

cd $PBS_O_WORKDIR
cd LDRU

MODE=${MODE}
DATASET=2
BATCH=32
VOCAB_SIZE=50000
RUN_NUM=1
EXPERIMENT_NAME=openwebtext_tied_v${VOCAB_SIZE}_${RUN_NUM}_GRC
TF_LOGDIR=tensorboard_logs/${EXPERIMENT_NAME}
mkdir -p $TF_LOGDIR

for MAX_LEN in 512
do
    TEXT_LOGDIR=logs/${EXPERIMENT_NAME}/seq${MAX_LEN}
    mkdir -p $TEXT_LOGDIR

    EXTRA_ARGS=(
        --num_layers 1
        --hidden_dim 1028
        --dropout_prob 0.15
        --lr 2e-4
        --l2_lambda 1e-5
        --binary_operator grc
    )
    COMMON_ARGS=(
        --embedding_dim 768
        --max_vocab_size $VOCAB_SIZE
        --model_name_prefix ${EXPERIMENT_NAME}_${MODE}_seq${MAX_LEN}
        --batch_size $BATCH
        --max_seq_len ${MAX_LEN}
        --print_log_file $TEXT_LOGDIR/ldru_logs_${RUN_NUM}.txt
        --tensorboard_log_dir $TF_LOGDIR
        --streaming_chunk_line_buffer 2048
        --streaming_shuffle_buffer_size 8192
        --optimizer adamw
        --compute_dtype bfloat16
        --train_steps_per_epoch 1000
        --validation_steps_per_epoch 100
        --test_steps_per_epoch 100
        --train_stride $((MAX_LEN / 2))
        --train_seq_bin data/pretokenized/openwebtext2_new/owt2_gpt2_new_train.bin
        --val_seq_bin data/pretokenized/openwebtext2_new/owt2_gpt2_new_val.bin
        --test_seq_bin data/pretokenized/openwebtext2_new/owt2_gpt2_new_test.bin
        --seq_meta_json data/pretokenized/openwebtext2_new/owt2_gpt2_new_meta.json
        --seq_bin_format token_stream
        --seq_bin_dtype uint16
        --tie_embeddings_ldru
        --nanogpt_ppl_metric
        --use_multi_operator_ldru
        --num_operators 8
        --ldru_prenorm_gelu_block
    )

    echo "Running: python train_causal_ldru.py ${COMMON_ARGS[@]} ${EXTRA_ARGS[@]}"
    python train_causal_ldru.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
done