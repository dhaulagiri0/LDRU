#!/bin/bash
#SBATCH --job-name=opw-train-tf-alibi
#SBATCH --output=opw-train-tf-alibi_0.out
#SBATCH --error=opw-train-tf-alibi_0.err
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --partition=compute
#SBATCH --gres=gpu:1

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate agldru

cd ~/Projects/AG-LDRU/LDRU

MODE=${MODE}
DATASET=2
BATCH=32
VOCAB_SIZE=50000
RUN_NUM=1
EXPERIMENT_NAME=openwebtext_tied_v${VOCAB_SIZE}_${RUN_NUM}_ALiBi
TF_LOGDIR=tensorboard_logs/${EXPERIMENT_NAME}
mkdir -p $TF_LOGDIR

for MAX_LEN in 512
do
    TEXT_LOGDIR=logs/${EXPERIMENT_NAME}/seq${MAX_LEN}
    mkdir -p $TEXT_LOGDIR

    EXTRA_ARGS=(
        --hidden_dim 768
        --dropout_prob 0.15
        --lr 2e-4
        --l2_lambda 1e-5
        --transformer
        --use_alibi
        --num_transformer_layers 8
        --num_transformer_heads 12
    )
    COMMON_ARGS=(
        --embedding_dim 768
        --max_vocab_size $VOCAB_SIZE
        --model_name_prefix ${EXPERIMENT_NAME}_${MODE}_seq${MAX_LEN}
        --batch_size $BATCH
        --max_seq_len ${MAX_LEN}
        --print_log_file $TEXT_LOGDIR/tf_logs_${RUN_NUM}.txt
        --tensorboard_log_dir $TF_LOGDIR
        --streaming_chunk_line_buffer 2048
        --streaming_shuffle_buffer_size 8192
        --optimizer adamw
        --compute_dtype bfloat16
        --train_steps_per_epoch 1000
        --validation_steps_per_epoch 100
        --test_steps_per_epoch 100
        --train_stride $((MAX_LEN / 2))
        --train_seq_bin data/pretokenized/openwebtext2_sp_50/owt2_sp_50_2k_train.bin
        --val_seq_bin data/pretokenized/openwebtext2_sp_50/owt2_sp_50_2k_val.bin
        --test_seq_bin data/pretokenized/openwebtext2_sp_50/owt2_sp_50_2k_test.bin
        --seq_meta_json data/pretokenized/openwebtext2_sp_50/owt2_sp_50_2k_meta.json
        --seq_bin_format token_stream
        --seq_bin_dtype uint16
        --tie_embeddings_transformer
        --nanogpt_ppl_metric
    )

    echo "Running: python train_causal_ldru.py ${COMMON_ARGS[@]} ${EXTRA_ARGS[@]}"
    python train_causal_ldru.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
done