#PBS -l walltime=26:30:00
#PBS -l select=1:ncpus=32:mem=128gb
#PBS -o logs/
#PBS -e logs/
#PBS -N pretok50

module purge
module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0
source ~/venv/ldru-venv/bin/activate

cd \$PBS_O_WORKDIR
cd LDRU

python open_webtext_scripts/pretokenize_sequences.py \
  --train_text data/openwebtext2/prepared/openwebtext2_train.txt \
  --val_text data/openwebtext2/prepared/openwebtext2_val.txt \
  --test_text data/openwebtext2/prepared/openwebtext2_test.txt \
  --tokenizer_type sentencepiece \
  --vocab_size 50000 \
  --append_eos \
  --out_dir data/pretokenized/openwebtext2_sp_50 \
  --basename owt2_sp_50_2k \
  --train_sentencepiece