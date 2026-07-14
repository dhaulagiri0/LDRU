# OpenWebText2 Data Pipeline and LDRU Training

## Prerequisites

All job bash scripts load the following environment:

```bash
module purge
module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0
source ~/venv/ldru-venv/bin/activate
```
Where ldru-venv contains all the necessary libraries found in requirements.txt

```bash
pip install -r requirements.txt
```

---

## 1. Download and Prepare OpenWebText2

```bash
python open_webtext_scripts/download_prepare_openwebtext2.py \
  --source_backend hf \
  --hf_dataset segyges/OpenWebText2 \
  --hf_split train \
  --output_dir data/openwebtext2/prepared \
  --overwrite_outputs
```

What this does:
- Downloads OpenWebText2 tar shards (default source is Hugging Face).
- Extracts and processes records.
- Writes:
  - `data/openwebtext2/prepared/openwebtext2_train.txt`
  - `data/openwebtext2/prepared/openwebtext2_val.txt`
  - `data/openwebtext2/prepared/openwebtext2_test.txt`
- `--hf_split train` is used because segyges/OpenWebText2 is effectively a single-source corpus on Hugging Face, and this pipeline deterministically creates local train/val/test splits from that source.

A bash script to run this is provided: open_webtext_scripts/get_opw.sh
---

## 2. Pretokenize the Prepared Text Files

Run:

```bash
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
```

What this does:
- Converts each split text file into `.bin` token streams for training.
- Writes:
  - `data/pretokenized/openwebtext2_sp_50/owt2_sp_50_2k_train.bin`
  - `data/pretokenized/openwebtext2_sp_50/owt2_sp_50_2k_val.bin`
  - `data/pretokenized/openwebtext2_sp_50/owt2_sp_50_2k_test.bin`
  - `data/pretokenized/openwebtext2_sp_50/owt2_sp_50_2k_meta.json`

A bash script to run this is provided: open_webtext_scripts/pretok_opw.sh

### SentencePiece note

When `--train_sentencepiece` is enabled, the script trains and outputs a sentencepiece model to:

`<out_dir>/<basename>_spm_vocab<vocab_size>.model`

For the command above, that is:

`data/pretokenized/openwebtext2_sp_50/owt2_sp_50_2k_spm_vocab50000.model`

This model is then used for pretokenisation.
The output: data/pretokenized/openwebtext2_sp_50/owt2_sp_50_2k_meta.json stores information about what tokenizer to use during training.

---

## 3. Train LDRU

Training job script:

`open_webtext_scripts/opw_train_ldru.sh`

Submit/run that script in your PBS environment. It launches `train_causal_ldru.py` with pretokenized `.bin` inputs and metadata. You might want to adjust the walltime accordingly.

Current script settings include:
- Batch size of 32
- Sequences are created with max_seq_len=512 and train_stride=256 (512/2), so each new training window starts 256 tokens after the previous one. This gives 50% overlap between consecutive windows.
- 1000 training steps and 100 validation steps
- Token-stream bin inputs (`--train_seq_bin`, `--val_seq_bin`, `--test_seq_bin`)
- Training uses bf16 datatype