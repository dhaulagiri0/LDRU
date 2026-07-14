#!/usr/bin/env bash
#PBS -N openwebtext2
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=4:mem=20gb
#PBS -o logs/
#PBS -e logs/

module purge
module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0
source ~/venv/ldru-venv/bin/activate
cd $PBS_O_WORKDIR
cd LDRU

set -euo pipefail

python open_webtext_scripts/download_prepare_openwebtext2.py \
  --source_backend hf \
  --hf_dataset segyges/OpenWebText2 \
  --hf_split train \
  --output_dir data/openwebtext2/prepared \
  --overwrite_outputs