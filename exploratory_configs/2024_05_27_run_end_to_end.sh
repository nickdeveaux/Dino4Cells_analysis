#!/bin/bash
source ~/miniconda3/bin/activate
conda activate dino4cells
export CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=0 python /mnt/vast/hpc/homes/ndv2107/Dino4Cells_analysis/run_end_to_end.py --config /mnt/vast/hpc/homes/ndv2107/Dino4Cells_analysis/exploratory_configs/2024_05_27_config_HPA_FOV_rerun_end_to_end.yaml 2>&1 | tee log.txt
