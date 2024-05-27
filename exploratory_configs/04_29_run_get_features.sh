#!/bin/bash
source ~/miniconda3/bin/activate
conda activate dino4cells
export CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=0 python /mnt/vast/hpc/homes/ndv2107/ndv_dino4cells_analysis/run_get_features.py --config /mnt/vast/hpc/homes/ndv2107/ndv_dino4cells_analysis/exploratory_configs/config_HPA_FOV_4channel_create_1channel_features_from_train_data_04_14.yaml 2>&1 | tee log.txt
