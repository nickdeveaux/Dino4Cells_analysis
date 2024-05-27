#!/bin/bash
#SBATCH --job-name=dino_training
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=GEN-GPU
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

# Load necessary modules
module load anaconda/3
source activate dino4cells

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0

# Run the training script using torchrun
python3 -m torch.distributed.launch --master_port=29501 --nproc_per_node=1 main_dino.py --arch vit_base --output_dir output/ --data_path /mnt/vast/hpc/LDB/LDB_data/microscopy/HPA/whole_images/Master_fovHPA_512.csv  --saveckp_freq 50 --batch_size_per_gpu 24 --num_channels 5 --patch_size 16 --local_crops_scale 0.2 0.5 --epochs 10 --config /mnt/vast/hpc/homes/ndv2107/Dino4Cells_analysis/exploratory_configs/2024_05_27_config_HPA_FOV_rerun.yaml --center_momentum 0.9 --lr 0.0005
