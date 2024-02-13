#!/bin/bash
#SBATCH --job-name=ConvNeXt         # job name
#SBATCH --partition=short           # nodes on gpu_p1 have 4 GPUs each
#SBATCH --ntasks=40                 # number of MPI task
#SBATCH --ntasks-per-node=4         # number of MPI task per node
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --cpus-per-task=1           # number of CPUs per task
#SBATCH --time=00:20:00             # job length
#SBATCH --output=train.out          # std out
#SBATCH --error=train.out           # std err
#SBATCH --exclusive                 # we reserve the entire node four our job
#SBATCH --reservation=CHPS          # (Optional) Used Slurm reservation

# Re-exports environment variables
# export NCCL_DEBUG=INFO                   # Enable NCCL debug messages
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH  # Preserve existing LD_LIBRARY_PATH
# export PATH=$PATH                        # Preserve existing PATH

# Modules
source env.sh

# Show commands
set -x

# Launch the container on each node
mpirun \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x PATH -x LD_LIBRARY_PATH \
    -mca pml ob1 -mca btl ^openib \
    singularity exec --nv swarm.sif python main.py \
        --model convnext_tiny --drop_path 0.1 \
        --batch_size 32 --lr 4e-3 --update_freq 16 \
        --warmup_epochs 5 --epochs 90 \
        --data_set CIFAR --nb_classes 100 --disable_eval true \
        --data_path datasets/ \
        --output_dir results/