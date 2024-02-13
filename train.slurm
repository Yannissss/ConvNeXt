#!/bin/bash
#SBATCH --job-name=ConvNeXt         # job name
#SBATCH --partition=long            # nodes on gpu_p1 have 4 GPUs each
#SBATCH --ntasks=16                 # number of MPI task
#SBATCH --ntasks-per-node=4         # number of MPI task per node
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --cpus-per-task=2           # number of CPUs per task
#SBATCH --time=28:00:00             # job length
#SBATCH --output=train.out          # std out
#SBATCH --error=train.out           # std err
#NOSBATCH --exclusive                 # we reserve the entire node four our job
#NOSBATCH --reservation=CHPS        # (Optional) Used Slurm reservation

# Re-exports environment variables
# export NCCL_DEBUG=INFO                   # Enable NCCL debug messages
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH  # Preserve existing LD_LIBRARY_PATH
# export PATH=$PATH                        # Preserve existing PATH

# Modules
echo "Loading required modules..."
module purge
module load cuda/11.7
module load singularity
module load gcc/10.2
module load openmpi/4.0.5.1-gnu10.2.0
module list
echo "Done."

# Show commands
set -x

# Launch the container on each node
mpirun \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x PATH -x LD_LIBRARY_PATH \
    -mca pml ob1 -mca btl ^openib \
    singularity exec --nv swarm.sif python main.py \
        --batch_size 4096 \
        --epochs 300 \
        --drop_path 0.4 \
        --update_freq 16 \
        --model convnext_base \
        --model_ema True \
        --model_ema_decay 0.9999 \
        --model_ema_force_cpu True \
        --model_ema_eval False \
        --opt adamw \
        --data_set IMNET \
        --data_path datasets/ImageNet-1K \
        --imagenet_default_mean_and_std True \
        --data_set IMNET \
        --output_dir 'training_base_IMNET-1K' \
        --disable_eval False \
        --enable_wandb False \