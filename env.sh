echo "Loading required modules..."
module purge
module load cuda/11.7
module load singularity
module load gcc/10.2
module load openmpi/4.0.5.1-gnu10.2.0
module list
echo "Done."