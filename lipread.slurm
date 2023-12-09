#!/bin/bash

##SBATCH --gres=gpu:1
##SBATCH --gres=gpu:v100:1
##SBATCH --time=12:00:00
##SBATCH --output=demo_%j.out
##SBATCH --error=demo_%j.err
##SBATCH --exclusive
##SBATCH --requeue

##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
##SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=20GB
#SBATCH --output=demo_%j.out
#SBATCH --error=demo_%j.err
#SBATCH --job-name=myTest

singularity exec --nv \
--bind /scratch \
--overlay /scratch/sb9509/overlay-50G-10M.ext3:ro \
/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
/bin/bash -c "

source /ext3/env.sh
cd /scratch/sb9509/Lip_Reading_Project
python lipreading.py"
