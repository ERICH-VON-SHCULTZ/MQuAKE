#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=6:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=qwensft
#SBATCH --output=./qwensft.out
#SBATCH --error=./qwensft.err

module purge
cd /scratch/yw8866/MQuAKE
OVERLAY_FILE=/scratch/yw8866/sftenv/overlay-15GB-500K.ext3:rw
SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
singularity exec --nv \
        --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
        /bin/bash -c "source /ext3/env.sh; python sftqwen3.py"