#!/bin/bash
#SBATCH --job-name=h1_ase
#SBATCH --output=h1_ase.out
#SBATCH --error=h1_ase.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread

#SBATCH --time=10:00:00 # maximum execution time (HH:MM:SS)

# values for seeds (42, 73, 19, 7929, 6177) (--video was additionally used with 42)

module purge
module load miniforge
conda activate isaaclab

set -x

SEED=6177
EXPERIMENT_ID=manipulation_4

python ase/run.py --task H1-ASE-CaT --motion_file ase/data/motions/h1/dataset.yaml --headless --seed=$SEED --experiment_name="h1_ase_cat" --experiment_id="$EXPERIMENT_ID" --checkpoint="logs/rl_games/h1_ase/h1_ase_$EXPERIMENT_ID/nn/h1_ase_00005000.pth"

python ase/run.py --task H1-Manipulation-HRL-ASE-CaT --motion_file ase/data/motions/h1/dataset.yaml --headless --seed=$SEED --llc_checkpoint="logs/rl_games/h1_ase_cat/h1_ase_cat_$EXPERIMENT_ID/nn/h1_ase_cat.pth" --experiment_name="h1_hrl_ase_cat" --experiment_id="$EXPERIMENT_ID"
