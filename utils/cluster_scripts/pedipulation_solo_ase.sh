#!/bin/bash
#SBATCH --job-name=solo_ase_drail
#SBATCH --output=run_logs/solo_ase_drail%j.out
#SBATCH --error=run_logs/solo_ase_drail%j.err

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

SEED=19
EXPERIMENT_ID=pedipulation_2

python ase/run.py --task Solo-ASE --motion_file ase/data/motions/solo/dataset.yaml --headless --seed=$SEED --experiment_name="solo_ase" --experiment_id="$EXPERIMENT_ID"

python ase/run.py --task Solo-Pedipulation-HRL-ASE --motion_file ase/data/motions/solo/dataset.yaml --headless --seed=$SEED --llc_checkpoint="logs/rl_games/solo_ase/solo_ase_$EXPERIMENT_ID/nn/solo_ase.pth" --experiment_name="solo_hrl_ase" --experiment_id="$EXPERIMENT_ID"
