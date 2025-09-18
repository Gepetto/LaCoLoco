#!/bin/bash
#SBATCH --job-name=solo_ase
#SBATCH --output=run_logs/solo_ase%j.out
#SBATCH --error=run_logs/solo_ase%j.err

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

SEED=73
EXPERIMENT_ID=dog_imitation_1

python ase/run.py --task Solo-ASE-CaT --motion_file ase/data/motions/solo_mocap/dataset_valid.yaml --headless --seed=$SEED --experiment_name="solo_ase_cat" --experiment_id="$EXPERIMENT_ID" --checkpoint="logs/rl_games/solo_ase/solo_ase_$EXPERIMENT_ID/nn/solo_ase_00005000.pth" env.events.reset_from_dataset.params.set_initial_velocities_from_dataset=False

python ase/run.py --task Solo-Pedipulation-HRL-ASE-CaT --motion_file ase/data/motions/solo_mocap/dataset_valid.yaml --headless --seed=$SEED --llc_checkpoint="logs/rl_games/solo_ase_cat/solo_ase_cat_$EXPERIMENT_ID/nn/solo_ase_cat.pth" --experiment_name="solo_hrl_ase_cat" --experiment_id="$EXPERIMENT_ID"
