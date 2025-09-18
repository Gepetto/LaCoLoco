# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# Copyright (c) 2025, LAAS-CNRS
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# based on https://github.com/isaac-sim/IsaacLab

import gymnasium as gym

from . import agents
from ase_envs.tasks.utils.cat.cat_env import CaTEnv
from ase_envs.tasks.utils.envs.ase_env import ASEEnv

##
# Register Gym environments.
##


gym.register(
    id="Manipulate-H1",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_manipulation_env_cfg:H1ManipulateEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_manipulate_h1.yaml",
    },
)


gym.register(
    id="Manipulate-H1-Play",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_manipulation_env_cfg:H1ManipulateEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_manipulate_h1.yaml",
    },
)
