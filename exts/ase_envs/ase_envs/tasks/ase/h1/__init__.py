# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# Copyright (c) 2025, LAAS-CNRS
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# based on https://github.com/isaac-sim/IsaacLab

import gymnasium as gym

from . import agents
from ase_envs.tasks.utils.envs.ase_env import ASEEnv

##
# Register Gym environments.
##

# H1 low level controller

gym.register(
    id="H1-ASE-DRAIL",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_env_ase_cfg:H1_ASE",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ase_drail_h1.yaml",
    },
)

gym.register(
    id="H1-ASE-DRAIL-Play",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_env_ase_cfg:H1_ASE_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ase_drail_h1.yaml",
    },
)

gym.register(
    id="H1-ASE-DRAIL-CaT",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_env_ase_cfg:H1_ASE_CaT",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ase_drail_h1.yaml",
    },
)

gym.register(
    id="H1-ASE-DRAIL-CaT-Play",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_env_ase_cfg:H1_ASE_CaT_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ase_drail_h1.yaml",
    },
)

gym.register(
    id="H1-ASE",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_env_ase_cfg:H1_ASE",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ase_h1.yaml",
    },
)

gym.register(
    id="H1-ASE-Play",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_env_ase_cfg:H1_ASE_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ase_h1.yaml",
    },
)

gym.register(
    id="H1-ASE-CaT",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_env_ase_cfg:H1_ASE_CaT",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ase_h1.yaml",
    },
)

gym.register(
    id="H1-ASE-CaT-Play",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_env_ase_cfg:H1_ASE_CaT_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ase_h1.yaml",
    },
)
