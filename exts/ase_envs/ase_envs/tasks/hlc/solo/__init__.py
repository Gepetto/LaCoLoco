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

# High level controllers


gym.register(
    id="Solo-Pedipulation-HRL-ASE",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_env_hrl_cfg:Solo_Pedipulation_HRL_ASE",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_hrl_solo.yaml",
    },
)


gym.register(
    id="Solo-Pedipulation-HRL-ASE-Play",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_env_hrl_cfg:Solo_Pedipulation_HRL_ASE_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_hrl_solo.yaml",
    },
)

gym.register(
    id="Solo-Pedipulation-HRL-ASE-CaT",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_env_hrl_cfg:Solo_Pedipulation_HRL_ASE",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_hrl_solo.yaml",
    },
)


gym.register(
    id="Solo-Pedipulation-HRL-ASE-CaT-Play",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_env_hrl_cfg:Solo_Pedipulation_HRL_ASE_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_hrl_solo.yaml",
    },
)


gym.register(
    id="Solo-Pedipulation-HRL-ASE-DRAIL",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_env_hrl_cfg:Solo_Pedipulation_HRL_ASE",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_hrl_drail_solo.yaml",
    },
)

gym.register(
    id="Solo-Pedipulation-HRL-ASE-DRAIL-Play",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_env_hrl_cfg:Solo_Pedipulation_HRL_ASE_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_hrl_drail_solo.yaml",
    },
)

gym.register(
    id="Solo-Pedipulation-HRL-ASE-DRAIL-CaT",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_env_hrl_cfg:Solo_Pedipulation_HRL_ASE",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_hrl_drail_solo.yaml",
    },
)

gym.register(
    id="Solo-Pedipulation-HRL-ASE-DRAIL-CaT-Play",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_env_hrl_cfg:Solo_Pedipulation_HRL_ASE_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_hrl_drail_solo.yaml",
    },
)

gym.register(
    id="Solo-Pedipulation-HRL-ASE-DRAIL-Clipping",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_env_hrl_cfg:Solo_Pedipulation_HRL_ASE_Clipping",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_hrl_drail_solo.yaml",
    },
)
