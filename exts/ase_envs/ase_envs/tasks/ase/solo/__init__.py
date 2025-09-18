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

# Low level controllers

gym.register(
    id="Solo-ASE",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_env_ase_cfg:Solo12_ASE",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ase_solo.yaml",
    },
)

gym.register(
    id="Solo-ASE-Play",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_env_ase_cfg:Solo12_ASE_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ase_solo.yaml",
    },
)

gym.register(
    id="Solo-ASE-CaT",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_env_ase_cfg:Solo12_ASE_CaT",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ase_solo.yaml",
    },
)

gym.register(
    id="Solo-ASE-CaT-Play",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_env_ase_cfg:Solo12_ASE_CaT_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ase_solo.yaml",
    },
)

gym.register(
    id="Solo-ASE-DRAIL",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_env_ase_cfg:Solo12_ASE",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ase_drail_solo.yaml",
    },
)

gym.register(
    id="Solo-ASE-DRAIL-Play",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_env_ase_cfg:Solo12_ASE_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ase_drail_solo.yaml",
    },
)

gym.register(
    id="Solo-ASE-DRAIL-CaT",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_env_ase_cfg:Solo12_ASE_CaT",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ase_drail_solo.yaml",
    },
)

gym.register(
    id="Solo-ASE-DRAIL-CaT-Play",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_env_ase_cfg:Solo12_ASE_CaT_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ase_drail_solo.yaml",
    },
)

gym.register(
    id="Solo-ASE-DRAIL-Clipping",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_env_ase_cfg:Solo12_ASE_Clipping",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ase_drail_solo.yaml",
    },
)

gym.register(
    id="Solo-ASE-DRAIL-CaT-Clipping",
    entry_point=ASEEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo_env_ase_cfg:Solo12_ASE_CaT_Clipping",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ase_drail_solo.yaml",
    },
)
