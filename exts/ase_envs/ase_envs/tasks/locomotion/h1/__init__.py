# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, LAAS-CNRS
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# based on https://github.com/isaac-sim/IsaacLab

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="H1-Velocity-Flat",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_flat_env_cfg:H1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_h1:H1FlatPPORunnerCfg",
    },
)


gym.register(
    id="H1-Velocity-Flat-Play",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_flat_env_cfg:H1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_h1:H1FlatPPORunnerCfg",
    },
)
