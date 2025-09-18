# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# Copyright (c) 2025, LAAS-CNRS
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# based on https://github.com/isaac-sim/IsaacLab
# based on https://github.com/Gepetto/constraints-as-terminations

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def modify_constraint_p(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_names: Sequence[str],
    num_steps: int,
    init_max_p: float,
):
    if env.common_step_counter <= env.cfg.constraints_num_steps_start:
        init_max_p = 0.0
    else:
        progress = min(
            (env.common_step_counter - env.cfg.constraints_num_steps_start) / num_steps,
            1.0,
        )
        # Linearly interpolate the expected time for episode end: soft_p is the maximum
        # termination probability so it is an image of the expected time of death.
        T_start = 20
        T_end = 1 / init_max_p
        init_max_p = 1 / (T_start + progress * (T_end - T_start))

    for name in term_names:
        term_cfg = env.constraint_manager.get_term_cfg(name)
        term_cfg.max_p = init_max_p
        env.constraint_manager.set_term_cfg(name, term_cfg)

    return init_max_p


def modify_constraint_p_linear(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_names: Sequence[str],
    num_steps: int,
    init_max_p: float,
):
    if env.common_step_counter <= env.cfg.constraints_num_steps_start:
        init_max_p = 0.0
    else:
        init_max_p = (
            min(
                (env.common_step_counter - env.cfg.constraints_num_steps_start)
                / num_steps,
                1.0,
            )
            * init_max_p
        )

    for name in term_names:
        term_cfg = env.constraint_manager.get_term_cfg(name)
        term_cfg.max_p = init_max_p
        env.constraint_manager.set_term_cfg(name, term_cfg)

    return init_max_p


def modify_constraint_p_quadratic(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_names: Sequence[str],
    num_steps: int,
    init_max_p: float,
):
    if env.common_step_counter <= env.cfg.constraints_num_steps_start:
        init_max_p = 0.0
    else:
        init_max_p = (
            min(
                (env.common_step_counter - env.cfg.constraints_num_steps_start)
                / num_steps,
                1.0,
            )
            ** 2
        ) * init_max_p

    for name in term_names:
        term_cfg = env.constraint_manager.get_term_cfg(name)
        term_cfg.max_p = init_max_p
        env.constraint_manager.set_term_cfg(name, term_cfg)

    return init_max_p
