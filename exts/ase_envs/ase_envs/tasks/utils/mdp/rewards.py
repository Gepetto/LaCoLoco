# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# Copyright (c) 2025, LAAS-CNRS
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# based on https://github.com/isaac-sim/IsaacLab

import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import RigidObject
import omni.isaac.lab_tasks.manager_based.classic.humanoid.mdp as humanoid_mdp
from omni.isaac.lab.utils.math import quat_rotate


def position_command_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    eef_offset: float = 0.0,
    quadratic_distance: bool = False,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    des_pos_w = env.command_manager.get_term(command_name).command_w

    body_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    body_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]

    if abs(eef_offset) > 0.0:
        offset_local = torch.tensor([eef_offset, 0.0, 0.0], device=body_pos_w.device)
        offset_local_expanded = offset_local.unsqueeze(0).expand_as(body_pos_w)
        offset_world = quat_rotate(body_quat_w, offset_local_expanded)
        new_body_pos_w = body_pos_w + offset_world
    else:
        new_body_pos_w = body_pos_w

    if quadratic_distance:
        distance = ((new_body_pos_w - des_pos_w) ** 2).sum(dim=1)
    else:
        distance = torch.norm(new_body_pos_w - des_pos_w, dim=1)

    return torch.exp(-distance / std**2)


def move_to_target_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving to the target heading."""
    des_pos_w = env.command_manager.get_term(command_name).command_w
    heading_proj = humanoid_mdp.observations.base_heading_proj(
        env, des_pos_w, asset_cfg
    ).squeeze(-1)
    # return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)
    return torch.exp((heading_proj - 1.0) / std**2)
