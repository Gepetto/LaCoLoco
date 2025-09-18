# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# Copyright (c) 2025, LAAS-CNRS
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# based on https://github.com/isaac-sim/IsaacLab
# based on https://github.com/Gepetto/constraints-as-terminations

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def joint_position(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    cstr = torch.abs(data.joint_pos[:, asset_cfg.joint_ids]) - limit
    return cstr


def joint_position_min(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    cstr = limit - data.joint_pos[:, asset_cfg.joint_ids]
    return cstr


def joint_position_max(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    cstr = data.joint_pos[:, asset_cfg.joint_ids] - limit
    return cstr


def joint_position_when_moving_forward(
    env: ManagerBasedRLEnv,
    limit: float,
    velocity_deadzone: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    cstr = (
        torch.abs(
            data.joint_pos[:, asset_cfg.joint_ids]
            - data.default_joint_pos[:, asset_cfg.joint_ids]
        )
        - limit
    )
    cstr *= (
        (
            torch.abs(env.command_manager.get_command("base_velocity")[:, 1])
            < velocity_deadzone
        )
        .float()
        .unsqueeze(1)
    )
    return cstr


def joint_torque(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    cstr = torch.abs(data.applied_torque[:, asset_cfg.joint_ids]) - limit
    return cstr


def joint_velocity(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    return torch.abs(data.joint_vel[:, asset_cfg.joint_ids]) - limit


def joint_acceleration(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    return torch.abs(data.joint_acc[:, asset_cfg.joint_ids]) - limit


def upsidedown(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    return data.projected_gravity_b[:, 2] > limit


def contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor = env.scene[sensor_cfg.name]

    net_contact_forces = contact_sensor.data.net_forces_w_history
    return torch.any(
        torch.max(
            torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1),
            dim=1,
        )[0]
        > 1.0,
        dim=1,
    )


def base_orientation(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    return torch.norm(data.projected_gravity_b[:, :2], dim=1) - limit


def air_time(
    env: ManagerBasedRLEnv,
    limit: float,
    velocity_deadzone: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor = env.scene[sensor_cfg.name]
    touchdown = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # Like in CaT
    command_more_than_limit = (
        (
            torch.norm(env.command_manager.get_command("base_velocity")[:, :3], dim=1)
            > velocity_deadzone
        )
        .float()
        .unsqueeze(1)
    )
    # Like in Isaaclab
    # command_more_than_limit = (
    #     torch.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1) > 0.1
    # )
    cstr = (limit - last_air_time) * touchdown.float() * command_more_than_limit
    return cstr


def n_foot_contact(
    env: ManagerBasedRLEnv,
    number_of_desired_feet: int,
    min_command_value: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor = env.scene[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_cstr = torch.abs(
        (
            torch.max(
                torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1),
                dim=1,
            )[0]
            > 1.0
        ).sum(1)
        - number_of_desired_feet
    )
    command_more_than_limit = (
        torch.norm(env.command_manager.get_command("base_velocity")[:, :3], dim=1)
        > min_command_value
    ).float()
    return contact_cstr * command_more_than_limit


def joint_range(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    return (
        torch.abs(
            data.joint_pos[:, asset_cfg.joint_ids]
            - data.default_joint_pos[:, asset_cfg.joint_ids]
        )
        - limit
    )


def action_rate(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    return (
        torch.abs(
            env.action_manager._action[:, asset_cfg.joint_ids]
            - env.action_manager._prev_action[:, asset_cfg.joint_ids]
        )
        / env.step_dt
        - limit
    )


def foot_contact_force(
    env: ManagerBasedRLEnv,
    limit: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor = env.scene[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    return (
        torch.max(
            torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1
        )[0]
        - limit
    )


def min_base_height(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return limit - robot.data.root_pos_w[:, 2]


def no_move(
    env: ManagerBasedRLEnv,
    velocity_deadzone: float,
    joint_vel_limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    cstr_nomove = (
        torch.abs(data.joint_vel[:, asset_cfg.joint_ids]) - joint_vel_limit
    ) * (
        torch.norm(env.command_manager.get_command("base_velocity")[:, :3], dim=1)
        < velocity_deadzone
    ).float().unsqueeze(
        1
    )
    return cstr_nomove
