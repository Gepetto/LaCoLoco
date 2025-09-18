# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# Copyright (c) 2025, LAAS-CNRS
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# based on https://github.com/isaac-sim/IsaacLab

import torch
from typing import TYPE_CHECKING, Literal


from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.envs import ManagerBasedEnv
import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp


def reset_robot_from_dataset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    joint_position_range: tuple[float, float],
    joint_velocity_range: tuple[float, float],
    joint_num: int = 12,
    default_config_prob: float = 0.0,
    set_initial_velocities_from_dataset: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints by scaling the default position and velocity by the given ranges.

    This function samples random values from the given ranges and scales the default joint positions and velocities
    by these values. The scaled values are then set into the physics simulation.
    """

    feature_sizes = [1, 3, 3, 3, joint_num, joint_num]  # Individual feature group sizes

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get default joint state

    num_envs = env_ids.shape[0]

    if num_envs <= 0:
        return

    motion_ids = env.unwrapped._motion_lib.sample_motions(num_envs)
    motion_times = env.unwrapped._motion_lib.sample_time(motion_ids)
    motion_state = env.unwrapped._motion_lib.get_motion_states(motion_ids, motion_times)

    if default_config_prob > 0.0:
        r = torch.empty(len(env_ids), device=env.unwrapped.device)
        default_config_envs = r.uniform_(0.0, 1.0) <= default_config_prob
        default_config_env_ids = default_config_envs.nonzero(as_tuple=False).flatten()
        non_default_config_env_ids = (
            (~default_config_envs).nonzero(as_tuple=False).flatten()
        )

        mdp.reset_root_state_uniform(
            env, env_ids[default_config_env_ids], pose_range, velocity_range, asset_cfg
        )
        mdp.reset_joints_by_scale(
            env,
            env_ids[default_config_env_ids],
            joint_position_range,
            joint_velocity_range,
            asset_cfg,
        )

    # Set x,y to just zeros
    motion_state[:, 0:2] = 0.0
    positions = (
        motion_state[non_default_config_env_ids, 0:3]
        + env.scene.env_origins[env_ids[non_default_config_env_ids]]
    )
    orientations = motion_state[non_default_config_env_ids, 3:7]
    velocities = motion_state[non_default_config_env_ids, 7:13]
    joint_pos = motion_state[non_default_config_env_ids, 13 : 13 + joint_num]
    joint_vel = motion_state[non_default_config_env_ids, 13 + joint_num :]

    # set into the physics simulation
    asset.write_root_pose_to_sim(
        torch.cat([positions, orientations], dim=-1),
        env_ids=env_ids[non_default_config_env_ids],
    )
    if set_initial_velocities_from_dataset:
        asset.write_root_velocity_to_sim(
            velocities, env_ids=env_ids[non_default_config_env_ids]
        )
    asset.write_joint_state_to_sim(
        joint_pos, joint_vel, env_ids=env_ids[non_default_config_env_ids]
    )

    dt = env.unwrapped.step_dt
    motion_ids = torch.tile(
        motion_ids.unsqueeze(-1),
        [1, env.unwrapped._num_amp_obs_steps],
    ).to(env.unwrapped.device)
    motion_times = motion_times.unsqueeze(-1)
    time_steps = -dt * torch.arange(
        0, env.unwrapped._num_amp_obs_steps, device=env.unwrapped.device
    )
    motion_times = motion_times + time_steps

    motion_ids = motion_ids.view(-1)
    motion_times = motion_times.view(-1)
    amp_obs_demo = env.unwrapped._motion_lib.get_motion_obs(motion_ids, motion_times)

    amp_obs_default_config_ids = (
        default_config_env_ids * env.unwrapped._num_amp_obs_steps
    )

    if default_config_prob > 0.0:
        for i in range(env.unwrapped._num_amp_obs_steps):
            amp_obs_demo[amp_obs_default_config_ids + i, 0] = (
                asset.data.root_link_pos_w[env_ids[default_config_env_ids], 2].clone()
            )
            amp_obs_demo[amp_obs_default_config_ids + i, 1:4] = (
                math_utils.quat_rotate_inverse(
                    asset.data.root_link_quat_w[default_config_env_ids, :],
                    asset.data.GRAVITY_VEC_W[default_config_env_ids],
                )
            )
            amp_obs_demo[amp_obs_default_config_ids + i, 4:7] = (
                asset.data.root_lin_vel_b[env_ids[default_config_env_ids], :].clone()
            )
            amp_obs_demo[amp_obs_default_config_ids + i, 7:10] = (
                asset.data.root_ang_vel_b[env_ids[default_config_env_ids], :].clone()
            )
            amp_obs_demo[amp_obs_default_config_ids + i, 10 : 10 + joint_num] = (
                asset.data.joint_pos[env_ids[default_config_env_ids]].clone()
            )
            amp_obs_demo[amp_obs_default_config_ids + i, 10 + joint_num :] = (
                asset.data.joint_vel[env_ids[default_config_env_ids]].clone()
            )

    timesteps = env.unwrapped._num_amp_obs_steps

    # Reshape into (512, 2, 34) so we can work with (t, t+1) together
    obs = amp_obs_demo.view(len(env_ids), timesteps, sum(feature_sizes))

    # Split the features into components
    split_obs = obs.split(feature_sizes, dim=2)  # List of tensors per feature

    # Interleave timesteps (t, t+1) **within each feature group**
    interleaved_features = [
        feat.view(len(env_ids), timesteps, size)  # Shape: (batch_size, 2, feature_size)
        for feat, size in zip(split_obs, feature_sizes)
    ]

    for name, feature in zip(
        env.unwrapped.observation_manager._group_obs_term_names["critic"],
        interleaved_features,
    ):
        env.unwrapped._starting_amp_obs[name][env_ids] = feature

    # env.unwrapped._envs_reference_motion_ids[env_ids] = motion_ids


def randomize_body_coms(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    max_displacement: float,
    asset_cfg: SceneEntityCfg,
):
    """.. tip::
    This function uses CPU tensors to assign the body masses. It is recommended to use this function
    only during the initialization of the environment.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    coms = asset.root_physx_view.get_coms().clone()[:, body_ids, :3]
    coms += torch.rand_like(coms) * 2 * max_displacement - max_displacement

    new_coms = asset.root_physx_view.get_coms().clone()
    new_coms[:, asset_cfg.body_ids, 0:3] = coms
    asset.root_physx_view.set_coms(new_coms, env_ids)


def random_scale_inertia_tensors(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scale_min: float,
    scale_max: float,
    asset_cfg: SceneEntityCfg,
):
    """.. tip::
    This function uses CPU tensors to assign the body masses. It is recommended to use this function
    only during the initialization of the environment.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    I_URDF = asset.root_physx_view.get_inertias().clone()[env_ids, :, :]
    N_envs, N_bodies, _ = I_URDF.shape
    device = I_URDF.device

    scale_factors = (
        torch.rand(N_envs, N_bodies, 1, device=device) * (scale_max - scale_min)
        + scale_min
    )

    I_URDF_scaled = I_URDF * scale_factors
    I_URDF_scaled_flattened = I_URDF_scaled.reshape(N_envs, N_bodies, 9)
    asset.root_physx_view.set_inertias(I_URDF_scaled_flattened, env_ids)
