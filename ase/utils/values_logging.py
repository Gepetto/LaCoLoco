import torch
from omni.isaac.lab.managers import SceneEntityCfg

from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.utils.noise.noise_model import uniform_noise


def get_obs(env):
    asset_cfg = SceneEntityCfg("robot")
    asset = env.unwrapped.scene[asset_cfg.name]
    base_lin_vel = asset.data.root_lin_vel_b
    base_ang_vel = asset.data.root_ang_vel_b
    base_gravity = asset.data.projected_gravity_b
    base_height = (asset.data.root_pos_w - env.unwrapped.scene.env_origins)[:, 2]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    obs_components = [
        base_height.unsqueeze(-1),
        base_gravity,
        base_lin_vel,
        base_ang_vel,
        joint_pos,
        joint_vel,
    ]

    obs_vector = torch.cat(obs_components, dim=-1)

    return obs_vector


def get_noisy_obs(env):
    asset_cfg = SceneEntityCfg("robot")
    asset = env.unwrapped.scene[asset_cfg.name]
    base_lin_vel = asset.data.root_lin_vel_b
    base_ang_vel = asset.data.root_ang_vel_b
    base_gravity = asset.data.projected_gravity_b
    base_height = (asset.data.root_pos_w - env.unwrapped.scene.env_origins)[:, 2]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    obs_components = [
        uniform_noise(base_height.unsqueeze(-1), Unoise(n_min=-0.005, n_max=0.005)),
        uniform_noise(base_gravity, Unoise(n_min=-0.05, n_max=0.05)),
        uniform_noise(base_lin_vel, Unoise(n_min=-0.1, n_max=0.1)),
        uniform_noise(base_ang_vel, Unoise(n_min=-0.2, n_max=0.2)),
        uniform_noise(joint_pos, Unoise(n_min=-0.01, n_max=0.01)),
        uniform_noise(joint_vel, Unoise(n_min=-0.2, n_max=0.2)),
    ]

    obs_vector = torch.cat(obs_components, dim=-1)

    return obs_vector


def get_states(env):
    asset_cfg = SceneEntityCfg("robot")
    asset = env.unwrapped.scene[asset_cfg.name]

    base_pos = asset.data.root_pos_w - env.unwrapped.scene.env_origins
    base_orientation = asset.data.root_quat_w

    base_lin_vel = asset.data.root_lin_vel_b
    base_ang_vel = asset.data.root_ang_vel_b

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    states_components = [
        base_pos,
        base_orientation,
        base_lin_vel,
        base_ang_vel,
        joint_pos,
        joint_vel,
    ]

    states_vector = torch.cat(states_components, dim=-1)

    return states_vector


def get_constraints(env):
    asset_cfg = SceneEntityCfg("robot")
    asset = env.unwrapped.scene[asset_cfg.name]
    contact_sensor = env.unwrapped.scene[SceneEntityCfg("contact_forces").name]

    joint_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    joint_velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    joint_acceleration = asset.data.joint_acc[:, asset_cfg.joint_ids]
    action_rate = (
        torch.abs(
            env.unwrapped.action_manager._action[:, asset_cfg.joint_ids]
            - env.unwrapped.action_manager._prev_action[:, asset_cfg.joint_ids]
        )
        / env.unwrapped.step_dt
    )

    # H1, solo detection, TODO: some better approach
    if joint_torque.shape[1] == 12:
        undesired_contact_body_ids, _ = contact_sensor.find_bodies(
            ["base_link", ".*_UPPER_LEG"], preserve_order=True
        )
        feet_ids, _ = contact_sensor.find_bodies([".*_FOOT"], preserve_order=True)
    else:
        undesired_contact_body_ids, _ = contact_sensor.find_bodies(
            [".*torso_link"], preserve_order=True
        )
        feet_ids, _ = contact_sensor.find_bodies([".*_ankle_link"], preserve_order=True)

    undesired_contact = torch.any(
        torch.max(
            torch.norm(
                contact_sensor.data.net_forces_w_history[
                    :, :, undesired_contact_body_ids
                ],
                dim=-1,
            ),
            dim=1,
        )[0]
        > 1.0,
        dim=1,
    )
    foot_contact_forces = torch.max(
        torch.norm(contact_sensor.data.net_forces_w_history[:, :, feet_ids], dim=-1),
        dim=1,
    )[0]
    upsidedown = asset.data.projected_gravity_b[:, 2]

    constraints_components = [
        joint_torque,
        joint_velocity,
        joint_acceleration,
        action_rate,
        undesired_contact.unsqueeze(-1),
        foot_contact_forces,
        upsidedown.unsqueeze(-1),
    ]

    constraints_vector = torch.cat(constraints_components, dim=-1)

    return constraints_vector


def get_position_errors(env):
    return (
        env.unwrapped.command_manager.get_term("ee_pose")
        .metrics["position_error"]
        .unsqueeze(-1)
    )
