# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# Copyright (c) 2025, LAAS-CNRS
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# based on https://github.com/isaac-sim/IsaacLab

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.utils.math as math_utils

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

from collections.abc import Sequence
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers import VisualizationMarkers
from dataclasses import MISSING
from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.math import quat_rotate


class UniformVelocityCommandWithDeadzone(mdp.UniformVelocityCommand):
    """velocity command sampling class ported directly from CaT"""

    cfg: "UniformVelocityCommandWithDeadzoneCfg"

    def __init__(
        self, cfg: "UniformVelocityCommandWithDeadzoneCfg", env: ManagerBasedEnv
    ):
        """Initializes the command generator.

        Args:
            cfg: The command generator configuration.
            env: The environment.
        """
        super().__init__(cfg, env)

        self.velocity_deadzone = cfg.velocity_deadzone
        self.dt = env.physics_dt
        self.max_episode_length_s = env.max_episode_length_s
        print(self.dt, self.max_episode_length_s)

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(
                self.heading_target[env_ids] - self.robot.data.heading_w[env_ids]
            )
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        # Enforce standing (i.e., zero velocity command) for standing envs

        # set small commands to zero
        self.vel_command_b *= (
            torch.any(
                torch.abs(self.vel_command_b[:, :3]) > self.velocity_deadzone, dim=1
            )
        ).unsqueeze(1)

        # standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        # self.vel_command_b[standing_env_ids, :] = 0.0

        # Random velocity command resampling
        no_vel_command = (
            torch.norm(self.vel_command_b[:, :3], dim=1) < self.velocity_deadzone
        ).float()
        p_resample_command = 0.01 * no_vel_command + (
            self.dt / self.max_episode_length_s
        ) * (1 - no_vel_command)
        resample_command_idx = (
            torch.bernoulli(p_resample_command).nonzero(as_tuple=False).flatten()
        )
        if len(resample_command_idx) > 0:
            self._resample(resample_command_idx)

        # Random angular velocity inversion during the episode to avoid having the robot moving in circle
        p_ang_vel = (
            self.dt / self.max_episode_length_s
        )  # <- time step / duration of X seconds
        # There will be a probability of 0.63 of having at least one swap after X seconds have elapsed
        # (1 / p) policy steps for X seconds, and the probability of having no swap at all is (1 - p)**(1 / p) = 0.37
        # The mean number of swaps for (1 / p) steps with probability p is 1.
        self.vel_command_b[:, 2] *= (
            1
            - 2
            * torch.bernoulli(
                torch.full_like(self.vel_command_b[:, 2], p_ang_vel)
            ).float()
        )


@configclass
class UniformVelocityCommandWithDeadzoneCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the normal velocity command generator."""

    class_type: type = UniformVelocityCommandWithDeadzone
    velocity_deadzone: float = 0.1


class UniformPositionCommandPedipulate(CommandTerm):
    cfg: "UniformPositionCommandPedipulateCfg"
    """Configuration for the command generator."""

    def __init__(
        self, cfg: "UniformPositionCommandPedipulateCfg", env: ManagerBasedEnv
    ):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.pose_command_w = torch.zeros_like(self.pose_command_b)

        self.offsets = torch.zeros(self.num_envs, 3, device=self.device)
        self.offsets[:, 2] = 0.22

        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["current_ranges_level"] = torch.zeros(
            self.num_envs, device=self.device
        )

        self._current_level = (
            torch.ones(self.num_envs, dtype=torch.int, device=self.device)
            * self.cfg.starting_level
        )
        self._mean_errors = torch.zeros(self.num_envs, device=self.device)

        self._first_it = True
        self._resample_command(range(self.num_envs))

    def __str__(self) -> str:
        msg = "UniformPositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 3).

        The first three elements correspond to the position
        """
        return self.pose_command_b

    @property
    def command_w(self) -> torch.Tensor:
        """The desired position command in world coordinates. Shape is (num_envs, 3).

        The first three elements correspond to the position
        """
        return self.pose_command_w

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # compute the error

        body_pos_w = self.robot.data.body_state_w[:, self.body_idx, :3]
        body_quat_w = self.robot.data.body_state_w[:, self.body_idx, 3:7]

        if abs(self.cfg.eef_offset) > 0.0:
            offset_local = torch.tensor(
                [self.cfg.eef_offset, 0.0, 0.0], device=body_pos_w.device
            )
            offset_local_expanded = offset_local.unsqueeze(0).expand_as(body_pos_w)
            offset_world = quat_rotate(body_quat_w, offset_local_expanded)
            new_body_pos_w = body_pos_w + offset_world
        else:
            new_body_pos_w = body_pos_w

        pos_error, _ = math_utils.compute_pose_error(
            self.pose_command_w[:, :3],
            torch.zeros(self.num_envs, 4, device=self.device),
            new_body_pos_w,
            body_quat_w,
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["current_ranges_level"][:] = self._current_level.to(torch.float)

        zero_mean = self._mean_errors < 1.0e-12
        if torch.any(zero_mean):
            indices = torch.nonzero(zero_mean, as_tuple=True)
            self._mean_errors[indices] = self.metrics["position_error"][indices]

        self._mean_errors = (
            self.cfg.position_mean_alfa * self.metrics["position_error"]
            + (1.0 - self.cfg.position_mean_alfa) * self._mean_errors
        )

    def _resample_command(self, env_ids: Sequence[int]):
        error_condition = (
            self._mean_errors < self.cfg.curriculum_position_error_threshold
        )

        if torch.any(error_condition) and not self._first_it:
            error_indices = torch.nonzero(error_condition, as_tuple=True)[0]
            env_ids_tensor = torch.tensor(env_ids, device=self.device)
            filtered_indices = error_indices[torch.isin(error_indices, env_ids_tensor)]

            ids_to_promote = []
            for id in filtered_indices:
                level = self._current_level[id]
                if level == 0:
                    ids_to_promote.append(id)
                else:
                    cmd_x = (
                        self.pose_command_w[id, 0] - self._env.scene.env_origins[id, 0]
                    )
                    cmd_y = (
                        self.pose_command_w[id, 1] - self._env.scene.env_origins[id, 1]
                    )

                    cmd_x_in_high_level = (
                        cmd_x < self.cfg.ranges[level - 1].pos_x[0]
                    ) or (cmd_x > self.cfg.ranges[level - 1].pos_x[1])
                    cmd_y_in_high_level = (
                        cmd_y < self.cfg.ranges[level - 1].pos_y[0]
                    ) or (cmd_y > self.cfg.ranges[level - 1].pos_y[1])

                    if cmd_x_in_high_level or cmd_y_in_high_level:
                        ids_to_promote.append(id)

            if ids_to_promote:  # Check if the list is not empty
                ids_to_promote_tensor = torch.tensor(
                    ids_to_promote, device=self._current_level.device
                )

                # Ensure it's a 1D tensor
                ids_to_promote_tensor = ids_to_promote_tensor.squeeze()

                self._current_level[ids_to_promote_tensor] = torch.min(
                    self._current_level[ids_to_promote_tensor] + 1,
                    torch.tensor(
                        len(self.cfg.ranges) - 1, device=self._current_level.device
                    ),
                )

        self._first_it = False

        self.pose_command_b[env_ids, 0] = torch.stack(
            [
                torch.empty(1, device=self.device).uniform_(
                    *self.cfg.ranges[level].pos_x
                )
                for level in self._current_level[env_ids]
            ]
        ).squeeze()
        self.pose_command_b[env_ids, 1] = torch.stack(
            [
                torch.empty(1, device=self.device).uniform_(
                    *self.cfg.ranges[level].pos_y
                )
                for level in self._current_level[env_ids]
            ]
        ).squeeze()
        self.pose_command_b[env_ids, 2] = torch.stack(
            [
                torch.empty(1, device=self.device).uniform_(
                    *self.cfg.ranges[level].pos_z
                )
                for level in self._current_level[env_ids]
            ]
        ).squeeze()

        # Only take into account the x y offset
        self.pose_command_w[env_ids, 0] = (
            self.pose_command_b[env_ids, 0] + self._env.scene.env_origins[env_ids, 0]
        )
        self.pose_command_w[env_ids, 1] = (
            self.pose_command_b[env_ids, 1] + self._env.scene.env_origins[env_ids, 1]
        )
        self.pose_command_w[env_ids, 2] = self.pose_command_b[env_ids, 2]

        body_pos_w = self.robot.data.body_state_w[:, self.body_idx, :3]
        body_quat_w = self.robot.data.body_state_w[:, self.body_idx, 3:7]

        if abs(self.cfg.eef_offset) > 0.0:
            offset_local = torch.tensor(
                [self.cfg.eef_offset, 0.0, 0.0], device=body_pos_w.device
            )
            offset_local_expanded = offset_local.unsqueeze(0).expand_as(body_pos_w)
            offset_world = quat_rotate(body_quat_w, offset_local_expanded)
            new_body_pos_w = body_pos_w + offset_world
        else:
            new_body_pos_w = body_pos_w

        pos_error, _ = math_utils.compute_pose_error(
            self.pose_command_w[:, :3],
            torch.zeros(self.num_envs, 4, device=self.device),
            new_body_pos_w,
            body_quat_w,
        )
        self._mean_errors[env_ids] = torch.norm(pos_error[env_ids], dim=-1)

        # TODO
        self.offsets[env_ids, 0] = self._env.scene.env_origins[env_ids, 0] + 0.3
        self.offsets[env_ids, 1] = self._env.scene.env_origins[env_ids, 1] - 0.08

    def _update_command(self):
        self.pose_command_b[:, :3], _ = math_utils.subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_w[:, :3],
            None,
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(
                    self.cfg.goal_pose_visualizer_cfg
                )
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(
                    self.cfg.current_pose_visualizer_cfg
                )
                self.range_visualizer = VisualizationMarkers(
                    self.cfg.range_visualizer_cfg
                )
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
            self.range_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)
                self.range_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(
            self.pose_command_w[:, :3],
            None,
        )
        # -- current body pose
        body_pos_w = self.robot.data.body_state_w[:, self.body_idx, :3]
        body_quat_w = self.robot.data.body_state_w[:, self.body_idx, 3:7]

        if abs(self.cfg.eef_offset) > 0.0:
            offset_local = torch.tensor(
                [self.cfg.eef_offset, 0.0, 0.0], device=body_pos_w.device
            )
            offset_local_expanded = offset_local.unsqueeze(0).expand_as(body_pos_w)
            offset_world = quat_rotate(body_quat_w, offset_local_expanded)
            new_body_pos_w = body_pos_w + offset_world
        else:
            new_body_pos_w = body_pos_w

        # self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])
        self.current_pose_visualizer.visualize(new_body_pos_w[:, :3], None)

        # self.range_visualizer.visualize(self.offsets[:], None)
        self.range_visualizer.visualize(self._env.scene.env_origins[:], None)


@configclass
class UniformPositionCommandPedipulateCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = UniformPositionCommandPedipulate

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    curriculum_position_error_threshold: float = MISSING
    position_mean_alfa: float = MISSING
    min_episode_number_before_level_change: float = MISSING
    starting_level: int = MISSING
    eef_offset: float = 0.0

    visualization_range: tuple[float, float, float] = (3.9, 2.0, 0.01)

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        pos_z: tuple[float, float] = MISSING
        """Range for the z position (in m)."""

    ranges: list[Ranges] = MISSING
    """Ranges for the commands."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        markers={
            "target": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0)
                ),
            ),
        },
        prim_path="/Visuals/Command/goal_pose",
    )

    current_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        markers={
            "target": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0)
                ),
            ),
        },
        prim_path="/Visuals/Command/body_pose",
    )

    # range_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
    #     markers={
    #         "range": sim_utils.CuboidCfg(
    #             size=(0.2, 0.32, 0.4),
    #             visual_material=sim_utils.PreviewSurfaceCfg(
    #                 diffuse_color=(0.0, 1.0, 0.0), opacity=0.1
    #             ),
    #         ),
    #     },
    #     prim_path="/Visuals/Command/current_range",
    # )

    range_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        markers={
            "range": sim_utils.CuboidCfg(
                size=visualization_range,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.0, 1.0), opacity=0.1
                ),
            ),
        },
        prim_path="/Visuals/Command/subenv",
    )
