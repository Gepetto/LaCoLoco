# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
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


class UniformPositionCommandHRL(CommandTerm):
    cfg: "UniformPositionCommandHRLCfg"
    """Configuration for the command generator."""

    def __init__(self, cfg: "UniformPositionCommandHRLCfg", env: ManagerBasedEnv):
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

        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

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
        pos_error, _ = math_utils.compute_pose_error(
            self.pose_command_w[:, :3],
            torch.zeros(self.num_envs, 4, device=self.device),
            self.robot.data.body_state_w[:, self.body_idx, :3],
            self.robot.data.body_state_w[:, self.body_idx, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        theta = r.uniform_(*self.cfg.ranges.angles)

        self.pose_command_b[env_ids, 0] = self.cfg.sampling_radius * torch.cos(
            torch.tensor(theta)
        )
        self.pose_command_b[env_ids, 1] = self.cfg.sampling_radius * torch.sin(
            torch.tensor(theta)
        )
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)

        # Only take into account the x y offset
        self.pose_command_w[env_ids, 0] = (
            self.pose_command_b[env_ids, 0] + self._env.scene.env_origins[env_ids, 0]
        )
        self.pose_command_w[env_ids, 1] = (
            self.pose_command_b[env_ids, 1] + self._env.scene.env_origins[env_ids, 1]
        )
        self.pose_command_w[env_ids, 2] = self.pose_command_b[env_ids, 2]

        pos_error, _ = math_utils.compute_pose_error(
            self.pose_command_w[:, :3],
            torch.zeros(self.num_envs, 4, device=self.device),
            self.robot.data.body_state_w[:, self.body_idx, :3],
            self.robot.data.body_state_w[:, self.body_idx, 3:7],
        )

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
                self.subenv_visualizer = VisualizationMarkers(
                    self.cfg.subenv_visualizer_cfg
                )
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
            self.range_visualizer.set_visibility(True)
            self.subenv_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)
                self.range_visualizer.set_visibility(False)
                self.subenv_visualizer.set_visibility(False)

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
        body_pose_w = self.robot.data.body_state_w[:, self.body_idx]
        # self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])
        self.current_pose_visualizer.visualize(body_pose_w[:, :3], None)

        self.range_visualizer.visualize(self._env.scene.env_origins[:], None)
        self.subenv_visualizer.visualize(self._env.scene.env_origins[:], None)


@configclass
class UniformPositionCommandHRLCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = UniformPositionCommandHRL

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    sampling_radius: float = MISSING

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        angles: tuple[float, float] = MISSING
        """Range for the angles (in rad)."""

        pos_z: tuple[float, float] = MISSING
        """Range for the z position (in m)."""

    ranges: Ranges = MISSING
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

    range_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        markers={
            "range": sim_utils.CylinderCfg(
                radius=1.5,
                height=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0), opacity=0.1
                ),
            ),
        },
        prim_path="/Visuals/Command/current_range",
    )

    subenv_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        markers={
            "range": sim_utils.CuboidCfg(
                size=(3.0, 3.0, 0.01),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.0, 1.0), opacity=0.1
                ),
            ),
        },
        prim_path="/Visuals/Command/subenv",
    )
