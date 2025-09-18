# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# Copyright (c) 2025, LAAS-CNRS
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# based on https://github.com/isaac-sim/IsaacLab

import math

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as locomotion_mdp


import ase_envs.tasks.utils.mdp.commands as commands
import ase_envs.tasks.utils.mdp.rewards as rewards


from ase_envs.assets.unitree import H1_ASE_MINIMAL_CFG  # isort: skip


@configclass
class MySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="exts/ase_envs/ase_envs/assets/Materials/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = H1_ASE_MINIMAL_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


def generate_ranges():
    x_offset = 0.5
    y_offset = -0.2
    z_offset = 1.0
    ranges = [
        commands.UniformPositionCommandPedipulateCfg.Ranges(
            pos_x=(-0.1 + x_offset, 0.0 + x_offset),
            pos_y=(-0.08 + y_offset, 0.08 + y_offset),
            pos_z=(-0.2 + z_offset, 0.2 + z_offset),
        ),
        commands.UniformPositionCommandPedipulateCfg.Ranges(
            pos_x=(-0.1 + x_offset, 0.05 + x_offset),
            pos_y=(-0.12 + y_offset, 0.12 + y_offset),
            pos_z=(-0.4 + z_offset, 0.4 + z_offset),
        ),
        commands.UniformPositionCommandPedipulateCfg.Ranges(
            pos_x=(-0.1 + x_offset, 0.1 + x_offset),
            pos_y=(-0.16 + y_offset, 0.16 + y_offset),
            pos_z=(-0.6 + z_offset, 0.8 + z_offset),
        ),
    ]

    return ranges


@configclass
class CommandsCfg:
    ee_pose = commands.UniformPositionCommandPedipulateCfg(
        asset_name="robot",
        body_name="right_elbow_link",
        eef_offset=0.3,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        # Only cartesian now
        curriculum_position_error_threshold=0.06,
        position_mean_alfa=0.05,
        min_episode_number_before_level_change=200,
        starting_level=0,
        # Increased difficulty
        ranges=generate_ranges(),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.5, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5))
        ee_pose_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "ee_pose"},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    end_effector_position_tracking = RewTerm(
        func=rewards.position_command_error_exp,
        weight=15.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_elbow_link"),
            "command_name": "ee_pose",
            "std": math.sqrt(0.8),
            "eef_offset": 0.3,
        },
    )

    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=0.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    lin_vel_z_l2 = None

    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso")},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    pass


##
# Environment configuration
##


@configclass
class H1ManipulateEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    constraints = None
    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


class H1ManipulateEnvCfg_PLAY(H1ManipulateEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 20

        # Set to the hardest level
        # self.commands.ee_pose.starting_level = len(self.commands.ee_pose.ranges) - 1
        self.commands.ee_pose.starting_level = 0
        self.commands.ee_pose.resampling_time_range = (2.0, 4.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        self.episode_length_s = 10.0

        # remove random pushing
        self.events.base_external_force_torque = None
