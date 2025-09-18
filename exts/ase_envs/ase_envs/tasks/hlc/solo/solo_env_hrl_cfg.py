# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# Copyright (c) 2025, LAAS-CNRS
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# based on https://github.com/isaac-sim/IsaacLab

import math
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from ase_envs.tasks.utils.cat.manager_constraint_cfg import (
    ConstraintTermCfg as ConstraintTerm,
)
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp

import ase_envs.tasks.utils.mdp.observations as observations
import ase_envs.tasks.utils.mdp.events as events
import ase_envs.tasks.utils.mdp.terminations as terminations
from omni.isaac.lab.utils import modifiers

import ase_envs.tasks.utils.cat.curriculums as curriculums
import ase_envs.tasks.utils.cat.constraints as constraints

import ase_envs.tasks.utils.mdp.commands as mdp_commands
import ase_envs.tasks.utils.ase.commands as ase_commands
import ase_envs.tasks.utils.mdp.rewards as rewards


from ase_envs.assets.odri import SOLO12_MINIMAL_CFG, SOLO12_MINIMAL_MOTOR_LIMITS_CFG


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

    robot: ArticulationCfg = SOLO12_MINIMAL_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


def generate_ranges():
    ranges = [
        mdp_commands.UniformPositionCommandPedipulateCfg.Ranges(
            pos_x=(0.3, 1.5),
            pos_y=(-0.08, -0.08),
            pos_z=(0.16, 0.32),
        ),
        mdp_commands.UniformPositionCommandPedipulateCfg.Ranges(
            pos_x=(0.3, 1.5),
            pos_y=(-1.0, 1.0),
            pos_z=(0.16, 0.32),
        ),
        mdp_commands.UniformPositionCommandPedipulateCfg.Ranges(
            pos_x=(-1.5, 1.5),
            pos_y=(-1.0, 1.0),
            pos_z=(0.16, 0.32),
        ),
    ]
    return ranges


@configclass
class CommandsPedipulationCurriculumCfg:
    ee_pose = mdp_commands.UniformPositionCommandPedipulateCfg(
        asset_name="robot",
        body_name="FR_FOOT",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        # Only cartesian now
        curriculum_position_error_threshold=0.06,
        position_mean_alfa=0.05,
        min_episode_number_before_level_change=100,
        starting_level=0,
        # Increased difficulty
        ranges=generate_ranges(),
    )


@configclass
class CommandsPedipulationCircleCfg:
    ee_pose = ase_commands.UniformPositionCommandHRLCfg(
        asset_name="robot",
        body_name="FR_FOOT",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        sampling_radius=1.5,
        ranges=ase_commands.UniformPositionCommandHRLCfg.Ranges(
            angles=(-3.14, 3.14),
            pos_z=(0.16, 0.42),
        ),
    )


@configclass
class CommandsVelocityCfg:
    base_velocity = mdp_commands.UniformVelocityCommandWithDeadzoneCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.78, 0.78),
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FL_HAA",
            "FL_HFE",
            "FL_KFE",
            "FR_HAA",
            "FR_HFE",
            "FR_KFE",
            "HL_HAA",
            "HL_HFE",
            "HL_KFE",
            "HR_HAA",
            "HR_HFE",
            "HR_KFE",
        ],
        scale=1.5,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsASECfg:
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
        base_height = ObsTerm(
            func=mdp.base_pos_z,
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )
        joint_pos = ObsTerm(
            func=observations.joint_pos,
            params={
                "names": [
                    "FL_HAA",
                    "FL_HFE",
                    "FL_KFE",
                    "FR_HAA",
                    "FR_HFE",
                    "FR_KFE",
                    "HL_HAA",
                    "HL_HFE",
                    "HL_KFE",
                    "HR_HAA",
                    "HR_HFE",
                    "HR_KFE",
                ]
            },
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_vel = ObsTerm(
            func=observations.joint_vel,
            params={
                "names": [
                    "FL_HAA",
                    "FL_HFE",
                    "FL_KFE",
                    "FR_HAA",
                    "FR_HFE",
                    "FR_KFE",
                    "HL_HAA",
                    "HL_HFE",
                    "HL_KFE",
                    "HR_HAA",
                    "HR_HFE",
                    "HR_KFE",
                ]
            },
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class AseCfg(ObsGroup):
        base_height = ObsTerm(
            func=mdp.base_pos_z,
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        joint_pos = ObsTerm(
            func=observations.joint_pos,
            params={
                "names": [
                    "FL_HAA",
                    "FR_HAA",
                    "HL_HAA",
                    "HR_HAA",
                    "FL_HFE",
                    "FR_HFE",
                    "HL_HFE",
                    "HR_HFE",
                    "FL_KFE",
                    "FR_KFE",
                    "HL_KFE",
                    "HR_KFE",
                ]
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=observations.joint_vel,
            params={
                "names": [
                    "FL_HAA",
                    "FR_HAA",
                    "HL_HAA",
                    "HR_HAA",
                    "FL_HFE",
                    "FR_HFE",
                    "HL_HFE",
                    "HR_HFE",
                    "FL_KFE",
                    "FR_KFE",
                    "HL_KFE",
                    "HR_KFE",
                ]
            },
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 2

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: AseCfg = AseCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.5),
            "dynamic_friction_range": (0.4, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 100,
        },
    )

    scale_inertia_tensors = EventTerm(
        func=events.random_scale_inertia_tensors,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "scale_min": 0.8,
            "scale_max": 1.2,
        },
    )

    scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "recompute_inertia": False,
        },
    )

    move_base_com = EventTerm(
        func=events.randomize_body_coms,
        mode="startup",
        params={
            "max_displacement": 0.02,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.75, 1.25),
            "velocity_range": (0.9, 1.1),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 8.0),
        params={
            "velocity_range": {
                "x": (-0.25, 0.25),
                "y": (-0.25, 0.25),
                "yaw": (-0.25, 0.25),
                "pitch": (-0.25, 0.25),
                "roll": (-0.25, 0.25),
            }
        },
    )


@configclass
class RewardsPedipulationCfg:
    end_effector_position_tracking = RewTerm(
        func=rewards.position_command_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="FR_FOOT"),
            "command_name": "ee_pose",
            "std": math.sqrt(0.1),
        },
    )


@configclass
class RewardsVelocityCfg:
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )


HARD_MAX_P = 0.2


@configclass
class ConstraintsCfg:
    # Safety Hard constraints
    foot_contact_force = ConstraintTerm(
        func=constraints.foot_contact_force,
        max_p=HARD_MAX_P,
        params={
            "limit": 25.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    curriculum_hard_constraints = CurrTerm(
        func=curriculums.modify_constraint_p_linear,
        params={
            "term_names": ["foot_contact_force"],
            "num_steps": 30000,
            "init_max_p": HARD_MAX_P,
        },
    )


##
# Environment configuration
##


@configclass
class FlatEnvCommon(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=3.0)
    actions: ActionsCfg = ActionsCfg()
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


@configclass
class Velocity_HRL_ASE(FlatEnvCommon):
    commands = CommandsVelocityCfg()
    rewards = RewardsVelocityCfg()

    observations = ObservationsASECfg()

    constraints = None
    curriculum = None


class Velocity_HRL_ASE_PLAY(Velocity_HRL_ASE):
    def __post_init__(self) -> None:
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 4096
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False

        self.scene.robot = SOLO12_MINIMAL_MOTOR_LIMITS_CFG.replace(
            prim_path="/World/envs/env_.*/Robot"
        )


@configclass
class Velocity_HRL_ASE_CaT(FlatEnvCommon):
    commands = CommandsVelocityCfg()
    rewards = RewardsVelocityCfg()

    observations = ObservationsASECfg()

    constraints = ConstraintsCfg()
    constraints_num_steps_start = 0
    curriculum = CurriculumCfg()


class Velocity_HRL_ASE_CaT_PLAY(Velocity_HRL_ASE_CaT):
    def __post_init__(self) -> None:
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 4096
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False

        self.scene.robot = SOLO12_MINIMAL_MOTOR_LIMITS_CFG.replace(
            prim_path="/World/envs/env_.*/Robot"
        )


@configclass
class Solo_Pedipulation_HRL_ASE(FlatEnvCommon):
    commands: CommandsPedipulationCircleCfg = CommandsPedipulationCircleCfg()
    rewards: RewardsPedipulationCfg = RewardsPedipulationCfg()

    observations = ObservationsASECfg()

    constraints = None
    curriculum = None


class Solo_Pedipulation_HRL_ASE_PLAY(Solo_Pedipulation_HRL_ASE):
    def __post_init__(self) -> None:
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 4096
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False

        self.scene.robot = SOLO12_MINIMAL_MOTOR_LIMITS_CFG.replace(
            prim_path="/World/envs/env_.*/Robot"
        )


@configclass
class Solo_Pedipulation_HRL_ASE_CaT(FlatEnvCommon):
    commands: CommandsPedipulationCircleCfg = CommandsPedipulationCircleCfg()
    rewards: RewardsPedipulationCfg = RewardsPedipulationCfg()

    observations = ObservationsASECfg()

    constraints = ConstraintsCfg()
    constraints_num_steps_start = 0
    curriculum = CurriculumCfg()


class Solo_Pedipulation_HRL_ASE_CaT_PLAY(Solo_Pedipulation_HRL_ASE_CaT):
    def __post_init__(self) -> None:
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 4096
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False

        self.scene.robot = SOLO12_MINIMAL_MOTOR_LIMITS_CFG.replace(
            prim_path="/World/envs/env_.*/Robot"
        )


# self.commands.ee_pose.starting_level = len(self.commands.ee_pose.ranges) - 1
# print(f"Starting level: {self.commands.ee_pose.starting_level}")


@configclass
class Solo_Pedipulation_HRL_ASE_Clipping(FlatEnvCommon):
    commands: CommandsPedipulationCircleCfg = CommandsPedipulationCircleCfg()
    rewards: RewardsPedipulationCfg = RewardsPedipulationCfg()

    observations = ObservationsASECfg()

    constraints = None
    curriculum = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot = SOLO12_MINIMAL_MOTOR_LIMITS_CFG.replace(
            prim_path="/World/envs/env_.*/Robot"
        )
