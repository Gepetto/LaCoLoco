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
from ase_envs.tasks.utils.cat.manager_constraint_cfg import (
    ConstraintTermCfg as ConstraintTerm,
)
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.terrains import (
    TerrainImporterCfg,
)
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp


import ase_envs.tasks.utils.cat.constraints as constraints
import ase_envs.tasks.utils.cat.curriculums as curriculums
import ase_envs.tasks.utils.mdp.observations as observations

import ase_envs.tasks.utils.mdp.commands as commands
import ase_envs.tasks.utils.mdp.events as events
import ase_envs.tasks.utils.mdp.rewards as rewards


from ase_envs.assets.odri import SOLO12_MINIMAL_CFG


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
    x_offset = 0.3
    y_offset = -0.08
    ranges = [
        commands.UniformPositionCommandPedipulateCfg.Ranges(
            pos_x=(-0.1 + x_offset, 0.0 + x_offset),
            pos_y=(-0.08 + y_offset, 0.08 + y_offset),
            pos_z=(0.12, 0.32),
        ),
        commands.UniformPositionCommandPedipulateCfg.Ranges(
            pos_x=(-0.1 + x_offset, 0.05 + x_offset),
            pos_y=(-0.12 + y_offset, 0.12 + y_offset),
            pos_z=(0.07, 0.37),
        ),
        commands.UniformPositionCommandPedipulateCfg.Ranges(
            pos_x=(-0.1 + x_offset, 0.1 + x_offset),
            pos_y=(-0.16 + y_offset, 0.16 + y_offset),
            pos_z=(0.02, 0.42),
        ),
    ]

    return ranges


@configclass
class CommandsCfg:
    ee_pose = commands.UniformPositionCommandPedipulateCfg(
        asset_name="robot",
        body_name="FR_FOOT",
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
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
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
            noise=Unoise(n_min=-0.01, n_max=0.01),
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
            "static_friction_range": (0.4, 1.5),
            "dynamic_friction_range": (0.4, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
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
            "position_range": (0.9, 1.1),
            "velocity_range": (0.9, 1.1),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 3.0),
        params={
            "velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)}
        },
    )


@configclass
class RewardsCfg:
    end_effector_position_tracking = RewTerm(
        func=rewards.position_command_error_exp,
        weight=15.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="FR_FOOT"),
            "command_name": "ee_pose",
            "std": math.sqrt(0.8),
        },
    )

    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-5.0e-2)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-6)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-2)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_LOWER_LEG", ".*_UPPER_LEG"],
            ),
            "threshold": 1.0,
        },
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-800.0)


@configclass
class ConstraintsCfg:
    # Safety Soft constraints
    joint_position_min_rhaa = ConstraintTerm(
        func=constraints.joint_position_min,
        max_p=0.25,
        # hard limit -2.25
        params={
            "limit": -1.75,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*R_HAA"]),
        },
    )
    joint_position_max_rhaa = ConstraintTerm(
        func=constraints.joint_position_max,
        max_p=0.25,
        # hard limit 0.95
        params={
            "limit": 0.7,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*R_HAA"]),
        },
    )
    joint_position_min_lhaa = ConstraintTerm(
        func=constraints.joint_position_min,
        max_p=0.25,
        # hard limit -0.95
        params={
            "limit": -0.7,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*L_HAA"]),
        },
    )
    joint_position_max_lhaa = ConstraintTerm(
        func=constraints.joint_position_max,
        max_p=0.25,
        # hard limit 2.25
        params={
            "limit": 1.75,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*L_HAA"]),
        },
    )
    joint_position_min_hfe = ConstraintTerm(
        func=constraints.joint_position_min,
        max_p=0.25,
        # hard limit -4.22 - robot should take step back instead of reaching limit
        params={
            "limit": -3.14,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HFE"]),
        },
    )
    joint_position_max_hfe = ConstraintTerm(
        func=constraints.joint_position_max,
        max_p=0.25,
        # hard limit 1.57
        params={
            "limit": 1.3,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HFE"]),
        },
    )
    joint_position_min_kfe = ConstraintTerm(
        func=constraints.joint_position_min,
        max_p=0.25,
        # hard limit -3.14
        params={
            "limit": -2.75,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_KFE"]),
        },
    )
    joint_position_max_kfe = ConstraintTerm(
        func=constraints.joint_position_max,
        max_p=0.25,
        # hard limit 3.14
        params={
            "limit": 2.75,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_KFE"]),
        },
    )

    joint_torque = ConstraintTerm(
        func=constraints.joint_torque,
        max_p=0.25,
        params={
            "limit": 3.0,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_HAA", ".*_HFE", ".*_KFE"]
            ),
        },
    )
    joint_velocity = ConstraintTerm(
        func=constraints.joint_velocity,
        max_p=0.25,
        params={
            # "limit": 2.0, # slower
            "limit": 16.0,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_HAA", ".*_HFE", ".*_KFE"]
            ),
        },
    )
    joint_acceleration = ConstraintTerm(
        func=constraints.joint_acceleration,
        max_p=0.25,
        params={
            # "limit": 50.0, # slower
            "limit": 800.0,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_HAA", ".*_HFE", ".*_KFE"]
            ),
        },
    )
    action_rate = ConstraintTerm(
        func=constraints.action_rate,
        max_p=0.25,
        params={
            "limit": 80.0,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_HAA", ".*_HFE", ".*_KFE"]
            ),
        },
    )
    contact_soft = ConstraintTerm(
        func=constraints.contact,
        max_p=0.25,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[".*_LOWER_LEG", ".*_UPPER_LEG"]
            )
        },
    )

    # Safety Hard constraints
    # only base for pedipulation
    contact = ConstraintTerm(
        func=constraints.contact,
        max_p=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"])
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
    # Safety Soft constraints
    curriculum_soft_constraints = CurrTerm(
        func=curriculums.modify_constraint_p_linear,
        params={
            "term_names": [
                "joint_position_min_rhaa",
                "joint_position_max_rhaa",
                "joint_position_min_lhaa",
                "joint_position_max_lhaa",
                "joint_position_min_hfe",
                "joint_position_max_hfe",
                "joint_position_min_kfe",
                "joint_position_max_kfe",
                "joint_torque",
                "joint_velocity",
                "joint_acceleration",
                "action_rate",
                "contact_soft",
            ],
            "num_steps": 10000,
            "init_max_p": 0.25,
        },
    )


##
# Environment configuration
##


@configclass
class Solo12PedipulateEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=3.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    constraints: ConstraintsCfg = ConstraintsCfg()
    constraints_num_steps_start = 0
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


class Solo12PedipulateEnvCfg_PLAY(Solo12PedipulateEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 20

        # Set to the hardest level
        self.commands.ee_pose.starting_level = len(self.commands.ee_pose.ranges) - 1
        self.commands.ee_pose.resampling_time_range = (2.0, 4.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        self.episode_length_s = 10.0
