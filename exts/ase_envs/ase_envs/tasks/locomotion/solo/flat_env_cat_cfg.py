# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# Copyright (c) 2025, LAAS-CNRS
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# based on https://github.com/isaac-sim/IsaacLab
# based on https://github.com/Gepetto/constraints-as-terminations

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
from omni.isaac.lab.terrains import (
    TerrainImporterCfg,
    HfRandomUniformTerrainCfg,
    TerrainGeneratorCfg,
)
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
import ase_envs.tasks.utils.cat.constraints as constraints
import ase_envs.tasks.utils.cat.curriculums as curriculums
import ase_envs.tasks.utils.mdp.observations as observations

import ase_envs.tasks.utils.mdp.terminations as terminations
import ase_envs.tasks.utils.mdp.events as events
import ase_envs.tasks.utils.mdp.commands as commands


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


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 8.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.78, 0.78)
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
        scale=0.5,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
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
            "pose_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.3, 0.3),
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
            "position_range": (0.5, 1.5),
            "velocity_range": (0.9, 1.1),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 8.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.1, 0.1),
                "yaw": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
            }
        },
    )


@configclass
class RewardsCfg:
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


@configclass
class ConstraintsCfg:
    # Safety Soft constraints
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

    # Safety Hard constraints
    # Knee and base
    contact = ConstraintTerm(
        func=constraints.contact,
        max_p=1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["base_link", ".*_UPPER_LEG"]
            )
        },
    )
    foot_contact_force = ConstraintTerm(
        func=constraints.foot_contact_force,
        max_p=1.0,
        params={
            "limit": 50.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
        },
    )
    front_hfe_position = ConstraintTerm(
        func=constraints.joint_position,
        max_p=1.0,
        params={
            "limit": 1.3,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["FL_HFE", "FR_HFE"]),
        },
    )
    upsidedown = ConstraintTerm(
        func=constraints.upsidedown,
        max_p=1.0,
        params={"limit": 0.0, "asset_cfg": SceneEntityCfg("robot")},
    )

    # Style constraints
    hip_position = ConstraintTerm(
        func=constraints.joint_position_when_moving_forward,
        max_p=0.25,
        params={
            "limit": 0.2,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HAA"]),
            "velocity_deadzone": 0.1,
        },
    )
    base_orientation = ConstraintTerm(
        func=constraints.base_orientation,
        max_p=0.25,
        params={"limit": 0.1, "asset_cfg": SceneEntityCfg("robot")},
    )
    air_time = ConstraintTerm(
        func=constraints.air_time,
        max_p=0.25,
        params={
            "limit": 0.25,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
            "velocity_deadzone": 0.1,
        },
    )
    no_move = ConstraintTerm(
        func=constraints.no_move,
        max_p=0.25,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_HAA", ".*_HFE", ".*_KFE"]
            ),
            "velocity_deadzone": 0.1,
            "joint_vel_limit": 4.0,
        },
    )
    two_foot_contact = ConstraintTerm(
        func=constraints.n_foot_contact,
        max_p=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
            "number_of_desired_feet": 2,
            "min_command_value": 0.5,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["base_link", ".*_UPPER_LEG"]
            ),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    # Safety Soft constraints
    curriculum_soft_constraints = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_names": [
                "joint_torque",
                "joint_velocity",
                "joint_acceleration",
                "action_rate",
                "hip_position",
                "base_orientation",
                "air_time",
                "two_foot_contact",
            ],
            "num_steps": 24 * 800,
            "init_max_p": 0.25,
        },
    )


##
# Environment configuration
##


@configclass
class Solo12FlatEnvCaTCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)

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
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


class Solo12FlatEnvCaTCfg_PLAY(Solo12FlatEnvCaTCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 120
        self.scene.env_spacing = 3.0

        # disable randomization for play
        self.observations.policy.enable_corruption = True

        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.78, 0.78)
        self.commands.base_velocity.rel_standing_envs = 0.0

        self.scene.terrain = TerrainImporterCfg(
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
