# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, LAAS-CNRS
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# based on https://github.com/isaac-sim/IsaacLab

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument("--play", action="store_true", default=False, help="")
parser.add_argument(
    "--video_length",
    type=int,
    default=500,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=10000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Path to model checkpoint."
)
parser.add_argument(
    "--track",
    action="store_true",
    default=False,
    help="if toggled, this experiment will be tracked with Weights and Biases",
)
parser.add_argument(
    "--wandb_project_name",
    type=str,
    default="rl_games",
    help="the wandb's project name",
)
parser.add_argument("--motion_file", type=str, default=None, help="")
parser.add_argument(
    "--llc_checkpoint",
    type=str,
    default=None,
    help="LLC policy to use when running HLC training",
)
parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
parser.add_argument("--experiment_id", type=str, default=None, help="Experiment id")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from utils.build_alg_runner import build_alg_runner

import gymnasium as gym
import math
import os
import random
from datetime import datetime

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab.utils.assets import retrieve_file_path
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

from omni.isaac.lab_tasks.utils.hydra import hydra_task_config

import omni.isaac.lab_tasks  # noqa: F401
import ase_envs.tasks  # noqa: F401

import yaml


from utils.rlgames_env import RlGamesAseGpuEnv, RlGamesVecEnvWrapperCaT


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict
):
    """Train with RL-Games agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1 or (
        args_cli.seed is None and agent_cfg["params"]["seed"] == -1
    ):
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = (
        args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    )
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(
            f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}"
        )

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    agent_cfg["params"]["config"]["name"] = (
        args_cli.experiment_name
        if args_cli.experiment_name is not None
        else agent_cfg["params"]["config"]["name"]
    )

    # specify directory for logging experiments
    log_root_path = os.path.join(
        "logs", "rl_games", agent_cfg["params"]["config"]["name"]
    )
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs

    if args_cli.experiment_id is not None:
        full_experiment_name = (
            f"{agent_cfg['params']['config']['name']}_{args_cli.experiment_id}"
        )
    else:
        full_experiment_name = agent_cfg["params"]["config"].get(
            "full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )

    if os.path.exists(os.path.join(log_root_path, full_experiment_name)):
        print("Requested experiment name already exists! Falling back to date")
        full_experiment_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"Full experiment name: {full_experiment_name}")
    log_dir = full_experiment_name
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    if args_cli.llc_checkpoint is not None:
        print(f"Using LLC: {args_cli.llc_checkpoint}")
        agent_cfg["params"]["config"]["llc_checkpoint"] = args_cli.llc_checkpoint

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]

    if "env" in agent_cfg["params"]:
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    else:
        clip_obs = math.inf
        clip_actions = math.inf

    if args_cli.motion_file is None:
        print("No motion file specified!")

    # create isaac environment
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
        motion_file=args_cli.motion_file,
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos_train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapperCaT(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesAseGpuEnv(
            config_name, num_actors, **kwargs
        ),
    )
    env_configurations.register(
        "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env}
    )

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)

    # create runner from rl-games
    runner = build_alg_runner(IsaacAlgoObserver())
    runner.load(agent_cfg)

    # reset the agent and env
    runner.reset()

    if args_cli.play:
        args = {"train": False, "play": True}
    else:
        args = {"train": True, "play": False}

    if args_cli.checkpoint is not None:
        args["checkpoint"] = resume_path

    if args_cli.motion_file is not None:
        args["motion_file"] = args_cli.motion_file

        with open(os.path.join(os.getcwd(), args_cli.motion_file), "r") as f:
            motion_config = yaml.load(f, Loader=yaml.SafeLoader)
            dump_yaml(
                os.path.join(log_root_path, log_dir, "params", "dataset.yaml"),
                motion_config,
            )

    if args_cli.track:
        import wandb

        wandb.init(
            project=args_cli.wandb_project_name,
            # entity=args_cli.wandb_entity,
            sync_tensorboard=True,
            config=agent_cfg,
            monitor_gym=True,
            save_code=True,
            name=full_experiment_name,
        )

    runner.run(args)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
