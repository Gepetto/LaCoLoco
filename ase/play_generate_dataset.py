# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, LAAS-CNRS
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# based on https://github.com/isaac-sim/IsaacLab

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Play a checkpoint of an RL agent from RL-Games."
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=500,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
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
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--save_latents", action="store_true", default=False)
parser.add_argument("--latents_change", action="store_true", default=False)

parser.add_argument("--motion_file", type=str, default=None, help="")
parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
parser.add_argument("--motion_steps", type=int, default=500, help="")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from utils.build_alg_runner import build_alg_runner

import gymnasium as gym
import math
import os
import random
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.assets import retrieve_file_path
from omni.isaac.lab.utils.dict import print_dict


from omni.isaac.lab_tasks.utils import (
    get_checkpoint_path,
    load_cfg_from_registry,
    parse_env_cfg,
)

import omni.isaac.lab_tasks  # noqa: F401
import ase_envs.tasks  # noqa: F401

import torch
import numpy as np
from utils.values_logging import get_obs, get_states, get_constraints


from utils.rlgames_env import RlGamesAseGpuEnv, RlGamesVecEnvWrapperCaT


def main():
    """Play with RL-Games agent."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1 or (
        args_cli.seed is None and agent_cfg["params"]["seed"] == -1
    ):
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = (
        args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    )
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
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(
            log_root_path, run_dir, checkpoint_file, other_dirs=["nn"]
        )
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # wrap around environment for rl-games
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
            "video_folder": os.path.join(log_root_path, log_dir, "videos_play"),
            "step_trigger": lambda step: step == 0,
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

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    # create runner from rl-games
    runner = build_alg_runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.

    NUM_STEPS = args_cli.motion_steps
    NUM_ENVS = env.unwrapped.num_envs
    LATENTS_TRANSITIONS = 4

    obs_buffer = torch.zeros((NUM_STEPS, NUM_ENVS, get_obs(env).shape[-1]))
    states_buffer = torch.zeros((NUM_STEPS, NUM_ENVS, get_states(env).shape[-1]))
    constraints_buffer = torch.zeros(
        (NUM_STEPS, NUM_ENVS, get_constraints(env).shape[-1])
    )
    dones_buffer = torch.zeros((NUM_STEPS, NUM_ENVS, 1))

    if args_cli.save_latents:
        agent._reset_latents()

        if hasattr(agent, "_ase_latents"):
            latents = [agent._ase_latents.cpu().numpy()]

    for step in range(NUM_STEPS):
        # run everything in inference mode
        with torch.inference_mode():
            if (
                args_cli.latents_change
                and step % (NUM_STEPS // LATENTS_TRANSITIONS) == 0
            ):
                agent._reset_latents()
                if hasattr(agent, "_ase_latents"):
                    latents.append(agent._ase_latents.cpu().numpy())

            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            # env stepping
            obs, _, dones, _ = env.step(actions)

            obs_buffer[step] = get_obs(env)
            states_buffer[step] = get_states(env)
            constraints_buffer[step] = get_constraints(env)
            dones_buffer[step] = dones.unsqueeze(-1)

            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0

    obs_numpy = obs_buffer.cpu().numpy()
    states_numpy = states_buffer.cpu().numpy()
    constraints_numpy = constraints_buffer.cpu().numpy()

    np.save(os.path.join(log_dir, "obs_dataset.npy"), obs_numpy)
    np.save(os.path.join(log_dir, "states_dataset.npy"), states_numpy)
    np.save(
        os.path.join(log_dir, "constraints_dataset.npy"),
        constraints_numpy,
    )
    np.save(
        os.path.join(log_dir, "dones_dataset.npy"),
        dones_buffer.cpu().numpy(),
    )

    if args_cli.save_latents:
        np.save(
            os.path.join(log_dir, "latents_dataset.npy"),
            np.array(latents),
        )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
