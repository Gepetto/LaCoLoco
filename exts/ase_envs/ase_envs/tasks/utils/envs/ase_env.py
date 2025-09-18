# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# Copyright (c) 2025, LAAS-CNRS
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# based on https://github.com/isaac-sim/IsaacLab

from __future__ import annotations

import gymnasium as gym
import torch
from collections.abc import Sequence

from omni.isaac.lab.envs.common import VecEnvStepReturn
from omni.isaac.lab.envs.manager_based_rl_env import ManagerBasedRLEnv

from utils.motion_lib import MotionLib

from ase_envs.tasks.utils.cat.constraint_manager import ConstraintManager


class ASEEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._amp_obs_space = self._get_observation_space()["critic"]
        self._num_amp_obs_steps = cfg.observations.critic.history_length
        assert self._num_amp_obs_steps >= 2

        self._num_amp_obs_per_step = int(
            self._amp_obs_space.shape[0] / self._num_amp_obs_steps
        )

        motion_file = kwargs["motion_file"]
        self._load_motion(motion_file)

        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        self._starting_amp_obs = {}

        obs_terms = zip(
            self.observation_manager._group_obs_term_names["critic"],
            self.observation_manager._group_obs_term_dim["critic"],
        )
        for index, (name, dims) in enumerate(obs_terms):
            print(name)
            print(dims)
            self._starting_amp_obs[name] = torch.zeros(
                (
                    self.num_envs,
                    self._num_amp_obs_steps,
                    dims[0] // self._num_amp_obs_steps,
                ),
                device=self.device,
                dtype=torch.float32,
            )

        self.last_reset_envs = None

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def fetch_amp_obs_demo(self, num_samples):

        motion_ids = self._motion_lib.sample_motions(num_samples)

        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.step_dt * (self._num_amp_obs_steps - 1)
        motion_times0 = self._motion_lib.sample_time(
            motion_ids, truncate_time=truncate_time
        )
        motion_times0 += truncate_time

        amp_obs_demo_flat = (
            self._motion_lib.build_amp_obs_demo(
                motion_ids, motion_times0, self._num_amp_obs_steps
            )
            .to(self.device)
            .view(-1, self.get_num_amp_obs())
        )

        return amp_obs_demo_flat

    def _load_motion(self, motion_file):
        self._motion_lib = MotionLib(
            motion_file=motion_file,
            device=self.device,
            step_dt=self.step_dt,
        )

    def _cat_reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # update the curriculum for environments that need a reset
        self.curriculum_manager.compute(env_ids=env_ids)
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)
        # apply events such as randomizations for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(
                mode="reset", env_ids=env_ids, global_env_step_count=env_step_count
            )

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- rewards manager
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- constraints manager
        if self.cfg.constraints:
            info = self.constraint_manager.reset(env_ids)
            self.extras["log"].update(info)
        # -- curriculum manager
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- command manager
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- termination manager
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- recorder manager
        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0

    def _reset_idx(self, env_ids: Sequence[int]):
        self.progress_buf[env_ids] = 0
        # Call the super class reset
        self._cat_reset_idx(env_ids)

        # Manually reset buffer - has to be done, because of using resets from dataset, which modify
        # default Isaaclab behaviour
        for key, buf in self.observation_manager._group_obs_term_history_buffer[
            "critic"
        ].items():
            if buf._buffer is not None:
                buf._num_pushes[env_ids] = self._num_amp_obs_steps

                buf._buffer[:, env_ids, :] = (
                    self._starting_amp_obs[key][env_ids, :, :]
                    .permute(1, 0, 2)
                    .to(self.device)
                )

        self.last_reset_envs = env_ids

    def cat_step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if (
                self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
            ):
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- CaT constraints prob computation
        if (
            self.cfg.constraints
            and self.common_step_counter > self.cfg.constraints_num_steps_start
        ):
            cstr_prob = self.constraint_manager.compute()
            self.extras["cstr_prob"] = cstr_prob
            # -- constrained reward computation
            self.reward_buf = torch.clip(
                self.reward_manager.compute(dt=self.step_dt) * (1.0 - cstr_prob),
                min=0.0,
                max=None,
            )
            dones = cstr_prob.clone()
        else:
            self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
            dones = torch.zeros(self.num_envs, device=self.device)
            self.extras["cstr_prob"] = torch.zeros(self.num_envs, device=self.device)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(reset_env_ids) > 0:
            dones[reset_env_ids] = 1.0
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)

            # this is needed to make joint positions set from reset events effective
            self.scene.write_data_to_sim()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, dones, self.reset_time_outs, self.extras

    def step(self, actions):
        self.progress_buf += 1
        obs, rewards, terminated, time_outs, extras = self.cat_step(actions)
        return obs, rewards, terminated, time_outs, extras

    def _get_observation_space(self):
        obs_manager = self.observation_manager
        obs_space = gym.spaces.Dict()
        for group_name, group_term_names in obs_manager.active_terms.items():
            # extract quantities about the group
            has_concatenated_obs = obs_manager.group_obs_concatenate[group_name]
            obs_dim = obs_manager.group_obs_dim[group_name]

            # assuming all terms in the group have same dimension
            # can implement a more sophisticated logic if not
            if has_concatenated_obs:
                group_obs_dim = sum(obs_dim)
            else:
                group_obs_dim = sum(dim[0] for dim in obs_dim)

            obs_space[group_name] = gym.spaces.Box(
                low=-float("inf"), high=float("inf"), shape=(group_obs_dim,)
            )
        return obs_space

    @property
    def amp_observation_space(self):
        return self._amp_obs_space

    def load_managers(self):
        super().load_managers()
        # prepare the managers
        # -- constraint manager

        if self.cfg.constraints:
            self.constraint_manager = ConstraintManager(self.cfg.constraints, self)
            print("[INFO] Constraint Manager: ", self.constraint_manager)
