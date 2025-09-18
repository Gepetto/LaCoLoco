# Copyright (c) 2018-2022, NVIDIA Corporation
# Copyright (c) 2025, LAAS-CNRS
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# based on https://github.com/nv-tlabs/ASE

import copy
import os
import yaml
import numpy as np
import torch

from rl_games.common import a2c_common

import learning.common_agent as common_agent


class CommonHRLAgent(common_agent.CommonAgent):
    def __init__(self, base_name, params):
        with open(os.path.join(os.getcwd(), params["config"]["llc_config"]), "r") as f:
            llc_config = yaml.load(f, Loader=yaml.SafeLoader)
            llc_config_params = llc_config["params"]
            self._latent_dim = llc_config_params["config"]["latent_dim"]

        self._task_size = params["config"]["task_obs_size"]
        self._llc_action_size = params["config"]["llc_action_size"]
        self._command_name = params["config"]["command_name"]

        super().__init__(base_name, params)

        self._llc_steps = params["config"]["llc_steps"]
        llc_checkpoint = params["config"]["llc_checkpoint"]
        llc_config_params["config"]["full_experiment_name"] = params["config"][
            "full_experiment_name"
        ]
        llc_config_params["config"]["train_dir"] = params["config"]["train_dir"]
        assert llc_checkpoint != ""
        self._build_llc(llc_config_params, llc_checkpoint)

        # self._random_latents = self._llc_agent.model.a2c_network.sample_latents(
        #     self.vec_env.env.unwrapped.num_envs
        # )

    def _load_config_params(self, config):
        super()._load_config_params(config)

        self._task_reward_w = config["task_reward_w"]
        self._disc_reward_w = config["disc_reward_w"]

    def _build_net_config(self):
        obs_size = (
            self.obs_shape[0]
            - self._llc_action_size
            + self._latent_dim
            + self._task_size
        )
        obs_shape = (obs_size,)
        self._hlc_obs_shape = obs_shape

        print(f"{self._llc_action_size} {self._latent_dim} {self._task_size}")
        print(f"Original: {self.obs_shape} edited: {obs_shape}")
        config = {
            "actions_num": self.actions_num,
            "input_shape": obs_shape,
            "num_seqs": self.num_actors * self.num_agents,
            "value_size": self.env_info.get("value_size", 1),
            "normalize_value": self.normalize_value,
            "normalize_input": self.normalize_input,
        }
        return config

    def init_tensors(self):
        super().init_tensors()

        del self.experience_buffer.tensor_dict["actions"]
        del self.experience_buffer.tensor_dict["mus"]
        del self.experience_buffer.tensor_dict["sigmas"]
        del self.experience_buffer.tensor_dict["obses"]

        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict["actions"] = torch.zeros(
            batch_shape + (self._latent_dim,),
            dtype=torch.float32,
            device=self.ppo_device,
        )
        self.experience_buffer.tensor_dict["mus"] = torch.zeros(
            batch_shape + (self._latent_dim,),
            dtype=torch.float32,
            device=self.ppo_device,
        )
        self.experience_buffer.tensor_dict["sigmas"] = torch.zeros(
            batch_shape + (self._latent_dim,),
            dtype=torch.float32,
            device=self.ppo_device,
        )
        print(f"Batch shape {batch_shape}, hlc obs shape {self._hlc_obs_shape}")
        self.experience_buffer.tensor_dict["obses"] = torch.zeros(
            batch_shape + self._hlc_obs_shape,
            dtype=torch.float32,
            device=self.ppo_device,
        )

        self.experience_buffer.tensor_dict["disc_rewards"] = torch.zeros_like(
            self.experience_buffer.tensor_dict["rewards"]
        )
        self.tensor_list += ["disc_rewards"]

    def play_steps(self):
        update_list = self.update_list

        for n in range(self.horizon_length):
            self.hlc_obs = self.obs.copy()  # Shallow copy of the dict
            obs_trimmed = self.obs["obs"][:, : -self._llc_action_size]
            last_action = self.experience_buffer.tensor_dict["actions"][-1]
            command = self.vec_env.env.unwrapped.command_manager.get_command(
                self._command_name
            )

            self.hlc_obs["obs"] = torch.cat([obs_trimmed, command, last_action], dim=1)

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.hlc_obs, masks)
            else:
                res_dict = self.get_action_values(self.hlc_obs)

            self.experience_buffer.update_data("obses", n, self.hlc_obs["obs"])
            self.experience_buffer.update_data("dones", n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data("states", n, self.hlc_obs["states"])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and "time_outs" in infos:
                shaped_rewards += (
                    self.gamma
                    * res_dict["values"]
                    * self.cast_obs(infos["time_outs"]).unsqueeze(1).float()
                )
            self.experience_buffer.update_data("rewards", n, shaped_rewards)

            if self._disc_reward_w > 0:
                self.experience_buffer.update_data(
                    "disc_rewards", n, infos["disc_rewards"]
                )

            self.dones = self.dones.float()  # <-- Modified line CaT
            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1

            all_done_indices = self.dones.ge(1.0).nonzero(
                as_tuple=False
            )  # <-- Modified line CaT
            env_done_indices = all_done_indices[:: self.num_agents]

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(
                self.current_shaped_rewards[env_done_indices]
            )
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones  # <-- Modified line CaT

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = (
                self.current_shaped_rewards * not_dones.unsqueeze(1)
            )
            self.current_lengths[env_done_indices] = 0

        last_values = self.get_values(self.hlc_obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_rewards = self.experience_buffer.tensor_dict["rewards"]

        if self._disc_reward_w > 0:
            mb_disc_rewards = self.experience_buffer.tensor_dict["disc_rewards"]
            mb_rewards = self._combine_rewards(mb_rewards, mb_disc_rewards)

        mb_advs = self.discount_values(
            fdones, last_values, mb_fdones, mb_values, mb_rewards
        )
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(
            a2c_common.swap_and_flatten01, self.tensor_list
        )
        batch_dict["returns"] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict["played_frames"] = self.batch_size

        return batch_dict

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs = self.obs

        rewards = 0.0
        disc_rewards = 0.0
        done_count = 0.0
        for t in range(self._llc_steps):
            llc_actions = self._compute_llc_action(obs["obs"], actions)
            obs, curr_rewards, curr_dones, infos = self.vec_env.step(llc_actions)

            rewards += curr_rewards
            done_count += curr_dones

            amp_obs = obs["states"]

            if self._disc_reward_w > 0:
                curr_disc_reward = self._calc_disc_reward(amp_obs)
                disc_rewards += curr_disc_reward

        rewards /= self._llc_steps

        if self._disc_reward_w > 0:
            disc_rewards /= self._llc_steps
            infos["disc_rewards"] = disc_rewards

        dones = torch.zeros_like(done_count)
        dones[done_count > 0] = 1.0

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return (
                self.obs_to_tensors(obs),
                rewards.to(self.ppo_device),
                dones.to(self.ppo_device),
                infos,
            )
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return (
                self.obs_to_tensors(obs),
                torch.from_numpy(rewards).to(self.ppo_device).float(),
                torch.from_numpy(dones).to(self.ppo_device),
                infos,
            )

    def cast_obs(self, obs):
        obs = super().cast_obs(obs)
        self._llc_agent.is_tensor_obses = self.is_tensor_obses
        return obs

    def _get_mean_rewards(self):
        rewards = super()._get_mean_rewards()
        rewards *= self._llc_steps
        return rewards

    def _setup_action_space(self):
        super()._setup_action_space()
        self.actions_num = self._latent_dim

    def _build_llc(self, config_params, checkpoint_file):
        raise NotImplementedError

    def _build_llc_agent_config(self, config_params, network):
        llc_env_info = copy.deepcopy(self.env_info)

        config = config_params
        config["config"]["network"] = network
        config["config"]["num_actors"] = self.num_actors
        config["config"]["features"] = {"observer": self.algo_observer}
        config["config"]["env_info"] = llc_env_info
        config["config"]["env_info"][
            "num_amp_obs_steps"
        ] = self.vec_env.env.unwrapped._num_amp_obs_steps

        return config

    def _compute_llc_action(self, llc_obs, actions):
        processed_obs = self._llc_agent._preproc_obs(llc_obs)

        ase_latents = torch.nn.functional.normalize(actions, dim=-1)

        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": processed_obs,
            "rnn_states": self.states,
            "ase_latents": ase_latents,
        }
        with torch.no_grad():
            res_dict = self._llc_agent.model(input_dict)

        return res_dict["mus"]

    def _calc_disc_reward(self, amp_obs):
        disc_reward = self._llc_agent._calc_disc_rewards(amp_obs)
        return disc_reward

    def _combine_rewards(self, task_rewards, disc_rewards):
        combined_rewards = (
            self._task_reward_w * task_rewards + self._disc_reward_w * disc_rewards
        )
        return combined_rewards

    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        if self._disc_reward_w > 0:
            train_info["disc_rewards"] = batch_dict["disc_rewards"]

    def _log_train_info(self, train_info, frame):
        # Info already logged by llc, adding it here causes clash
        # and problems in wandb with displaying necessary plots
        pass
