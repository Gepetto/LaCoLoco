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

from rl_games.algos_torch import players

import learning.common_player as common_player

from utils.values_logging import get_constraints, get_position_errors


class CommonHRLPlayer(common_player.CommonPlayer):
    def __init__(self, params):
        with open(os.path.join(os.getcwd(), params["config"]["llc_config"]), "r") as f:
            llc_config = yaml.load(f, Loader=yaml.SafeLoader)
            llc_config_params = llc_config["params"]
            self._latent_dim = llc_config_params["config"]["latent_dim"]

        self._task_size = params["config"]["task_obs_size"]
        self._llc_action_size = params["config"]["llc_action_size"]
        self._command_name = params["config"]["command_name"]

        super().__init__(params)

        self._llc_steps = params["config"]["llc_steps"]
        llc_checkpoint = params["config"]["llc_checkpoint"]
        assert llc_checkpoint != ""
        self._build_llc(llc_config_params, llc_checkpoint)

        if "train_dir" in params["config"]:
            self.log_dir = os.path.join(
                params["config"]["train_dir"], params["config"]["full_experiment_name"]
            )
        self.max_steps = params["config"]["player"]["max_steps"]
        self.save_metric_logs = params["config"]["player"]["save_metric_logs"]

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obs_dict = self.env_reset(self.env)
            batch_size = 1
            if len(obs_dict["obs"].size()) > len(self.obs_shape):
                batch_size = obs_dict["obs"].size()[0]
            self.batch_size = batch_size
            self.last_action = torch.zeros(
                (batch_size, self.actions_num), dtype=torch.float32, device=self.device
            )

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            # Initialize per-environment accumulators if balance_env_rewards is enabled
            if self.balance_env_rewards:
                per_env_rewards = torch.zeros(batch_size, dtype=torch.float32)
                per_env_steps = torch.zeros(batch_size, dtype=torch.float32)
                per_env_game_res = torch.zeros(batch_size, dtype=torch.float32)
                per_env_games_played = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            done_indices = []

            if self.save_metric_logs:
                constraints_buffer = torch.zeros(
                    (
                        self.max_steps,
                        self.env.unwrapped.num_envs,
                        get_constraints(self.env).shape[-1],
                    )
                )

                errors_buffer = torch.zeros(
                    (
                        self.max_steps,
                        self.env.unwrapped.num_envs,
                        get_position_errors(self.env).shape[-1],
                    )
                )

                dones_buffer = torch.zeros(
                    (
                        self.max_steps,
                        self.env.unwrapped.num_envs,
                        1,
                    )
                )

            print(f"Playing {self.max_steps} steps and {n_games} games")

            for n in range(self.max_steps):
                hlc_obs = copy.deepcopy(obs_dict)
                obs_trimmed = hlc_obs["obs"][:, : -self._llc_action_size]
                last_action = self.last_action
                hlc_obs["obs"] = torch.cat(
                    [
                        obs_trimmed,
                        self.env.unwrapped.command_manager.get_command(
                            self._command_name
                        ),
                        last_action,
                    ],
                    dim=1,
                )

                if self.evaluation and n % self.update_checkpoint_freq == 0:
                    self.maybe_load_new_checkpoint()

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(hlc_obs, masks, is_deterministic)
                else:
                    action = self.get_action(hlc_obs, is_deterministic)
                self.last_action = action
                obs_dict, r, done, info = self.env_step(self.env, obs_dict, action)
                cr += r
                steps += 1

                self._post_step(info)

                if render:
                    self.env.render(mode="human")
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[:: self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if self.save_metric_logs:
                    dones_buffer[n] = done.unsqueeze(-1)
                    constraints_buffer[n] = get_constraints(self.env)
                    errors_buffer[n] = get_position_errors(self.env)

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                    game_res = 0.0
                    if isinstance(info, dict):
                        if "battle_won" in info:
                            print_game_res = True
                            game_res = info.get("battle_won", 0.5)
                        if "scores" in info:
                            print_game_res = True
                            game_res = info.get("scores", 0.5)

                    if self.balance_env_rewards:
                        # Update per-environment accumulators
                        per_env_rewards[done_indices] += cr[done_indices]
                        per_env_steps[done_indices] += steps[done_indices]
                        per_env_games_played[done_indices] += 1
                        if print_game_res:
                            per_env_game_res[done_indices] += game_res

                        # Reset current rewards and steps for done environments
                        cr[done_indices] = 0
                        steps[done_indices] = 0
                    else:
                        # Original accumulation
                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps
                        sum_game_res += game_res

                        if self.print_stats:
                            cur_rewards_done = cur_rewards / done_count
                            cur_steps_done = cur_steps / done_count
                            if print_game_res:
                                print(
                                    f"reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f} w: {game_res}"
                                )
                            else:
                                print(
                                    f"reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f}"
                                )

                done_indices = done_indices[:, 0]

            if self.save_metric_logs:
                np.save(
                    os.path.join(self.log_dir, "constraints_dataset.npy"),
                    constraints_buffer.cpu().numpy(),
                )
                np.save(
                    os.path.join(self.log_dir, "errors_dataset.npy"),
                    errors_buffer.cpu().numpy(),
                )
                np.save(
                    os.path.join(self.log_dir, "dones_dataset.npy"),
                    dones_buffer.cpu().numpy(),
                )

        if self.save_metric_logs:
            with open(os.path.join(self.log_dir, "rewards_dataset.yaml"), "w") as f:
                yaml.dump({"Sum rewards": sum_rewards}, f)

        print(sum_rewards)
        if print_game_res:
            print(
                "av reward:",
                sum_rewards / games_played * n_game_life,
                "av steps:",
                sum_steps / games_played * n_game_life,
                "winrate:",
                sum_game_res / games_played * n_game_life,
            )
        else:
            print(
                "av reward:",
                sum_rewards / games_played * n_game_life,
                "av steps:",
                sum_steps / games_played * n_game_life,
            )

    def get_action(self, obs_dict, is_deterministic=False):
        obs = obs_dict["obs"]

        if len(obs.size()) == len(self.obs_shape):
            obs = obs.unsqueeze(0)
        proc_obs = self._preproc_obs(obs)
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": proc_obs,
            "rnn_states": self.states,
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict["mus"]
        action = res_dict["actions"]
        self.states = res_dict["rnn_states"]
        if is_deterministic:
            current_action = mu
        else:
            current_action = action

        current_action = current_action.detach()
        if self.clip_actions:
            return players.rescale_actions(
                self.actions_low,
                self.actions_high,
                torch.clamp(current_action, -1.0, 1.0),
            )
        else:
            return current_action

    def env_step(self, env, obs_dict, action):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        obs = obs_dict

        rewards = 0.0
        done_count = 0.0
        for t in range(self._llc_steps):
            llc_actions = self._compute_llc_action(obs["obs"], action)
            obs, curr_rewards, curr_dones, infos = env.step(llc_actions)

            rewards += curr_rewards
            done_count += curr_dones

        rewards /= self._llc_steps
        dones = torch.zeros_like(done_count)
        dones[done_count > 0] = 1.0

        if hasattr(obs, "dtype") and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return obs, rewards.to(self.device), dones.to(self.device), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return (
                self.obs_to_torch(obs),
                torch.from_numpy(rewards),
                torch.from_numpy(dones),
                infos,
            )

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

    def _build_llc(self, config_params, checkpoint_file):
        raise NotImplementedError

    def _build_llc_agent_config(self, config_params, network):
        llc_env_info = copy.deepcopy(self.env_info)
        llc_env_info["amp_observation_space"] = self.env.unwrapped.amp_observation_space
        llc_env_info["num_envs"] = self.env.unwrapped.num_envs

        config = config_params
        config["config"]["network"] = network
        config["config"]["env_info"] = llc_env_info
        config["config"]["env_info"][
            "num_amp_obs_steps"
        ] = self.env.unwrapped._num_amp_obs_steps

        return config

    def _setup_action_space(self):
        super()._setup_action_space()
        self.actions_num = self._latent_dim

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
            "num_seqs": self.num_agents,
            "value_size": self.env_info.get("value_size", 1),
            "normalize_value": self.normalize_value,
            "normalize_input": self.normalize_input,
        }
        return config
