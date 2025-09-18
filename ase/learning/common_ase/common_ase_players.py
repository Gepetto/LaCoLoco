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

import numpy as np
import torch

from rl_games.algos_torch import players
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from omni.isaac.lab.utils.array import convert_to_torch

import learning.common_player as common_player


class CommonASEPlayer(common_player.CommonPlayer):
    def __init__(self, params):
        self._latent_dim = params["config"]["latent_dim"]
        self._latent_steps_min = params["config"].get("latent_steps_min", np.inf)
        self._latent_steps_max = params["config"].get("latent_steps_max", np.inf)
        self._random_latents_resets = params["config"]["player"].get(
            "random_latents_resets", True
        )

        self._enc_reward_scale = params["config"]["enc_reward_scale"]

        self._normalize_amp_input = params["config"].get("normalize_amp_input", True)
        self._disc_reward_scale = params["config"]["disc_reward_scale"]

        super().__init__(params)

        if hasattr(self, "env") and self.env != None:
            self.num_envs = self.env.unwrapped.num_envs
        else:
            self.num_envs = self.env_info["num_envs"]

        batch_size = self.num_envs
        self._ase_latents = torch.zeros(
            (batch_size, self._latent_dim), dtype=torch.float32, device=self.device
        )

        self._reset_latent_step_count()
        self._reset_latents()

    def run(self):
        self._reset_latent_step_count()

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

        self.wait_for_checkpoint()

        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obs_dict = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict["obs"], batch_size)

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

            for n in range(self.max_steps):
                if len(done_indices) > 0:
                    self._reset_latents(done_indices)

                if self.evaluation and n % self.update_checkpoint_freq == 0:
                    self.maybe_load_new_checkpoint()

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obs_dict, masks, is_deterministic)
                else:
                    action = self.get_action(obs_dict, is_deterministic)

                obs_dict, r, done, info = self.env_step(self.env, action)
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

                    if batch_size // self.num_agents == 1 or games_played >= n_games:
                        break

                done_indices = done_indices[:, 0]

        if self.balance_env_rewards:
            # Calculate per-environment average rewards
            valid_envs = per_env_games_played > 0
            per_env_avg_rewards = torch.zeros(batch_size, dtype=torch.float32)
            per_env_avg_steps = torch.zeros(batch_size, dtype=torch.float32)
            per_env_avg_game_res = torch.zeros(batch_size, dtype=torch.float32)

            per_env_avg_rewards[valid_envs] = (
                per_env_rewards[valid_envs] / per_env_games_played[valid_envs]
            )
            per_env_avg_steps[valid_envs] = (
                per_env_steps[valid_envs] / per_env_games_played[valid_envs]
            )

            overall_avg_reward = per_env_avg_rewards[valid_envs].mean().item()
            overall_avg_steps = per_env_avg_steps[valid_envs].mean().item()

            if print_game_res:
                per_env_avg_game_res[valid_envs] = (
                    per_env_game_res[valid_envs] / per_env_games_played[valid_envs]
                )
                overall_winrate = per_env_avg_game_res[valid_envs].mean().item()
                print(
                    "av reward:",
                    overall_avg_reward * n_game_life,
                    "av steps:",
                    overall_avg_steps * n_game_life,
                    "winrate:",
                    overall_winrate * n_game_life,
                )
            else:
                print(
                    "av reward:",
                    overall_avg_reward * n_game_life,
                    "av steps:",
                    overall_avg_steps * n_game_life,
                )
        else:
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
        if self._random_latents_resets:
            self._update_latents()

        obs = obs_dict["obs"]
        if len(obs.size()) == len(self.obs_shape):
            obs = obs.unsqueeze(0)
        obs = self._preproc_obs(obs)
        ase_latents = self._ase_latents

        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs,
            "rnn_states": self.states,
            "ase_latents": ase_latents,
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

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _reset_latents(self, done_env_ids=None):
        print("Resetting latents!")
        if done_env_ids is None:
            done_env_ids = convert_to_torch(
                np.arange(self.num_envs), dtype=torch.long, device=self.device
            )

        rand_vals = self.model.a2c_network.sample_latents(len(done_env_ids))
        self._ase_latents[done_env_ids] = rand_vals

    def _update_latents(self):
        if self._latent_step_count <= 0:
            self._reset_latents()
            self._reset_latent_step_count()
        else:
            self._latent_step_count -= 1

    def _reset_latent_step_count(self):
        self._latent_step_count = np.random.randint(
            self._latent_steps_min, self._latent_steps_max
        )

    def _build_net(self, config):
        super()._build_net(config)

        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(config["amp_input_shape"]).to(
                self.device
            )
            self._amp_input_mean_std.eval()

    def _build_net_config(self):
        config = super()._build_net_config()
        if hasattr(self, "env") and self.env != None:
            config["amp_input_shape"] = self.env.unwrapped.amp_observation_space.shape
            config["amp_obs_steps"] = self.env.unwrapped._num_amp_obs_steps
        else:
            config["amp_input_shape"] = self.env_info["amp_observation_space"].shape
            config["amp_obs_steps"] = self.env_info["num_amp_obs_steps"]

        self._amp_observation_space = config["amp_input_shape"]
        self._amp_obs_steps = config["amp_obs_steps"]

        config["ase_latent_shape"] = (self._latent_dim,)
        return config

    def restore(self, fn):
        if fn != "Base":
            super().restore(fn)
            if self._normalize_amp_input:
                checkpoint = torch_ext.load_checkpoint(fn)
                self._amp_input_mean_std.load_state_dict(
                    checkpoint["amp_input_mean_std"]
                )
