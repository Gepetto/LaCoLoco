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

import time
import numpy as np
import torch
import torch.nn as nn

from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.common import common_losses

from omni.isaac.lab.utils.array import convert_to_torch

import learning.replay_buffer as replay_buffer
import learning.common_agent as common_agent


class CommonASEAgent(common_agent.CommonAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)

        # better to do common normalization for same kinds of observations but in isaaclab there
        # is a different order of history, so it can't be done with
        #  // self.vec_env.env.unwrapped._num_amp_obs_steps,
        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(
                self._amp_observation_space.shape
            ).to(self.ppo_device)

    def _load_config_params(self, config):
        super()._load_config_params(config)

        # when eps greedy is enabled, rollouts will be generated using a mixture of
        # a deterministic and stochastic actions. The deterministic actions help to
        # produce smoother, less noisy, motions that can be used to train a better
        # discriminator. If the discriminator is only trained with jittery motions
        # from noisy actions, it can learn to phone in on the jitteriness to
        # differential between real and fake samples.
        self._enable_eps_greedy = bool(config["enable_eps_greedy"])

        self._task_reward_w = config["task_reward_w"]
        self._disc_reward_w = config["disc_reward_w"]

        self._amp_observation_space = self.env_info["amp_observation_space"]

        if self.vec_env is not None:
            self._num_amp_obs_steps = self.vec_env.env.unwrapped._num_amp_obs_steps
        else:
            self._num_amp_obs_steps = self.env_info["num_amp_obs_steps"]

        self._amp_batch_size = int(config["amp_batch_size"])
        self._amp_minibatch_size = int(config["amp_minibatch_size"])
        assert self._amp_minibatch_size <= self.minibatch_size

        self._disc_coef = config["disc_coef"]
        self._disc_logit_reg = config["disc_logit_reg"]
        self._disc_grad_penalty = config["disc_grad_penalty"]
        self._disc_weight_decay = config["disc_weight_decay"]
        self._disc_reward_scale = config["disc_reward_scale"]
        self._normalize_amp_input = config.get("normalize_amp_input", True)

        self._latent_dim = config["latent_dim"]
        self._latent_steps_min = config.get("latent_steps_min", np.inf)
        self._latent_steps_max = config.get("latent_steps_max", np.inf)
        self._latent_dim = config["latent_dim"]
        self._amp_diversity_bonus = config["amp_diversity_bonus"]
        self._amp_diversity_tar = config["amp_diversity_tar"]

        self._enc_coef = config["enc_coef"]
        self._enc_weight_decay = config["enc_weight_decay"]
        self._enc_reward_scale = config["enc_reward_scale"]
        self._enc_grad_penalty = config["enc_grad_penalty"]

        self._enc_reward_w = config["enc_reward_w"]

    def _build_net_config(self):
        config = super()._build_net_config()
        config["amp_input_shape"] = self._amp_observation_space.shape
        config["amp_obs_steps"] = self._num_amp_obs_steps
        config["ase_latent_shape"] = (self._latent_dim,)
        return config

    def init_tensors(self):
        super().init_tensors()
        self._build_amp_buffers()

        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict["ase_latents"] = torch.zeros(
            batch_shape + (self._latent_dim,),
            dtype=torch.float32,
            device=self.ppo_device,
        )

        self.experience_buffer.tensor_dict["cstr_prob"] = torch.zeros_like(
            self.experience_buffer.tensor_dict["dones"]
        )
        self._ase_latents = torch.zeros(
            (batch_shape[-1], self._latent_dim),
            dtype=torch.float32,
            device=self.ppo_device,
        )

        self.tensor_list += ["ase_latents"]

        self._latent_reset_steps = torch.zeros(
            batch_shape[-1], dtype=torch.int32, device=self.ppo_device
        )
        num_envs = self.vec_env.env.unwrapped.num_envs
        env_ids = convert_to_torch(
            np.arange(num_envs), dtype=torch.long, device=self.ppo_device
        )
        self._reset_latent_step_count(env_ids)

    def _init_train(self):
        super()._init_train()
        self._init_amp_demo_buf()

    def train_epoch(self):
        self.vec_env.set_train_info(self.frame, self)

        self.set_eval()
        play_time_start = time.perf_counter()

        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.perf_counter()
        update_time_start = time.perf_counter()
        rnn_masks = batch_dict.get("rnn_masks", None)

        self._update_amp_demos()
        num_obs_samples = batch_dict["amp_obs"].shape[0]
        amp_obs_demo = self._amp_obs_demo_buffer.sample(num_obs_samples)["amp_obs"]
        batch_dict["amp_obs_demo"] = amp_obs_demo

        if self._amp_replay_buffer.get_total_count() == 0:
            batch_dict["amp_obs_replay"] = batch_dict["amp_obs"]
        else:
            batch_dict["amp_obs_replay"] = self._amp_replay_buffer.sample(
                num_obs_samples
            )["amp_obs"]

        self.set_train()

        self.curr_frames = batch_dict.pop("played_frames")
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        train_info = None

        if self.is_rnn:
            frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())
            print(frames_mask_ratio)

        kls = []

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])
                ep_kls.append(curr_train_info["kl"])
                self.dataset.update_mu_sigma(
                    curr_train_info["mu"], curr_train_info["sigma"]
                )
                if self.schedule_type == "legacy":
                    av_kls = curr_train_info["kl"]
                    self.last_lr, self.entropy_coef = self.scheduler.update(
                        self.last_lr,
                        self.entropy_coef,
                        self.epoch_num,
                        0,
                        av_kls.item(),
                    )
                    self.update_lr(self.last_lr)

                if train_info is None:
                    train_info = dict()
                    for k, v in curr_train_info.items():
                        train_info[k] = [v]
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)

            av_kls = torch_ext.mean_list(ep_kls)

            if self.schedule_type == "standard":
                self.last_lr, self.entropy_coef = self.scheduler.update(
                    self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item()
                )
                self.update_lr(self.last_lr)
            kls.append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_std.eval()  # don't need to update statstics more than one miniepoch

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        self._store_replay_amp_obs(batch_dict["amp_obs"])

        train_info["play_time"] = play_time
        train_info["update_time"] = update_time
        train_info["total_time"] = total_time
        self._record_train_batch_info(batch_dict, train_info)

        return train_info

    def play_steps(self):
        update_list = self.update_list
        done_indices = []

        for n in range(self.horizon_length):
            if len(done_indices) > 0:
                self._reset_latents(done_indices)
                self._reset_latent_step_count(done_indices)

            self._update_latents()

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(
                    self.obs, self._ase_latents, masks
                )
            else:
                res_dict = self.get_action_values(
                    self.obs, self._ase_latents, self._rand_action_probs
                )
            self.experience_buffer.update_data("obses", n, self.obs["obs"])
            self.experience_buffer.update_data("dones", n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data("states", n, self.obs["states"])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])

            shaped_rewards = self.rewards_shaper(rewards)

            if self.value_bootstrap and "time_outs" in infos:
                shaped_rewards += (
                    self.gamma
                    * res_dict["values"]
                    * self.cast_obs(infos["time_outs"]).unsqueeze(1).float()
                )

            self.experience_buffer.update_data("rewards", n, shaped_rewards)
            self.experience_buffer.update_data("next_obses", n, self.obs["obs"])
            self.experience_buffer.update_data("amp_obs", n, self.obs["states"])
            self.experience_buffer.update_data("ase_latents", n, self._ase_latents)
            self.experience_buffer.update_data(
                "rand_action_mask", n, res_dict["rand_action_mask"]
            )
            self.experience_buffer.update_data("cstr_prob", n, infos["cstr_prob"])

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

            done_indices = env_done_indices[:, 0]

        last_values = self.get_values(self.obs, self._ase_latents)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_rewards = self.experience_buffer.tensor_dict["rewards"]
        mb_amp_obs = self.experience_buffer.tensor_dict["amp_obs"]
        mb_ase_latents = self.experience_buffer.tensor_dict["ase_latents"]

        mb_cstr_prob = self.experience_buffer.tensor_dict["cstr_prob"]
        amp_rewards = self._calc_amp_rewards(mb_amp_obs, mb_ase_latents)

        amp_rewards["disc_rewards"] = amp_rewards["disc_rewards"] * (
            1.0 - mb_cstr_prob.unsqueeze(-1)
        )  # <-- Modified line CaT

        amp_rewards["enc_rewards"] = amp_rewards["enc_rewards"] * (
            1.0 - mb_cstr_prob.unsqueeze(-1)
        )  # <-- Modified line CaT

        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)
        mb_advs = self.discount_values(
            fdones, last_values, mb_fdones, mb_values, mb_rewards
        )
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(
            a2c_common.swap_and_flatten01, self.tensor_list
        )
        batch_dict["returns"] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict["played_frames"] = self.batch_size

        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)

        return batch_dict

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        return_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = input_dict["obs"]
        obs_batch = self._preproc_obs(obs_batch)

        amp_obs = input_dict["amp_obs"][0 : self._amp_minibatch_size]
        amp_obs = self._preproc_amp_obs(amp_obs)
        if self._enable_enc_grad_penalty():
            amp_obs.requires_grad_(True)

        amp_obs_replay = input_dict["amp_obs_replay"][0 : self._amp_minibatch_size]
        amp_obs_replay = self._preproc_amp_obs(amp_obs_replay)

        amp_obs_demo = input_dict["amp_obs_demo"][0 : self._amp_minibatch_size]
        amp_obs_demo = self._preproc_amp_obs(amp_obs_demo)
        amp_obs_demo.requires_grad_(True)

        ase_latents = input_dict["ase_latents"]

        rand_action_mask = input_dict["rand_action_mask"]
        rand_action_sum = torch.sum(rand_action_mask)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch,
            "amp_obs": amp_obs,
            "amp_obs_replay": amp_obs_replay,
            "amp_obs_demo": amp_obs_demo,
            "ase_latents": ase_latents,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict["rnn_masks"]
            batch_dict["rnn_states"] = input_dict["rnn_states"]
            batch_dict["seq_length"] = self.seq_length

            if self.zero_rnn_on_done:
                batch_dict["dones"] = input_dict["dones"]

        a_info = {}

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]
            disc_agent_logit = res_dict["disc_agent_logit"]
            disc_agent_replay_logit = res_dict["disc_agent_replay_logit"]
            disc_demo_logit = res_dict["disc_demo_logit"]
            enc_pred = res_dict["enc_pred"]

            a_loss = self.actor_loss_func(
                old_action_log_probs_batch,
                action_log_probs,
                advantage,
                self.ppo,
                curr_e_clip,
            )

            c_loss = common_losses.default_critic_loss(
                value_preds_batch, values, curr_e_clip, return_batch, self.clip_value
            )

            if self.bound_loss_type == "regularisation":
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == "bound":
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)

            c_loss = torch.mean(c_loss)
            a_loss = torch.sum(rand_action_mask * a_loss) / rand_action_sum
            entropy = torch.sum(rand_action_mask * entropy) / rand_action_sum
            b_loss = torch.sum(rand_action_mask * b_loss) / rand_action_sum

            disc_agent_cat_logit = torch.cat(
                [disc_agent_logit, disc_agent_replay_logit], dim=0
            )
            disc_loss, disc_info = self._disc_loss(
                disc_agent_cat_logit, disc_demo_logit, amp_obs_demo
            )

            enc_latents = batch_dict["ase_latents"][0 : self._amp_minibatch_size]
            enc_loss_mask = rand_action_mask[0 : self._amp_minibatch_size]
            enc_loss, enc_info = self._enc_loss(
                enc_pred, enc_latents, batch_dict["amp_obs"], enc_loss_mask
            )

            loss = (
                a_loss
                + 0.5 * c_loss * self.critic_coef
                - entropy * self.entropy_coef
                + b_loss * self.bounds_loss_coef
                + disc_loss * self._disc_coef
                + enc_loss * self._enc_coef
            )

            if self._enable_amp_diversity_bonus():
                diversity_loss = self._diversity_loss(
                    batch_dict["obs"], mu, batch_dict["ase_latents"]
                )
                diversity_loss = (
                    torch.sum(rand_action_mask * diversity_loss) / rand_action_sum
                )
                loss += self._amp_diversity_bonus * diversity_loss
                a_info["amp_diversity_loss"] = diversity_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        if self.truncate_grads:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(
                mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl
            )
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  # / sum_mask

        self.train_result = {
            "actor_loss": a_loss,
            "critic_loss": c_loss,
            "entropy": entropy,
            "kl": kl_dist,
            "last_lr": self.last_lr,
            "lr_mul": lr_mul,
            "b_loss": b_loss,
            "disc_loss": disc_loss,
            "enc_loss": enc_loss,
            "mu": mu.detach(),
            "sigma": sigma.detach(),
        }
        self.train_result.update(a_info)
        self.train_result.update(enc_info)
        self.train_result.update(disc_info)

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)

        self.dataset.values_dict["amp_obs"] = batch_dict["amp_obs"]
        self.dataset.values_dict["amp_obs_demo"] = batch_dict["amp_obs_demo"]
        self.dataset.values_dict["amp_obs_replay"] = batch_dict["amp_obs_replay"]

        rand_action_mask = batch_dict["rand_action_mask"]
        self.dataset.values_dict["rand_action_mask"] = rand_action_mask

        ase_latents = batch_dict["ase_latents"]
        self.dataset.values_dict["ase_latents"] = ase_latents

    def set_eval(self):
        super().set_eval()
        if self._normalize_amp_input:
            self._amp_input_mean_std.eval()

    def set_train(self):
        super().set_train()
        if self._normalize_amp_input:
            self._amp_input_mean_std.train()

    def get_stats_weights(self):
        state = super().get_stats_weights()
        if self._normalize_amp_input:
            state["amp_input_mean_std"] = self._amp_input_mean_std.state_dict()
        return state

    def set_stats_weights(self, weights):
        super().set_stats_weights(weights)
        if self._normalize_amp_input:
            self._amp_input_mean_std.load_state_dict(weights["amp_input_mean_std"])

    def get_values(self, obs, ase_latents):
        with torch.no_grad():
            if self.has_central_value:
                states = obs["states"]
                self.central_value_net.eval()
                input_dict = {
                    "is_train": False,
                    "states": states,
                    "actions": None,
                    "is_done": self.dones,
                    "ase_latents": ase_latents,
                }
                value = self.get_central_value(input_dict)
            else:
                self.model.eval()
                processed_obs = self._preproc_obs(obs["obs"])
                input_dict = {
                    "is_train": False,
                    "prev_actions": None,
                    "obs": processed_obs,
                    "rnn_states": self.rnn_states,
                    "ase_latents": ase_latents,
                }
                result = self.model(input_dict)
                value = result["values"]
            return value

    def get_action_values(self, obs_dict, ase_latents, rand_action_probs):
        processed_obs = self._preproc_obs(obs_dict["obs"])
        self.model.eval()
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": processed_obs,
            "rnn_states": self.rnn_states,
            "ase_latents": ase_latents,
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs_dict["states"]
                input_dict = {
                    "is_train": False,
                    "states": states,
                }
                value = self.get_central_value(input_dict)
                res_dict["values"] = value

        rand_action_mask = torch.bernoulli(rand_action_probs)
        det_action_mask = rand_action_mask == 0.0
        res_dict["actions"][det_action_mask] = res_dict["mus"][det_action_mask]
        res_dict["rand_action_mask"] = rand_action_mask

        return res_dict

    def env_reset(self):
        obs = super().env_reset()

        num_envs = self.vec_env.env.unwrapped.num_envs
        env_ids = convert_to_torch(
            np.arange(num_envs), dtype=torch.long, device=self.ppo_device
        )

        if len(env_ids) > 0:
            self._reset_latents(env_ids)
            self._reset_latent_step_count(env_ids)

        return obs

    def _reset_latent_step_count(self, env_ids):
        self._latent_reset_steps[env_ids] = torch.randint_like(
            self._latent_reset_steps[env_ids],
            low=self._latent_steps_min,
            high=self._latent_steps_max,
        )

    def _reset_latents(self, env_ids):
        n = len(env_ids)
        z = self._sample_latents(n)
        self._ase_latents[env_ids] = z

    def _sample_latents(self, n):
        z = self.model.a2c_network.sample_latents(n)
        return z

    def _update_latents(self):
        new_latent_envs = (
            self._latent_reset_steps <= self.vec_env.env.unwrapped.progress_buf
        )

        need_update = torch.any(new_latent_envs)
        if need_update:
            new_latent_env_ids = new_latent_envs.nonzero(as_tuple=False).flatten()
            self._reset_latents(new_latent_env_ids)
            self._latent_reset_steps[new_latent_env_ids] += torch.randint_like(
                self._latent_reset_steps[new_latent_env_ids],
                low=self._latent_steps_min,
                high=self._latent_steps_max,
            )

    def _eval_actor(self, obs, ase_latents):
        output = self.model.a2c_network.eval_actor(obs=obs, ase_latents=ase_latents)
        return output

    def _calc_amp_rewards(self, amp_obs, ase_latents):
        raise NotImplementedError

    def _calc_enc_rewards(self, amp_obs, ase_latents):
        with torch.no_grad():
            enc_pred = self._eval_enc(amp_obs)
            err = self._calc_enc_error(enc_pred, ase_latents)
            enc_r = torch.clamp_min(-err, 0.0)
            enc_r *= self._enc_reward_scale

        return enc_r

    def _enc_loss(self, enc_pred, ase_latent, enc_obs, loss_mask):
        enc_err = self._calc_enc_error(enc_pred, ase_latent)
        enc_loss = torch.mean(enc_err)

        # weight decay
        if self._enc_weight_decay != 0:
            enc_weights = self.model.a2c_network.get_enc_weights()
            enc_weights = torch.cat(enc_weights, dim=-1)
            enc_weight_decay = torch.sum(torch.square(enc_weights))
            enc_loss += self._enc_weight_decay * enc_weight_decay

        enc_info = {}

        if self._enable_enc_grad_penalty():
            enc_obs_grad = torch.autograd.grad(
                enc_err,
                enc_obs,
                grad_outputs=torch.ones_like(enc_err),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )
            enc_obs_grad = enc_obs_grad[0]
            enc_obs_grad = torch.sum(torch.square(enc_obs_grad), dim=-1)
            enc_grad_penalty = torch.mean(enc_obs_grad)

            enc_loss += self._enc_grad_penalty * enc_grad_penalty

            enc_info["enc_grad_penalty"] = enc_grad_penalty.detach()

        return enc_loss, enc_info

    def _diversity_loss(self, obs, action_params, ase_latents):
        assert self.model.a2c_network.is_continuous

        n = obs.shape[0]
        assert n == action_params.shape[0]

        new_z = self._sample_latents(n)
        mu, sigma = self._eval_actor(obs=obs, ase_latents=new_z)

        a_diff = action_params - mu
        a_diff = torch.mean(torch.square(a_diff), dim=-1)

        z_diff = new_z * ase_latents
        z_diff = torch.sum(z_diff, dim=-1)
        z_diff = 0.5 - 0.5 * z_diff

        diversity_bonus = a_diff / (z_diff + 1e-5)
        diversity_loss = torch.square(self._amp_diversity_tar - diversity_bonus)

        return diversity_loss

    def _calc_enc_error(self, enc_pred, ase_latent):
        err = enc_pred * ase_latent
        err = -torch.sum(err, dim=-1, keepdim=True)
        return err

    def _enable_enc_grad_penalty(self):
        return self._enc_grad_penalty != 0

    def _enable_amp_diversity_bonus(self):
        return self._amp_diversity_bonus != 0

    def _eval_enc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_enc(proc_amp_obs)

    def _combine_rewards(self, task_rewards, amp_rewards):
        disc_r = amp_rewards["disc_rewards"]
        enc_r = amp_rewards["enc_rewards"]
        combined_rewards = (
            self._task_reward_w * task_rewards
            + self._disc_reward_w * disc_r
            + self._enc_reward_w * enc_r
        )
        return combined_rewards

    def _build_rand_action_probs(self):
        num_envs = self.vec_env.env.unwrapped.num_envs
        env_ids = convert_to_torch(
            np.arange(num_envs), dtype=torch.float32, device=self.ppo_device
        )

        self._rand_action_probs = 1.0 - torch.exp(
            10 * (env_ids / (num_envs - 1.0) - 1.0)
        )
        self._rand_action_probs[0] = 1.0
        self._rand_action_probs[-1] = 0.0

        if not self._enable_eps_greedy:
            self._rand_action_probs[:] = 1.0

    def _init_amp_demo_buf(self):
        buffer_size = self._amp_obs_demo_buffer.get_buffer_size()
        num_batches = int(np.ceil(buffer_size / self._amp_batch_size))

        for i in range(num_batches):
            curr_samples = self._fetch_amp_obs_demo(self._amp_batch_size)
            self._amp_obs_demo_buffer.store({"amp_obs": curr_samples})

    def _fetch_amp_obs_demo(self, num_samples):
        amp_obs_demo = self.vec_env.env.unwrapped.fetch_amp_obs_demo(num_samples)
        return amp_obs_demo

    def _build_amp_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict["amp_obs"] = torch.zeros(
            batch_shape + self._amp_observation_space.shape, device=self.ppo_device
        )
        self.experience_buffer.tensor_dict["rand_action_mask"] = torch.zeros(
            batch_shape, dtype=torch.float32, device=self.ppo_device
        )

        amp_obs_demo_buffer_size = int(self.config["amp_obs_demo_buffer_size"])
        self._amp_obs_demo_buffer = replay_buffer.ReplayBuffer(
            amp_obs_demo_buffer_size, self.ppo_device
        )

        self._amp_replay_keep_prob = self.config["amp_replay_keep_prob"]
        replay_buffer_size = int(self.config["amp_replay_buffer_size"])
        self._amp_replay_buffer = replay_buffer.ReplayBuffer(
            replay_buffer_size, self.ppo_device
        )

        self._build_rand_action_probs()

        self.tensor_list += ["amp_obs", "rand_action_mask"]

    def _update_amp_demos(self):
        new_amp_obs_demo = self._fetch_amp_obs_demo(self._amp_batch_size)
        self._amp_obs_demo_buffer.store({"amp_obs": new_amp_obs_demo})

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _calc_advs(self, batch_dict):
        returns = batch_dict["returns"]
        values = batch_dict["values"]
        rand_action_mask = batch_dict["rand_action_mask"]

        advantages = returns - values
        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            advantages = torch_ext.normalization_with_masks(
                advantages, rand_action_mask
            )

        return advantages

    def _store_replay_amp_obs(self, amp_obs):
        buf_size = self._amp_replay_buffer.get_buffer_size()
        buf_total_count = self._amp_replay_buffer.get_total_count()
        if buf_total_count > buf_size:
            keep_probs = convert_to_torch(
                np.array([self._amp_replay_keep_prob] * amp_obs.shape[0]),
                device=self.ppo_device,
            )
            keep_mask = torch.bernoulli(keep_probs) == 1.0
            amp_obs = amp_obs[keep_mask]

        if amp_obs.shape[0] > buf_size:
            rand_idx = torch.randperm(amp_obs.shape[0])
            rand_idx = rand_idx[:buf_size]
            amp_obs = amp_obs[rand_idx]

        self._amp_replay_buffer.store({"amp_obs": amp_obs})

    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        train_info["disc_rewards"] = batch_dict["disc_rewards"]
        train_info["enc_rewards"] = batch_dict["enc_rewards"]

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)

        if self._disc_reward_w > 0:
            self.writer.add_scalar(
                "losses/disc_loss",
                torch_ext.mean_list(train_info["disc_loss"]).item(),
                frame,
            )

            self.writer.add_scalar(
                "info/disc_agent_acc",
                torch_ext.mean_list(train_info["disc_agent_acc"]).item(),
                frame,
            )
            self.writer.add_scalar(
                "info/disc_demo_acc",
                torch_ext.mean_list(train_info["disc_demo_acc"]).item(),
                frame,
            )
            self.writer.add_scalar(
                "info/disc_agent_logit",
                torch_ext.mean_list(train_info["disc_agent_logit"]).item(),
                frame,
            )
            self.writer.add_scalar(
                "info/disc_demo_logit",
                torch_ext.mean_list(train_info["disc_demo_logit"]).item(),
                frame,
            )

            disc_reward_std, disc_reward_mean = torch.std_mean(
                train_info["disc_rewards"]
            )
            self.writer.add_scalar(
                "info/disc_reward_mean", disc_reward_mean.item(), frame
            )
            self.writer.add_scalar(
                "info/disc_reward_std", disc_reward_std.item(), frame
            )

            if "disc_grad_penalty" in train_info:
                self.writer.add_scalar(
                    "info/disc_grad_penalty",
                    torch_ext.mean_list(train_info["disc_grad_penalty"]).item(),
                    frame,
                )
            if "disc_logit_loss" in train_info:
                self.writer.add_scalar(
                    "info/disc_logit_loss",
                    torch_ext.mean_list(train_info["disc_logit_loss"]).item(),
                    frame,
                )

        self.writer.add_scalar(
            "losses/enc_loss", torch_ext.mean_list(train_info["enc_loss"]).item(), frame
        )

        if self._enable_amp_diversity_bonus():
            self.writer.add_scalar(
                "losses/amp_diversity_loss",
                torch_ext.mean_list(train_info["amp_diversity_loss"]).item(),
                frame,
            )

        enc_reward_std, enc_reward_mean = torch.std_mean(train_info["enc_rewards"])
        self.writer.add_scalar("info/enc_reward_mean", enc_reward_mean.item(), frame)
        self.writer.add_scalar("info/enc_reward_std", enc_reward_std.item(), frame)

        if self._enable_enc_grad_penalty():
            self.writer.add_scalar(
                "info/enc_grad_penalty",
                torch_ext.mean_list(train_info["enc_grad_penalty"]).item(),
                frame,
            )
