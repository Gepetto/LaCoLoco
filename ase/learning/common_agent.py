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

import os
import time

import torch
import torch.nn as nn
from torch import optim

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from rl_games.common import a2c_common
from rl_games.common import datasets
from rl_games.common import common_losses

from ase_envs.tasks.utils.rl_games_cat.cat_experience import CaTExperienceBuffer


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


class CommonAgent(a2c_common.A2CBase):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)

        self._load_config_params(params["config"])

        self.is_discrete = False
        self._setup_action_space()
        self.bounds_loss_coef = params["config"].get("bounds_loss_coef", None)
        self.clip_actions = params["config"].get("clip_actions", True)
        self._save_intermediate = params["config"].get("save_intermediate", False)
        self._save_intermediate_freq = params["config"].get(
            "save_intermediate_freq", 4000
        )

        net_config = self._build_net_config()
        self.model = self.network.build(net_config)
        self.model.to(self.ppo_device)
        self.states = None

        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)

        self.bound_loss_type = self.config.get(
            "bound_loss_type", "bound"
        )  # 'regularisation' or 'bound'

        self.optimizer = optim.Adam(
            self.model.parameters(),
            float(self.last_lr),
            eps=1e-08,
            weight_decay=self.weight_decay,
        )

        if self.has_central_value:
            cv_config = {
                "state_shape": torch_ext.shape_whc_to_cwh(self.state_shape),
                "value_size": self.value_size,
                "ppo_device": self.ppo_device,
                "num_agents": self.num_agents,
                "horizon_length": self.horizon_length,
                "num_actors": self.num_actors,
                "num_actions": self.actions_num,
                "seq_length": self.seq_length,
                "normalize_value": self.normalize_value,
                "network": self.central_value_config["network"],
                "config": self.central_value_config,
                "writter": self.writer,
                "max_epochs": self.max_epochs,
                "multi_gpu": self.multi_gpu,
                "zero_rnn_on_done": self.zero_rnn_on_done,
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(
                self.ppo_device
            )

        self.use_experimental_cv = self.config.get("use_experimental_cv", True)
        self.dataset = datasets.PPODataset(
            self.batch_size,
            self.minibatch_size,
            self.is_discrete,
            self.is_rnn,
            self.ppo_device,
            self.seq_length,
        )
        if self.normalize_value:
            self.value_mean_std = (
                self.central_value_net.model.value_mean_std
                if self.has_central_value
                else self.model.value_mean_std
            )

        self.has_value_loss = self.use_experimental_cv or not self.has_central_value
        self.algo_observer.after_init(self)

    def _load_config_params(self, config):
        self.last_lr = config["learning_rate"]

    def _build_net_config(self):
        obs_shape = self.obs_shape
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
        self.update_list = ["actions", "neglogpacs", "values", "mus", "sigmas"]
        self.tensor_list = self.update_list + ["obses", "states", "dones"]

        batch_size = self.num_agents * self.num_actors
        algo_info = {
            "num_actors": self.num_actors,
            "horizon_length": self.horizon_length,
            "has_central_value": self.has_central_value,
            "use_action_masks": self.use_action_masks,
        }

        # Experience buffer that uses float32 dones
        self.experience_buffer = CaTExperienceBuffer(
            self.env_info, algo_info, self.ppo_device
        )

        # Redefining dones as float32 instead of uint8
        self.dones = torch.ones(
            (batch_size,), dtype=torch.float32, device=self.ppo_device
        )

        self.experience_buffer.tensor_dict["next_obses"] = torch.zeros_like(
            self.experience_buffer.tensor_dict["obses"]
        )

        self.tensor_list += ["next_obses"]

    def _init_train(self):
        pass

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.frame = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        model_output_file = os.path.join(self.nn_dir, self.config["name"])

        self._init_train()

        while True:
            epoch_num = self.update_epoch()
            train_info = self.train_epoch()

            sum_time = train_info["total_time"]
            total_time += sum_time
            frame = self.frame

            scaled_time = sum_time
            scaled_play_time = train_info["play_time"]
            curr_frames = self.curr_frames
            self.frame += curr_frames
            if self.print_stats:
                print(f"Epoch {epoch_num} / {self.max_epochs}")

            if self.writer:
                self.writer.add_scalar(
                    "performance/total_fps", curr_frames / scaled_time, frame
                )
                self.writer.add_scalar(
                    "performance/step_fps", curr_frames / scaled_play_time, frame
                )
                self.writer.add_scalar("info/epochs", epoch_num, frame)
                self._log_train_info(train_info, frame)

            self.algo_observer.after_print_stats(frame, epoch_num, total_time)

            if self.game_rewards.current_size > 0:
                mean_rewards = self.game_rewards.get_mean()
                mean_lengths = self.game_lengths.get_mean()

                if self.writer:
                    for i in range(self.value_size):
                        self.writer.add_scalar(
                            "rewards{0}/frame".format(i), mean_rewards[i], frame
                        )
                        self.writer.add_scalar(
                            "rewards{0}/iter".format(i), mean_rewards[i], epoch_num
                        )
                        self.writer.add_scalar(
                            "rewards{0}/time".format(i), mean_rewards[i], total_time
                        )

                    self.writer.add_scalar("episode_lengths/frame", mean_lengths, frame)
                    self.writer.add_scalar(
                        "episode_lengths/iter", mean_lengths, epoch_num
                    )

                if self.has_self_play_config:
                    self.self_play_manager.update(self)

            if self.save_freq > 0:
                if epoch_num % self.save_freq == 0:
                    self.save(model_output_file)

            if self._save_intermediate and self._save_intermediate_freq > 0:
                if epoch_num % self._save_intermediate_freq == 0:
                    int_model_output_file = (
                        model_output_file + "_" + str(epoch_num).zfill(8)
                    )
                    self.save(int_model_output_file)

            if epoch_num > self.max_epochs:
                self.save(model_output_file)
                print("MAX EPOCHS NUM!")
                return self.last_mean_rewards, epoch_num

            update_time = 0

    def train_epoch(self):
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

        train_info["play_time"] = play_time
        train_info["update_time"] = update_time
        train_info["total_time"] = total_time
        self._record_train_batch_info(batch_dict, train_info)

        return train_info

    def play_steps(self):
        raise NotImplementedError

    def train_actor_critic(self, input_dict):
        self.calc_gradients(input_dict)
        return self.train_result

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

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {"is_train": True, "prev_actions": actions_batch, "obs": obs_batch}

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict["rnn_masks"]
            batch_dict["rnn_states"] = input_dict["rnn_states"]
            batch_dict["seq_length"] = self.seq_length

            if self.zero_rnn_on_done:
                batch_dict["dones"] = input_dict["dones"]

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]

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

            a_loss = torch.mean(a_loss)
            c_loss = torch.mean(c_loss)
            b_loss = torch.mean(b_loss)
            entropy = torch.mean(entropy)

            loss = (
                a_loss
                + 0.5 * c_loss * self.critic_coef
                - entropy * self.entropy_coef
                + b_loss * self.bounds_loss_coef
            )

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
            "mu": mu.detach(),
            "sigma": sigma.detach(),
        }

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num

    def prepare_dataset(self, batch_dict):
        obses = batch_dict["obses"]
        returns = batch_dict["returns"]
        dones = batch_dict["dones"]
        values = batch_dict["values"]
        actions = batch_dict["actions"]
        neglogpacs = batch_dict["neglogpacs"]
        mus = batch_dict["mus"]
        sigmas = batch_dict["sigmas"]
        rnn_states = batch_dict.get("rnn_states", None)
        rnn_masks = batch_dict.get("rnn_masks", None)

        advantages = self._calc_advs(batch_dict)

        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()

        dataset_dict = {}
        dataset_dict["old_values"] = values
        dataset_dict["old_logp_actions"] = neglogpacs
        dataset_dict["advantages"] = advantages
        dataset_dict["returns"] = returns
        dataset_dict["actions"] = actions
        dataset_dict["obs"] = obses
        dataset_dict["rnn_states"] = rnn_states
        dataset_dict["rnn_masks"] = rnn_masks
        dataset_dict["mu"] = mus
        dataset_dict["sigma"] = sigmas

        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict["old_values"] = values
            dataset_dict["advantages"] = advantages
            dataset_dict["returns"] = returns
            dataset_dict["actions"] = actions
            dataset_dict["obs"] = batch_dict["states"]
            dataset_dict["rnn_masks"] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def reg_loss(self, mu):
        if self.bounds_loss_coef is not None:
            reg_loss = (mu * mu).sum(axis=-1)
        else:
            reg_loss = 0
        return reg_loss

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss

    def preprocess_actions(self, actions):
        if self.clip_actions:
            clamped_actions = torch.clamp(actions, -1.0, 1.0)
            rescaled_actions = rescale_actions(
                self.actions_low, self.actions_high, clamped_actions
            )
        else:
            rescaled_actions = actions

        if not self.is_tensor_obses:
            rescaled_actions = rescaled_actions.cpu().numpy()

        return rescaled_actions

    def _setup_action_space(self):
        action_space = self.env_info["action_space"]
        self.actions_num = action_space.shape[0]

        self.actions_low = (
            torch.from_numpy(action_space.low.copy()).float().to(self.ppo_device)
        )
        self.actions_high = (
            torch.from_numpy(action_space.high.copy()).float().to(self.ppo_device)
        )

    def _calc_advs(self, batch_dict):
        returns = batch_dict["returns"]
        values = batch_dict["values"]

        advantages = returns - values
        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def restore(self, fn, set_epoch=True):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch)

    def _record_train_batch_info(self, batch_dict, train_info):
        pass

    def _log_train_info(self, train_info, frame):
        self.writer.add_scalar(
            "performance/update_time", train_info["update_time"], frame
        )
        self.writer.add_scalar("performance/play_time", train_info["play_time"], frame)
        self.writer.add_scalar(
            "losses/a_loss", torch_ext.mean_list(train_info["actor_loss"]).item(), frame
        )
        self.writer.add_scalar(
            "losses/c_loss",
            torch_ext.mean_list(train_info["critic_loss"]).item(),
            frame,
        )

        self.writer.add_scalar(
            "losses/bounds_loss",
            torch_ext.mean_list(train_info["b_loss"]).item(),
            frame,
        )
        self.writer.add_scalar(
            "losses/entropy", torch_ext.mean_list(train_info["entropy"]).item(), frame
        )
        self.writer.add_scalar(
            "info/last_lr", train_info["last_lr"][-1] * train_info["lr_mul"][-1], frame
        )
        self.writer.add_scalar("info/lr_mul", train_info["lr_mul"][-1], frame)
        self.writer.add_scalar(
            "info/e_clip", self.e_clip * train_info["lr_mul"][-1], frame
        )
        self.writer.add_scalar(
            "info/kl", torch_ext.mean_list(train_info["kl"]).item(), frame
        )
