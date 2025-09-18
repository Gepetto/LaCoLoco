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


import torch
import torch.nn as nn

from rl_games.algos_torch import network_builder
from rl_games.algos_torch import torch_ext


class HRLBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.get("actions_num")
            input_shape = kwargs.get("input_shape")
            self.value_size = kwargs.get("value_size", 1)
            self.num_seqs = num_seqs = kwargs.get("num_seqs", 1)

            network_builder.NetworkBuilder.BaseNetwork.__init__(self)

            self.load(params)
            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()

            if self.has_cnn:
                if self.permute_input:
                    input_shape = torch_ext.shape_whc_to_cwh(input_shape)
                cnn_args = {
                    "ctype": self.cnn["type"],
                    "input_shape": input_shape,
                    "convs": self.cnn["convs"],
                    "activation": self.cnn["activation"],
                    "norm_func_name": self.normalization,
                }
                self.actor_cnn = self._build_conv(**cnn_args)

                if self.separate:
                    self.critic_cnn = self._build_conv(**cnn_args)

            mlp_input_shape = self._calc_input_size(input_shape, self.actor_cnn)

            in_mlp_shape = mlp_input_shape
            if len(self.units) == 0:
                out_size = mlp_input_shape
            else:
                out_size = self.units[-1]

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    rnn_in_size = out_size
                    out_size = self.rnn_units
                    if self.rnn_concat_input:
                        rnn_in_size += in_mlp_shape
                else:
                    rnn_in_size = in_mlp_shape
                    in_mlp_shape = self.rnn_units

                if self.separate:
                    self.a_rnn = self._build_rnn(
                        self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers
                    )
                    self.c_rnn = self._build_rnn(
                        self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers
                    )
                    if self.rnn_ln:
                        self.a_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                        self.c_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                else:
                    self.rnn = self._build_rnn(
                        self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers
                    )
                    if self.rnn_ln:
                        self.layer_norm = torch.nn.LayerNorm(self.rnn_units)

            mlp_args = {
                "input_size": in_mlp_shape,
                "units": self.units,
                "activation": self.activation,
                "norm_func_name": self.normalization,
                "dense_func": torch.nn.Linear,
                "d2rl": self.is_d2rl,
                "norm_only_first_layer": self.norm_only_first_layer,
            }
            self.actor_mlp = self._build_mlp(**mlp_args)
            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)

            self.value = self._build_value_layer(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            if self.is_discrete:
                self.logits = torch.nn.Linear(out_size, actions_num)
            """
                for multidiscrete actions num is a tuple
            """
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList(
                    [torch.nn.Linear(out_size, num) for num in actions_num]
                )
            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(
                    self.space_config["mu_activation"]
                )
                mu_init = self.init_factory.create(**self.space_config["mu_init"])
                self.sigma_act = self.activations_factory.create(
                    self.space_config["sigma_activation"]
                )

                sigma_init = self.init_factory.create(**self.space_config["sigma_init"])

                if self.fixed_sigma:
                    sigma_grad = self.space_config["sigma_grad"]
                    self.sigma = nn.Parameter(
                        torch.zeros(
                            actions_num, requires_grad=sigma_grad, dtype=torch.float32
                        ),
                        requires_grad=sigma_grad,
                    )
                else:
                    self.sigma = torch.nn.Linear(out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn["initializer"])

            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)

        def forward(self, obs_dict):
            mu, sigma, value, states = super().forward(obs_dict)
            norm_mu = torch.tanh(mu)
            return norm_mu, sigma, value, states

        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)
            value = self.value_act(self.value(c_out))
            return value

    def build(self, name, **kwargs):
        net = HRLBuilder.Network(self.params, **kwargs)
        return net
