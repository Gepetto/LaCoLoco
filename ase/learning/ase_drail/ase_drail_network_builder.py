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
# based on https://github.com/NVlabs/DRAIL

import torch
import torch.nn as nn

from learning.common_ase import common_ase_network_builder


class MLPConditionDiffusion(nn.Module):
    def __init__(
        self,
        amp_obs_size,
        conditional_dim,
        units,
        activation,
        initializer,
        diffusion_steps,
    ):
        super().__init__()

        input_size = amp_obs_size + conditional_dim

        self._units = units
        self._initializer = initializer
        self._mlp = []

        in_size = input_size
        for i in range(len(units)):
            unit = units[i]
            curr_dense = torch.nn.Linear(in_size, unit)
            self._mlp.append(curr_dense)
            self._mlp.append(activation)
            in_size = unit
        self._mlp.append(torch.nn.Linear(in_size, amp_obs_size))
        self._mlp = nn.ModuleList(self._mlp)

        self._step_embeddings = nn.ModuleList(
            [nn.Embedding(diffusion_steps, unit) for unit in units]
        )

        self.init_params()

    def forward(self, amp_obs, conditional, t):
        x = torch.concat([amp_obs, conditional], dim=1)
        for idx, embedding_layer in enumerate(self._step_embeddings):
            t_embedding = embedding_layer(t)
            x = self._mlp[2 * idx](x)
            x += t_embedding
            x = self._mlp[2 * idx + 1](x)

        x = self._mlp[-1](x)

        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self._initializer(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class Discriminator(nn.Module):
    def __init__(
        self,
        amp_obs_size,
        conditional_dim,
        units,
        activation,
        initializer,
        diffusion_steps,
        sample_strategy,
        sample_strategy_value,
        device,
    ):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.sample_strategy = sample_strategy
        self.sample_strategy_value = sample_strategy_value
        self.conditional_dim = conditional_dim

        self.model = MLPConditionDiffusion(
            amp_obs_size,
            conditional_dim,
            units,
            activation,
            initializer,
            diffusion_steps,
        ).to(device)

        self.device = device

        betas = cosine_beta_schedule(diffusion_steps)
        self.betas = betas.to(self.device)
        alphas = 1 - betas
        alphas_prod = torch.cumprod(alphas, 0)
        self.alphas_bar_sqrt = torch.sqrt(alphas_prod).to(self.device)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod).to(self.device)

    def diffusion_loss(
        self, label, amp_obs, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps
    ):
        batch_size = amp_obs.shape[0]

        if self.sample_strategy == "constant":
            step = self.sample_strategy_value
            if step >= n_steps:
                step = n_steps - 1
            t = torch.full((batch_size,), step, device=self.device)
            t = t.unsqueeze(-1)
        else:
            t = torch.randint(0, n_steps, size=(batch_size // 2,)).to(self.device)
            t = torch.cat([t, n_steps - 1 - t], dim=0)  # [batch_size, 1]
            t = t.unsqueeze(-1)

        # # coefficient of x0
        a = alphas_bar_sqrt[t]

        # # coefficient of eps
        aml = one_minus_alphas_bar_sqrt[t]

        label_input = torch.full((batch_size, self.conditional_dim), label).to(
            self.device
        )

        # generate random noise eps
        e = torch.randn_like(amp_obs).to(self.device)

        # model input
        x = amp_obs * a + e * aml

        # get predicted randome noise at time t
        output = self.model(x, label_input, t.squeeze(-1))

        return (e - output).square().mean(dim=1, keepdim=True)

    def forward(self, amp_obs, label):
        return self.diffusion_loss(
            label,
            amp_obs,
            self.alphas_bar_sqrt,
            self.one_minus_alphas_bar_sqrt,
            self.diffusion_steps,
        )


class ASEDRAILBuilder(
    common_ase_network_builder.CommonASEBuilder,
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    class Network(
        common_ase_network_builder.CommonASEBuilder.Network,
    ):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)

        def load(self, params):
            super().load(params)

            self._disc_units = params["disc"]["units"]
            self._disc_activation = params["disc"]["activation"]
            self._disc_initializer = params["disc"]["initializer"]
            self._disc_label_dim = params["disc"]["label_dim"]
            self._disc_diffusion_steps = params["disc"]["diffusion_steps"]
            self._disc_sample_strategy = params["disc"]["sample_strategy"]
            self._disc_sample_strategy_value = params["disc"]["sample_strategy_value"]

        def eval_disc(self, amp_obs):
            label_one = self._disc_net(amp_obs, 1.0)
            label_zero = self._disc_net(amp_obs, 0.0)
            output = nn.functional.softmax(
                torch.stack([-label_one, -label_zero]), dim=0
            )[0]
            return output

        def get_disc_weights(self):
            weights = []
            for m in self._disc_net.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))

            return weights

        def _build_disc(self, input_shape, device):
            self._disc_net = Discriminator(
                input_shape[0],
                self._disc_label_dim,
                self._disc_units,
                self.activations_factory.create(self._disc_activation),
                self.init_factory.create(**self._disc_initializer),
                self._disc_diffusion_steps,
                self._disc_sample_strategy,
                self._disc_sample_strategy_value,
                device,
            )

    def build(self, name, **kwargs):
        net = ASEDRAILBuilder.Network(self.params, **kwargs)
        return net
