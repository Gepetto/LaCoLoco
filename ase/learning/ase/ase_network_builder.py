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

from learning.common_ase import common_ase_network_builder

DISC_LOGIT_INIT_SCALE = 1.0


class ASEBuilder(common_ase_network_builder.CommonASEBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    class Network(common_ase_network_builder.CommonASEBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)

        def load(self, params):
            super().load(params)

            self._disc_units = params["disc"]["units"]
            self._disc_activation = params["disc"]["activation"]
            self._disc_initializer = params["disc"]["initializer"]

        def eval_disc(self, amp_obs):
            disc_mlp_out = self._disc_mlp(amp_obs)
            disc_logits = self._disc_logits(disc_mlp_out)
            return disc_logits

        def get_disc_logit_weights(self):
            return torch.flatten(self._disc_logits.weight)

        def get_disc_weights(self):
            weights = []
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))

            weights.append(torch.flatten(self._disc_logits.weight))
            return weights

        def _build_disc(self, input_shape, device=None):
            self._disc_mlp = nn.Sequential()

            mlp_args = {
                "input_size": input_shape[0],
                "units": self._disc_units,
                "activation": self._disc_activation,
                "dense_func": torch.nn.Linear,
            }
            self._disc_mlp = self._build_mlp(**mlp_args)

            mlp_out_size = self._disc_units[-1]
            self._disc_logits = torch.nn.Linear(mlp_out_size, 1)

            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            torch.nn.init.uniform_(
                self._disc_logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE
            )
            torch.nn.init.zeros_(self._disc_logits.bias)

    def build(self, name, **kwargs):
        net = ASEBuilder.Network(self.params, **kwargs)
        return net
