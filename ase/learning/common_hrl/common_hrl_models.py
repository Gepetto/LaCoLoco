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

from rl_games.algos_torch.models import ModelA2CContinuousLogStd


class ModelHRLContinuous(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)

    def build(self, config):
        net = self.network_builder.build("amp", **config)
        for name, param in net.named_parameters():
            print(name, param.shape)

        return ModelHRLContinuous.Network(
            net,
            obs_shape=config["input_shape"],
            normalize_value=config["normalize_value"],
            normalize_input=config["normalize_input"],
            value_size=config["value_size"],
        )

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(
            self, a2c_network, obs_shape, normalize_value, normalize_input, value_size
        ):
            super().__init__(
                a2c_network,
                obs_shape=obs_shape,
                normalize_value=normalize_value,
                normalize_input=normalize_input,
                value_size=value_size,
            )
