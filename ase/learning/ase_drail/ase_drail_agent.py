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

from rl_games.common import a2c_common

from learning.common_ase import common_ase_agent


class ASEDRAILAgent(common_ase_agent.CommonASEAgent):
    def _calc_amp_rewards(self, amp_obs, ase_latents):
        disc_r = self._calc_disc_rewards(a2c_common.swap_and_flatten01(amp_obs))
        enc_r = self._calc_enc_rewards(amp_obs, ase_latents)
        output = {
            "disc_rewards": disc_r.reshape(
                amp_obs.shape[1], amp_obs.shape[0], 1
            ).transpose(0, 1),
            "enc_rewards": enc_r,
        }
        return output

    def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo):
        # prediction loss
        disc_loss_agent = self._compute_agent_loss(disc_agent_logit)
        disc_loss_demo = self._compute_expert_loss(disc_demo_logit)
        disc_loss = disc_loss_agent + disc_loss_demo

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(
            disc_agent_logit, disc_demo_logit
        )

        info = {
            "disc_agent_acc": disc_agent_acc.detach(),
            "disc_demo_acc": disc_demo_acc.detach(),
            "disc_agent_logit": disc_agent_logit.detach(),
            "disc_demo_logit": disc_demo_logit.detach(),
        }
        return disc_loss, info

    def _compute_agent_loss(self, disc_logits):
        return torch.nn.functional.binary_cross_entropy(
            disc_logits, torch.zeros_like(disc_logits)
        )

    def _compute_expert_loss(self, disc_logits):
        return torch.nn.functional.binary_cross_entropy(
            disc_logits, torch.ones_like(disc_logits)
        )

    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        agent_acc = disc_agent_logit < 0.5
        agent_acc = torch.mean(agent_acc.float())
        demo_acc = disc_demo_logit > 0.5
        demo_acc = torch.mean(demo_acc.float())
        return agent_acc, demo_acc

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = disc_logits
            disc_r = -torch.log(
                torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device))
            )
            disc_r *= self._disc_reward_scale

        return disc_r
