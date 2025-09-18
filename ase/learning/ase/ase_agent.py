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

from learning.common_ase import common_ase_agent


class ASEAgent(common_ase_agent.CommonASEAgent):
    def _calc_amp_rewards(self, amp_obs, ase_latents):
        disc_r = self._calc_disc_rewards(amp_obs)
        enc_r = self._calc_enc_rewards(amp_obs, ase_latents)
        output = {"disc_rewards": disc_r, "enc_rewards": enc_r}
        return output

    def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo):
        # prediction loss
        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # logit reg
        logit_weights = self.model.a2c_network.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty
        disc_demo_grad = torch.autograd.grad(
            disc_demo_logit,
            obs_demo,
            grad_outputs=torch.ones_like(disc_demo_logit),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        # weight decay
        if self._disc_weight_decay != 0:
            disc_weights = self.model.a2c_network.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(
            disc_agent_logit, disc_demo_logit
        )

        info = {
            "disc_grad_penalty": disc_grad_penalty.detach(),
            "disc_logit_loss": disc_logit_loss.detach(),
            "disc_agent_acc": disc_agent_acc.detach(),
            "disc_demo_acc": disc_demo_acc.detach(),
            "disc_agent_logit": disc_agent_logit.detach(),
            "disc_demo_logit": disc_demo_logit.detach(),
        }
        return disc_loss, info

    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.zeros_like(disc_logits))
        return loss

    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits))
        return loss

    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        agent_acc = disc_agent_logit < 0
        agent_acc = torch.mean(agent_acc.float())
        demo_acc = disc_demo_logit > 0
        demo_acc = torch.mean(demo_acc.float())
        return agent_acc, demo_acc

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(
                torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device))
            )
            disc_r *= self._disc_reward_scale

        return disc_r
