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
import os
import yaml

import torch


class MotionLib:
    def __init__(self, motion_file, device, step_dt):
        self._device = device

        self._load_motions(motion_file)

        self.dataset_obs = (
            torch.cat([m for m in self._obs_motions], dim=0).float().to(self._device)
        )
        self.dataset_states = (
            torch.cat([m for m in self._states_motions], dim=0).float().to(self._device)
        )

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

        self.motion_ids = torch.arange(
            len(self._obs_motions), dtype=torch.long, device=self._device
        )

        self._step_dt = step_dt

    def num_motions(self):
        return len(self._obs_motions)

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion(self, motion_id):
        return self._obs_motions[motion_id]

    def sample_motions(self, n):
        motion_ids = torch.multinomial(
            self._motion_weights, num_samples=n, replacement=True
        )
        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        phase = torch.rand(motion_ids.shape, device=self._device)

        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time

    def sample_nearby_time(
        self, motion_ids, motion_time, time_delta, truncate_time=None
    ):
        nearby_time = (
            torch.rand(motion_ids.shape, device=self._device) - 0.5
        ) * time_delta + motion_time

        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time

        nearby_time = torch.clamp(torch.min(nearby_time, motion_len), min=0)

        return nearby_time

    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]

    def get_motion_obs(self, motion_ids, motion_times):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt
        )

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        dataset_obs_0 = self.dataset_obs[f0l]
        dataset_obs_1 = self.dataset_obs[f1l]

        vals = [
            dataset_obs_0,
            dataset_obs_1,
        ]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)
        dataset_obs = (1.0 - blend) * dataset_obs_0 + blend * dataset_obs_1

        return dataset_obs

    def get_motion_states(self, motion_ids, motion_times):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt
        )

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        dataset_states_0 = self.dataset_states[f0l]
        dataset_states_1 = self.dataset_states[f1l]

        vals = [
            dataset_states_0,
            dataset_states_1,
        ]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)
        dataset_states = (1.0 - blend) * dataset_states_0 + blend * dataset_states_1

        return dataset_states

    def _load_motions(self, motion_file):
        self._obs_motions = []
        self._states_motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []

        total_len = 0.0

        (
            motion_obs_files,
            motion_states_files,
            motion_weights,
            use_full_motion,
            starting_ids,
            ending_ids,
            motion_dts,
        ) = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_obs_files)
        for f in range(num_motion_files):
            print(
                "Loading {:d}/{:d} motion files: {:s}".format(
                    f + 1, num_motion_files, motion_obs_files[f]
                )
            )
            if not motion_obs_files[f].endswith(".npy"):
                print("Wrong motion file format")

            loaded_obs = np.load(motion_obs_files[f]).transpose(1, 0, 2)
            full_motion_obs = torch.from_numpy(loaded_obs)
            full_motion_obs.to(device=self._device)

            for curr_motion_obs in full_motion_obs:
                if use_full_motion[f]:
                    curr_motion_obs = curr_motion_obs.to(device=self._device)
                else:
                    curr_motion_obs = curr_motion_obs[
                        starting_ids[f] : ending_ids[f], :
                    ].to(device=self._device)

                motion_fps = 1.0 / motion_dts[f]
                curr_dt = 1.0 / motion_fps

                num_frames = curr_motion_obs.shape[0]
                curr_len = 1.0 / motion_fps * (num_frames - 1)

                self._motion_fps.append(motion_fps)
                self._motion_dt.append(curr_dt)
                self._motion_num_frames.append(num_frames)

                self._obs_motions.append(curr_motion_obs)
                self._motion_lengths.append(curr_len)

                curr_weight = motion_weights[f]
                self._motion_weights.append(curr_weight)

            loaded_states = np.load(motion_states_files[f]).transpose(1, 0, 2)
            full_motion_states = torch.from_numpy(loaded_states)
            full_motion_states.to(device=self._device)

            for curr_motion_states in full_motion_states:
                print(curr_motion_states.shape)

                if use_full_motion[f]:
                    curr_motion_states = curr_motion_states.to(device=self._device)
                else:
                    curr_motion_states = curr_motion_states[
                        starting_ids[f] : ending_ids[f], :
                    ].to(device=self._device)
                self._states_motions.append(curr_motion_states)

        self._motion_lengths = torch.tensor(
            self._motion_lengths, device=self._device, dtype=torch.float32
        )

        # TODO: used for full replay of states, enable with param
        # self._states_motions_tensor = torch.stack(self._states_motions).to(self._device)

        self._motion_weights = torch.tensor(
            self._motion_weights, dtype=torch.float32, device=self._device
        )
        self._motion_weights /= self._motion_weights.sum()

        self._motion_fps = torch.tensor(
            self._motion_fps, device=self._device, dtype=torch.float32
        )
        self._motion_dt = torch.tensor(
            self._motion_dt, device=self._device, dtype=torch.float32
        )
        self._motion_num_frames = torch.tensor(
            self._motion_num_frames, device=self._device
        )

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print(
            "Loaded {:d} motions with a total length of {:.3f}s.".format(
                num_motions, total_len
            )
        )

    def _fetch_motion_files(self, motion_file):
        ext = os.path.splitext(motion_file)[1]
        assert ext == ".yaml"

        dir_name = os.path.dirname(motion_file)
        motion_obs_files = []
        motion_states_files = []
        motion_weights = []
        use_full_motion = []
        starting_ids = []
        ending_ids = []
        motion_dts = []

        with open(os.path.join(os.getcwd(), motion_file), "r") as f:
            motion_config = yaml.load(f, Loader=yaml.SafeLoader)

        motion_list = motion_config["motions"]
        for motion_entry in motion_list:
            curr_obs_file = motion_entry["obs_file"]
            curr_states_file = motion_entry["states_file"]
            curr_weight = motion_entry["weight"]
            assert curr_weight >= 0

            if ("starting_id" in motion_entry) and ("ending_id" in motion_entry):
                use_full_motion.append(False)
                starting_ids.append(motion_entry["starting_id"])
                ending_ids.append(motion_entry["ending_id"])
            else:
                use_full_motion.append(True)
                starting_ids.append(0)
                ending_ids.append(0)

            motion_weights.append(curr_weight)
            curr_obs_file = os.path.join(dir_name, curr_obs_file)
            curr_states_file = os.path.join(dir_name, curr_states_file)
            motion_obs_files.append(curr_obs_file)
            motion_states_files.append(curr_states_file)
            motion_dts.append(motion_entry["dt"])

        return (
            motion_obs_files,
            motion_states_files,
            motion_weights,
            use_full_motion,
            starting_ids,
            ending_ids,
            motion_dts,
        )

    def _calc_frame_blend(self, time, len, num_frames, dt):

        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def build_amp_obs_demo(self, motion_ids, motion_times0, num_steps):
        dt = self._step_dt

        timesteps = num_steps
        batch_size = motion_ids.shape[0]

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, num_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, num_steps, device=self._device)
        motion_times = torch.clip(motion_times + time_steps, min=0)

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        amp_obs_demo = self.get_motion_obs(motion_ids, motion_times)

        # TODO config with parameter
        if amp_obs_demo.shape[1] == 48:
            feature_sizes = [1, 3, 3, 3, 19, 19]
        else:
            feature_sizes = [1, 3, 3, 3, 12, 12]

        obs = amp_obs_demo.view(batch_size, timesteps, sum(feature_sizes))

        # Split the features into components
        split_obs = obs.split(feature_sizes, dim=2)

        # Interleave timesteps (t, t+1) **within each feature group**
        interleaved_features = [
            feat.permute(0, 1, 2).reshape(batch_size, size * timesteps)
            for feat, size in zip(split_obs, feature_sizes)
        ]

        # Concatenate all interleaved features along feature dimension
        obs_final = torch.cat(interleaved_features, dim=1)

        return obs_final
