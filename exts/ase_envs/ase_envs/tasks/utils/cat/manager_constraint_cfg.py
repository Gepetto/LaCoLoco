# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# Copyright (c) 2025, LAAS-CNRS
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# based on https://github.com/isaac-sim/IsaacLab
# based on https://github.com/Gepetto/constraints-as-terminations

"""Configuration terms for different managers."""

from __future__ import annotations

import torch
from collections.abc import Callable
from dataclasses import MISSING

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers.manager_term_cfg import ManagerTermBaseCfg


##
# Constraint manager.
##


@configclass
class ConstraintTermCfg(ManagerTermBaseCfg):
    func: Callable[..., torch.Tensor] = MISSING

    max_p: float = MISSING
