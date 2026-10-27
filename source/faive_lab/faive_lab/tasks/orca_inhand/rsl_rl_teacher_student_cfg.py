# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runner cfgs for teacher -> student distillation. Kept out of agents/ (owned by S0); PPO settings are the baseline's."""

from isaaclab.utils import configclass

from .agents.rsl_rl_ppo_cfg import PPORunnerCfgOrcaV2


@configclass
class PPORunnerCfgOrcaV2Teacher(PPORunnerCfgOrcaV2):
    """Baseline PPO; actor and critic both read the privileged ``teacher`` observation group."""

    obs_groups = {"policy": ["teacher"], "critic": ["teacher"]}
