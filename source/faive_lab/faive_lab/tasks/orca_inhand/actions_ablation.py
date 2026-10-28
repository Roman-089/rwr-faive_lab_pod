# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action-space ablations (docs/ablation/action_space.md).

Relative joint targets as in HORA / LEAP: each step the policy moves the previous joint target by a bounded delta,
instead of choosing an absolute target that the baseline then low-passes with EMA alpha 0.2 (a ~250 ms time constant
at 20 Hz). Small, fast finger motions are easy to express and the target never jumps. The policy observation is
unchanged: ``last_joint_pos_cmd`` reads ``_prev_applied_actions``, which here is the current target, so the policy
always sees the state it is moving from.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import EMAJointPositionToLimitsActionCfg
from isaaclab.envs.mdp.actions.joint_actions_to_limits import EMAJointPositionToLimitsAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class RelativeJointPositionToLimitsAction(EMAJointPositionToLimitsAction):
    """target_t = clamp(target_{t-1} + delta_scale * clamp(a_t, -1, 1) * (upper - lower) / 2, lower, upper).

    Reuses the EMA term's target buffer (``_prev_applied_actions``) and its reset, which sets the target to the
    current joint position; ``alpha`` is not used.
    """

    cfg: RelativeJointPositionToLimitsActionCfg

    def __init__(self, cfg: RelativeJointPositionToLimitsActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._delta_scale = cfg.delta_scale

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        lower = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0]
        upper = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1]
        delta = self._delta_scale * actions.clamp(-1.0, 1.0) * 0.5 * (upper - lower)
        self._processed_actions[:] = torch.clamp(self._prev_applied_actions + delta, lower, upper)
        self._prev_applied_actions[:] = self._processed_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is not None and not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device)
        super().reset(env_ids)


@configclass
class RelativeJointPositionToLimitsActionCfg(EMAJointPositionToLimitsActionCfg):
    class_type: type = RelativeJointPositionToLimitsAction

    alpha: float = 1.0
    """Unused (kept so the EMA parent initializes)."""

    delta_scale: float = 0.1
    """Max target change per policy step as a fraction of the joint half-range (0.1: full range in 20 steps = 1 s;
    HORA uses 1/24 rad per step at 20 Hz)."""
