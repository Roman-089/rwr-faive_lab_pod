# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim2real terms for the ablations: delayed joint targets and per-episode joint-position observation bias."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.actions.actions_cfg import EMAJointPositionToLimitsActionCfg
from isaaclab.envs.mdp.actions.joint_actions_to_limits import EMAJointPositionToLimitsAction
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass, math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DelayedEMAJointPositionToLimitsAction(EMAJointPositionToLimitsAction):
    """EMA joint position action whose targets reach the joints ``d`` physics substeps late (d sampled per episode).

    The delay has physics-step resolution (8.3 ms at 120 Hz), so it also covers servo lag below one policy step
    (Aero Hand: sim servo 10-40 ms too fast). The target applied at substep ``s`` (0..decimation-1) after the policy
    step is the one produced ``ceil((d - s) / decimation)`` policy steps ago.

    The EMA chain and ``_prev_applied_actions`` (read by the ``last_joint_pos_cmd`` observation) stay those of the
    command the policy *produced*, as on the real robot; only the target sent to the joints is delayed.
    """

    cfg: DelayedEMAJointPositionToLimitsActionCfg

    def __init__(self, cfg: DelayedEMAJointPositionToLimitsActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._decimation = env.cfg.decimation
        lo, hi = cfg.delay_substeps_range
        # FIFO of produced targets, newest at index 0
        n_buf = -(-hi // self._decimation) + 1
        self._target_buffer = torch.zeros(n_buf, self.num_envs, self.action_dim, device=self.device)
        self._delay = torch.randint(lo, hi + 1, (self.num_envs,), device=self.device)
        self._env_idx = torch.arange(self.num_envs, device=self.device)
        self._substep = 0

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        # until the first command arrives, the joints hold their reset position
        self._target_buffer[:, ids] = self._prev_applied_actions[ids]
        lo, hi = self.cfg.delay_substeps_range
        n = self.num_envs if env_ids is None else len(env_ids)
        self._delay[ids] = torch.randint(lo, hi + 1, (n,), device=self.device)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._target_buffer = torch.roll(self._target_buffer, 1, dims=0)
        self._target_buffer[0] = self._processed_actions
        self._substep = 0

    def apply_actions(self):
        age = torch.clamp(self._delay - self._substep + self._decimation - 1, min=0) // self._decimation
        self._asset.set_joint_position_target(self._target_buffer[age, self._env_idx], joint_ids=self._joint_ids)
        self._substep += 1


@configclass
class DelayedEMAJointPositionToLimitsActionCfg(EMAJointPositionToLimitsActionCfg):
    class_type: type = DelayedEMAJointPositionToLimitsAction

    delay_substeps_range: tuple[int, int] = MISSING
    """Inclusive range of the per-episode delay in physics substeps (8.3 ms at 120 Hz; 6 substeps = 1 policy step)."""


class joint_pos_limit_normalized_biased(ManagerTermBase):
    """``joint_pos_limit_normalized`` with a per-episode, per-joint offset (encoder zero / calibration error).

    The offset is sampled uniformly in ``±bias_rad`` radians at reset and added to the joint position before the
    normalization, so the per-step noise of the term cfg comes on top.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        self.bias_rad = cfg.params["bias_rad"]
        self.bias = torch.zeros(env.num_envs, self.asset.num_joints, device=env.device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = slice(None) if env_ids is None else env_ids
        n = self.num_envs if env_ids is None else len(env_ids)
        self.bias[ids] = (torch.rand(n, self.asset.num_joints, device=self.device) * 2 - 1) * self.bias_rad

    def __call__(
        self, env: ManagerBasedEnv, bias_rad: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        limits = self.asset.data.soft_joint_pos_limits
        return math_utils.scale_transform(self.asset.data.joint_pos + self.bias, limits[..., 0], limits[..., 1])
