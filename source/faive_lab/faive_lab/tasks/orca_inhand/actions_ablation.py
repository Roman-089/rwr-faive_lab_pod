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

Random action holding (docs/ablation/sim2real.md; TeleDexter's "random action masking", arXiv 2607.11481): the joint
target sent to the servos is randomly not updated for a policy step, as when a command packet is lost or late.
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


class HoldingEMAJointPositionToLimitsAction(EMAJointPositionToLimitsAction):
    """EMA joint position action whose joint targets are randomly held: the servos keep the previous target.

    Per episode each env samples p ~ U(0, hold_prob_max). Each policy step a hold starts with probability p (if none is
    active) and lasts d ~ U{hold_steps_range} policy steps, during which the joints keep the target applied before it.
    By default all joints of an env are held together (one lost sync-write packet carries every servo's target);
    ``per_joint`` holds each joint independently (TeleDexter freezes 3 random joints, i.e. desynchronized actuators).

    Only the target sent to the joints (``processed_actions``, written by the parent's ``apply_actions``) is held. The
    EMA chain and ``_prev_applied_actions`` (read by the ``last_joint_pos_cmd`` observation) stay those of the command
    the policy node computed and sent, as on the real robot, where the node does not know a packet was lost. This
    works because the parent's EMA recursion reads only ``_prev_applied_actions``; ``_processed_actions`` is rebuilt
    every step and only consumed by ``apply_actions``.

    On reset the held target is the reset joint position (= the parent's reset EMA state). ``hold_prob_max = 0`` runs
    the parent's code path with no extra random draws, so it reproduces the parent exactly. The delay term composes as
    ``class _(DelayedEMAJointPositionToLimitsAction, HoldingEMAJointPositionToLimitsAction)``: the held target then
    enters the delay FIFO.
    """

    cfg: HoldingEMAJointPositionToLimitsActionCfg

    def __init__(self, cfg: HoldingEMAJointPositionToLimitsActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        lo, hi = cfg.hold_steps_range
        if not (0.0 <= cfg.hold_prob_max <= 1.0 and 1 <= lo <= hi):
            raise ValueError(f"Need 0 <= hold_prob_max <= 1 and 1 <= lo <= hi, got {cfg.hold_prob_max}, {(lo, hi)}.")
        self._hold_prob_max = cfg.hold_prob_max
        self._hold_steps = (lo, hi)
        # per-env hold probability (resampled per episode) and remaining held steps per env (or per joint)
        self._hold_prob = torch.zeros(self.num_envs, 1, device=self.device)
        n_cols = self.action_dim if cfg.per_joint else 1
        self._hold_left = torch.zeros(self.num_envs, n_cols, dtype=torch.long, device=self.device)
        if self._hold_prob_max > 0:
            self._hold_prob.uniform_(0.0, self._hold_prob_max)

    def process_actions(self, actions: torch.Tensor):
        if self._hold_prob_max <= 0:
            super().process_actions(actions)
            return
        held_target = self._processed_actions.clone()  # what the joints track now
        super().process_actions(actions)
        start = (self._hold_left == 0) & (torch.rand(self._hold_left.shape, device=self.device) < self._hold_prob)
        lo, hi = self._hold_steps
        dur = torch.randint(lo, hi + 1, self._hold_left.shape, device=self.device)
        self._hold_left = torch.where(start, dur, self._hold_left)
        held = self._hold_left > 0
        self._processed_actions[:] = torch.where(held, held_target, self._processed_actions)
        self._hold_left = (self._hold_left - 1).clamp_(min=0)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is not None and not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device)
        super().reset(env_ids)
        if self._hold_prob_max <= 0:
            return
        ids = slice(None) if env_ids is None else env_ids
        # until the first command arrives, the joints hold their reset position
        self._processed_actions[ids] = self._prev_applied_actions[ids]
        self._hold_left[ids] = 0
        self._hold_prob[ids] = torch.rand_like(self._hold_prob[ids]) * self._hold_prob_max


@configclass
class HoldingEMAJointPositionToLimitsActionCfg(EMAJointPositionToLimitsActionCfg):
    class_type: type = HoldingEMAJointPositionToLimitsAction

    hold_prob_max: float = 0.1
    """Per-episode hold probability p ~ U(0, hold_prob_max) per policy step (0: the EMA parent exactly). 0.1 = up to
    one 50 ms step in 10 stale; TeleDexter uses p_mask = 0.15 per step but freezes only 3 of 16-22 joints."""

    hold_steps_range: tuple[int, int] = (1, 1)
    """Inclusive range of a hold's length in policy steps (50 ms each), sampled per hold. (1, 1): independent single
    lost / late packets; longer holds model node stalls (TeleDexter ramps its freeze length from 1 up to 10 steps)."""

    per_joint: bool = False
    """Hold each joint independently instead of all joints of an env together."""
