# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Automatic domain randomization (ADR, OpenAI arXiv 1910.07113) for the two deployment gaps the perturbed eval tests:
action delay and a per-episode joint-encoder offset (docs/ablation/sim2real.md).

One global level ``env.adr_level`` in [0, 1] scales both ranges: the per-episode delay is drawn from
``[0, round(L * max_delay)]`` physics substeps and the encoder offset from ``±L * bias_rad``. ``adr_level`` (a
curriculum term) raises L by ``step`` every ``interval_steps`` env steps while the share of ended episodes that ended in
a drop stays below ``low``, and lowers it above ``high``; so the randomization grows only as fast as the policy copes.
A new episode samples with the current L, running episodes keep their values. The env's physics DR (gains, friction,
masses, gravity) is unchanged.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from .mdp.sim2real import (
    DelayedEMAJointPositionToLimitsAction,
    DelayedEMAJointPositionToLimitsActionCfg,
    joint_pos_limit_normalized_biased,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def _level(env) -> float:
    return getattr(env, "adr_level", 0.0)


class ADRDelayedEMAJointPositionToLimitsAction(DelayedEMAJointPositionToLimitsAction):
    """Delayed EMA joint targets whose per-episode delay is drawn from ``[0, round(L * delay_substeps_range[1])]``.
    ``delay_substeps_range[1]`` is the delay at L = 1 and sizes the target buffer."""

    def __init__(self, cfg: ADRDelayedEMAJointPositionToLimitsActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._delay.zero_()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        n = self.num_envs if env_ids is None else len(env_ids)
        hi = int(round(_level(self._env) * self.cfg.delay_substeps_range[1]))
        self._delay[ids] = torch.randint(0, hi + 1, (n,), device=self.device)


@configclass
class ADRDelayedEMAJointPositionToLimitsActionCfg(DelayedEMAJointPositionToLimitsActionCfg):
    class_type: type = ADRDelayedEMAJointPositionToLimitsAction


class joint_pos_limit_normalized_adr_biased(joint_pos_limit_normalized_biased):
    """``joint_pos_limit_normalized_biased`` with the offset range scaled by the ADR level (``bias_rad`` at L = 1)."""

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = slice(None) if env_ids is None else env_ids
        n = self.num_envs if env_ids is None else len(env_ids)
        scale = self.bias_rad * _level(self._env)
        self.bias[ids] = (torch.rand(n, self.asset.num_joints, device=self.device) * 2 - 1) * scale


def adr_level(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    interval_steps: int = 480,
    step: float = 0.05,
    low: float = 0.25,
    high: float = 0.40,
    min_episodes: int = 1000,
    drop_term: str = "object_dropped",
) -> float:
    """Curriculum term: update ``env.adr_level`` from the drop share of the episodes that ended (see module doc).
    Called on every reset; returns the level for logging (``Curriculum/adr``)."""
    if not hasattr(env, "_adr_last"):
        env.adr_level = _level(env)
        env._adr_drops, env._adr_episodes, env._adr_last = 0, 0, env.common_step_counter
    if len(env_ids) > 0:
        env._adr_drops += int(env.termination_manager.get_term(drop_term)[env_ids].sum())
        env._adr_episodes += len(env_ids)
    if env.common_step_counter - env._adr_last >= interval_steps and env._adr_episodes >= min_episodes:
        share = env._adr_drops / env._adr_episodes
        if share < low:
            env.adr_level = min(1.0, env.adr_level + step)
        elif share > high:
            env.adr_level = max(0.0, env.adr_level - step)
        env._adr_drops, env._adr_episodes, env._adr_last = 0, 0, env.common_step_counter
    return env.adr_level
