# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the reward ablations in ``orca_v2_repose_ablation_env_cfg.py`` (docs/ablation/rewards.md).

The baseline rewards rotation with a speed-independent direction cosine (weight 60) and a magnitude kernel
exp(-|err| / 0.2) that is ~0 far below the commanded speed. The literature (HORA, LEAP, RotateIt, AnyRotate, DLR) uses
a rotation reward that is LINEAR in the angular velocity about the target axis, clipped; DLR 2024 (2411.13148) sets
the clip from the target speed. Both terms reuse the baseline's finite-difference angular velocity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .mdp.rewards import _get_numeric_root_ang_vel

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _target_axis(env: ManagerBasedRLEnv, command_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit target axis (N, 3) and commanded speed (N,); the command speed is never 0 (sampled in 60-100 % of max)."""
    target = env.command_manager.get_term(command_name).command[:, 3:6]
    speed = torch.norm(target, dim=-1).clamp(min=1e-6)
    return target / speed[:, None], speed


def rotation_progress_clipped(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    min_vel: float = -0.5,
) -> torch.Tensor:
    """clip(w . k, min_vel, |w*|) / |w*|: constant gradient up to the commanded speed, no reward for overspeed.

    In [min_vel / |w*|, 1]. HORA / LEAP form (clipped linear) with DLR 2024's command-dependent upper clip.
    """
    axis, speed = _target_axis(env, command_name)
    w_axis = (_get_numeric_root_ang_vel(env, object_cfg) * axis).sum(-1)
    return torch.clamp(w_axis, min=torch.full_like(speed, min_vel), max=speed) / speed


def off_axis_angular_velocity_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """L1 norm of the cube angular velocity orthogonal to the target axis [rad/s] (RotateIt's off-axis penalty)."""
    axis, _ = _target_axis(env, command_name)
    w = _get_numeric_root_ang_vel(env, object_cfg)
    return torch.abs(w - (w * axis).sum(-1, keepdim=True) * axis).sum(-1)


def rotation_progress_linear(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ref_speed: float = 1.5,
    min_vel: float = -0.5,
    max_vel: float | None = None,
) -> torch.Tensor:
    """clip(w . k, min_vel, max_vel) / ref_speed: linear in the speed about the target axis, independent of the commanded
    speed ("as fast as possible" up to ``max_vel``); ``ref_speed`` only sets the scale (1 at ref_speed).

    ``max_vel=None`` (no upper clip) is exploited: the policy flings the cube, which spins at ~30 rad/s while it falls,
    and the short burst pays more than holding it (run 2026-10-29: 99.9 % drops, episodes of 2 s). Use a finite clip."""
    axis, _ = _target_axis(env, command_name)
    w_axis = (_get_numeric_root_ang_vel(env, object_cfg) * axis).sum(-1)
    return torch.clamp(w_axis, min=min_vel, max=max_vel) / ref_speed
