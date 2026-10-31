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
    ramp_s: float = 0.0,
    ramp_start_vel: float = 0.5,
) -> torch.Tensor:
    """clip(w . k, min_vel, max_vel) / ref_speed: linear in the speed about the target axis, independent of the commanded
    speed ("as fast as possible" up to ``max_vel``); ``ref_speed`` only sets the scale (1 at ref_speed).

    ``max_vel=None`` (no upper clip) is exploited: the policy flings the cube, which spins at ~30 rad/s while it falls,
    and the short burst pays more than holding it (run 2026-10-29: 99.9 % drops, episodes of 2 s). Use a finite clip.

    ``ramp_s > 0``: the clip rises linearly from ``ramp_start_vel`` to ``max_vel`` over the first ``ramp_s`` seconds of
    each episode, so a fast spin-up from rest pays no more than a gentle one (a third of onedir's drops happen in the
    first 3 s after spawn, docs/ablation/rewards.md)."""
    axis, _ = _target_axis(env, command_name)
    w_axis = (_get_numeric_root_ang_vel(env, object_cfg) * axis).sum(-1)
    w_axis = torch.clamp(w_axis, min=min_vel, max=max_vel)
    if ramp_s > 0.0 and max_vel is not None:
        frac = torch.clamp(env.episode_length_buf * env.step_dt / ramp_s, max=1.0)
        w_axis = torch.minimum(w_axis, ramp_start_vel + (max_vel - ramp_start_vel) * frac)
    return w_axis / ref_speed


def joint_work_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Mechanical power of the joints, sum |tau_i * qdot_i| [W] (HORA's work penalty, arXiv 2210.04887): favours smooth,
    low-effort finger gaits. ``applied_torque`` is the PD drive's torque estimate for the implicit actuators."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=1)


def cube_outside_region(
    env: ManagerBasedRLEnv,
    center: tuple[float, float],
    radius: float,
    sigma: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """exp(-e / sigma) - 1 for the cube's rotation axis being e = max(0, |c_xy - center| - radius) outside an allowed
    circle on the palm: 0 inside, falling exponentially towards -1 outside (use a positive weight). The cube turns about
    world z, so its axis is the vertical through the cube centre c_xy (env frame). The region was chosen by the user
    with docs/ablation/tools/palm_axis_region.html."""
    obj = env.scene[object_cfg.name]
    xy = obj.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    d = torch.linalg.norm(xy - torch.tensor(center, device=xy.device), dim=-1)
    return torch.exp(-torch.clamp(d - radius, min=0.0) / sigma) - 1.0


def cube_tilt(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """1 - max_i |(R e_i) . z|: zero while one cube face points up (any yaw, so a clean z-rotation is free), 1 - cos(tilt)
    otherwise (45 deg -> 0.29). The +z policies tumble the cube (mean tilt 25-36 deg, the top face changes ~100-170
    times per minute, off-axis spin above the z spin) while rotlin keeps it flat (6 deg); docs/ablation/rewards.md."""
    w, x, y, z = env.scene[object_cfg.name].data.root_quat_w.unbind(-1)
    world_z_in_body = torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1)
    return 1.0 - world_z_in_body.abs().amax(dim=-1)
