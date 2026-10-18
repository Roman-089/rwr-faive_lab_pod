# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Custom rewards for the continuous rotation task
"""

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from .commands import InHandReOrientationCommand
    from .commands import ContinuousRotationCommand
    

def _get_numeric_root_ang_vel(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Compute angular velocity by differentiating root orientation.
    avoids instability when reading from asset.data.root_ang_vel_w
    The result is cached per simulation step for reuse across reward terms.
    """

    asset: RigidObject = env.scene[object_cfg.name]
    current_quat = asset.data.root_quat_w

    cache_attr = f"_numeric_ang_vel_cache_{object_cfg.name}"
    cache_step_attr = f"_numeric_ang_vel_cache_step_{object_cfg.name}"
    prev_quat_attr = f"_numeric_ang_vel_prev_quat_{object_cfg.name}"

    current_step = getattr(env, "common_step_counter", 0)

    if hasattr(env, cache_step_attr):
        # just return cached value if available
        cached_step = getattr(env, cache_step_attr)
        if cached_step == current_step:
            return getattr(env, cache_attr)

    if not hasattr(env, prev_quat_attr):
        # first time initialization
        prev_quat_init = current_quat.clone()
        setattr(env, prev_quat_attr, prev_quat_init)
        zero_vel = torch.zeros_like(current_quat[:, :3])
        setattr(env, cache_attr, zero_vel)
        setattr(env, cache_step_attr, current_step)
        return zero_vel

    prev_quat_tensor: torch.Tensor = getattr(env, prev_quat_attr)
    prev_quat_for_calc = prev_quat_tensor

    # On the first step after a reset, force zero angular velocity for stability.
    first_step_mask = None
    if hasattr(env, "episode_length_buf"):
        first_step_mask = env.episode_length_buf == 1
        if torch.any(first_step_mask):
            prev_quat_for_calc = prev_quat_tensor.clone()
            prev_quat_for_calc[first_step_mask] = current_quat[first_step_mask]

    delta_quat = math_utils.quat_mul(current_quat, math_utils.quat_conjugate(prev_quat_for_calc))
    delta_quat = math_utils.quat_unique(delta_quat)

    vec_part = delta_quat[:, 1:]
    w_part = torch.clamp(delta_quat[:, :1], min=-1.0, max=1.0)
    sin_half_theta = torch.norm(vec_part, p=2, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half_theta, w_part)

    non_zero_mask = sin_half_theta > 1e-8
    axis = torch.zeros_like(vec_part)
    axis = torch.where(
        non_zero_mask.expand_as(vec_part),
        vec_part / torch.clamp(sin_half_theta, min=1e-8),
        axis,
    )

    dt = torch.clamp_min(current_quat.new_tensor(getattr(env, "step_dt", 1.0)), 1e-8)
    numeric_ang_vel = axis * (angle / dt)

    if first_step_mask is not None and torch.any(first_step_mask):
        numeric_ang_vel = numeric_ang_vel.clone()
        numeric_ang_vel[first_step_mask] = 0.0

    prev_quat_tensor.copy_(current_quat)

    setattr(env, cache_attr, numeric_ang_vel)
    setattr(env, cache_step_attr, current_step)

    return numeric_ang_vel



def angular_acceleration_penalty_exp(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    sigma: float = 1.0,
) -> torch.Tensor:
    """Exponential smoothness reward based on angular acceleration.
    
    Penalizes rapid changes in angular velocity to encourage smooth rotation.
    Uses exponential form for numerical stability.
    
    Args:
        env: The environment object
        object_cfg: The scene entity config
        sigma: Tolerance parameter for acceleration magnitude
        
    Returns:
        Smoothness reward in range [0, 1]
    """
    current_ang_vel = _get_numeric_root_ang_vel(env, object_cfg)

    prev_ang_vel_attr = f"_last_numeric_ang_vel_{object_cfg.name}"

    if not hasattr(env, prev_ang_vel_attr):
        setattr(env, prev_ang_vel_attr, current_ang_vel.clone())
        return torch.ones(env.num_envs, device=current_ang_vel.device)

    prev_ang_vel = getattr(env, prev_ang_vel_attr)
    ang_accel = current_ang_vel - prev_ang_vel

    if hasattr(env, "episode_length_buf"):
        first_step_mask = env.episode_length_buf == 1
        if torch.any(first_step_mask):
            ang_accel = ang_accel.clone()
            ang_accel[first_step_mask] = 0.0
    accel_magnitude = torch.norm(ang_accel, p=2, dim=-1)
    
    # Exponential reward: small acceleration -> reward near 1
    reward = torch.exp(-accel_magnitude / sigma)

    # Update for next step
    setattr(env, prev_ang_vel_attr, current_ang_vel.clone())
    
    return torch.clamp(reward, min=0.0, max=1.0)



def angular_velocity_alignment(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for angular velocity direction alignment.
    
    Args:
        env: The environment object
        command_name: Command term name
        object_cfg: Scene entity config
        
    Returns:
        Alignment reward in [0, 1]
    """
    command_term: ContinuousRotationCommand = env.command_manager.get_term(command_name)
    
    target_ang_vel = command_term.command[:, 3:6]
    current_ang_vel = _get_numeric_root_ang_vel(env, object_cfg)
    
    cosine_similarity = torch.nn.functional.cosine_similarity(
        current_ang_vel, target_ang_vel, dim=-1, eps=1e-8
    )
    
    # alignment_reward = cosine_similarity * 0.5 + 0.5
    
    return cosine_similarity


def track_angular_velocity_magnitude_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    sigma: float = 0.4,
) -> torch.Tensor:
    """Exponential reward for angular velocity magnitude tracking.
    
    Args:
        env: The environment object
        command_name: Command term name
        object_cfg: Scene entity config
        sigma: Tolerance parameter
        
    Returns:
        Magnitude tracking reward in [0, 1]
    """
    command_term: ContinuousRotationCommand = env.command_manager.get_term(command_name)
    
    target_ang_vel = command_term.command[:, 3:6]
    current_ang_vel = _get_numeric_root_ang_vel(env, object_cfg)
    
    target_magnitude = torch.norm(target_ang_vel, p=2, dim=-1)
    current_magnitude = torch.norm(current_ang_vel, p=2, dim=-1)
    magnitude_error = torch.abs(current_magnitude - target_magnitude)
    
    reward = torch.exp(-magnitude_error / sigma)
    return torch.clamp(reward, min=0.0, max=1.0)


def track_position_exp(  
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    sigma: float = 0.02,
) -> torch.Tensor:
    """Exponential reward for position tracking.
    
    Args:
        env: The environment object
        command_name: Command term name
        object_cfg: Scene entity config
        sigma: Tolerance parameter
        
    Returns:
        Position tracking reward in [0, 1]
    """
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: ContinuousRotationCommand = env.command_manager.get_term(command_name)
    
    target_pos = command_term.command[:, 0:3]
    current_pos = asset.data.root_pos_w - env.scene.env_origins
    pos_error = torch.norm(current_pos - target_pos, p=2, dim=-1)
    
    reward = torch.exp(-pos_error / sigma)
    return torch.clamp(reward, min=0.0, max=1.0)


def adaptive_angular_velocity_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    good_threshold: float = 0.06,
    bad_threshold: float = 0.15,
    sigma: float = 0.35,
) -> torch.Tensor:
    """Adaptive angular velocity tracking with progressive weighting.
    
    Weight strategy based on position stability:
    - pos_error < 6cm: 100% weight (position good, fully track angular velocity)
    - 6cm < pos_error < 15cm: linear decay 100% -> 30%
    - pos_error > 15cm: 30% weight (position poor, but keep learning)
    
    Args:
        env: The environment object
        command_name: Command term name
        object_cfg: Scene entity config
        good_threshold: Position error threshold for "good" (0.06m)
        bad_threshold: Position error threshold for "bad" (0.15m)
        sigma: Tolerance for angular velocity error (0.35 rad/s)
        
    Returns:
        Weighted angular velocity tracking reward
    """
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: ContinuousRotationCommand = env.command_manager.get_term(command_name)
    
    # Calculate position error for adaptive weighting
    target_pos = command_term.command[:, 0:3]
    current_pos = asset.data.root_pos_w - env.scene.env_origins
    pos_error = torch.norm(current_pos - target_pos, p=2, dim=-1)
    
    # Piecewise linear weight function
    weight = torch.ones_like(pos_error)
    
    # Middle region: linear interpolation
    middle_mask = (pos_error > good_threshold) & (pos_error < bad_threshold)
    weight[middle_mask] = 1.0 - (pos_error[middle_mask] - good_threshold) / \
                          (bad_threshold - good_threshold) * 0.7
    
    # Bad region: fixed 30%
    bad_mask = pos_error >= bad_threshold
    weight[bad_mask] = 0.3
    
    # Angular velocity tracking
    target_ang_vel = command_term.command[:, 3:6]
    current_ang_vel = _get_numeric_root_ang_vel(env, object_cfg)
    
    # Magnitude matching
    target_magnitude = torch.norm(target_ang_vel, p=2, dim=-1)
    current_magnitude = torch.norm(current_ang_vel, p=2, dim=-1)
    magnitude_error = torch.abs(current_magnitude - target_magnitude)
    magnitude_reward = torch.exp(-magnitude_error / sigma)
    
    # Direction alignment
    cosine_similarity = torch.nn.functional.cosine_similarity(
        current_ang_vel, target_ang_vel, dim=-1, eps=1e-8
    )
    direction_reward = cosine_similarity * 0.5 + 0.5
    
    # Combined angular reward
    angular_reward = (magnitude_reward + direction_reward) / 2.0
    
    # Apply adaptive weight
    return angular_reward * weight

