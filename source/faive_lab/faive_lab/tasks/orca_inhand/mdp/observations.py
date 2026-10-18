# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera


if TYPE_CHECKING:
    from .commands import InHandReOrientationCommand


def goal_quat_diff(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, make_quat_unique: bool
) -> torch.Tensor:
    """Goal orientation relative to the asset's root frame.

    The quaternion is represented as (w, x, y, z). The real part is always positive.
    """
    # extract useful elements
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)

    # obtain the orientations
    goal_quat_w = command_term.command[:, 3:7]
    asset_quat_w = asset.data.root_quat_w

    # compute quaternion difference
    quat = math_utils.quat_mul(asset_quat_w, math_utils.quat_conjugate(goal_quat_w))
    # make sure the quaternion real-part is always positive
    return math_utils.quat_unique(quat) if make_quat_unique else quat


def rgb_CHWF(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    convert_perspective_to_orthogonal: bool = False,
) -> torch.Tensor:
    """
    Adapted from Isaac Lab's image observation. Instead of normalize, we just scale the image to (0, 1) and return it as C H W shape
    Images of a specific datatype from the camera sensor.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]
    # obtain the input image
    images = sensor.data.output["rgb"]
    images = (images.float() / 255.0).permute(0, 3, 1, 2)
    return images.clone()


def angular_vel_error(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    object_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Calculate the angular velocity tracking error.
    
    Args:
        env: The environment object.
        command_name: The command term name for extracting the target angular velocity.
        object_cfg: The configuration for the scene entity.
        
    Returns:
        Angular velocity error magnitude as a tensor of shape (num_envs, 1).
    """
    asset: RigidObject = env.scene[object_cfg.name]
    current_ang_vel = asset.data.root_ang_vel_w   # (num_envs, 3)

    # 目标角速度
    command_term = env.command_manager.get_term(command_name)
    target_ang_vel = command_term.angular_vel_command  # (num_envs, 3)

    # L2 norm of difference
    error_vector = current_ang_vel - target_ang_vel
    return error_vector


def position_error_w(env, asset_cfg: SceneEntityCfg, command_name: str):
    """Compute position error between object and commanded target position.

    Args:
        env: The environment instance (ManagerBasedRLEnv).
        asset_cfg: SceneEntityCfg, identifies the object asset.
        command_name: str, name of the command providing target position.

    Returns:
        Tensor of shape (num_envs, 3): position error in world frame.
    """
    # 当前物体位置
    obj_pos = env.scene[asset_cfg.name].data.root_pos_w  # (num_envs, 3)

    # 从 command manager 里取出目标命令
    cmd = env.command_manager.get_command(command_name)  # (num_envs, 6)
    # 前三维是期望位置
    target_pos = cmd[:, :3]

    # 误差（当前位置 - 目标位置）
    return obj_pos - target_pos


def vertical_position_status(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Vertical position error - critical for preventing drops.
    
    单独提供垂直方向信息,因为这是最容易掉的方向。
    """
    asset: RigidObject = env.scene[object_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    
    target_z = command_term.command[:, 2:3]
    current_z = (asset.data.root_pos_w - env.scene.env_origins)[:, 2:3]
    
    z_error = current_z - target_z
    
    return z_error


def angular_velocity_quality_metrics(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Multiple angular velocity quality indicators.
    
    返回 [magnitude_error, direction_alignment] 两个指标。
    帮助机器人理解速度大小和方向两个维度。
    """
    asset: RigidObject = env.scene[object_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    
    target_ang_vel = command_term.command[:, 3:6]
    current_ang_vel = asset.data.root_ang_vel_w
    
    # Magnitude error
    target_magnitude = torch.norm(target_ang_vel, p=2, dim=-1, keepdim=True)
    current_magnitude = torch.norm(current_ang_vel, p=2, dim=-1, keepdim=True)
    magnitude_error = current_magnitude - target_magnitude
    
    # Direction alignment (cosine similarity)
    cosine_sim = torch.nn.functional.cosine_similarity(
        current_ang_vel, target_ang_vel, dim=-1, eps=1e-8
    ).unsqueeze(-1)
    
    return torch.cat([magnitude_error, cosine_sim], dim=-1)


def time_until_command_resample(
    env: ManagerBasedRLEnv,
    command_name: str
) -> torch.Tensor:
    """Time remaining until next command resample (normalized to [0, 1]).
    
    Returns:
        Tensor of shape (num_envs, 1): normalized time [0=just resampled, 1=about to resample]
    """
    command_term = env.command_manager.get_term(command_name)
    
    # 计算距离下次重采样的时间
    time_since_resample = env.episode_length_buf * env.step_dt - command_term.time_left
    max_resample_time = command_term.cfg.resampling_time_range[1]
    
    # 归一化到 [0, 1]
    normalized_time = torch.clamp(time_since_resample / max_resample_time, 0.0, 1.0)
    
    return normalized_time.unsqueeze(-1)

def last_joint_pos_cmd(env: ManagerBasedRLEnv, action_name: str | None = None) -> torch.Tensor:
    """
    The commanded joint position of the last time step
    https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions_to_limits.py
    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    return env.action_manager.get_term(action_name)._prev_applied_actions
