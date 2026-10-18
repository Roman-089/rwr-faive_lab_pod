# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .orientation_command import InHandReOrientationCommand
from .continuous_command import ContinuousRotationCommand


@configclass
class InHandReOrientationCommandCfg(CommandTermCfg):
    """Configuration for the uniform 3D orientation command term.

    Please refer to the :class:`InHandReOrientationCommand` class for more details.
    """

    class_type: type = InHandReOrientationCommand
    resampling_time_range: tuple[float, float] = (1e6, 1e6)  # no resampling based on time

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    init_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset of the asset from its default position.

    This is used to account for the offset typically present in the object's default position
    so that the object is spawned at a height above the robot's palm. When the position command
    is generated, the object's default position is used as the reference and the offset specified
    is added to it to get the desired position of the object.
    """

    make_quat_unique: bool = MISSING
    """Whether to make the quaternion unique or not.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    orientation_success_threshold: float = MISSING
    """Threshold for the orientation error to consider the goal orientation to be reached."""

    update_goal_on_success: bool = MISSING
    """Whether to update the goal orientation when the goal orientation is reached."""

    marker_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset of the marker from the object's desired position.

    This is useful to position the marker at a height above the object's desired position.
    Otherwise, the marker may occlude the object in the visualization.
    """

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.0, 1.0, 1.0),
            ),
        },
    )
    """The configuration for the goal pose visualization marker. Defaults to a DexCube marker."""

@configclass
class ContinuousRotationCommandCfg(CommandTermCfg):
    """Configuration for the continuous rotation command term.

    This command term generates continuous rotation commands with target angular velocity.
    The command includes both position (to keep object in place) and angular velocity
    (to specify rotation speed and direction).

    Please refer to the :class:`ContinuousRotationCommand` class for more details.
    """

    class_type: type = ContinuousRotationCommand
    resampling_time_range: tuple[float, float] = (5.0, 10.0)  # resample angular velocity every 5-10 seconds

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    init_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset of the asset from its default position.

    This is used to account for the offset typically present in the object's default position
    so that the object is spawned at a height above the robot's palm. The object should
    maintain this position while rotating continuously.
    """

    angular_vel_range: tuple[float, float, float] = (2.0, 2.0, 2.0)
    """Range of angular velocities for each axis (rad/s).

    Specifies the maximum angular velocity magnitude for rotation around each axis:
    - [0]: Maximum angular velocity around X-axis (rad/s)
    - [1]: Maximum angular velocity around Y-axis (rad/s) 
    - [2]: Maximum angular velocity around Z-axis (rad/s)
    
    The actual angular velocity commands will be sampled uniformly from 
    [-angular_vel_range[i], +angular_vel_range[i]] for each axis i.
    """

    angular_vel_magnitude_threshold: float = 0.3
    """Threshold for angular velocity magnitude error (rad/s).
    
    When the difference between commanded and actual angular speed magnitude
    is below this threshold, it is considered successful.
    """

    angular_vel_direction_threshold: float = 0.7
    """Threshold for angular velocity direction alignment (cosine similarity).
    
    Values close to 1.0 mean the actual and commanded angular velocities
    are well aligned. Typically set between 0.8 and 0.95.
    """

    position_success_threshold: float = 0.08
    """Threshold for position error to consider position holding successful (m).
    
    When the position error (distance between commanded and actual position) falls
    below this threshold, the position holding is considered successful.
    """
    debug_vis: bool = True
