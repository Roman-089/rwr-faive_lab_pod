# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from faive_lab.assets.orca_v2 import ORCA_V2_CFG
from faive_lab import FAIVE_ASSETS_DATA_DIR

from . import mdp
from .mdp.noise_cfg import GaussianNoiseAndNormalizeQuatCfg as QuatNoise
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import TiledCameraCfg #CameraCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

from dataclasses import MISSING

import numpy as np

CAMERA_CFG = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        update_period=0.1,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=[500,   0.          , 112,
                                  0.   , 500,          112,
                                  0.   ,   0.        , 1.          ],
                width=224,
                height=224,
                clipping_range=(0.001, 2.0),
                # focal_length=24
            ),
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.07, 0.0, 1.5), rot=(0, 0, 1.0, 0.0,), convention="ros"),
    )

@configclass
class HandAndObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a multi-object scene."""

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.95, 0.95, 0.95), intensity=1000.0),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/domeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.02, 0.02, 0.02), intensity=1000.0),
    )

    camera = CAMERA_CFG

    # # rigid object
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(45./60, 45./60, 45./60),   # the cube USD is 60mm, make it 45 mm
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=3.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.11),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    robot: ArticulationCfg = ORCA_V2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ORCA_V2_CFG.init_state.replace(
            pos=(0.0004, 0.1512, -0.0187),  # 6 mm towards -x so the cube sits in the middle of the v2 palm, clear of the thumb
            rot=(-0.2607, 0.9654, 0.0, 0.0),
        )
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    # object_pose = mdp.InHandReOrientationCommandCfg(
    #     asset_name="object",
    #     init_pos_offset=(0.0, 0.0, -0.04),
    #     update_goal_on_success=True,
    #     # from 0.1 to 0.15
    #     orientation_success_threshold=0.15,
    #     make_quat_unique=False,
    #     marker_pos_offset=(-0.15, -0.06, 0.08),
    #     debug_vis=True,
    # )

    continuous_rotation = mdp.ContinuousRotationCommandCfg(
        asset_name="object",
        # the target position should be where the cube can be comfortably held
        init_pos_offset=(0.0, 0.0, -0.05),
        angular_vel_range=(0.0, 0.0, 1.5),  # clamp rotation to only around z axis
        angular_vel_magnitude_threshold=0.3, 
        angular_vel_direction_threshold=0.7,   
        position_success_threshold=0.08,
        resampling_time_range=(8.0, 8.0),
    ) 

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=[
            # "(?!right_wrist).*",  # Exclude right_wrist, include all other joints
            ".*",  # Include all joints
        ],
        alpha=0.2,  # smoothing factor for exponential moving average
        rescale_to_limits=True,  # scale [-1, 1] to joint limits
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class KinematicObsGroupCfg(ObsGroup):
        """
        Observations for policy group.
        some pre-implemented observations can be found in
        https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html
        """

        # observation terms (order preserved)
        # -- robot terms
        # reduce the noise
        joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, noise=Gnoise(std=0.10))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2, noise=Gnoise(std=0.05))

        # -- object terms
        # object_pos = ObsTerm(
        #     func=mdp.root_pos_w, noise=Gnoise(std=0.01), params={"asset_cfg": SceneEntityCfg("object")}
        # )

        # object_quat = ObsTerm(
        #     func=mdp.root_quat_w, noise=QuatNoise(std=0.05), params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False}
        # )

        # object_lin_vel = ObsTerm(
        #     func=mdp.root_lin_vel_w, noise=Gnoise(std=0.1), params={"asset_cfg": SceneEntityCfg("object")}
        # )

        # Remove angular velocity scaling to allow the robot to load complete angular velocity information
        # object_ang_vel = ObsTerm(
        #     func=mdp.root_ang_vel_w,
        #     scale=1.0,  # Change from 0.2 to 1.0, do not compress angular velocity information
        #     noise=Gnoise(std=0.01),
        #     params={"asset_cfg": SceneEntityCfg("object")},
        # )
        # position_error = ObsTerm(
        #     func=mdp.position_error_w,  # 需要：pos - pos_cmd
        #     noise=Gnoise(std=0.001),
        #     params={"asset_cfg": SceneEntityCfg("object"), "command_name": "continuous_rotation"},
        # )
        
        # -- command terms
        
        goal_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "continuous_rotation"})

        last_action = ObsTerm(func=mdp.last_action)
        last_joint_pos_cmd = ObsTerm(func=mdp.last_joint_pos_cmd, params={"action_name": "joint_pos"})

        enable_corruption = True
        concatenate_terms = True
    
    @configclass
    class TeacherObsGroupCfg(ObsGroup):
        """Noisy-free observations for teacher policy."""
        
        joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized)  
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)    
        
        object_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False})
        # object_lin_vel = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("object")})
        # object_ang_vel = ObsTerm(func=mdp.root_ang_vel_w, scale=1.0, params={"asset_cfg": SceneEntityCfg("object")})
        # position_error = ObsTerm(func=mdp.position_error_w, params={"asset_cfg": SceneEntityCfg("object"), "command_name": "continuous_rotation"})
        # angular_vel_error = ObsTerm(func=mdp.angular_vel_error, params={"command_name": "continuous_rotation", "object_cfg": SceneEntityCfg("object")})
        
        goal_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "continuous_rotation"})
        # time_to_resample = ObsTerm(func=mdp.time_until_command_resample, params={"command_name": "continuous_rotation"})
        
        last_action = ObsTerm(func=mdp.last_action)
        last_joint_pos_cmd = ObsTerm(func=mdp.last_joint_pos_cmd, params={"action_name": "joint_pos"})
        enable_corruption = False  
        concatenate_terms = True

    @configclass
    class CameraCfg(ObsGroup):
        """Observations for camera group."""

        camera = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera"), "normalize":False},
        )
        
        enable_corruption = False
        concatenate_terms = False

    # observation groups
    # policy: TeacherObsGroupCfg = TeacherObsGroupCfg()
    policy: KinematicObsGroupCfg = KinematicObsGroupCfg()
    camera: CameraCfg = CameraCfg()
    # student: KinematicObsGroupCfg = KinematicObsGroupCfg()


@configclass
class RewardsCfg:
    """
    Reward terms for the MDP.
    Useful pre-implemented reward terms can be found in
    https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#module-isaaclab.envs.mdp.rewards
    """
    # 1. Tracking object position (also penalizes drops)
    object_position_tracking = RewTerm(
        func=mdp.track_position_exp,
        weight=15.0,
        params={
            "command_name": "continuous_rotation",
            "object_cfg": SceneEntityCfg("object"),
            "sigma": 0.02,
        },
    )

    # 2. Matching rotation speed
    angular_velocity_magnitude = RewTerm(
        func=mdp.track_angular_velocity_magnitude_exp,
        weight=30.0,
        params={
            "command_name": "continuous_rotation",
            "object_cfg": SceneEntityCfg("object"),
            "sigma": 0.2,
        },
    )
    # 3. Match the direction of rotation
    angular_velocity_direction = RewTerm(
        func=mdp.angular_velocity_alignment,
        weight=60.0,
        params={
            "command_name": "continuous_rotation",
            "object_cfg": SceneEntityCfg("object"),
        },
    )

    angular_acceleration = RewTerm(
        func=mdp.angular_acceleration_penalty_exp,
        weight=1.0,  
        params={
            "object_cfg": SceneEntityCfg("object"),
            "sigma": 0.5
        },  
    )

    # -- penalties
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-5)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-2e-4)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.075)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropped = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -0.2, "asset_cfg": SceneEntityCfg("object")})

@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- robot
    joint_reset = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.01, 0.01),
            "velocity_range": (-0.0, 0.0),
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 2.0),
            "dynamic_friction_range": (0.8, 2.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    
    robot_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    robot_joint_pos_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        },
    )


    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- scene
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(24.0, 360.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

    # reset
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.005, 0.005], "y": [-0.005, 0.005], "z": [-0.019, -0.021],
                           "roll": [-3.145, 3.145], "pitch": [-3.145, 3.145], "yaw": [-3.145, 3.145]},
            "velocity_range": {
            "x": [-0.001, 0.001],       
            "y": [-0.001, 0.001],      
            "z": [-0.001, 0.001],
            # "roll": [-0.02, 0.02],   
            # "pitch": [-0.02, 0.02],  
            # "yaw": [-0.02, 0.02],  
            },
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
        },
    )
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [0.0, 0.0],          # Keep x fixed ????????????
                # "y": [-0.0, 0.0],           # Keep y fixed 
                # "z": [-0.0, 0.0],           # Keep z fixed
                "roll": [-0.05, 0.05],  # ±2 degrees (2° = 0.0349 rad)
                # "pitch": [-0.1, 0.1], # ±5 degrees (5° = 0.0873 rad)
                # "yaw": [-0.0, 0.0],          # Keep yaw fixed
            },
            "velocity_range": {
                # "x": [0.0, 0.0],
                # "y": [0.0, 0.0],
                # "z": [0.0, 0.0],
                # "roll": [0.0, 0.0],
                # "pitch": [0.0, 0.0],
                # "yaw": [0.0, 0.0],
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_within_limits_range,
        mode="reset",
        params={
            "position_range": {".*": [0.2, 0.2]},
            "velocity_range": {".*": [0.0, 0.0]},
            "use_default_offset": True,
            "operation": "scale",
        },
    )


@configclass
class OrcaV2ReposeEnvCfg(ManagerBasedRLEnvCfg):
    """
    Configured for the ORCA v2 hand for the cube repose task
    For other hands and other tasks, child classes can be created from this with the necessary fields overridden
    """
    # scene
    scene: InteractiveSceneCfg = HandAndObjectSceneCfg(num_envs=2048, env_spacing=0.5, replicate_physics=False)

    # env
    # use 20Hz control, same as default for ORCA in faive_system and what was used for the characterization
    decimation = 6
    episode_length_s = 32.0
    is_finite_horizon = True
    enable_cam = False
    cam_to_01 = False # change from 0 255 to 0. 1. and permute to C H W

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**20,
            gpu_max_rigid_patch_count=2**23, 
        ),
    )
    # viewer
    viewer = ViewerCfg(eye=(2.0, -2.0, 2.0))

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # xr: XrCfg = XrCfg( # TODO: this needs to be configured properly
    #     anchor_pos=(0.0, 0.0, -0.6),
    #     anchor_rot=(-0.7071, 0.0, 0.0, 0.7071),
    # )

    def __post_init__(self):
        super().__post_init__()
        if not self.enable_cam:
            self.scene.camera = None
            self.observations.camera = None
        else:

            self.scene.camera = CAMERA_CFG
            self.observations.camera = ObservationsCfg.CameraCfg()
            if self.cam_to_01:
                for k, v in self.observations.camera.__dict__.items():
                    if isinstance(v, ObsTerm) and v.func == mdp.image:
                        v.func = mdp.rgb_CHWF
                        v.params.pop("normalize", None)
        # self.observations.policy.concatenate_terms = False
        # self.observations.student.concatenate_terms = False

                

