# Owner: X (Task X / FAST)
"""FAST screw task: the hand hangs palm-down over a fixture plate and spins an M8 nut on an upright bolt –
screw in, screw out, as fast as commanded (reference: "Robot fingers go invisible mode on screws").

Same pipeline as Task 1 (manager-based env, rsl_rl PPO, same train.py / play.py); reuses Task 1's actions,
proprio observations and robot domain randomization, so S0's improvements carry over automatically.

World geometry (plate, bolt, nut, hand placement) comes from world.py – the SAME numbers generate the
printable fixture (scripts/fast_world/export_cad.py), so sim and real setup match.

Tasks:
  fast-screw-v0              blind policy (proprio + command only) -> deployable without a nut sensor
  fast-screw-privileged-v0   + nut speed & height in the observation (sim upper bound / teacher)
  fast-screw-60hz-v0         blind, 60 Hz control (uses the QDD bandwidth)
  fast-screw-play-v0         16 envs, no noise, for play.py videos
"""
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

from faive_lab.assets.orca_v2_fast import ORCA_V2_FAST_CFG
from faive_lab.tasks.orca_inhand import mdp as orca_mdp
from faive_lab.tasks.orca_inhand.orca_v2_repose_env_cfg import ActionsCfg, EventCfg, ObservationsCfg

from . import mdp as screw_mdp
from .world import WorldGeometry, hand_pose
from .world_usd import world_usd_path

WORLD = WorldGeometry()
HAND_POS, HAND_ROT = hand_pose(WORLD)
NUT_START_LIFT = WORLD.nut_start_height * WORLD.nut_travel

THREAD_DAMPING = 2e-4        # Nm/(rad/s) on nut_spin – free-running thread, randomized
THREAD_FRICTION = 2e-3       # Nm on nut_spin, randomized
FINGERS_FOR_PROXIMITY = ["right_index_fingertip_assembly", "right_middle_fingertip_assembly", "right_thumb_dp"]

SPIN = SceneEntityCfg("fixture", joint_names=["nut_spin"])
LIFT = SceneEntityCfg("fixture", joint_names=["nut_lift"])


# --------------------------------------------------------------------------- scene
@configclass
class ScrewSceneCfg(InteractiveSceneCfg):
    light = AssetBaseCfg(prim_path="/World/light",
                         spawn=sim_utils.DistantLightCfg(color=(0.95, 0.95, 0.95), intensity=1000.0))
    dome_light = AssetBaseCfg(prim_path="/World/domeLight",
                              spawn=sim_utils.DomeLightCfg(color=(0.02, 0.02, 0.02), intensity=1000.0))

    robot: ArticulationCfg = ORCA_V2_FAST_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ORCA_V2_FAST_CFG.init_state.replace(pos=HAND_POS, rot=HAND_ROT),
    )

    fixture: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Fixture",
        spawn=sim_utils.UsdFileCfg(
            usd_path=world_usd_path(WORLD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0),
                                                   joint_pos={"nut_spin": 0.0, "nut_lift": NUT_START_LIFT}),
        actuators={
            # passive spin: only the thread's resistance
            "thread": ImplicitActuatorCfg(joint_names_expr=["nut_spin"], stiffness=0.0,
                                          damping=THREAD_DAMPING, friction=THREAD_FRICTION),
            # software thread: stiff position drive, target set by mdp.ScrewCommand every step
            "lift": ImplicitActuatorCfg(joint_names_expr=["nut_lift"], stiffness=2000.0, damping=20.0,
                                        effort_limit_sim=100.0),
        },
    )


# --------------------------------------------------------------------------- MDP
@configclass
class CommandsCfg:
    screw = screw_mdp.ScrewCommandCfg(asset_name="fixture", resampling_time_range=(4.0, 4.0),
                                      speed_range=(5.0, 15.0), alternate_direction=True,
                                      thread_pitch=WORLD.thread_pitch, debug_vis=False)


@configclass
class ScrewObservationsCfg:
    @configclass
    class PolicyCfg(ObservationsCfg.KinematicObsGroupCfg):
        """Task 1 proprio terms in the same order; goal_command = screw command (1 value instead of 6)."""

        goal_command = ObsTerm(func=orca_mdp.generated_commands, params={"command_name": "screw"})

    policy: PolicyCfg = PolicyCfg()


@configclass
class PrivilegedObservationsCfg:
    @configclass
    class PolicyCfg(ScrewObservationsCfg.PolicyCfg):
        nut_speed = ObsTerm(func=screw_mdp.joint_vel_of, scale=0.1, noise=Gnoise(std=0.05), params={"asset_cfg": SPIN})
        nut_height = ObsTerm(func=screw_mdp.joint_pos_of, scale=50.0, params={"asset_cfg": LIFT})

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    speed_tracking = RewTerm(func=screw_mdp.spin_speed_tracking_exp, weight=30.0,
                             params={"command_name": "screw", "sigma": 3.0, "asset_cfg": SPIN})
    direction_progress = RewTerm(func=screw_mdp.spin_direction_progress, weight=10.0,
                                 params={"command_name": "screw", "max_speed": 15.0, "asset_cfg": SPIN})
    fingers_near_nut = RewTerm(func=screw_mdp.bodies_near_body, weight=2.0,
                               params={"sigma": 0.03,
                                       "robot_cfg": SceneEntityCfg("robot", body_names=FINGERS_FOR_PROXIMITY),
                                       "target_cfg": SceneEntityCfg("fixture", body_names=["nut"])})
    # same smoothness penalties as Task 1
    joint_vel_l2 = RewTerm(func=orca_mdp.joint_vel_l2, weight=-1e-5)
    action_l2 = RewTerm(func=orca_mdp.action_l2, weight=-2e-4)
    action_rate_l2 = RewTerm(func=orca_mdp.action_rate_l2, weight=-0.075)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=orca_mdp.time_out, time_out=True)


@configclass
class ScrewEventCfg(EventCfg):
    """Task 1 robot randomization + fixture randomization (Task 1 cube terms are switched off below)."""

    fixture_pose = EventTerm(
        func=orca_mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": [-0.005, 0.005], "y": [-0.005, 0.005], "z": [-0.003, 0.003],
                               "yaw": [-math.pi, math.pi]},
                "velocity_range": {}, "asset_cfg": SceneEntityCfg("fixture")})
    nut_state = EventTerm(
        func=orca_mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("fixture", joint_names=["nut_spin", "nut_lift"]),
                "position_range": (-0.4 * WORLD.nut_travel, 0.4 * WORLD.nut_travel),  # spin offset is irrelevant (hex)
                "velocity_range": (0.0, 0.0)})
    thread_damping = EventTerm(
        func=orca_mdp.randomize_actuator_gains, mode="reset",
        params={"asset_cfg": SPIN, "stiffness_distribution_params": (1.0, 1.0),
                "damping_distribution_params": (0.3, 5.0), "operation": "scale", "distribution": "log_uniform"})
    thread_friction = EventTerm(
        func=orca_mdp.randomize_joint_parameters, mode="reset",
        params={"asset_cfg": SPIN, "friction_distribution_params": (0.3, 5.0),
                "operation": "scale", "distribution": "log_uniform"})
    nut_material = EventTerm(
        func=orca_mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("fixture", body_names=".*"), "static_friction_range": (0.3, 0.9),
                "dynamic_friction_range": (0.25, 0.8), "restitution_range": (0.0, 0.0), "num_buckets": 64})


@configclass
class FastScrewEnvCfg(ManagerBasedRLEnvCfg):
    scene: InteractiveSceneCfg = ScrewSceneCfg(num_envs=4096, env_spacing=0.4, replicate_physics=False)
    control_hz: float = 20.0                   # Task 1 rate; see fast-screw-60hz-v0
    episode_length_s = 16.0                    # 4 x 4 s: in, out, in, out
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physx=PhysxCfg(bounce_threshold_velocity=0.2, gpu_max_rigid_contact_count=2**20,
                       gpu_max_rigid_patch_count=2**23))
    viewer = ViewerCfg(eye=(0.25, -0.25, 0.25), lookat=(0.0, 0.0, 0.05))

    observations: ScrewObservationsCfg = ScrewObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: ScrewEventCfg = ScrewEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = max(1, round(1.0 / (self.control_hz * self.sim.dt)))
        self.sim.render_interval = self.decimation
        # Task 1 cube terms do not exist here; the hand-roll reset would move the hand off the fixed nut
        for t in ("object_physics_material", "object_scale_mass", "reset_object", "reset_robot_base"):
            setattr(self.events, t, None)


@configclass
class FastScrewPrivilegedEnvCfg(FastScrewEnvCfg):
    observations: PrivilegedObservationsCfg = PrivilegedObservationsCfg()


@configclass
class FastScrew60HzEnvCfg(FastScrewEnvCfg):
    control_hz: float = 60.0


@configclass
class FastScrewPlayEnvCfg(FastScrewEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
