# Owner: X (Task X / FAST)
"""FAST screw task: turn a bolt between thumb and index as fast as commanded – in, out, in, out.

Same pipeline as Task 1 (manager-based env, rsl_rl PPO, same train.py / play.py), and it reuses Task 1's
actions, proprio observations and domain randomization, so S0's improvements carry over automatically.
Differences to Task 1:
  - robot = ORCA v2 with QDD actuators on QDD_JOINTS (assets/orca_v2_fast.py)
  - object = fixed screw fixture with a bolt on a continuous joint (bolt_usd.py), thread = joint damping/friction
  - command = signed bolt speed, alternating in/out (mdp.ScrewCommand)
  - rewards = bolt speed tracking + direction progress + fingertip proximity + Task 1 smoothness penalties

Placement of the bolt relative to the hand (BOLT_POS_HAND) is a first guess – check it in the play video
and move it so thumb tip and index tip can both reach the head.
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
from .bolt_usd import BoltGeometry, bolt_usd_path

# --------------------------------------------------------------------------- placement
HAND_POS = (0.0, 0.0, 0.5)
HAND_ROT = (0.0, 0.0, 0.0, 1.0)                 # (w, x, y, z), ORCA default
# bolt head centre in the hand (forearm) frame: fingers point +y, thumb on the -z (palm) side,
# index on -x. Bolt axis = hand x, shaft pointing out of the hand on the index side.
BOLT_POS_HAND = (-0.045, 0.165, -0.040)         # TODO: tune with the play video
BOLT_ROT_HAND = (1.0, 0.0, 0.0, 0.0)
BOLT = BoltGeometry()

THREAD_DAMPING = 2e-4                           # Nm/(rad/s) – loose thread, randomized below
THREAD_FRICTION = 2e-3                          # Nm – randomized below


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def _rotate(q, v):
    w, x, y, z = _qmul(_qmul(q, (0.0, *v)), (q[0], -q[1], -q[2], -q[3]))
    return (x, y, z)


def bolt_world_pose(hand_pos=HAND_POS, hand_rot=HAND_ROT, pos_hand=BOLT_POS_HAND, rot_hand=BOLT_ROT_HAND):
    p = tuple(h + r for h, r in zip(hand_pos, _rotate(hand_rot, pos_hand)))
    return p, _qmul(hand_rot, rot_hand)


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

    bolt: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Bolt",
        spawn=sim_utils.UsdFileCfg(
            usd_path=bolt_usd_path(BOLT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=bolt_world_pose()[0], rot=bolt_world_pose()[1], joint_pos={"bolt_joint": 0.0}),
        actuators={
            # passive joint: no position spring, only the thread's resistance
            "thread": ImplicitActuatorCfg(joint_names_expr=["bolt_joint"], stiffness=0.0,
                                          damping=THREAD_DAMPING, friction=THREAD_FRICTION),
        },
    )


# --------------------------------------------------------------------------- MDP
@configclass
class CommandsCfg:
    screw = screw_mdp.ScrewCommandCfg(asset_name="bolt", resampling_time_range=(4.0, 4.0),
                                      speed_range=(5.0, 15.0), alternate_direction=True, debug_vis=False)


@configclass
class ScrewObservationsCfg:
    @configclass
    class PolicyCfg(ObservationsCfg.KinematicObsGroupCfg):
        """Task 1 proprio terms in the same order; goal_command now = screw command (1 value)."""

        goal_command = ObsTerm(func=orca_mdp.generated_commands, params={"command_name": "screw"})
        # extra terms are appended at the end
        bolt_vel = ObsTerm(func=screw_mdp.bolt_joint_vel, scale=0.1, noise=Gnoise(std=0.05),
                           params={"asset_cfg": SceneEntityCfg("bolt", joint_names=["bolt_joint"])})

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    speed_tracking = RewTerm(func=screw_mdp.screw_speed_tracking_exp, weight=30.0,
                             params={"command_name": "screw", "sigma": 3.0,
                                     "asset_cfg": SceneEntityCfg("bolt", joint_names=["bolt_joint"])})
    direction_progress = RewTerm(func=screw_mdp.screw_direction_progress, weight=10.0,
                                 params={"command_name": "screw", "max_speed": 15.0,
                                         "asset_cfg": SceneEntityCfg("bolt", joint_names=["bolt_joint"])})
    fingertips_near_bolt = RewTerm(func=screw_mdp.fingertips_near_bolt, weight=2.0,
                                   params={"sigma": 0.03,
                                           "robot_cfg": SceneEntityCfg("robot", body_names=[
                                               "right_index_fingertip_assembly", "right_thumb_dp"]),
                                           "bolt_cfg": SceneEntityCfg("bolt", body_names=["bolt"])})
    # same smoothness penalties as Task 1
    joint_vel_l2 = RewTerm(func=orca_mdp.joint_vel_l2, weight=-1e-5)
    action_l2 = RewTerm(func=orca_mdp.action_l2, weight=-2e-4)
    action_rate_l2 = RewTerm(func=orca_mdp.action_rate_l2, weight=-0.075)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=orca_mdp.time_out, time_out=True)


@configclass
class ScrewEventCfg(EventCfg):
    """Task 1 robot randomization + bolt randomization (Task 1 cube terms are switched off below)."""

    bolt_pose = EventTerm(
        func=orca_mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": [-0.005, 0.005], "y": [-0.005, 0.005], "z": [-0.005, 0.005]},
                "velocity_range": {}, "asset_cfg": SceneEntityCfg("bolt")})
    bolt_angle = EventTerm(
        func=orca_mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("bolt"), "position_range": (-math.pi, math.pi),
                "velocity_range": (0.0, 0.0)})
    thread_damping = EventTerm(
        func=orca_mdp.randomize_actuator_gains, mode="reset",
        params={"asset_cfg": SceneEntityCfg("bolt", joint_names=["bolt_joint"]),
                "stiffness_distribution_params": (1.0, 1.0), "damping_distribution_params": (0.3, 5.0),
                "operation": "scale", "distribution": "log_uniform"})
    thread_friction = EventTerm(
        func=orca_mdp.randomize_joint_parameters, mode="reset",
        params={"asset_cfg": SceneEntityCfg("bolt", joint_names=["bolt_joint"]),
                "friction_distribution_params": (0.3, 5.0), "operation": "scale", "distribution": "log_uniform"})
    bolt_material = EventTerm(
        func=orca_mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("bolt", body_names=".*"), "static_friction_range": (0.4, 1.0),
                "dynamic_friction_range": (0.3, 0.9), "restitution_range": (0.0, 0.0), "num_buckets": 64})


@configclass
class FastScrewEnvCfg(ManagerBasedRLEnvCfg):
    scene: InteractiveSceneCfg = ScrewSceneCfg(num_envs=4096, env_spacing=0.5, replicate_physics=False)
    control_hz: float = 20.0                   # Task 1 rate; the QDD loop can go much faster -> try 60 Hz
    episode_length_s = 16.0                    # 4 x 4 s: in, out, in, out
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physx=PhysxCfg(bounce_threshold_velocity=0.2, gpu_max_rigid_contact_count=2**20,
                       gpu_max_rigid_patch_count=2**23))
    viewer = ViewerCfg(eye=(0.35, -0.35, 0.75), lookat=bolt_world_pose()[0])

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
        # Task 1 cube terms do not exist here; hand roll reset would move the hand away from the fixed bolt
        for t in ("object_physics_material", "object_scale_mass", "reset_object", "reset_robot_base"):
            setattr(self.events, t, None)


@configclass
class FastScrew60HzEnvCfg(FastScrewEnvCfg):
    """Same task at 60 Hz control – lets the policy use the QDD bandwidth (EMA alpha kept at 0.2)."""

    control_hz: float = 60.0


@configclass
class FastScrewPlayEnvCfg(FastScrewEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
