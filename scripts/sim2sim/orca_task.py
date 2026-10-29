"""Everything the sim2sim harness copies from the Isaac Lab task, in one place, with where it comes from.

Source of truth (keep in sync; the CPU test in docs/ablation/sim2sim.md checks the values against these files):
  T = source/faive_lab/faive_lab/tasks/orca_inhand/orca_v2_repose_env_cfg.py   (OrcaV2ReposeEnvCfg, S0 / ablation B)
  R = source/faive_lab/faive_lab/assets/orca_v2.py                              (ORCA_V2_CFG, read at import, not copied)
  U = scripts/sim2sim/orca_v2_right_usd.json                                     (extracted from the hand USD)

Joint order: the policy's observation and action order is the articulation order PhysX gives the joints
(robot.joint_names, breadth-first over the link tree, siblings by name), printed in every training log
(logs/.../sim/isaaclab_*.log, "Joint names:"). It is NOT the order of the joint list in orca_v2.py.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ORCA_V2_PY = REPO_ROOT / "source/faive_lab/faive_lab/assets/orca_v2.py"
USD_JSON = Path(__file__).resolve().parent / "orca_v2_right_usd.json"

JOINT_NAMES = (
    "right_wrist",
    "right_index_abd", "right_middle_abd", "right_pinky_abd", "right_ring_abd", "right_thumb_cmc",
    "right_index_mcp", "right_middle_mcp", "right_pinky_mcp", "right_ring_mcp", "right_thumb_abd",
    "right_index_pip", "right_middle_pip", "right_pinky_pip", "right_ring_pip", "right_thumb_mcp",
    "right_thumb_dip",
)  # fmt: skip
"""Isaac Lab / PhysX articulation order = policy obs and action order (17)."""

# --- simulation (T: OrcaV2ReposeEnvCfg) ----------------------------------------------------------------------------
PHYSICS_DT = 1.0 / 120.0  # sim.dt
DECIMATION = 6  # control at 20 Hz
CONTROL_DT = PHYSICS_DT * DECIMATION
GRAVITY = (0.0, 0.0, -9.81)  # Isaac default; hand links have disable_gravity=True (R), the cube does not

# --- scene (T: HandAndObjectSceneCfg) -------------------------------------------------------------------------------
HAND_POS = (0.0004, 0.1512, -0.0187)  # robot init_state.pos (root link right_forearm, fixed base)
HAND_QUAT = (-0.2607, 0.9654, 0.0, 0.0)  # robot init_state.rot, w x y z
CUBE_SIZE = 0.045  # DexCube 60 mm scaled by 45/60
CUBE_MASS = 0.11  # object mass_props.mass
CUBE_DEFAULT_POS = (0.0, 0.0, 0.0)  # object init_state.pos (= default root state)

# --- MDP ------------------------------------------------------------------------------------------------------------
EMA_ALPHA = 0.2  # ActionsCfg.joint_pos (EMAJointPositionToLimitsAction, rescale_to_limits=True)
JOINT_VEL_SCALE = 0.2  # policy obs joint_vel scale
OBS_NOISE_STD = {"joint_pos": 0.10, "joint_vel": 0.05}  # Gnoise on the policy group, added BEFORE the scale
COMMAND_POS = (0.0, 0.0, -0.05)  # ContinuousRotationCommand.pos_command = default cube pos + init_pos_offset
DROP_HEIGHT = -0.2  # TerminationsCfg.object_dropped: cube root z (world) below this

# --- reset distribution (T: EventCfg, mode "reset") ----------------------------------------------------------------
CUBE_RESET_POS_RANGE = ((-0.005, 0.005), (-0.005, 0.005), (-0.021, -0.019))  # reset_object pose_range x y z
CUBE_RESET_EULER_RANGE = (-3.145, 3.145)  # roll / pitch / yaw, each uniform (quat_from_euler_xyz)
CUBE_RESET_LIN_VEL = 0.001  # +- per axis; angular velocity 0
JOINT_RESET_SCALE = 0.2  # reset_robot_joints: U(0.2 * lower, 0.2 * upper) per joint (operation "scale", default 0)
HAND_RESET_ROLL = 0.05  # reset_robot_base: roll U(+-0.05) about the hand root's own x axis (DR group)

# --- domain randomization (T: EventCfg), used with --dr isaac ------------------------------------------------------
DR_HAND_FRICTION = (0.8, 2.0)  # robot_physics_material (startup), static and dynamic
DR_CUBE_FRICTION = (0.4, 1.0)  # object_physics_material dynamic (static 0.3-1.0); PhysX combines by average
DR_CUBE_MASS_SCALE = (0.5, 1.5)  # object_scale_mass
DR_LINK_MASS_SCALE = (0.95, 1.05)  # robot_scale_mass (startup)
DR_STIFFNESS_SCALE = (0.75, 1.5)  # robot_joint_stiffness_and_damping, log-uniform, per joint
DR_DAMPING_SCALE = (0.3, 3.0)
DR_JOINT_LIMIT_STD = 0.01  # robot_joint_pos_limits: + N(0, 0.01) on lower and upper
DR_MIN_STEPS_BETWEEN = 720  # min_step_count_between_reset of the reset-mode DR terms (control steps)

NOMINAL_HAND_FRICTION = 0.5 * sum(DR_HAND_FRICTION)
NOMINAL_CUBE_FRICTION = 0.5 * sum(DR_CUBE_FRICTION)


# --- actuator parameters (R) ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class JointParams:
    """Per joint, in JOINT_NAMES order: Isaac's implicit PD drive and joint properties (SI units, rad)."""

    stiffness: np.ndarray  # N m / rad (PhysX angular drive, Isaac Lab writes rad-based values)
    damping: np.ndarray  # N m s / rad
    friction: np.ndarray  # N m: Isaac Sim 5 static friction effort (dynamic / viscous friction are 0 in the log)
    armature: np.ndarray  # kg m^2
    effort_limit: np.ndarray  # N m, clamps the drive torque (effort_limit_sim)
    lower: np.ndarray  # rad, USD limits (soft_joint_pos_limit_factor 1.0)
    upper: np.ndarray


def _resolve(values, joint: str) -> float:
    """Value of a joint in an Isaac Lab per-joint dict (keys are regexes, re.fullmatch, exactly one must match)."""
    if not isinstance(values, dict):
        return float(values)
    hits = [v for k, v in values.items() if re.fullmatch(k, joint)]
    if len(hits) != 1:
        raise ValueError(f"{joint}: {len(hits)} matches in {list(values)}")
    return float(hits[0])


def read_joint_params(orca_v2_py: Path = ORCA_V2_PY, usd_json: Path = USD_JSON) -> JointParams:
    """Parse the ImplicitActuatorCfg literals in orca_v2.py (no Isaac Lab import) and the USD joint limits."""
    tree = ast.parse(Path(orca_v2_py).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "ImplicitActuatorCfg"]
    if len(calls) != 1:
        raise ValueError(f"expected one ImplicitActuatorCfg in {orca_v2_py}, found {len(calls)}")
    kw = {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords}
    missing = set(JOINT_NAMES) ^ set(kw["joint_names_expr"])
    if missing:
        raise ValueError(f"orca_v2.py actuator joints differ from JOINT_NAMES: {sorted(missing)}")
    limits = {j["name"]: j["limits_deg"] for j in json.loads(Path(usd_json).read_text())["joints"] if "limits_deg" in j}

    def col(key, default=None):
        return np.array([_resolve(kw.get(key, default), j) for j in JOINT_NAMES])

    lim = np.deg2rad(np.array([limits[j] for j in JOINT_NAMES]))
    return JointParams(
        stiffness=col("stiffness"),
        damping=col("damping"),
        friction=col("friction", 0.0),
        armature=col("armature", 0.0),
        effort_limit=col("effort_limit_sim"),
        lower=lim[:, 0],
        upper=lim[:, 1],
    )


# --- reset and DR sampling ------------------------------------------------------------------------------------------


def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Isaac Lab's math_utils.quat_from_euler_xyz (w x y z)."""
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    return np.array([
        cy * cr * cp + sy * sr * sp,
        cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp,
        sy * cr * cp - cy * sr * sp,
    ])  # fmt: skip


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])  # fmt: skip


@dataclass
class ResetState:
    """One (re)spawn as Isaac's reset events sample it."""

    joint_pos: np.ndarray
    cube_pos: np.ndarray
    cube_quat: np.ndarray
    cube_lin_vel: np.ndarray
    hand_quat: np.ndarray


def sample_reset(rng: np.random.Generator, jp: JointParams, hand_roll: bool) -> ResetState:
    """reset_robot_joints + reset_object (+ reset_robot_base when ``hand_roll``; it is a DR term in the task)."""
    joint_pos = rng.uniform(JOINT_RESET_SCALE * jp.lower, JOINT_RESET_SCALE * jp.upper)
    lo, hi = np.array(CUBE_RESET_POS_RANGE).T
    cube_pos = np.array(CUBE_DEFAULT_POS) + rng.uniform(lo, hi)
    cube_quat = quat_from_euler_xyz(*rng.uniform(*CUBE_RESET_EULER_RANGE, size=3))
    cube_lin_vel = rng.uniform(-CUBE_RESET_LIN_VEL, CUBE_RESET_LIN_VEL, size=3)
    roll = rng.uniform(-HAND_RESET_ROLL, HAND_RESET_ROLL) if hand_roll else 0.0
    hand_quat = quat_mul(np.array(HAND_QUAT), quat_from_euler_xyz(roll, 0.0, 0.0))
    hand_quat /= np.linalg.norm(hand_quat)  # the cfg's (-0.2607, 0.9654, 0, 0) has norm 0.99998; PhysX normalizes too
    return ResetState(joint_pos, cube_pos, cube_quat, cube_lin_vel, hand_quat)


@dataclass
class PhysicsParams:
    """Physics of one trial; defaults are the centers of Isaac's DR ranges ("nominal")."""

    hand_friction: float = NOMINAL_HAND_FRICTION
    cube_friction: float = NOMINAL_CUBE_FRICTION
    cube_mass: float = CUBE_MASS
    link_mass_scale: np.ndarray | None = None  # per link (USD order), None = 1
    stiffness_scale: np.ndarray = field(default_factory=lambda: np.ones(len(JOINT_NAMES)))
    damping_scale: np.ndarray = field(default_factory=lambda: np.ones(len(JOINT_NAMES)))
    limit_offset: np.ndarray = field(default_factory=lambda: np.zeros((len(JOINT_NAMES), 2)))  # rad, lower / upper

    @property
    def contact_friction(self) -> float:
        """Hand-cube friction: PhysX's default combine mode averages the two materials."""
        return 0.5 * (self.hand_friction + self.cube_friction)


def sample_startup_dr(rng: np.random.Generator, p: PhysicsParams, n_links: int) -> None:
    """Isaac's startup-mode terms (once per env): hand friction, link masses."""
    p.hand_friction = rng.uniform(*DR_HAND_FRICTION)
    p.link_mass_scale = rng.uniform(*DR_LINK_MASS_SCALE, size=n_links)


def sample_reset_dr(rng: np.random.Generator, p: PhysicsParams) -> None:
    """Isaac's reset-mode DR terms: drive gains, joint limits, cube friction and mass."""
    n = len(JOINT_NAMES)
    p.stiffness_scale = np.exp(rng.uniform(*np.log(DR_STIFFNESS_SCALE), size=n))
    p.damping_scale = np.exp(rng.uniform(*np.log(DR_DAMPING_SCALE), size=n))
    p.limit_offset = rng.normal(0.0, DR_JOINT_LIMIT_STD, size=(n, 2))
    p.cube_friction = rng.uniform(*DR_CUBE_FRICTION)
    p.cube_mass = CUBE_MASS * rng.uniform(*DR_CUBE_MASS_SCALE)
