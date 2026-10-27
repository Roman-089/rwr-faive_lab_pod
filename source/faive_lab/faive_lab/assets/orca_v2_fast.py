# Owner: X (Task X / FAST) – see TEAM_FAST.md
"""ORCA v2 with QDD direct-drive joints (CubeMars GL30 KV290 + moteus-c1) – the FAST hand.

In the sim the tendons are not modelled: every joint is driven directly by an implicit PD actuator.
So "FAST" only changes the actuator parameters of the QDD joints; the hand geometry (USD) stays the same.

The QDD parameters below are ORDER-OF-MAGNITUDE ESTIMATES from the transmission, not measured.
Replace them with the joint fit (rwr_faive_lab/scripts/motor_param_characterization/optimization.py) on chirps
recorded with rwr-fast/taskx/fast/tools/chirp.py, exactly like Task 1 does for the servo joints.

Physics note (why the estimate matters): without a gearbox the motor torque reaches the joint only through
the spool -> tendon -> joint-pulley ratio. Power is conserved, so a joint that is ~30x faster than the servo
version is also roughly ~30x weaker. Check the effort below against what screwing actually needs.
"""
import re

from isaaclab.actuators import ImplicitActuatorCfg

from faive_lab.assets.orca_v2 import ORCA_V2_CFG

# TODO (Thu 29.10. with Aurel): which joints get a QDD motor. Must match rwr-fast/taskx/fast/config.yaml.
QDD_JOINTS = ["right_index_abd", "right_index_mcp", "right_index_pip", "right_thumb_mcp"]

# --- transmission estimate (TODO: take r_spool / r_joint from the FAST CAD) ---------------------------
MOTOR_RATED_TORQUE_NM = 0.08      # GL30 KV290 rated torque (Aurel's QDD post)
MOTOR_NOLOAD_RAD_S = 290 * 12 * 2 * 3.141592653589793 / 60    # KV290 at 12 V ≈ 364 rad/s
R_SPOOL_M = 0.0095                # tendon radius on the motor spool (estimate)
R_JOINT_M = 0.006                 # tendon moment arm at the finger joint (estimate)
# tendon force F = tau_m / r_spool  ->  joint torque = F * r_joint ; tendon speed = w_m * r_spool -> w_j = v / r_joint
TORQUE_RATIO = R_JOINT_M / R_SPOOL_M                  # joint torque per motor torque (= motor rad per joint rad)

QDD_EFFORT_NM = MOTOR_RATED_TORQUE_NM * TORQUE_RATIO          # ≈ 0.05 Nm at the joint
QDD_VEL_LIMIT_RAD_S = MOTOR_NOLOAD_RAD_S / TORQUE_RATIO       # ≈ 575 rad/s no-load (effort-limited in practice)
QDD_STIFFNESS = QDD_EFFORT_NM / 0.2          # saturates at ~0.2 rad (11 deg) error
QDD_DAMPING = 0.1 * QDD_STIFFNESS            # same damping/stiffness ratio as the servo fingers (~0.1 s)

_base = ORCA_V2_CFG.actuators["fingers"]
ALL_JOINTS = list(_base.joint_names_expr)
SERVO_JOINTS = [j for j in ALL_JOINTS if j not in QDD_JOINTS]
assert set(QDD_JOINTS) <= set(ALL_JOINTS), f"unknown QDD joints: {set(QDD_JOINTS) - set(ALL_JOINTS)}"


def _per_joint(values, joints):
    """Resolve ORCA's regex-keyed parameter dict to explicit per-joint values."""
    out = {}
    for j in joints:
        hits = [v for k, v in values.items() if re.fullmatch(k, j)]
        if len(hits) != 1:
            raise ValueError(f"{j}: {len(hits)} matching keys in {list(values)}")
        out[j] = hits[0]
    return out


def _qdd(value):
    return {j: value for j in QDD_JOINTS}


ORCA_V2_FAST_CFG = ORCA_V2_CFG.replace(
    actuators={
        # unchanged ORCA servo joints (Task 1 parameters; S1's fit flows in automatically)
        "servos": ImplicitActuatorCfg(
            joint_names_expr=SERVO_JOINTS,
            stiffness=_per_joint(_base.stiffness, SERVO_JOINTS),
            damping=_per_joint(_base.damping, SERVO_JOINTS),
            friction=_per_joint(_base.friction, SERVO_JOINTS),
            armature=_per_joint(_base.armature, SERVO_JOINTS),
            effort_limit_sim=_per_joint(_base.effort_limit_sim, SERVO_JOINTS),
        ),
        # QDD direct-drive joints – ESTIMATES until the FAST chirps are fitted
        "qdd": ImplicitActuatorCfg(
            joint_names_expr=list(QDD_JOINTS),
            stiffness=_qdd(QDD_STIFFNESS),
            damping=_qdd(QDD_DAMPING),
            friction=_qdd(0.0),            # cogging is compensated on the moteus
            armature=_qdd(0.0),
            effort_limit_sim=_qdd(QDD_EFFORT_NM),
            velocity_limit_sim=_qdd(QDD_VEL_LIMIT_RAD_S),
        ),
    }
)
"""ORCA v2 with the QDD joints of the FAST hand."""
