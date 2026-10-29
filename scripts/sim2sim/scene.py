"""MuJoCo scene of the Isaac Lab cube-rotation task: ORCA v2 right hand + 45 mm cube, built as an MJCF string.

The hand is rebuilt from what Isaac Lab simulates rather than from the vendor MJCF (orcahand/orcahand_description v2),
see docs/ablation/sim2sim.md for why:
  - link tree, joint frames / axes / limits, masses and inertias, and the 53 convex collision hulls (exactly the
    64-vertex hulls PhysX collides) from the hand USD, via orca_v2_right_usd.json (extract_usd_model.py);
  - drive stiffness / damping, joint friction, armature and effort limits from orca_v2.py (orca_task.read_joint_params);
  - mount pose, cube, gravity on the cube only (hand links have disable_gravity=True in Isaac) from the env cfg.

Isaac's implicit PD drive (PhysX articulation drive, torque = k (target - q) - d qdot, clamped to the effort limit)
becomes a MuJoCo position actuator with kp = k, kv = d and forcerange = effort limit. The drives are heavily
overdamped (d / k = 70-140 ms) on finger links of 2e-6..7e-5 kg m^2, so the damping must be implicit: integrator
implicitfast treats kv implicitly (kp stays explicit, stable while dt * k / (d + dt k) < 2, true for any dt here).
But MuJoCo drops that implicit term while the actuator force is clamped at forcerange, and a clamped joint with no
armature then integrates F / I explicitly (I ~ 5e-6 kg m^2: the joint slams between its limits within one step).
PhysX keeps the drive implicit inside its clamp. The fix is an armature of ``armature_dt_factor * d * dt`` per joint
(on top of orca_v2.py's 0): one clamped step can then change the velocity by at most F / (2 d), less than the
damping band, so the clamp engages smoothly. It adds a 2 dt (1.7 ms) mechanical lag to a 70-140 ms first-order
response (joint check in docs/ablation/sim2sim.md).

Self-collisions: Isaac enables them (orca_v2.py) and PhysX skips pairs of links connected by a joint. The MJCF
excludes exactly those pairs explicitly: MuJoCo's parent filter alone misses the wrist (its parent, the tower, is
welded to the world, and filterparent does not apply to world-welded parents).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import orca_task as task

LINK_RGBA = {"carpals": "0.25 0.25 0.27 1", "skin": "0.92 0.90 0.86 1", "default": "0.35 0.35 0.38 1"}
CUBE_FACES = [  # (axis, sign, rgba): colored plates so the cube's rotation is visible (as scripts/replay)
    (0, 1, "0.90 0.20 0.20 1"), (0, -1, "0.20 0.75 0.85 1"),
    (1, 1, "0.25 0.75 0.30 1"), (1, -1, "0.80 0.30 0.80 1"),
    (2, 1, "0.20 0.35 0.90 1"), (2, -1, "0.95 0.80 0.15 1"),
]  # fmt: skip


@dataclass
class SceneCfg:
    """MuJoCo-specific choices (no Isaac counterpart); defaults justified in docs/ablation/sim2sim.md."""

    timestep: float = task.PHYSICS_DT / 10  # 1.2 kHz: 10 MuJoCo steps per Isaac physics step, 60 per control step
    contact_solref: tuple[float, float] = (0.004, 1.0)  # stiff, critically damped contacts (PhysX contacts are rigid)
    contact_solimp: tuple[float, float, float] = (0.95, 0.99, 0.001)
    limit_solref: tuple[float, float] = (0.004, 1.0)  # joint limits: PhysX limits are hard
    impratio: float = 10.0  # with the elliptic cone: friction close to hard (no creeping slip), as PhysX's
    cube_condim: int = 4  # sliding + torsional friction (PhysX torsional patch friction on the DexCube collider)
    cube_torsional_radius: float = 0.008  # m; DexCube's physxCollision:minTorsionalPatchRadius (torsional = mu * radius)
    armature_dt_factor: float = 2.0  # numerical armature = factor * damping * timestep (see module docstring)


def _fmt(a) -> str:
    return " ".join(f"{x:.9g}" for x in np.ravel(a))


def _quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def _mat_to_quat(r: np.ndarray) -> np.ndarray:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(r, dtype=np.float64).ravel())
    return q


def load_usd_model(path: Path = task.USD_JSON) -> dict:
    return json.loads(Path(path).read_text())


def child_pose_in_parent(joint: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pose of joint['child'] in joint['parent'] at q = 0: T(pos0, rot0) * T(pos1, rot1)^-1 (PhysX joint frames)."""
    r0, r1 = _quat_to_mat(joint["rot0"]), _quat_to_mat(joint["rot1"])
    r = r0 @ r1.T
    p = np.asarray(joint["pos0"]) - r @ np.asarray(joint["pos1"])
    return p, r


def joint_axis_in_child(joint: dict) -> np.ndarray:
    """Revolute axis in the child link frame (USD axis token is in the joint frame rot1)."""
    e = {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}[joint["axis"]]
    return _quat_to_mat(joint["rot1"]) @ np.array(e)


def build_mjcf(cfg: SceneCfg | None = None, usd: dict | None = None, jp: task.JointParams | None = None,
               render: bool = False) -> str:
    """MJCF of hand + cube. Bodies are named like the USD links, joints / actuators like JOINT_NAMES."""
    cfg = cfg or SceneCfg()
    usd = usd or load_usd_model()
    jp = jp or task.read_joint_params()
    by_child = {j["child"]: j for j in usd["joints"]}
    children: dict[str, list[str]] = {}
    for j in usd["joints"]:
        if j["parent"] is not None:
            children.setdefault(j["parent"], []).append(j["child"])
    roots = [j["child"] for j in usd["joints"] if j["parent"] is None]
    assert roots == ["right_forearm"], roots
    ji = {n: i for i, n in enumerate(task.JOINT_NAMES)}

    assets = []
    for link, hulls in usd["collision_hulls"].items():
        for k, h in enumerate(hulls):
            assets.append(f'<mesh name="{link}_{k}" vertex="{_fmt(h["vertices"])}"/>')

    def body_xml(link: str, depth: int) -> str:
        ind = "  " * depth
        lk = usd["links"][link]
        jt = by_child[link]
        if jt["parent"] is None:  # root: fixed to the world at the hand mount pose
            pos, quat = np.array(task.HAND_POS), np.array(task.HAND_QUAT)
        else:
            p, r = child_pose_in_parent(jt)
            pos, quat = p, _mat_to_quat(r)
        lines = [f'{ind}<body name="{link}" pos="{_fmt(pos)}" quat="{_fmt(quat)}" gravcomp="1">']
        lines.append(
            f'{ind}  <inertial pos="{_fmt(lk["com"])}" quat="{_fmt(lk["principal_axes"])}" mass="{lk["mass"]:.9g}"'
            f' diaginertia="{_fmt(lk["diag_inertia"])}"/>'
        )
        if jt["type"] == "revolute":
            i = ji[jt["name"]]
            armature = jp.armature[i] + cfg.armature_dt_factor * jp.damping[i] * cfg.timestep
            lines.append(
                f'{ind}  <joint name="{jt["name"]}" pos="{_fmt(jt["pos1"])}" axis="{_fmt(joint_axis_in_child(jt))}"'
                f' range="{jp.lower[i]:.9g} {jp.upper[i]:.9g}" armature="{armature:.9g}"'
                f' frictionloss="{jp.friction[i]:.9g}"/>'
            )
        rgba = LINK_RGBA["carpals"] if link == "right_carpals" else LINK_RGBA["default"]
        for k, h in enumerate(usd["collision_hulls"].get(link, [])):
            col = LINK_RGBA["skin"] if "Skin" in h["name"] else rgba
            lines.append(f'{ind}  <geom name="{link}_{h["name"]}_{k}" class="hand" mesh="{link}_{k}" rgba="{col}"/>')
        for c in sorted(children.get(link, [])):
            lines.append(body_xml(c, depth + 1))
        lines.append(f"{ind}</body>")
        return "\n".join(lines)

    actuators = []
    for i, name in enumerate(task.JOINT_NAMES):
        actuators.append(
            f'<position name="{name}" joint="{name}" kp="{jp.stiffness[i]:.9g}" kv="{jp.damping[i]:.9g}"'
            f' ctrlrange="{jp.lower[i]:.9g} {jp.upper[i]:.9g}" forcerange="{-jp.effort_limit[i]:.9g} {jp.effort_limit[i]:.9g}"/>'
        )

    # PhysX never collides two links joined by a joint. MuJoCo's parent filter misses the wrist (its parent, the tower,
    # is welded to the world, and filterparent does not apply to world-welded parents), so list every joined pair.
    excludes = [
        f'<exclude body1="{j["parent"]}" body2="{j["child"]}"/>' for j in usd["joints"] if j["parent"] is not None
    ]

    s = task.CUBE_SIZE / 2
    cube_inertia = task.CUBE_MASS * task.CUBE_SIZE**2 / 6
    plates = []
    for axis, sign, rgba in CUBE_FACES:
        pos, size = np.zeros(3), np.full(3, 0.75 * s)
        pos[axis], size[axis] = sign * (s + 0.0002), 0.0002
        plates.append(f'<geom type="box" class="visual" pos="{_fmt(pos)}" size="{_fmt(size)}" rgba="{rgba}"/>')
    mu = 0.5 * (task.NOMINAL_HAND_FRICTION + task.NOMINAL_CUBE_FRICTION)
    visual = (
        '<visual><global offwidth="1280" offheight="1280"/><quality shadowsize="4096"/>'
        '<headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/></visual>'
        if render
        else ""
    )
    floor = (
        '<geom name="floor" type="plane" size="0.5 0.5 0.01" pos="0 0 -0.25" class="visual" rgba="0.3 0.32 0.35 1"/>'
        '<light pos="0.3 -0.3 1.0" dir="-0.3 0.3 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>'
        if render
        else ""
    )
    return f"""<mujoco model="orca_v2_cube_rotation_sim2sim">
  <compiler angle="radian" autolimits="true" inertiafromgeom="false"/>
  <option timestep="{cfg.timestep:.9g}" gravity="{_fmt(task.GRAVITY)}" integrator="implicitfast" cone="elliptic"
          impratio="{cfg.impratio:g}"/>
  {visual}
  <default>
    <geom solref="{_fmt(cfg.contact_solref)}" solimp="{_fmt(cfg.contact_solimp)}"/>
    <joint type="hinge" limited="true" solreflimit="{_fmt(cfg.limit_solref)}" damping="0"/>
    <position ctrllimited="true" forcelimited="true"/>
    <default class="hand"><geom type="mesh" condim="3" friction="{task.NOMINAL_HAND_FRICTION:.6g} 0 0"/></default>
    <default class="visual"><geom contype="0" conaffinity="0" group="2" mass="0"/></default>
  </default>
  <asset>
    {chr(10).join(assets)}
  </asset>
  <worldbody>
    {floor}
{body_xml("right_forearm", 2)}
    <body name="cube" pos="{_fmt(task.CUBE_DEFAULT_POS)}">
      <freejoint name="cube"/>
      <inertial pos="0 0 0" mass="{task.CUBE_MASS:.9g}" diaginertia="{_fmt([cube_inertia] * 3)}"/>
      <geom name="cube" type="box" size="{s} {s} {s}" condim="{cfg.cube_condim}" priority="1"
            friction="{mu:.6g} {mu * cfg.cube_torsional_radius:.6g} 0" rgba="0.95 0.95 0.95 1"/>
      {"".join(plates)}
    </body>
  </worldbody>
  <contact>
    {chr(10).join(excludes)}
  </contact>
  <actuator>
    {chr(10).join(actuators)}
  </actuator>
</mujoco>
"""


if __name__ == "__main__":
    import argparse

    import mujoco

    ap = argparse.ArgumentParser(description="Write the sim2sim MJCF (e.g. to inspect it with python -m mujoco.viewer)")
    ap.add_argument("--out", default="orca_v2_sim2sim.xml")
    a = ap.parse_args()
    xml = build_mjcf(render=True)
    m = mujoco.MjModel.from_xml_string(xml)
    Path(a.out).write_text(xml)
    print(f"wrote {a.out}: {m.nbody} bodies, {m.njnt} joints, {m.ngeom} geoms, {m.nu} actuators, dt {m.opt.timestep}")
