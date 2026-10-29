"""Consistency checks of the MuJoCo hand against Isaac Lab (CPU). Mismatches here would look like policy failures.

    python scripts/sim2sim/consistency_checks.py joints        # chirp per joint: MuJoCo vs Isaac's implicit PD
    python scripts/sim2sim/consistency_checks.py kinematics --rollout <run>/eval/model_X.npz [--vendor DIR]

joints: the chirp of scripts/motor_param_characterization/chirp_characterization.py (20 s, 0.1-5 Hz logarithmic,
+-10 deg about the middle of the range, 20 Hz commands, joint read before each command acts, 10 s settling) on every
joint of the full MuJoCo hand (other joints held at 0, no cube), against the 1-DoF replica of PhysX's implicit drive
(scripts/actuator_net/servo_model.py pd_step: implicit Euler of I qdd = k (target - q) - d qd - friction) with the
joint-space inertia of the same USD mass properties. Two references bracket PhysX's TGS solver: one implicit step per
1/120 s physics step, and 8 sub-steps. Reports position RMSE and FRF gain / phase differences
(scripts/motor_param_characterization/signal_processing.py).

kinematics: Isaac eval rollouts store every link's world pose (not the joint angles). The joint angles are recovered
from the parent / child link rotations, MuJoCo is put in that configuration with the hand root at Isaac's root pose,
and all 19 link poses are compared. The sign of the recovered angles is checked against the joint limits (a flipped
axis would push the angles out of the asymmetric ranges). With --vendor (an orcahand_description checkout), the vendor
v2 MJCF's kinematics are compared too: for random joint angles, every link pose relative to the forearm must differ
from ours by one constant transform per link (the frames differ by mesh-origin offsets, the joints must not).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import mujoco

import orca_task as task
from hand import OrcaHandSim
from scene import SceneCfg, child_pose_in_parent, joint_axis_in_child, load_usd_model

sys.path.insert(0, str(task.REPO_ROOT / "scripts/actuator_net"))
sys.path.insert(0, str(task.REPO_ROOT / "scripts/motor_param_characterization"))
from servo_model import pd_step  # noqa: E402
from signal_processing import compute_frequency_response, prepare_chirp_signals  # noqa: E402

CHIRP = {"duration": 20.0, "f0": 0.1, "f1": 5.0, "amplitude_deg": 10.0, "settle_s": 10.0}
REPORT_FREQS = (0.5, 1.0, 2.0, 4.0)


# --- joints ---------------------------------------------------------------------------------------------------------


def chirp_command() -> np.ndarray:
    from scipy import signal

    t = np.arange(0, CHIRP["duration"], task.CONTROL_DT)
    return np.deg2rad(CHIRP["amplitude_deg"]) * signal.chirp(t, CHIRP["f0"], CHIRP["duration"], CHIRP["f1"],
                                                             method="logarithmic")


def mujoco_chirp(hand: OrcaHandSim, j: int, cmd: np.ndarray) -> np.ndarray:
    """Measured joint j (rad) at each 20 Hz command, read before that command acts (as chirp_characterization.py)."""
    m = hand.model
    m.geom_contype[hand._cube_geom] = m.geom_conaffinity[hand._cube_geom] = 0
    state = task.sample_reset(np.random.default_rng(0), hand.joint_params, hand_roll=False)
    state.joint_pos[:] = 0.0
    state.cube_pos = np.array([0.0, 0.0, -1.0])
    hand.reset(state)
    center = 0.5 * (hand.joint_params.lower[j] + hand.joint_params.upper[j])
    target = np.zeros(len(task.JOINT_NAMES))
    target[j] = center + cmd[0]
    hand.set_joint_targets(target)
    for _ in range(int(CHIRP["settle_s"] / task.PHYSICS_DT)):
        hand.step_physics()
    meas = np.zeros_like(cmd)
    for k, c in enumerate(cmd):
        target[j] = center + c
        hand.set_joint_targets(target)
        meas[k] = hand.joint_pos()[j]
        for _ in range(task.DECIMATION):
            hand.step_physics()
    return meas - center


def replica_chirp(k: float, d: float, inertia: float, friction: float, cmd: np.ndarray, substeps: int) -> np.ndarray:
    dt = task.PHYSICS_DT / substeps
    q = v = torch.tensor(float(cmd[0]), dtype=torch.float64)
    v = torch.zeros((), dtype=torch.float64)
    f = lambda x: torch.tensor(x, dtype=torch.float64)  # noqa: E731
    for _ in range(int(CHIRP["settle_s"] / dt)):
        q, v = pd_step(q, v, f(cmd[0]), f(k), f(d), f(inertia), f(friction), dt)
    meas = np.zeros_like(cmd)
    for i, c in enumerate(cmd):
        meas[i] = q.item()
        for _ in range(task.DECIMATION * substeps):
            q, v = pd_step(q, v, f(c), f(k), f(d), f(inertia), f(friction), dt)
    return meas


def frf(cmd: np.ndarray, meas: np.ndarray):
    t = np.arange(len(cmd)) * task.CONTROL_DT
    return compute_frequency_response(prepare_chirp_signals(time=t, commanded=np.rad2deg(cmd), measured=np.rad2deg(meas)))


def at_freqs(fr, freqs=REPORT_FREQS) -> tuple[list[float], list[float]]:
    """Gain (dB) and phase (deg) averaged over the FFT bins within +-10 % of each frequency."""
    g, p = [], []
    for f in freqs:
        sel = np.abs(fr.frequency - f) <= 0.1 * f
        h = np.mean(fr.response[sel])
        g.append(float(20 * np.log10(np.abs(h))))
        p.append(float(np.rad2deg(np.angle(h))))
    return g, p


def check_joints(args) -> dict:
    hand = OrcaHandSim(SceneCfg() if args.timestep_div is None else SceneCfg(timestep=task.PHYSICS_DT / args.timestep_div))
    jp = hand.joint_params
    # joint-space inertia at the default pose from the same USD mass properties, without the numerical armature
    d = mujoco.MjData(hand.model)
    mujoco.mj_forward(hand.model, d)
    qm = np.zeros((hand.model.nv, hand.model.nv))
    mujoco.mj_fullM(hand.model, d, qm)
    inertia = np.diag(qm)[hand._vadr] - hand.model.dof_armature[hand._vadr]
    cmd = chirp_command()
    out = {}
    print(f"{'joint':18s} {'I [kg m2]':>9s} | RMSE vs PhysX 1 / 8 substeps [deg] | gain dB / phase deg at "
          f"{REPORT_FREQS} Hz: MuJoCo | PhysX-1 | PhysX-8")
    for j, name in enumerate(task.JOINT_NAMES):
        if args.joints and name.replace("right_", "") not in args.joints:
            continue
        mj = mujoco_chirp(hand, j, cmd)
        refs = {s: replica_chirp(jp.stiffness[j], jp.damping[j], inertia[j], jp.friction[j], cmd, s) for s in (1, 8)}
        g_mj, p_mj = at_freqs(frf(cmd, mj))
        row = {"inertia": float(inertia[j]), "mujoco": {"gain_db": g_mj, "phase_deg": p_mj}}
        for s, ref in refs.items():
            g, p = at_freqs(frf(cmd, ref))
            row[f"physx_{s}"] = {"rmse_deg": float(np.rad2deg(np.sqrt(np.mean((mj - ref) ** 2)))),
                                 "max_err_deg": float(np.rad2deg(np.max(np.abs(mj - ref)))),
                                 "gain_db": g, "phase_deg": p}
        out[name] = row
        fmt = lambda g, p: " ".join(f"{a:+5.1f}/{b:+5.0f}" for a, b in zip(g, p))  # noqa: E731
        print(f"{name:18s} {inertia[j]:9.2e} | {row['physx_1']['rmse_deg']:5.2f} / {row['physx_8']['rmse_deg']:5.2f}"
              f" | {fmt(g_mj, p_mj)} | {fmt(row['physx_1']['gain_db'], row['physx_1']['phase_deg'])}"
              f" | {fmt(row['physx_8']['gain_db'], row['physx_8']['phase_deg'])}")
    return out


# --- kinematics -----------------------------------------------------------------------------------------------------


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(..., 4) w x y z -> (..., 3, 3), normalizing (eval rollouts store float32)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)  # fmt: skip


def rot_angle(r: np.ndarray) -> np.ndarray:
    """Rotation angle (rad) of (..., 3, 3), stable near 0."""
    vee = np.stack([r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]], -1)
    return np.arctan2(0.5 * np.linalg.norm(vee, axis=-1), 0.5 * (np.trace(r, axis1=-2, axis2=-1) - 1))


def joint_angles_from_links(usd: dict, link_names, link_quat: np.ndarray) -> np.ndarray:
    """(..., L, 4) link world quats (Isaac link order) -> (..., 17) joint angles (JOINT_NAMES order)."""
    names = list(link_names)
    r = quat_to_mat(link_quat.astype(np.float64))
    q = {}
    for j in usd["joints"]:
        if j["type"] != "revolute":
            continue
        rp, rc = r[..., names.index(j["parent"]), :, :], r[..., names.index(j["child"]), :, :]
        r0 = child_pose_in_parent(j)[1]
        m = np.einsum("ji,...jk->...ik", r0, np.einsum("...ji,...jk->...ik", rp, rc))  # R0^T Rp^T Rc = Rot(axis, q)
        a = joint_axis_in_child(j)
        s = 0.5 * (a[0] * (m[..., 2, 1] - m[..., 1, 2]) + a[1] * (m[..., 0, 2] - m[..., 2, 0])
                   + a[2] * (m[..., 1, 0] - m[..., 0, 1]))
        q[j["name"]] = np.arctan2(s, 0.5 * (np.trace(m, axis1=-2, axis2=-1) - 1))
    return np.stack([q[n] for n in task.JOINT_NAMES], -1)


def check_isaac_rollout(args) -> dict:
    hand = OrcaHandSim()
    m, d = hand.model, mujoco.MjData(hand.model)
    z = np.load(args.rollout)
    names = list(z["link_names"])
    pos, quat = z["link_pos"][:: args.stride], z["link_quat"][:: args.stride]  # (T, E, L, ...)
    q = joint_angles_from_links(hand.usd, names, quat)  # (T, E, 17)
    bodies = [m.body(n).id for n in names]
    root = m.body("right_forearm").id
    perr, rerr = [], []
    for t in range(q.shape[0]):
        for e in range(q.shape[1]):
            m.body_pos[root], m.body_quat[root] = pos[t, e, 0], quat[t, e, 0] / np.linalg.norm(quat[t, e, 0])
            d.qpos[hand._qadr] = q[t, e]
            mujoco.mj_kinematics(m, d)
            perr.append(np.linalg.norm(d.xpos[bodies] - pos[t, e], axis=-1))
            rerr.append(rot_angle(np.einsum("lji,ljk->lik", d.xmat[bodies].reshape(-1, 3, 3), quat_to_mat(quat[t, e]))))
    perr, rerr = np.array(perr), np.array(rerr)
    jp = hand.joint_params
    tol = np.deg2rad(3.0)  # Isaac randomizes the limits by N(0, 0.01 rad) and its limits are slightly soft

    def inside(qq):
        return float(np.mean((qq >= jp.lower - tol) & (qq <= jp.upper + tol)))

    res = {
        "rollout": str(args.rollout), "configurations": int(q.shape[0] * q.shape[1]),
        "link_pos_err_um": {"max": float(perr.max() * 1e6), "mean": float(perr.mean() * 1e6)},
        "link_rot_err_deg": {"max": float(np.rad2deg(rerr.max())), "mean": float(np.rad2deg(rerr.mean()))},
        "within_limits_assumed_sign": inside(q), "within_limits_flipped_sign": inside(-q),
        "q_range_deg": {n: [float(np.rad2deg(q[..., i].min())), float(np.rad2deg(q[..., i].max()))]
                        for i, n in enumerate(task.JOINT_NAMES)},
    }
    print(f"[kinematics] Isaac rollout {args.rollout}: {res['configurations']} configurations, link position error "
          f"max {res['link_pos_err_um']['max']:.2f} um (mean {res['link_pos_err_um']['mean']:.2f}), rotation error max "
          f"{res['link_rot_err_deg']['max']:.4f} deg; recovered angles within limits (+-3 deg): "
          f"{100 * res['within_limits_assumed_sign']:.1f} % with our joint sign, "
          f"{100 * res['within_limits_flipped_sign']:.1f} % if every axis were flipped")
    return res


VENDOR_THUMB = {"cmc": "cmc", "abd": "abd", "mcp": "mcp", "dip": "pip"}  # vendor names the thumb's last joint pip


def vendor_joint_name(ours: str) -> str:
    """right_index_abd -> right_i-abd, right_thumb_dip -> right_t-pip (orcahand_description v2 names)."""
    if ours == "right_wrist":
        return ours
    finger, kind = ours.replace("right_", "").split("_")
    return f"right_t-{VENDOR_THUMB[kind]}" if finger == "thumb" else f"right_{finger[0]}-{kind}"


def check_vendor(args) -> dict:
    """Vendor v2 MJCF vs ours: per link, T_ours^-1 T_vendor (relative to each model's forearm) must be constant."""
    hand = OrcaHandSim()
    usd = hand.usd
    vm = mujoco.MjModel.from_xml_path(str(Path(args.vendor) / "v2/scene_right.xml"))
    vd, d = mujoco.MjData(vm), mujoco.MjData(hand.model)
    vq = np.array([vm.jnt_qposadr[vm.joint(vendor_joint_name(n)).id] for n in task.JOINT_NAMES])
    link_of_joint = {j["name"]: j["child"] for j in usd["joints"]}
    pairs = {"right_forearm": vm.body("right_ForeArmStructure-Model_e18f2368").id}
    pairs["right_tower"] = vm.body("right_TopTower-Model_4a80d30e").id
    for n in task.JOINT_NAMES:
        pairs[link_of_joint[n]] = vm.jnt_bodyid[vm.joint(vendor_joint_name(n)).id]
    ours = {k: hand.model.body(k).id for k in pairs}
    rng = np.random.default_rng(0)
    jp = hand.joint_params
    rel = []
    for k in range(args.vendor_samples):
        qq = np.zeros(17) if k == 0 else rng.uniform(jp.lower, jp.upper)
        d.qpos[hand._qadr] = qq
        vd.qpos[vq] = qq
        mujoco.mj_kinematics(hand.model, d)
        mujoco.mj_kinematics(vm, vd)
        row = []
        for link, vb in pairs.items():
            def in_forearm(model_d, b, root):
                r0, p0 = model_d.xmat[root].reshape(3, 3), model_d.xpos[root]
                return r0.T @ model_d.xmat[b].reshape(3, 3), r0.T @ (model_d.xpos[b] - p0)
            ro, po = in_forearm(d, ours[link], ours["right_forearm"])
            rv, pv = in_forearm(vd, vb, pairs["right_forearm"])
            row.append((ro.T @ rv, ro.T @ (pv - po)))  # vendor body frame expressed in our link frame
        rel.append(row)
    pos_dev, rot_dev, offsets = [], [], {}
    for i, link in enumerate(pairs):
        r_ref, p_ref = rel[0][i]
        offsets[link] = {"pos_mm": (p_ref * 1e3).round(3).tolist(), "rot_deg": float(np.rad2deg(rot_angle(r_ref)))}
        for row in rel[1:]:
            pos_dev.append(np.linalg.norm(row[i][1] - p_ref))
            rot_dev.append(rot_angle(r_ref.T @ row[i][0]))
    res = {
        "vendor": str(args.vendor), "samples": args.vendor_samples,
        "max_pos_dev_um": float(np.max(pos_dev) * 1e6), "max_rot_dev_deg": float(np.rad2deg(np.max(rot_dev))),
        "frame_offsets_at_q0": offsets,
        "vendor_masses_kg": {k: float(vm.body_mass[v]) for k, v in pairs.items()},
        "usd_masses_kg": {k: usd["links"][k]["mass"] for k in pairs},
    }
    print(f"[kinematics] vendor MJCF ({args.vendor_samples} random configurations): link pose relative to ours varies by "
          f"at most {res['max_pos_dev_um']:.2f} um / {res['max_rot_dev_deg']:.4f} deg (0 = identical kinematics)")
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="check", required=True)
    pj = sub.add_parser("joints")
    pj.add_argument("--joints", nargs="*", default=None, help="e.g. index_pip wrist (default: all 17)")
    pj.add_argument("--timestep_div", type=int, default=None)
    pk = sub.add_parser("kinematics")
    pk.add_argument("--rollout", required=True, help="Isaac eval rollout .npz (scripts/eval/eval_policy.py)")
    pk.add_argument("--stride", type=int, default=5, help="use every n-th control step")
    pk.add_argument("--vendor", default=None, help="orcahand_description checkout (needs v2/ incl. meshes)")
    pk.add_argument("--vendor_samples", type=int, default=50)
    for q in (pj, pk):
        q.add_argument("--out", default=None, help="write the results as JSON")
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.check == "joints":
        res = check_joints(args)
    else:
        res = {"isaac": check_isaac_rollout(args)}
        if args.vendor:
            res["vendor"] = check_vendor(args)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
