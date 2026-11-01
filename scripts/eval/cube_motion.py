"""How clean is the rotation? Cube tilt, top-face changes, off-axis spin and wrist motion from eval rollouts (.npz).

RPM only integrates the spin about the commanded axis, so a policy that tumbles the cube scores as fast as one that turns
it flat. This reports, over the alive steps of the nominal rollout:
  w_cmd / |w_xy|   mean spin about the commanded axis and mean off-axis spin [rad/s]
  tilt             1 - max_i |(R e_i) . z| as an angle: 0 deg while a face points up [deg, mean / p90]
  flips/min        changes of the face that points up (tumbling)
  wrist            range (p5-p95) [rad] and mean |qdot| [rad/s] of forearm -> carpals (the right_wrist joint)

  in region        share of alive steps whose cube centre (rotation axis) is inside --region CX CY R (metres, env frame)

Usage (CPU, numpy): python scripts/eval/cube_motion.py <npz or run dir> [...] [--region CX CY R] [--steady]
  --steady         adds drop-free after the spawn catch (share of envs without a drop / reset at t >= 2 s)
  --spread         adds the per-env speed distribution (each env = one draw of the randomised friction / mass, like one
                   hand trial): RPM p10 / median over envs, stall rate (envs below 5 RPM) and good-trial rate (envs with
                   no drop at t >= 2 s that turn at >= 8 RPM, ~60 % of the 1.5 rad/s command)
(run dir: its last eval/model_*.npz)
"""

import glob
import os
import sys

import numpy as np


def _qconj(q):
    return q * np.array([1, -1, -1, -1])


def _qmul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        -1,
    )


def _world_z_in_body(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)


def _resolve(path):
    if path.endswith(".npz"):
        return path
    files = [f for f in glob.glob(os.path.join(path, "eval", "model_*.npz")) if "perturbed" not in f and "_cmd" not in f]
    return sorted(files, key=lambda p: int(p.split("model_")[1].split(".")[0]))[-1]


def cube_motion(npz_path, region=None):
    d = np.load(npz_path, allow_pickle=True)
    dt = float(d["dt"])
    alive = ~(d["done"] > 0)
    w = d["obj_angvel"]
    sign = np.sign(d["cmd_angvel"][0, :, 2])
    w_cmd = w[..., 2] * sign
    w_xy = np.linalg.norm(w[..., :2], axis=-1)
    up = _world_z_in_body(d["obj_quat"])
    tilt = np.degrees(np.arccos(np.clip(np.abs(up).max(-1), 0, 1)))
    face = np.abs(up).argmax(-1)
    flips = ((face[1:] != face[:-1]) & alive[1:] & alive[:-1]).sum()
    names = list(d["link_names"])
    q = d["link_quat"]
    rel = _qmul(_qconj(q[:, :, names.index("right_forearm")]), q[:, :, names.index("right_carpals")])
    rel = rel * np.sign(rel[..., :1])
    drel = _qmul(_qconj(rel[:1]), rel)
    drel = drel * np.sign(drel[..., :1])
    ax = np.abs(drel[..., 1:]).reshape(-1, 3).sum(0).argmax()
    wrist = 2 * np.arctan2(drel[..., 1 + ax], drel[..., 0])
    out_region = {}
    if region is not None:
        cx, cy, r = region
        inside = np.hypot(d["obj_pos"][..., 0] - cx, d["obj_pos"][..., 1] - cy) <= r
        out_region = {"in_region": float(inside[alive].mean())}
    return out_region | {
        "w_cmd": float(w_cmd[alive].mean()),
        "w_xy": float(w_xy[alive].mean()),
        "tilt_mean": float(tilt[alive].mean()),
        "tilt_p90": float(np.percentile(tilt[alive], 90)),
        "flips_per_min": float(flips / (alive.sum() * dt / 60)),
        "wrist_range": float(np.percentile(wrist, 95) - np.percentile(wrist, 5)),
        "wrist_speed": float(np.abs(np.diff(wrist, axis=0) / dt).mean()),
        # share of envs without a drop / reset once the spawn catch is over (t >= 2 s): most eval drops happen 0.25-0.75 s
        # after the initial spawn (drop_analysis.md, 2026-11-01), which the hand, where the cube is placed by hand, never sees
        "drop_free_after_2s": float((~(d["done"][int(round(2.0 / dt)) :] > 0).any(0)).mean()),
        **_spread(d, w_cmd, alive, dt),
    }


def _spread(d, w_cmd, alive, dt):
    """Per-env RPM over the alive steps after the spawn catch (t >= 2 s): the mean over envs hides envs that nearly stall."""
    k = int(round(2.0 / dt))
    a = alive[k:]
    rpm = np.where(a, w_cmd[k:], 0.0).sum(0) / np.maximum(a.sum(0), 1) * 60 / (2 * np.pi)
    no_drop = ~(d["done"][k:] > 0).any(0)
    return {"rpm_env_p10": float(np.percentile(rpm, 10)), "rpm_env_median": float(np.median(rpm)),
            "stall_rate": float((rpm < 5.0).mean()), "good_trial_rate": float((no_drop & (rpm >= 8.0)).mean())}


if __name__ == "__main__":
    argv, region = sys.argv[1:], None
    steady = "--steady" in argv  # extra last column, opt-in so parsers of the default line keep working
    spread = "--spread" in argv
    argv = [a for a in argv if a not in ("--steady", "--spread")]
    if "--region" in argv:
        i = argv.index("--region")
        region = tuple(float(v) for v in argv[i + 1 : i + 4])
        argv = argv[:i] + argv[i + 4 :]
    print(f"{'rollout':60s} w_cmd |w_xy|  tilt mean/p90 flips/min  wrist range/speed" + ("  in region" if region else "") + ("  drop-free t>=2s" if steady else "")
          + ("  RPM p10 / median  stall  good-trial" if spread else ""))
    for arg in argv:
        path = _resolve(arg)
        m = cube_motion(path, region)
        name = os.path.relpath(path)[-60:]
        print(
            f"{name:60s} {m['w_cmd']:5.2f} {m['w_xy']:5.2f}  {m['tilt_mean']:5.1f} / {m['tilt_p90']:4.1f}"
            f"  {m['flips_per_min']:8.1f}  {m['wrist_range']:5.2f} / {m['wrist_speed']:4.2f}"
            + (f"  {100 * m['in_region']:9.1f} %" if region else "")
            + (f"  {100 * m['drop_free_after_2s']:14.1f} %" if steady else "")
            + (f"  {m['rpm_env_p10']:7.2f} / {m['rpm_env_median']:5.2f}  {100 * m['stall_rate']:4.1f} %  {100 * m['good_trial_rate']:8.1f} %"
               if spread else "")
        )
