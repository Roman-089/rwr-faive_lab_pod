"""Sim2sim eval: run an exported policy (TorchScript, as on the robot) on the MuJoCo ORCA v2 hand, score it like Isaac.

Same protocol and metrics as scripts/eval/eval_policy.py (docs/ablation/eval_protocol.md): |command| 1.5 rad/s about
world z, sign alternating over trials (+z only for "-onedir" tasks), 1200 control steps (60 s at 20 Hz) per trial, no
command resampling, a dropped cube (z < -0.2 m) re-spawns with Isaac's reset distribution and the drop is counted.
Isaac evaluates 64 envs in parallel; here trial i plays env i (even = +z, odd = -z).

The loop is the policy node's: obs (policy_io.PolicyIO) -> policy.forward once per 50 ms tick -> EMA targets ->
hand (hand.OrcaHandSim) for 6 physics steps. policy.reset() at every (re)spawn (deploy contract, docs/ablation/export.md).

Physics: --dr none (default) = centers of Isaac's DR ranges; --dr isaac = Isaac's DR terms sampled as in training
(startup terms once per trial, reset terms at spawn and again on a respawn >= 720 steps later). Obs noise as in
Isaac's eval (on) unless --no_obs_noise.

Usage (CPU, ~10 s per 60 s trial on one core):
    python scripts/sim2sim/run_sim2sim.py --policy logs/rsl_rl/orca_v2_repose/<run> [--trials 16] [--dr isaac]
        [--video out.mp4 --video_trials 2]
--policy takes an exported .pt or a run dir (-> <run>/exported/policy_history.pt, scripts/export/export_history_policy.py).
Output: <run>/sim2sim/metrics_<tag>.json (all settings + per-trial numbers) and rollout_<tag>.npz (tag: dr<none|isaac>...).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen video (Mesa EGL works headless on the login node); set before mujoco

import numpy as np
import torch

import orca_task as task
from hand import OrcaHandSim
from policy_io import PolicyIO, goal_command
from scene import SceneCfg


def metrics(w_z: np.ndarray, cmd_sign: np.ndarray, dropped: np.ndarray, dt: float) -> dict[str, float]:
    """w_z, dropped: (T, N); cmd_sign: (N,). Verbatim copy of scripts/eval/eval_policy.py metrics() (that file imports
    Isaac Sim at module level, so it cannot be imported here; the CPU test checks both are identical)."""
    signed = (w_z * cmd_sign).sum(0) * dt  # rad in the commanded direction, per env
    total = np.abs(w_z).sum(0) * dt
    minutes = w_z.shape[0] * dt / 60.0
    drops = dropped.sum(0)
    return {
        "rpm": float(np.mean(signed / (2 * np.pi) / minutes)),
        "rpm_std": float(np.std(signed / (2 * np.pi) / minutes)),
        "ra": float(100 * np.mean(np.clip(signed, 0, None) / np.maximum(total, 1e-6))),
        "dfsr": float(100 * np.mean(drops == 0)),
        "dc": float(np.mean(drops)),
    }


def load_policy(path: str) -> tuple[torch.jit.ScriptModule, dict, Path]:
    p = Path(path)
    if p.is_dir():
        p = p / "exported" / "policy_history.pt"
    extra = {"policy_history.json": ""}
    policy = torch.jit.load(str(p), map_location="cpu", _extra_files=extra)
    policy.eval()
    info = json.loads(extra["policy_history.json"]) if extra["policy_history.json"] else {}
    if info and (info.get("obs_dim") != 74 or info.get("action_dim") != 17):
        raise ValueError(f"{p}: obs/action dim {info.get('obs_dim')}/{info.get('action_dim')}, expected 74/17")
    a = info.get("action_term", {})
    cls = a.get("class_type", "").split(":")[-1]
    if a and (a.get("alpha") != task.EMA_ALPHA or not a.get("rescale_to_limits")
              or not cls.endswith("EMAJointPositionToLimitsAction")):
        raise ValueError(f"{p}: action term {a} is not the EMA alpha {task.EMA_ALPHA} term this harness implements")
    if a and cls != "EMAJointPositionToLimitsAction":  # -delay / -acthold subclasses: trained with extra sim effects
        print(f"[sim2sim] WARNING: trained with {cls}; the harness applies the plain EMA term (as the deploy node)")
    return policy, info, p


class Recorder:
    """Per-step log of one trial (control rate)."""

    keys = ("w", "cube_pos", "done", "joint_pos", "target", "action")

    def __init__(self):
        self.data = {k: [] for k in self.keys}

    def add(self, **kw):
        for k, v in kw.items():
            self.data[k].append(np.asarray(v, dtype=np.float32))

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.stack(v) for k, v in self.data.items()}


def run_trial(hand: OrcaHandSim, policy, trial: int, direction: float, args, frame_cb=None) -> dict[str, np.ndarray]:
    """One 60 s trial of one env. Returns per-step arrays (Recorder keys)."""
    jp = hand.joint_params
    rng = np.random.default_rng([args.seed, trial, 0])  # resets and DR
    io = PolicyIO(jp.lower, jp.upper, obs_noise=not args.no_obs_noise, rng=np.random.default_rng([args.seed, trial, 1]))
    phys = task.PhysicsParams()
    if args.dr == "isaac":
        task.sample_startup_dr(rng, phys, len(hand.links))
        task.sample_reset_dr(rng, phys)
    hand.set_physics(phys)
    last_dr = 0
    cmd = goal_command(direction, args.cmd)
    rec = Recorder()

    def spawn():
        state = task.sample_reset(rng, jp, hand_roll=args.dr == "isaac")
        hand.reset(state)
        io.reset(hand.joint_pos())
        policy.reset()

    spawn()
    for t in range(args.steps):
        obs = io.observation(hand.joint_pos(), hand.joint_vel(), cmd)
        with torch.inference_mode():
            action = policy(torch.from_numpy(obs)[None])[0].numpy()
        target = io.process_action(action)
        hand.set_joint_targets(target)
        for _ in range(task.DECIMATION):
            hand.step_physics()
        if frame_cb is not None:
            frame_cb(trial, t, cmd)
        dropped = hand.cube_pos()[2] < task.DROP_HEIGHT
        if dropped:
            if args.dr == "isaac" and t + 1 - last_dr >= task.DR_MIN_STEPS_BETWEEN:
                task.sample_reset_dr(rng, phys)
                hand.set_physics(phys)
                last_dr = t + 1
            spawn()
        # logged after the reset, as Isaac's eval logs the post-reset state of a re-spawned env
        rec.add(w=hand.cube_ang_vel_w(), cube_pos=hand.cube_pos(), done=float(dropped), joint_pos=hand.joint_pos(),
                target=target, action=action)
    return rec.arrays()


def make_video_writer(hand: OrcaHandSim, path: str, size: int):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw

    renderer = mujoco.Renderer(hand.model, size, size)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = task.COMMAND_POS
    cam.distance, cam.azimuth, cam.elevation = 0.32, 135.0, -30.0
    writer = imageio.get_writer(path, fps=round(1 / task.CONTROL_DT), codec="libx264", quality=8, macro_block_size=8)

    def frame(trial, t, cmd):
        renderer.update_scene(hand.data, cam)
        img = Image.fromarray(renderer.render())
        w = hand.cube_ang_vel_w()
        ImageDraw.Draw(img).text((8, 6), f"trial {trial}  t={t * task.CONTROL_DT:5.1f}s\ncmd  wz {cmd[5]:+.2f} rad/s\n"
                                 f"cube wz {w[2]:+.2f} rad/s", fill=(255, 255, 255))
        writer.append_data(np.asarray(img))

    def close():
        writer.close()
        renderer.close()

    return frame, close


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="exported TorchScript .pt or a run dir")
    p.add_argument("--trials", type=int, default=16)
    p.add_argument("--steps", type=int, default=1200, help="control steps per trial (1200 = 60 s at 20 Hz)")
    p.add_argument("--cmd", type=float, default=1.5, help="commanded |angular velocity| about z [rad/s]")
    p.add_argument("--onedir", action="store_true", help="+z only (default: from the policy's task id, '-onedir')")
    p.add_argument("--dr", choices=("none", "isaac"), default="none")
    p.add_argument("--no_obs_noise", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timestep_div", type=int, default=None, help="MuJoCo steps per Isaac physics step (default 10)")
    p.add_argument("--tag", default=None, help="output name suffix (default: from --dr / noise / cmd)")
    p.add_argument("--out_dir", default=None, help="default: <run>/sim2sim")
    p.add_argument("--video", default=None, help="write an MP4 of the first --video_trials trials (EGL / osmesa)")
    p.add_argument("--video_trials", type=int, default=1)
    p.add_argument("--video_size", type=int, default=480)
    args = p.parse_args()
    torch.set_num_threads(1)

    policy, info, policy_path = load_policy(args.policy)
    scene_cfg = SceneCfg() if args.timestep_div is None else SceneCfg(timestep=task.PHYSICS_DT / args.timestep_div)
    hand = OrcaHandSim(scene_cfg, render=args.video is not None)
    onedir = args.onedir or "-onedir" in info.get("task", "")
    signs = np.array([1.0 if (i % 2 == 0 or onedir) else -1.0 for i in range(args.trials)])
    tag = args.tag or "_".join(
        [f"dr{args.dr}"] + (["nonoise"] if args.no_obs_noise else []) + ([f"cmd{args.cmd:g}"] if args.cmd != 1.5 else [])
        + ([f"div{args.timestep_div}"] if args.timestep_div else [])
    )
    run_dir = policy_path.parent.parent if policy_path.parent.name == "exported" else policy_path.parent
    out_dir = Path(args.out_dir or run_dir / "sim2sim")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sim2sim] {policy_path} ({info.get('policy_class', '?')}, history {info.get('history_length', '?')}), "
          f"task {info.get('task', '?')}, {args.trials} trials x {args.steps} steps, dr {args.dr}, "
          f"obs noise {not args.no_obs_noise}, MuJoCo dt {hand.model.opt.timestep:.6g} s")

    frame_cb, close_video = (None, None)
    if args.video:
        frame_cb, close_video = make_video_writer(hand, args.video, args.video_size)
    logs, t0 = [], time.time()
    try:
        for i in range(args.trials):
            cb = frame_cb if (frame_cb is not None and i < args.video_trials) else None
            logs.append(run_trial(hand, policy, i, signs[i], args, cb))
            m = metrics(logs[-1]["w"][:, None, 2], signs[i : i + 1], logs[-1]["done"][:, None] > 0, task.CONTROL_DT)
            print(f"[sim2sim] trial {i:3d} cmd {signs[i]:+.0f}: rpm {m['rpm']:6.2f} ra {m['ra']:5.1f} drops "
                  f"{int(m['dc'])}  ({time.time() - t0:.0f} s)", flush=True)
    finally:
        if close_video is not None:
            close_video()

    w_z = np.stack([lg["w"][:, 2] for lg in logs], 1)
    done = np.stack([lg["done"] for lg in logs], 1) > 0
    res = metrics(w_z, signs, done, task.CONTROL_DT)
    for name, sel in (("plus", signs > 0), ("minus", signs < 0)):
        if sel.any():
            res[f"rpm_{name}"] = metrics(w_z[:, sel], signs[sel], done[:, sel], task.CONTROL_DT)["rpm"]
    held = np.stack([lg["cube_pos"][:, 2] for lg in logs], 1) > -0.15
    res["cube_z_mean"] = float(np.mean(np.stack([lg["cube_pos"][:, 2] for lg in logs], 1)[held]))
    res["off_axis_w_mean"] = float(np.mean(np.concatenate([np.linalg.norm(lg["w"][:, :2], axis=1) for lg in logs])))
    res.update(
        policy=str(policy_path), task=info.get("task"), checkpoint=info.get("checkpoint"), trials=args.trials,
        steps=args.steps, cmd=args.cmd, onedir=onedir, dr=args.dr, obs_noise=not args.no_obs_noise, seed=args.seed,
        mujoco_timestep=hand.model.opt.timestep, scene_cfg=vars(scene_cfg), wall_s=time.time() - t0,
        per_trial=[metrics(w_z[:, i : i + 1], signs[i : i + 1], done[:, i : i + 1], task.CONTROL_DT)
                   for i in range(args.trials)],
    )
    print(f"[sim2sim] {json.dumps({k: v for k, v in res.items() if k != 'per_trial' and k != 'scene_cfg'})}")
    (out_dir / f"metrics_{tag}.json").write_text(json.dumps(res, indent=1, default=str))
    np.savez_compressed(out_dir / f"rollout_{tag}.npz", cmd_sign=signs, dt=task.CONTROL_DT,
                        joint_names=np.array(task.JOINT_NAMES),
                        **{k: np.stack([lg[k] for lg in logs], 1) for k in Recorder.keys})
    print(f"[sim2sim] wrote {out_dir}/metrics_{tag}.json")


if __name__ == "__main__":
    main()
