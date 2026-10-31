"""Evaluate checkpoints with the ORCA PT paper's metrics (arXiv 2605.21330) and record rollouts for videos.

One Isaac Sim start for several checkpoints of one run. Fixed eval setting (docs/ablation/eval_protocol.md): the
task's own DR and observation noise, |command| = --cmd rad/s about z with the sign alternating over envs (+z only
for the one-direction "-onedir" tasks), no command
resampling, --steps control steps (1200 = 60 s at 20 Hz) without time-out; a dropped cube re-spawns and the
drop is counted.

--perturb adds sim2real effects at test time only (the policy was not trained with them): joint targets 50-100 ms
late (6-12 physics substeps) and a per-episode joint-position obs offset U(+-0.05 rad). It measures robustness to
unmodeled effects, e.g. whether a history-based policy degrades less than a single-observation MLP.

Metrics per checkpoint (mean over envs):
    rpm   cube rotation in the commanded direction per minute (integral of sign(cmd) * w_z)
    ra    rotation accuracy [%]: commanded-direction rotation / total absolute rotation
    dfsr  drop-free success rate [%]: envs without any drop in the trial
    dc    drop count per trial

Usage:
    python scripts/eval/eval_policy.py --task orca-v2-inhand-repose-v0 --headless \
        --checkpoints logs/rsl_rl/orca_v2_repose/<run>/model_1000.pt .../model_4999.pt
Output: <run>/eval/<ckpt>.npz (first --video_envs envs, for scripts/replay/render_rollout.py) and
<run>/eval/metrics.json (all checkpoints evaluated so far); with --perturb: *_perturbed.npz, metrics_perturbed.json;
with --cmd other than 1.5: *_cmd<cmd>.npz, metrics_cmd<cmd>.json.
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "replay"))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate checkpoints (RPM / RA / DFSR / DC) and record rollouts.")
parser.add_argument("--task", type=str, default="orca-v2-inhand-repose-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1200, help="control steps per trial (1200 = 60 s at 20 Hz)")
parser.add_argument("--cmd", type=float, default=1.5, help="commanded |angular velocity| about z [rad/s]")
parser.add_argument("--video_envs", type=int, default=4, help="envs whose link poses are saved for the video")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--perturb", action="store_true", help="test-time action delay + joint obs offset")
parser.add_argument("--servo", action="store_true",
                    help="real ORCA v2 servo model with its training DR (17-117 ms delay, per-joint acceleration limit, "
                         "play, slope; adr.use_orca_v2_servo_model); outputs *_servo")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import faive_lab.tasks  # noqa: F401
from faive_lab.tasks.orca_inhand import adr as orca_adr
from faive_lab.tasks.orca_inhand.mdp.sim2real import (
    DelayedEMAJointPositionToLimitsActionCfg,
    joint_pos_limit_normalized_biased,
)

from usd_meshes import export_meshes, object_size  # isort: skip


def metrics(w_z: np.ndarray, cmd_sign: np.ndarray, dropped: np.ndarray, dt: float) -> dict[str, float]:
    """w_z, dropped: (T, N); cmd_sign: (N,)."""
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


def apply_run_params(env_cfg, agent_cfg, ckpt: str):
    """Rebuild what the run was trained with, from its saved params/: training-time hydra overrides (network sizes,
    action-term settings) are not part of the task id, so without this a run trained with e.g.
    agent.policy.actor_hidden_dims=[1024,...] fails to load and env.actions.joint_pos.alpha=0.5 is evaluated at 0.2."""
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_multi_constructor("", lambda loader, suffix, node: None)  # ignore python/* tags
    params = os.path.join(os.path.dirname(os.path.abspath(ckpt)), "params")
    changed = []
    if os.path.exists(os.path.join(params, "agent.yaml")):
        policy = (yaml.load(open(os.path.join(params, "agent.yaml")), Loader=_Loader) or {}).get("policy") or {}
        for key, value in policy.items():
            if key.endswith("hidden_dims") and hasattr(agent_cfg.policy, key) and getattr(agent_cfg.policy, key) != value:
                setattr(agent_cfg.policy, key, value)
                changed.append(f"agent.policy.{key}={value}")
    if os.path.exists(os.path.join(params, "env.yaml")):
        action = ((yaml.load(open(os.path.join(params, "env.yaml")), Loader=_Loader) or {}).get("actions") or {}).get("joint_pos") or {}
        for key in ("alpha", "delta_scale"):
            if key in action and hasattr(env_cfg.actions.joint_pos, key) and getattr(env_cfg.actions.joint_pos, key) != action[key]:
                setattr(env_cfg.actions.joint_pos, key, action[key])
                changed.append(f"env.actions.joint_pos.{key}={action[key]}")
    if changed:
        print(f"[eval] from the run's params: {', '.join(changed)}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    apply_run_params(env_cfg, agent_cfg, args_cli.checkpoints[0])
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # one trial = one episode: no time-out inside the trial, no command resampling
    env_cfg.episode_length_s = (args_cli.steps + 10) * env_cfg.sim.dt * env_cfg.decimation
    env_cfg.commands.continuous_rotation.resampling_time_range = (1e6, 1e6)
    # non-default command speeds get their own outputs (speed-tracking check)
    suffix = "" if args_cli.cmd == 1.5 else f"_cmd{args_cli.cmd:g}"
    if args_cli.perturb:
        suffix += "_perturbed"
        a = env_cfg.actions.joint_pos
        env_cfg.actions.joint_pos = DelayedEMAJointPositionToLimitsActionCfg(
            asset_name=a.asset_name, joint_names=a.joint_names, alpha=a.alpha,
            rescale_to_limits=a.rescale_to_limits, delay_substeps_range=(6, 12),
        )
        env_cfg.observations.policy.joint_pos.func = joint_pos_limit_normalized_biased
        env_cfg.observations.policy.joint_pos.params = {"bias_rad": 0.05}
    if args_cli.servo:
        if args_cli.perturb:
            raise ValueError("--servo and --perturb are exclusive")
        suffix += "_servo"
        orca_adr.use_orca_v2_servo_model(env_cfg, delay_substeps_range=(2, 14), adr=False)

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    base = env.unwrapped
    robot, obj = base.scene["robot"], base.scene["object"]
    origins = base.scene.env_origins
    cmd_term = base.command_manager.get_term("continuous_rotation")
    n, nv = args_cli.num_envs, min(args_cli.video_envs, args_cli.num_envs)
    cmd_sign = torch.where(torch.arange(n, device=base.device) % 2 == 0, 1.0, -1.0)
    if "-onedir" in args_cli.task:  # trained on +z only (orca_v2_repose_ablation_env_cfg.OrcaV2ReposeAsymOneDirEnvCfg)
        cmd_sign = torch.ones_like(cmd_sign)

    meshes = export_meshes(robot.root_physx_view.link_paths[0])
    try:
        size = np.float32(object_size(obj.root_physx_view.prim_paths[0]))
    except Exception as e:  # noqa: BLE001
        print(f"[eval] could not read object size ({e}), using 45 mm")
        size = np.float32(0.045)

    runner_class = DistillationRunner if agent_cfg.class_name == "DistillationRunner" else OnPolicyRunner
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    results_path = None
    for ckpt in args_cli.checkpoints:
        ckpt = retrieve_file_path(ckpt)
        runner.load(ckpt)
        policy = runner.get_inference_policy(device=base.device)
        policy_nn = runner.alg.policy  # recurrent policies keep a hidden state per env
        # env buffers written during a rollout are inference tensors: reset + rollout must both run in inference mode
        with torch.inference_mode():
            torch.manual_seed(args_cli.seed)
            env.reset()
            policy_nn.reset()
            cmd_term.angular_vel_command[:] = 0.0
            cmd_term.angular_vel_command[:, 2] = cmd_sign * args_cli.cmd
            obs = env.get_observations()

            log = {k: [] for k in ("link_pos", "link_quat", "obj_pos", "obj_quat", "obj_angvel", "cmd_angvel",
                                   "cmd_pos", "reward", "done")}
            for step in range(args_cli.steps):
                obs, rew, dones, _ = env.step(policy(obs))
                policy_nn.reset(dones)
                # re-spawned envs resample their command: pin it again (takes effect from the next obs on)
                cmd_term.angular_vel_command[:, :2] = 0.0
                cmd_term.angular_vel_command[:, 2] = cmd_sign * args_cli.cmd
                log["link_pos"].append((robot.data.body_pos_w[:nv] - origins[:nv, None]).cpu().numpy())
                log["link_quat"].append(robot.data.body_quat_w[:nv].cpu().numpy())  # wxyz
                log["obj_pos"].append((obj.data.root_pos_w - origins).cpu().numpy())
                log["obj_quat"].append(obj.data.root_quat_w.cpu().numpy())
                log["obj_angvel"].append(obj.data.root_ang_vel_w.cpu().numpy())
                log["cmd_angvel"].append(cmd_term.angular_vel_command.cpu().numpy())
                log["cmd_pos"].append(cmd_term.pos_command.cpu().numpy())
                log["reward"].append(rew.cpu().numpy())
                log["done"].append(dones.cpu().numpy())

        data = {k: np.stack(v).astype(np.float32) for k, v in log.items()}
        m = metrics(data["obj_angvel"][..., 2], cmd_sign.cpu().numpy(), data["done"] > 0, float(base.step_dt))
        m["checkpoint"] = os.path.basename(ckpt)
        m.update(cmd=args_cli.cmd, steps=args_cli.steps, num_envs=n, task=args_cli.task, perturb=args_cli.perturb,
                 servo=args_cli.servo)
        print(f"[eval] {ckpt}: {m}")

        out_dir = os.path.join(os.path.dirname(ckpt), "eval")
        os.makedirs(out_dir, exist_ok=True)
        data.update(meshes)
        data["object_size"] = size
        data["link_names"] = np.array(robot.body_names)
        data["dt"] = np.float32(base.step_dt)
        data["checkpoint"] = np.array(ckpt)
        np.savez_compressed(os.path.join(out_dir, os.path.basename(ckpt).replace(".pt", f"{suffix}.npz")), **data)

        results_path = os.path.join(out_dir, f"metrics{suffix}.json")
        results = json.load(open(results_path)) if os.path.exists(results_path) else {}
        results[m["checkpoint"]] = m
        json.dump(results, open(results_path, "w"), indent=1)
    print(f"[eval] metrics in {results_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
