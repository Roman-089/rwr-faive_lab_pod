"""Check an exported history policy (export_history_policy.py) against the original policy inside the real env.

Needs a GPU node with Isaac Sim. The env runs the run's task with its own noise / DR; episodes are shortened
(--episode_s) and their start is staggered over the envs, so time-outs and drops reset envs at many different steps.
Every control step:
  - the exported TorchScript module (CPU, the deploy device) gets ONLY the newest single-step obs of each env, sliced
    from the env's flat policy obs (export_history_policy.current_obs), and reset(env_ids) for the envs the env has just
    reset (their obs is already the new episode's first obs), exactly the deploy contract;
  - (1) its rebuilt history (flat_history()) must equal the env's flat policy obs bit for bit: this checks history
    layout, start-up fill and reset timing against the ObservationManager of the real env, incl. noise;
  - (2) its actions are compared with a CPU copy of the original policy rebuilt from the run dir (load_run, same
    construction as the export; expected ~1e-6) and with the runner's policy on the sim device as in play.py / eval
    (expected small but nonzero: GPU vs CPU float32 kernels).
The env is stepped with the runner policy's actions (--drive policy) or the exported module's (--drive exported).
Recurrent policies: (2) only. Results: printed and <run>/exported/verify_in_sim.json.

Usage (GPU node; task and checkpoint default to <run>/exported/policy_history.json):
    python scripts/export/verify_export_in_sim.py --run logs/rsl_rl/orca_v2_repose/<run> --headless \
        [--num_envs 64 --steps 1000 --episode_s 4 --drive policy]
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Compare an exported history policy with the original in the real env.")
parser.add_argument("--run", type=str, required=True, help="run directory with exported/policy_history.pt")
parser.add_argument("--model", type=str, default=None, help="default: <run>/exported/policy_history.pt")
parser.add_argument("--task", type=str, default=None, help="default: from the export's json")
parser.add_argument("--checkpoint", type=str, default=None, help="default: the one the module was exported from")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--episode_s", type=float, default=4.0, help="episode length (more resets); <= 0: task default")
parser.add_argument("--drive", choices=("policy", "exported"), default="policy", help="whose actions step the env")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--tol", type=float, default=1e-5, help="max |action diff| vs the CPU copy of the policy")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

RUN = os.path.abspath(args_cli.run)
MODEL = args_cli.model or os.path.join(RUN, "exported", "policy_history.pt")
with open(os.path.splitext(MODEL)[0] + ".json") as f:
    META = json.load(f)
TASK = args_cli.task or META["task"]
CHECKPOINT = args_cli.checkpoint or META["checkpoint"]
if not os.path.isabs(CHECKPOINT):
    CHECKPOINT = os.path.join(RUN, CHECKPOINT)
if TASK is None:
    sys.exit("no task id in the export's json: pass --task")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import faive_lab.tasks  # noqa: F401

import export_history_policy as ehp  # noqa: E402  (after faive_lab: uses the real task modules)

RESULT = {"ok": False}  # hydra_task_config's wrapper drops main()'s return value


@hydra_task_config(TASK, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.episode_s > 0:
        env_cfg.episode_length_s = args_cli.episode_s
    env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    base = env.unwrapped

    # original policy, as play.py / eval_policy.py load it
    runner_class = DistillationRunner if agent_cfg.class_name == "DistillationRunner" else OnPolicyRunner
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(CHECKPOINT)
    policy = runner.get_inference_policy(device=base.device)
    policy_nn = runner.alg.policy
    # the same policy rebuilt from the run dir on CPU (what the export was made from), and the exported module
    cpu_policy, layout, _ = ehp.load_run(RUN, CHECKPOINT)
    exported = torch.jit.load(MODEL)  # CPU
    recurrent = getattr(cpu_policy, "is_recurrent", False)
    group = layout["group"]
    h = max(layout["history_length"], 1)

    stats = {"steps": 0, "resets": 0, "flat_mismatch_steps": 0, "max_da_cpu": 0.0, "max_da_sim": 0.0,
             "max_da_cpu_after_reset": 0.0}
    since_reset = torch.zeros(base.num_envs, dtype=torch.long)
    with torch.inference_mode():
        torch.manual_seed(args_cli.seed)
        env.reset()
        # stagger the time-outs over the envs (as OnPolicyRunner.learn(init_at_random_ep_len=True))
        base.episode_length_buf[:] = torch.randint_like(base.episode_length_buf, high=int(base.max_episode_length))
        policy_nn.reset()
        if recurrent:
            cpu_policy.reset()
        exported.reset()
        obs = env.get_observations()
        for step in range(args_cli.steps):
            flat = obs[group]
            a_exp = exported(ehp.current_obs(flat, layout).cpu())
            if not recurrent and not torch.equal(exported.flat_history(), flat.cpu()):
                stats["flat_mismatch_steps"] += 1
            a_sim = policy(obs)
            a_cpu = cpu_policy.act_inference(ehp.policy_inputs(cpu_policy, layout, flat.cpu()))
            d_cpu = (a_exp - a_cpu).abs().max(dim=-1).values
            stats["max_da_cpu"] = max(stats["max_da_cpu"], float(d_cpu.max()))
            stats["max_da_sim"] = max(stats["max_da_sim"], float((a_exp - a_sim.cpu()).abs().max()))
            fresh = since_reset < h  # history still (partly) made of the episode's first obs
            if fresh.any():
                stats["max_da_cpu_after_reset"] = max(stats["max_da_cpu_after_reset"], float(d_cpu[fresh].max()))
            obs, _, dones, _ = env.step(a_sim if args_cli.drive == "policy" else a_exp.to(base.device))
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            policy_nn.reset(dones)
            if recurrent:
                cpu_policy.reset(dones.cpu())
            if len(done_ids):
                exported.reset(done_ids.cpu())  # before the new episode's first obs, as on the robot
            since_reset += 1
            since_reset[done_ids.cpu()] = 0
            stats["resets"] += len(done_ids)
            stats["steps"] += 1

    ok = stats["flat_mismatch_steps"] == 0 and stats["max_da_cpu"] <= args_cli.tol
    stats.update(task=TASK, checkpoint=CHECKPOINT, model=MODEL, num_envs=args_cli.num_envs, drive=args_cli.drive,
                 episode_s=args_cli.episode_s, recurrent=recurrent, tol=args_cli.tol, ok=ok)
    print(f"[verify] {args_cli.num_envs} envs x {stats['steps']} steps, {stats['resets']} env resets")
    if not recurrent:
        print(f"[verify] flat history == env policy obs on all steps: {stats['flat_mismatch_steps'] == 0} "
              f"({stats['flat_mismatch_steps']} mismatching steps)")
    print(f"[verify] max |action diff| exported vs CPU policy {stats['max_da_cpu']:.2e} (first {h} steps after a "
          f"reset: {stats['max_da_cpu_after_reset']:.2e}), vs runner policy on {base.device}: "
          f"{stats['max_da_sim']:.2e}")
    print(f"[verify] {'OK' if ok else 'FAILED'}")
    with open(os.path.join(os.path.dirname(MODEL), "verify_in_sim.json"), "w") as f:
        json.dump(stats, f, indent=1)
    env.close()
    RESULT["ok"] = ok


if __name__ == "__main__":
    main()
    simulation_app.close()
    sys.exit(0 if RESULT["ok"] else 1)
