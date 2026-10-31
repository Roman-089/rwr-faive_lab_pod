"""Run the course's real-hand policy node, unchanged, against the MuJoCo hand (software in the loop), on CPU, without ROS.

The node file (rwr_system ``experiments/rl_inference/policy_inference_cube_spin_v2.py``) is imported from a checkout
of the deploy repo and run in-process under ``fake_ros`` (lock-step simulated time) against ``OrcaMujocoDriver``. So its
own joint mapping (config.yaml order <-> USD order), limits table, observation, EMA, timer and model loading are what
drives the hand. Protocol as ``run_sim2sim.py`` (nominal physics, +z at ``--goal_ang_vel``, 60 s trials, same spawns
from the same seed): a drop respawns the cube and hand and starts a fresh node instance (Isaac's eval resets the env).

``--acceptance`` also runs ``run_sim2sim.run_trial`` (our own observation / action code) on the same spawns without
observation noise and compares the first trial step by step: any difference in the first observation or targets is an
interface bug in the node (or in our sim side).

Usage (CPU, conda env rwr):
    python scripts/sim2sim/ros_sil/run_node_sil.py --rwr_system <checkout> \\
        --policy logs/rsl_rl/orca_v2_repose/<run>/exported/policy_history.pt --trials 8 [--acceptance] \\
        [--state_hz 100] [--cmd_latency_ms 10] [--out sil.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import fake_ros  # noqa: E402

fake_ros.install()
import orca_task as task  # noqa: E402
import torch  # noqa: E402
from hand import OrcaHandSim  # noqa: E402
from orca_mujoco_driver import OrcaMujocoDriver  # noqa: E402
from run_sim2sim import metrics, run_trial  # noqa: E402

NODE_REL = "experiments/rl_inference/policy_inference_cube_spin_v2.py"
NODE_NAME = "policy_inference_node_v2"
CONFIG_REL = "src/hand_control/orca/configs/orcahand_v2/right/config.yaml"


def load_node_class(rwr_system: Path):
    """Import the node file with ``faive_system`` = the checkout (its own imports: faive_system.src.common.ros_utils)."""
    link_dir = Path(tempfile.mkdtemp(prefix="faive_system_"))
    (link_dir / "faive_system").symlink_to(rwr_system.resolve(), target_is_directory=True)
    sys.path.insert(0, str(link_dir))
    spec = importlib.util.spec_from_file_location("policy_inference_cube_spin_v2", rwr_system / NODE_REL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PolicyInferenceNodeV2


def run_sil_trial(node_cls, hand: OrcaHandSim, rwr_system: Path, policy_path: str, trial: int, args) -> dict:
    """One trial: ``args.steps`` control steps (of 6 physics steps) of node + driver, drops respawn hand + cube."""
    ex = fake_ros.reset_executor()
    fake_ros.LATENCY_S.clear()
    if args.cmd_latency_ms > 0:
        fake_ros.LATENCY_S["/right/hand/policy_output"] = args.cmd_latency_ms / 1000
    csv0 = str(Path(args.csv_dir) / "trial0_episode0.csv") if args.csv_dir and trial == 0 else ""
    fake_ros.PARAMS[NODE_NAME] = {"policy_ckpt_path": policy_path, "goal_ang_vel": args.goal_ang_vel,
                                  "dry_run": False, "log_period_s": 0.0, "csv_log_path": csv0}
    rng = np.random.default_rng([args.seed, trial, 0])  # same spawn sequence as run_sim2sim.run_trial (dr none)
    hand.set_physics(task.PhysicsParams())
    hand.reset(task.sample_reset(rng, hand.joint_params, hand_roll=False))
    rec = {"w": [], "done": [], "joint_pos": [], "joint_vel": [], "target": [], "cube_pos": [],
           "spawn_joint_pos": [hand.joint_pos()]}
    state = {"physics": 0, "node": None, "restart": False}

    def world():
        state["physics"] += 1
        if state["physics"] % task.DECIMATION:
            return
        dropped = hand.cube_pos()[2] < task.DROP_HEIGHT
        if dropped:
            hand.reset(task.sample_reset(rng, hand.joint_params, hand_roll=False))
            state["restart"] = True
        rec["w"].append(hand.cube_ang_vel_w())
        rec["done"].append(float(dropped))
        rec["joint_pos"].append(hand.joint_pos())
        rec["joint_vel"].append(hand.joint_vel())
        rec["target"].append(driver.last_command_usd.copy())
        rec["cube_pos"].append(hand.cube_pos())

    driver = OrcaMujocoDriver(hand, str(rwr_system / CONFIG_REL), state_hz=args.state_hz, on_physics_step=world)
    state["node"] = node_cls()
    driver.publish_state()  # the node subscribes in its __init__: give it the current state
    n_phys = args.steps * task.DECIMATION
    while state["physics"] < n_phys:
        ex.run_until(ex.tick + 10 * fake_ros.TICKS_PER_S,
                     stop=lambda: state["restart"] or state["physics"] >= n_phys)
        if state["restart"]:  # fresh node = new episode (EMA from the measured pose, last action 0, policy reset)
            fake_ros.PARAMS[NODE_NAME]["csv_log_path"] = ""
            state["node"].destroy_node()
            driver.waiting = True
            state["node"] = node_cls()
            driver.publish_state()
            state["restart"] = False
    state["node"].destroy_node()
    driver.destroy_node()
    return {k: np.array(v) for k, v in rec.items()}


def check_node_io(trial: dict, csv_path: Path, jp, goal_ang_vel: float) -> dict:
    """Recompute the node's observation and joint command with our PolicyIO from the states the node saw (trial 0,
    first episode, from the node's own CSV log): same trajectory, so any difference is an interface difference."""
    from policy_io import PolicyIO, goal_command, obs_slices

    rows = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    obs_node, cmd_node = rows[:, 1:75], rows[:, 75:92]
    n = min(len(rows), len(trial["joint_pos"]))
    first_drop = next((i for i in range(len(trial["done"])) if trial["done"][i]), len(trial["done"]))
    n = min(n, first_drop) - 1  # the last row's raw action is only known from the next row
    q = np.concatenate([trial["spawn_joint_pos"], trial["joint_pos"][:n]])
    v = np.concatenate([np.zeros((1, 17)), trial["joint_vel"][:n]])
    io = PolicyIO(jp.lower, jp.upper)
    io.reset(q[0])
    cmd = goal_command(np.sign(goal_ang_vel), abs(goal_ang_vel)).astype(np.float32)
    worst = {k: 0.0 for k in obs_slices()} | {"joint_cmd": 0.0}
    for t in range(n):
        ours = io.observation(q[t], v[t], cmd)
        for k, sl in obs_slices().items():
            worst[k] = max(worst[k], float(np.abs(ours[sl] - obs_node[t, sl]).max()))
        target = io.process_action(obs_node[t + 1, obs_slices()["last_action"]])
        worst["joint_cmd"] = max(worst["joint_cmd"], float(np.abs(target - cmd_node[t]).max()))
    return {"steps_checked": n, **{f"max_diff_{k}": v for k, v in worst.items()}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rwr_system", required=True, type=Path)
    p.add_argument("--policy", required=True)
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--goal_ang_vel", type=float, default=1.5)
    p.add_argument("--state_hz", type=float, default=0.0, help="0: publish the joint state after every physics step")
    p.add_argument("--cmd_latency_ms", type=float, default=0.0)
    p.add_argument("--acceptance", action="store_true")
    p.add_argument("--out", default="")
    p.add_argument("--csv_dir", default="", help="log the node's CSV (trial 0, first episode) here; --acceptance sets it")
    args = p.parse_args()

    fake_ros.LOG_LEVEL["info"] = False
    if args.acceptance and not args.csv_dir:
        args.csv_dir = tempfile.mkdtemp(prefix="sil_csv_")
    node_cls = load_node_class(args.rwr_system)
    hand = OrcaHandSim()
    trials = [run_sil_trial(node_cls, hand, args.rwr_system, args.policy, i, args) for i in range(args.trials)]
    w = np.stack([t["w"][:, 2] for t in trials], 1)
    done = np.stack([t["done"] for t in trials], 1)
    m = metrics(w, np.sign(np.full(args.trials, args.goal_ang_vel)), done, task.CONTROL_DT)
    m["drop_times_s"] = [[round((i + 1) * task.CONTROL_DT, 2) for i in np.flatnonzero(t["done"])] for t in trials]
    m["rpm_per_trial"] = [round(float(x), 2) for x in w.sum(0) * task.CONTROL_DT / (2 * np.pi) / (args.steps * task.CONTROL_DT / 60)]
    m.update(policy=args.policy, trials=args.trials, steps=args.steps, goal_ang_vel=args.goal_ang_vel,
             state_hz=args.state_hz, cmd_latency_ms=args.cmd_latency_ms, node=str(args.rwr_system / NODE_REL))
    print("[sil] " + " ".join(f"{k} {v:.3f}" for k, v in m.items() if isinstance(v, float)))
    print(f"[sil] drop times [s] per trial: {m['drop_times_s']}")

    if args.acceptance:
        policy = torch.jit.load(args.policy, map_location="cpu").eval()
        ref_args = types.SimpleNamespace(seed=args.seed, dr="none", no_obs_noise=True, cmd=abs(args.goal_ang_vel),
                                         steps=args.steps)
        ref = run_trial(hand, policy, 0, float(np.sign(args.goal_ang_vel)), ref_args)
        sil = trials[0]
        n = min(len(ref["target"]), len(sil["target"]))
        first_drop = next((i for i in range(n) if ref["done"][i] or sil["done"][i]), n)
        dt = np.abs(ref["target"][:n] - sil["target"][:n]).max(1)
        dq = np.abs(ref["joint_pos"][:n] - sil["joint_pos"][:n]).max(1)
        k = next((i for i in range(n) if dt[i] > 1e-4), n)
        m["acceptance"] = {"first_target_diff_rad": float(dt[0]), "steps_target_within_1e-4": int(k),
                           "first_drop_step": int(first_drop), "max_target_diff_before_drop": float(dt[:first_drop].max()),
                           "max_joint_diff_before_drop": float(dq[:first_drop].max())}
        m["acceptance"]["same_trajectory"] = check_node_io(trials[0], Path(args.csv_dir) / "trial0_episode0.csv",
                                                           hand.joint_params, args.goal_ang_vel)
        print(f"[acceptance] {m['acceptance']}")
    if args.out:
        Path(args.out).write_text(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
