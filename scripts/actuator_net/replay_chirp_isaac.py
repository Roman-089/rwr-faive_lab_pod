# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Replay a recorded chirp on one ORCA v2 joint in Isaac Sim, with the implicit PD of orca_v2.py and with the servo
actuator net side by side, and compare both with the offline replica that train.py / evaluate.py use.

This is the check that the 1-DoF replica (servo_model.pd_step: implicit Euler PD + Coulomb friction, USD link
inertia) matches PhysX, i.e. that the offline numbers carry over to the simulator. Run it on a GPU node before the RL
runs (written on a CPU node, not run yet):

  cd scripts/actuator_net
  python replay_chirp_isaac.py --headless --csv ../motor_param_characterization/202510_orcav1b_realworld_data/\
chirp_index_mcp_20251018_120536_ctrl40Hz_20.0s_0.1-5.0Hz_10deg.csv

The command is the recording's command, re-centred on the middle of the simulated joint's range (as
chirp_characterization.py does), zero-order held on the 120 Hz physics grid, after 10 s of settling at its first
value. Writes logs/actuator_net_replay/<recording>_{implicit,actnet}.csv (timestamp, commanded_position,
measured_position in deg, re-centred like the recording) for plot_chirp_utils.py, and prints the RMSE of each model
against the measured joint and against the offline replica of the same model.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a recorded chirp with the implicit PD and the servo actuator net.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--csv", type=str, required=True, help="chirp recording (timestamp, commanded/measured deg)")
parser.add_argument("--naming", type=str, default="auto", choices=("auto", "v1b", "v2"), help="joint naming")
parser.add_argument("--network_file", type=str, default=None, help="servo net weights (default: the deployed ones)")
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import isaacsim.core.utils.prims as prim_utils  # noqa: E402
from faive_lab.assets.orca_v2 import ORCA_V2_CFG  # noqa: E402
from faive_lab.tasks.orca_inhand.actuator_net import SERVO_NET_FILE, servo_net_actuators  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from chirp_data import PHYSICS_DT, joint_name_from_file, load_recording, physics_grid, read_orca_v2_params  # noqa: E402
from chirp_data import resolve_naming  # noqa: E402
from servo_model import RolloutState, load_model, params_for, prior_from_meta, rollout, sample_at  # noqa: E402

SETTLE_STEPS = 1200  # 10 s at the first command, as in chirp_characterization.py


def main():
    csv = Path(args_cli.csv)
    rec = load_recording(csv, resolve_naming([joint_name_from_file(csv)[0]], args_cli.naming))
    joint_name = f"right_{rec.joint_type}"
    network_file = args_cli.network_file or SERVO_NET_FILE

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args_cli.device))
    robots = {}
    for i, kind in enumerate(("implicit", "actnet")):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=(0.5 * i, 0.0, 0.0))
        actuators = (
            dict(ORCA_V2_CFG.actuators)
            if kind == "implicit"
            else servo_net_actuators(ORCA_V2_CFG.actuators["fingers"], network_file=network_file)
        )
        robots[kind] = Articulation(ORCA_V2_CFG.replace(prim_path=f"/World/Origin{i}/Robot", actuators=actuators))
    sim.reset()

    robot = robots["implicit"]
    joint_idx = robot.find_joints(joint_name)[0][0]
    limits = robot.data.joint_pos_limits[0, joint_idx]
    center = float((limits[0] + limits[1]) / 2)
    cmd_ticks, sample_pos = physics_grid(rec)
    mean_cmd = float(np.mean(rec.commanded))
    cmd = center + np.concatenate([np.full(SETTLE_STEPS, cmd_ticks[0]), cmd_ticks]) - mean_cmd
    q_start = robot.data.joint_pos[0, joint_idx].item()

    # Isaac Sim: one physics step per command tick, the actuator runs in write_data_to_sim() as in the RL env
    q_isaac = {kind: [q_start] for kind in robots}
    for target in cmd:
        for kind, r in robots.items():
            targets = r.data.default_joint_pos.clone()
            targets[0, joint_idx] = float(target)
            r.set_joint_position_target(targets)
            r.write_data_to_sim()
        sim.step()
        for kind, r in robots.items():
            r.update(PHYSICS_DT)
            q_isaac[kind].append(r.data.joint_pos[0, joint_idx].item())

    # offline replica of both models on the same command sequence, from the same start state
    sim_params = read_orca_v2_params()
    net, meta = load_model(network_file)
    cmd_t = torch.tensor(cmd, dtype=torch.float32)[None]
    q0 = torch.tensor([q_start], dtype=torch.float32)
    q_offline = {}
    with torch.no_grad():
        for kind, prior, model in (("implicit", None, None), ("actnet", prior_from_meta(meta), net)):
            p = params_for([rec], sim_params, prior)
            state = RolloutState.at_rest(q0, meta["max_delay_steps"], meta["history_len"])
            q_offline[kind] = rollout(cmd_t, state, p, model)[0]

    out_dir = Path("logs/actuator_net_replay")
    os.makedirs(out_dir, exist_ok=True)
    pos = torch.tensor(sample_pos + SETTLE_STEPS, dtype=torch.float32)[None]
    measured = rec.measured - mean_cmd
    for kind in robots:
        qi = sample_at(torch.tensor(q_isaac[kind], dtype=torch.float32)[None], pos)[0].numpy() - center
        qo = sample_at(q_offline[kind], pos)[0].numpy() - center
        rmse_real = np.rad2deg(np.sqrt(np.mean((qi - measured) ** 2)))
        rmse_replica = np.rad2deg(np.sqrt(np.mean((qi - qo) ** 2)))
        print(f"[{kind:8s}] {joint_name}: RMSE vs measured {rmse_real:.2f} deg | Isaac Sim vs offline replica"
              f" {rmse_replica:.3f} deg")
        path = out_dir / f"{rec.name}_{kind}.csv"
        pd.DataFrame(
            {
                "timestamp": rec.time,
                "commanded_position": np.rad2deg(rec.commanded - mean_cmd),
                "measured_position": np.rad2deg(qi),
            }
        ).to_csv(path, index=False)
        print(f"[INFO]: wrote {path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
