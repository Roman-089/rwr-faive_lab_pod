# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Run a chirp signal on a specified joint and log the commanded and measured positions.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="This script runs a chirp signal on a specified joint.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# add custom arguments
parser.add_argument('--joint_name', type=str, default='right_index_mcp', help='Name of the joint to characterize')
parser.add_argument('--stiffness', type=float, nargs="+", default=[3.996], help='Stiffness of the actuator')
parser.add_argument('--damping', type=float, nargs="+", default=[0.245], help='Damping of the actuator')
parser.add_argument('--friction', type=float, nargs="+", default=[0.1], help='Friction of the actuator')
parser.add_argument('--armature', type=float, nargs="+", default=[0.008], help='Armature of the actuator')
parser.add_argument('--effort_limit', type=float, nargs="+", default=[0.575], help='Effort limit of the actuator')
parser.add_argument('--plot_data', action='store_true', default=False, help='Plot the data after characterization')

joint_name = parser.parse_args().joint_name
stiffness = parser.parse_args().stiffness
damping = parser.parse_args().damping
friction = parser.parse_args().friction
armature = parser.parse_args().armature
effort_limit = parser.parse_args().effort_limit

# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)

simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import signal

import isaacsim.core.utils.prims as prim_utils
from faive_lab.assets.orca_v2 import ORCA_V2_CFG

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg

from plot_chirp_utils import ChirpDataset, plot_chirp_frequency_response
from signal_processing import compute_frequency_response_from_dataframe
  

def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the scene for multiple robots."""
    env_origins = torch.zeros(num_origins, 3)
    num_cols = np.floor(np.sqrt(num_origins))
    num_rows = np.ceil(num_origins / num_cols)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0
    return env_origins.tolist()


def design_scene(
    stiffness: list[float],
    friction: list[float],
    damping: list[float],
    armature: list[float],
    effort_limit: list[float],
    joint_name: str,
    init_pos: float,
) -> tuple[dict, list[list[float]]]:
    """Designs the scene for all parameter sets."""
    if not prim_utils.is_prim_path_valid("/World/defaultGroundPlane"):
        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/defaultGroundPlane", cfg)
    if not prim_utils.is_prim_path_valid("/World/Light"):
        cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        cfg.func("/World/Light", cfg)

    num_robots = len(stiffness)
    assert num_robots == len(friction) == len(damping) == len(armature) == len(effort_limit), "All parameter lists must have the same length."
    origins = define_origins(num_robots, spacing=0.5)
    scene_entities = {}

    for i in range(num_robots):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origins[i])
        robot_cfg: ArticulationCfg = ORCA_V2_CFG.replace(prim_path=f"/World/Origin{i}/Robot").replace(
            actuators={
                # overwrite the properties of the specified joint
                "fingers": ImplicitActuatorCfg(
                    joint_names_expr=[
                        joint_name
                    ],
                    effort_limit_sim=effort_limit[i],
                    velocity_limit_sim=200.0,
                    stiffness=stiffness[i],
                    damping=damping[i],
                    friction=friction[i],
                    armature=armature[i],
                )
            },
        )
        orca_hand = Articulation(cfg=robot_cfg)
        scene_entities[f"hand_{i}"] = orca_hand

    return scene_entities, origins


def run_cmd_and_readback_multi(
    sim: sim_utils.SimulationContext,
    robots: list[Articulation],
    joint_name: str,
    cmd_angles,
) -> list[list[list[float]]]:
    """Run the chirp command on multiple robots in parallel."""

    sim_dt = sim.get_physics_dt()
    decimation = 6
    # make sure the control frequency is at 20Hz (same as real robot and the RL env)
    assert sim_dt * decimation == 1.0 / 20
    joint_idx = robots[0].data.joint_names.index(joint_name)

    # motion should be centered around the middle of the joint range
    joint_range = robots[0].data.joint_pos_limits[0][joint_idx]
    center_pos_rad = (joint_range[0] + joint_range[1]) / 2.0
    center_pos_rad = center_pos_rad.cpu().item()

    cmd_angles = np.asarray(cmd_angles, dtype=np.float32)
    measurements = [np.zeros_like(cmd_angles) for _ in robots]

    joint_targets = [r.data.joint_pos.clone() for r in robots]
    for jt, r in zip(joint_targets, robots):
        jt[0, joint_idx] = float(center_pos_rad + cmd_angles[0])
        r.set_joint_position_target(jt)
        r.write_data_to_sim()
    for _ in range(int(1.0 / sim_dt * 10)):
        sim.step()
        for r in robots:
            r.update(sim_dt)

    data_logs = [[] for _ in robots]
    start_sim_time = sim.current_time
    for k, tgt in enumerate(cmd_angles):
        for jt, r in zip(joint_targets, robots):
            jt[0, joint_idx] = float(center_pos_rad + tgt)
            r.set_joint_position_target(jt)
            r.write_data_to_sim()
        sim_time_since_start = sim.current_time - start_sim_time
        for i, r in enumerate(robots):
            measurements[i][k] = r.data.joint_pos[0, joint_idx].item()
            tgt_deg = np.rad2deg(tgt + center_pos_rad)
            meas_deg = np.rad2deg(measurements[i][k])
            data_logs[i].append([sim_time_since_start, tgt_deg, meas_deg])
        for _ in range(decimation):
            sim.step()
            for r in robots:
                r.update(sim_dt)

    return data_logs


def evaluate(
    joint_name: str,
    stiffness: list[float],
    damping: list[float],
    friction: list[float],
    armature: list[float],
    effort_limit: list[float],
) -> list[list[list[float]]]:
    """Evaluate all parameter sets in parallel."""
    # Chirp signal parameters - make sure it's the same as what was run on the real robot
    duration=20.0
    f0=0.1
    f1=5.0
    amplitude_deg=10
    dt_control = 1/20
    t = np.arange(0, duration, dt_control)
    chirp_deg = amplitude_deg * signal.chirp(t, f0, duration, f1, method='logarithmic')
    chirp_signal = np.deg2rad(chirp_deg)  # Convert to radians
    dt = 1 / 120
    sim_cfg = sim_utils.SimulationCfg(dt=dt, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[0.5, -0.5, 0.8], target=[0.0, -0.2, 0.5])

    scene_entities, scene_origins = design_scene(stiffness, friction, damping, armature, effort_limit, joint_name, 0.)
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    output = run_cmd_and_readback_multi(sim, list(scene_entities.values()), joint_name, chirp_signal)
    return output

def save_chirp_visualisations(csv_path: str, joint_name: str, *, output_dir: Path) -> None:
    """Generate plots and frequency response CSV for a chirp log."""

    df = pd.read_csv(csv_path)
    processed, frf = compute_frequency_response_from_dataframe(df)
    dataset = ChirpDataset(
        label=joint_name,
        source=Path(csv_path),
        processed=processed,
        frequency_response=frf,
    )

    fig = plot_chirp_frequency_response([dataset], title=joint_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_path = output_dir / f"{joint_name}_{dataset.source.stem}.png"
    fig.savefig(figure_path, facecolor=fig.get_facecolor())
    print(f"[INFO]: Plot saved to {figure_path}")
    plt.close(fig)

    freq_path = output_dir / f"{joint_name}_{dataset.source.stem}_freq_response.csv"
    pd.DataFrame(
        {
            "frequency": frf.frequency,
            "magnitude_db": frf.magnitude_db,
            "phase_deg": frf.phase_deg,
        }
    ).to_csv(freq_path, index=False)
    print(f"[INFO]: Frequency response data saved to {freq_path}")


def main():
    """Main function """
    print(f"[INFO]: Characterizing joint {joint_name} with {len(stiffness)} parameter sets")

    logs = evaluate(joint_name, stiffness, damping, friction, armature, effort_limit)
    # make sure the logs directory exists
    logs_dir = "logs/characterization"
    os.makedirs(logs_dir, exist_ok=True)
    for params, log in zip(zip(stiffness, damping, friction, armature, effort_limit), logs):
        s, d, f, a, e = params
        filepath = f"{logs_dir}/chirp_data{s:.6f}_{d:.6f}_{f:.6f}_{a:.6f}_{e:.6f}.csv"
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestamp", "commanded_position", "measured_position"])
            writer.writerows(log)
        if args_cli.plot_data:
            save_chirp_visualisations(filepath, joint_name, output_dir=Path("logs"))


if __name__ == "__main__":
    # run the main execution
    main()
