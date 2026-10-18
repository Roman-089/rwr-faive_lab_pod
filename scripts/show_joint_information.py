"""
Print joint and link mass properties of various hands including ORCA hand.
Useful as a sanity check during sim2real.

.. code-block:: bash

    python scripts/show_joint_information.py --headless

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Print joint properties for the demo dexterous hands.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
from prettytable import PrettyTable

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

##
# Pre-defined configs
##
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG  # isort:skip
from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG  # isort:skip
from faive_lab.assets.orca_v2 import ORCA_V2_CFG


def design_scene() -> tuple[dict[str, Articulation], list[list[float]]]:
    """Creates the ground plane, lighting, and demo hand assets."""

    origin = torch.zeros(3)

    prim_utils.create_prim("/World/Origin1", "Xform", translation=origin)
    allegro = Articulation(ALLEGRO_HAND_CFG.replace(prim_path="/World/Origin1/Robot"))

    prim_utils.create_prim("/World/Origin2", "Xform", translation=origin)
    shadow_hand = Articulation(SHADOW_HAND_CFG.replace(prim_path="/World/Origin2/Robot"))

    prim_utils.create_prim("/World/Origin3", "Xform", translation=origin)
    orca_hand = Articulation(ORCA_V2_CFG.replace(prim_path="/World/Origin3/Robot"))

    scene_entities = {
        "allegro": allegro,
        "shadow_hand": shadow_hand,
        "orca_v2": orca_hand,
    }
    return scene_entities


def extract_first_env(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """Returns the first environment slice on CPU (or None if unavailable)."""
    if tensor is None:
        return None
    if tensor.ndim == 0:
        return tensor.detach().cpu()
    return tensor[0].detach().cpu()


def format_range(range_tensor: torch.Tensor | None) -> str:
    if range_tensor is None:
        return "N/A"
    lower = float(range_tensor[0])
    upper = float(range_tensor[1])
    return f"[{lower:.4f}, {upper:.4f}]"


def format_scalar(values: torch.Tensor | None, index: int) -> str:
    if values is None:
        return "N/A"
    return f"{float(values[index]):.4f}"


def print_link_masses(robot: Articulation) -> None:
    masses = extract_first_env(robot.root_physx_view.get_masses())
    if masses is None:
        print("[INFO]: Link mass information unavailable.")
        return

    table = PrettyTable()
    table.field_names = ["Link", "Mass (kg)"]
    table.align = "r"
    table.align["Link"] = "l"

    for index, link_name in enumerate(robot.body_names):
        table.add_row([link_name, f"{float(masses[index]):.4f}"])

    print("[INFO]: Link masses:")
    print(table)


def print_joint_information(name: str, robot: Articulation) -> None:
    """Pretty-prints joint limits and gains for the provided articulation."""
    joint_names = robot.joint_names

    pos_limits = extract_first_env(robot.data.joint_pos_limits)
    vel_limits = extract_first_env(robot.data.joint_vel_limits)
    effort_limits = extract_first_env(robot.data.joint_effort_limits)

    stiffness = extract_first_env(robot.data.default_joint_stiffness)
    if stiffness is None:
        stiffness = extract_first_env(robot.data.joint_stiffness)
    damping = extract_first_env(robot.data.default_joint_damping)
    if damping is None:
        damping = extract_first_env(robot.data.joint_damping)

    table = PrettyTable()
    table.field_names = ["Joint", "Range (rad)", "Max Vel (rad/s)", "Effort", "P gain", "D gain"]
    table.align = "r"
    table.align["Joint"] = "l"

    for index, joint_name in enumerate(joint_names):
        pos_range = format_range(pos_limits[index]) if pos_limits is not None else "N/A"
        vel_limit = format_scalar(vel_limits, index)
        effort_limit = format_scalar(effort_limits, index)
        p_gain = format_scalar(stiffness, index)
        d_gain = format_scalar(damping, index)
        table.add_row([joint_name, pos_range, vel_limit, effort_limit, p_gain, d_gain])

    print(f"\n[INFO]: Joint properties for '{name}' ({robot.cfg.prim_path})")
    print(table)

    print_link_masses(robot)

    if robot.actuators:
        print("[INFO]: Actuator groups:")
        for actuator_name, actuator in robot.actuators.items():
            model_name = actuator.cfg.class_type.__name__
            joint_subset = ", ".join(actuator._joint_names)  # pylint: disable=protected-access
            print(
                f"  - {actuator_name}: {model_name} | stiffness range "
                f"[{actuator.stiffness.min().item():.4f}, {actuator.stiffness.max().item():.4f}] | "
                f"damping range [{actuator.damping.min().item():.4f}, {actuator.damping.max().item():.4f}]"
            )
            print(f"    joints: {joint_subset}")


def main() -> None:
    """Entry point for printing the joint summaries."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_entities = design_scene()

    sim.reset()

    print("[INFO]: Setup complete. Printing joint information...")
    for entity_name, robot in scene_entities.items():
        print_joint_information(entity_name, robot)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()