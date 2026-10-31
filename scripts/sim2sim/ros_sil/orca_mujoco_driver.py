"""MuJoCo stand-in for the real hand driver (rwr_system ``hand_control_node`` + ORCA v2 hand), as a ROS 2 node.

Same interface as ``hand_control_node`` for ``orca_v2`` (rwr_system f0f8cf6):
  subscribes  /<chirality>/hand/policy_output            Float32MultiArray, 17 joint targets [rad], config.yaml order
  publishes   /<chirality>/hand/joint_position_from_motor Float32MultiArray, 17 joint positions [rad], same order
              /<chirality>/hand/joint_velocity_from_motor Float32MultiArray, 17 joint velocities [rad/s]
The targets go to ``OrcaHandSim.set_joint_targets`` (MuJoCo PD position actuators with the Isaac task's gains, clamped
to the USD limits = config.yaml's joint_roms). Physics runs at the task's 120 Hz; the joint state is published after
every physics step (``state_hz`` 0) or by its own timer (the real driver: 100 Hz). With ``wait_for_first_command`` the
physics holds still until the first target arrives (the real hand holds its pose until the policy starts).

Runs under real rclpy (``python orca_mujoco_driver.py`` on a machine with ROS 2 and mujoco; a placed cube that drops
is respawned) or in-process under ``fake_ros`` (``run_node_sil.py``, lock-step, scored).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/sim2sim
import orca_task as task  # noqa: E402
from hand import OrcaHandSim  # noqa: E402

_FAKE = getattr(rclpy, "_fake", False)


class OrcaMujocoDriver(Node):
    def __init__(self, hand: OrcaHandSim, hand_config_path: str, chirality: str = "right", state_hz: float = 0.0,
                 wait_for_first_command: bool = True, on_physics_step=None):
        super().__init__("orca_mujoco_driver")
        self.hand = hand
        joint_ids = list(yaml.safe_load(Path(hand_config_path).read_text())["joint_ids"])
        usd = [n.removeprefix(f"{chirality}_") for n in task.JOINT_NAMES]
        if sorted(joint_ids) != sorted(usd):
            raise ValueError(f"config.yaml joint_ids {joint_ids} do not match the model's joints {usd}")
        self.robot_to_usd = np.array([joint_ids.index(n) for n in usd])  # usd[i] = robot[robot_to_usd[i]]
        self.usd_to_robot = np.array([usd.index(n) for n in joint_ids])
        self.waiting = wait_for_first_command
        self.on_physics_step = on_physics_step
        self.n_commands = 0
        self.last_command_usd = None
        prio = {"priority": 0} if _FAKE else {}  # lock-step: physics before the policy tick at the same time
        self.pos_pub = self.create_publisher(Float32MultiArray, f"/{chirality}/hand/joint_position_from_motor", 10)
        self.vel_pub = self.create_publisher(Float32MultiArray, f"/{chirality}/hand/joint_velocity_from_motor", 10)
        self.create_subscription(Float32MultiArray, f"/{chirality}/hand/policy_output", self.command_cb, 0)
        self.create_timer(task.PHYSICS_DT, self.physics_cb, **prio)
        self.state_hz = state_hz
        if state_hz > 0:
            self.create_timer(1.0 / state_hz, self.publish_state, **({"priority": 1} if _FAKE else {}))
        self.publish_state()

    def command_cb(self, msg: Float32MultiArray):
        q = np.asarray(msg.data, dtype=np.float64)
        if q.shape != (len(self.robot_to_usd),):
            self.get_logger().error(f"Expected {len(self.robot_to_usd)} joint angles, got {q.shape}")
            return
        self.last_command_usd = q[self.robot_to_usd]
        self.hand.set_joint_targets(self.last_command_usd)
        self.n_commands += 1
        self.waiting = False

    def physics_cb(self):
        if self.waiting:
            return
        self.hand.step_physics()
        if self.state_hz <= 0:
            self.publish_state()
        if self.on_physics_step is not None:
            self.on_physics_step()

    def publish_state(self):
        q, v = self.hand.joint_pos(), self.hand.joint_vel()
        self.pos_pub.publish(Float32MultiArray(data=q[self.usd_to_robot].astype(np.float32).tolist()))
        self.vel_pub.publish(Float32MultiArray(data=v[self.usd_to_robot].astype(np.float32).tolist()))


def main():
    """Real ROS 2: driver + world (cube spawn, respawn after a drop). Args: <rwr_system checkout> [--state_hz 100]."""
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("rwr_system", help="path of the rwr_system checkout (for config.yaml's joint order)")
    p.add_argument("--state_hz", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=0)
    args, ros_args = p.parse_known_args()
    rclpy.init(args=ros_args)
    hand = OrcaHandSim()
    rng = np.random.default_rng(args.seed)
    hand.reset(task.sample_reset(rng, hand.joint_params, hand_roll=False))
    cfg = Path(args.rwr_system) / "src/hand_control/orca/configs/orcahand_v2/right/config.yaml"
    steps = [0]

    def world():
        steps[0] += 1
        if steps[0] % task.DECIMATION == 0 and hand.cube_pos()[2] < task.DROP_HEIGHT:
            node.get_logger().warn("cube dropped: respawning it")
            state = task.sample_reset(rng, hand.joint_params, hand_roll=False)
            state.joint_pos = hand.joint_pos()  # keep the hand where it is, only the cube is placed again
            hand.reset(state)

    node = OrcaMujocoDriver(hand, str(cfg), state_hz=args.state_hz, wait_for_first_command=False, on_physics_step=world)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
