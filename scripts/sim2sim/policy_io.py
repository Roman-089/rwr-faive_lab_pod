"""The policy node's side of the interface: 74-dim observation in, EMA joint targets out (no physics, no MuJoCo).

Mirrors Isaac Lab exactly (Isaac Lab 2.3.2, paths under ~/IsaacLab/source/isaaclab/isaaclab/):
  obs  = [joint_pos 17 | joint_vel 17 | goal_command 6 | last_action 17 | last_joint_pos_cmd 17]
    joint_pos           envs/mdp/observations.py joint_pos_limit_normalized: 2 (q - (lo + hi) / 2) / (hi - lo)
    joint_vel           joint_vel_rel (default joint vel is 0), times 0.2
    goal_command        ContinuousRotationCommand.command: target cube position (0, 0, -0.05) in the env frame
                        (= world here) + target angular velocity (0, 0, +-1.5) rad/s, world frame
    last_action         the raw policy output of the previous step (not clipped; zeros after a reset)
    last_joint_pos_cmd  the EMA joint target of the previous step in rad (= joint positions after a reset)
  noise: N(0, 0.10) on the normalized joint_pos and N(0, 0.05) on joint_vel BEFORE the 0.2 scale
         (managers/observation_manager.py: func -> noise -> clip -> scale); the deploy node adds none.
  action (envs/mdp/actions/joint_actions_to_limits.py, EMAJointPositionToLimitsAction):
    a -> clip [-1, 1] -> lo + (a + 1) / 2 (hi - lo) -> target = 0.2 * that + 0.8 * previous target -> clip [lo, hi]
    reset: previous target = current joint positions.

Joint order everywhere: orca_task.JOINT_NAMES. The limits are the policy node's (nominal) limits, independent of
whatever the hand's physical limits are.
"""

from __future__ import annotations

import numpy as np

import orca_task as task

OBS_LAYOUT = (("joint_pos", 17), ("joint_vel", 17), ("goal_command", 6), ("last_action", 17), ("last_joint_pos_cmd", 17))
OBS_DIM = sum(n for _, n in OBS_LAYOUT)


def obs_slices() -> dict[str, slice]:
    out, i = {}, 0
    for name, n in OBS_LAYOUT:
        out[name] = slice(i, i + n)
        i += n
    return out


class PolicyIO:
    def __init__(self, lower: np.ndarray, upper: np.ndarray, alpha: float = task.EMA_ALPHA,
                 obs_noise: bool = False, rng: np.random.Generator | None = None):
        self.lower, self.upper = np.asarray(lower, np.float64), np.asarray(upper, np.float64)
        self.alpha = alpha
        self.obs_noise = obs_noise
        self.rng = rng or np.random.default_rng(0)
        n = len(self.lower)
        self.last_action = np.zeros(n)
        self.prev_target = np.zeros(n)

    def reset(self, joint_pos: np.ndarray) -> None:
        """Start of an episode: last action 0, EMA state = current joint positions (Isaac clamps targets, not this)."""
        self.last_action[:] = 0.0
        self.prev_target[:] = joint_pos

    def observation(self, joint_pos: np.ndarray, joint_vel: np.ndarray, command: np.ndarray) -> np.ndarray:
        """The 74-dim policy observation (float32) from the current joint state and the 6-dim goal command."""
        q = 2.0 * (joint_pos - 0.5 * (self.lower + self.upper)) / (self.upper - self.lower)
        v = np.asarray(joint_vel, np.float64)
        if self.obs_noise:
            q = q + self.rng.normal(0.0, task.OBS_NOISE_STD["joint_pos"], q.shape)
            v = v + self.rng.normal(0.0, task.OBS_NOISE_STD["joint_vel"], v.shape)
        obs = np.concatenate([q, task.JOINT_VEL_SCALE * v, command, self.last_action, self.prev_target])
        return obs.astype(np.float32)

    def process_action(self, action: np.ndarray) -> np.ndarray:
        """Raw policy action (17) -> joint position targets (rad); updates last_action and the EMA state."""
        self.last_action = np.asarray(action, np.float64).copy()
        a = np.clip(self.last_action, -1.0, 1.0)
        target = self.lower + 0.5 * (a + 1.0) * (self.upper - self.lower)
        target = self.alpha * target + (1.0 - self.alpha) * self.prev_target
        self.prev_target = np.clip(target, self.lower, self.upper)
        return self.prev_target.copy()


def goal_command(direction: float, speed: float = 1.5) -> np.ndarray:
    """Eval command: hold the cube at COMMAND_POS, rotate about world z at direction * speed rad/s."""
    return np.array([*task.COMMAND_POS, 0.0, 0.0, direction * speed])
