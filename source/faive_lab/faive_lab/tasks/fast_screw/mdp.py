# Owner: X (Task X / FAST)
"""MDP terms for the FAST screw task: command (screw in / out at a target speed), observations, rewards."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# --------------------------------------------------------------------------- command
class ScrewCommand(CommandTerm):
    """Signed target bolt speed [rad/s]: >0 = screw in, <0 = screw out. Shape (num_envs, 1)."""

    cfg: "ScrewCommandCfg"

    def __init__(self, cfg: "ScrewCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.bolt: Articulation = env.scene[cfg.asset_name]
        self.joint_id = self.bolt.find_joints(cfg.joint_name)[0][0]
        self.target = torch.zeros(self.num_envs, 1, device=self.device)
        for k in ("bolt_speed", "target_speed", "speed_error", "progress_mm", "direction_ok"):
            self.metrics[k] = torch.zeros(self.num_envs, device=self.device)
        self._angle_at_resample = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.target

    def _bolt_angle_vel(self):
        return (self.bolt.data.joint_pos[:, self.joint_id], self.bolt.data.joint_vel[:, self.joint_id])

    def _update_metrics(self):
        ang, vel = self._bolt_angle_vel()
        tgt = self.target[:, 0]
        self.metrics["bolt_speed"] = vel
        self.metrics["target_speed"] = tgt
        self.metrics["speed_error"] = torch.abs(vel - tgt)                  # absolute! (cf. Task 1 caveat)
        progress_rev = torch.sign(tgt) * (ang - self._angle_at_resample) / (2 * torch.pi)
        self.metrics["progress_mm"] = progress_rev * self.cfg.thread_pitch_mm
        self.metrics["direction_ok"] = (torch.sign(vel) == torch.sign(tgt)).float()

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        lo, hi = self.cfg.speed_range
        speed = lo + torch.rand(n, device=self.device) * (hi - lo)
        if self.cfg.alternate_direction:
            sign = -torch.sign(self.target[env_ids, 0])
            sign[sign == 0] = 1.0
        else:
            sign = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
        self.target[env_ids, 0] = sign * speed
        self._angle_at_resample[env_ids] = self._bolt_angle_vel()[0][env_ids]

    def _update_command(self):
        pass


@configclass
class ScrewCommandCfg(CommandTermCfg):
    class_type: type = ScrewCommand
    asset_name: str = MISSING
    joint_name: str = "bolt_joint"
    speed_range: tuple[float, float] = (5.0, 15.0)
    """|target bolt speed| in rad/s (sampled uniformly). 10 rad/s ≈ 1.6 rev/s."""
    alternate_direction: bool = True
    """True: in, out, in, ... at every resample (the demo); False: random sign."""
    thread_pitch_mm: float = 1.25
    """Only for the progress metric (M8 coarse = 1.25 mm/rev)."""


# --------------------------------------------------------------------------- observations
def bolt_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("bolt")) -> torch.Tensor:
    """Bolt speed [rad/s]. On the real fixture this needs a sensor (e.g. encoder on the bolt) – see README."""
    bolt: Articulation = env.scene[asset_cfg.name]
    return bolt.data.joint_vel[:, asset_cfg.joint_ids]


def bolt_joint_sincos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("bolt")) -> torch.Tensor:
    bolt: Articulation = env.scene[asset_cfg.name]
    a = bolt.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.cat((torch.sin(a), torch.cos(a)), dim=-1)


# --------------------------------------------------------------------------- rewards
def screw_speed_tracking_exp(env: ManagerBasedRLEnv, command_name: str, sigma: float,
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("bolt")) -> torch.Tensor:
    """exp(-(w - w*)^2 / sigma^2): turn the bolt at the commanded signed speed."""
    bolt: Articulation = env.scene[asset_cfg.name]
    w = bolt.data.joint_vel[:, asset_cfg.joint_ids[0]]
    w_star = env.command_manager.get_command(command_name)[:, 0]
    return torch.exp(-torch.square(w - w_star) / sigma**2)


def screw_direction_progress(env: ManagerBasedRLEnv, command_name: str, max_speed: float,
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("bolt")) -> torch.Tensor:
    """Dense shaping: signed speed in the commanded direction, clipped to [-1, 1] after scaling."""
    bolt: Articulation = env.scene[asset_cfg.name]
    w = bolt.data.joint_vel[:, asset_cfg.joint_ids[0]]
    sign = torch.sign(env.command_manager.get_command(command_name)[:, 0])
    return torch.clamp(sign * w / max_speed, -1.0, 1.0)


def fingertips_near_bolt(env: ManagerBasedRLEnv, sigma: float,
                         robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                         bolt_cfg: SceneEntityCfg = SceneEntityCfg("bolt")) -> torch.Tensor:
    """exp(-mean distance / sigma) between the listed finger bodies and the bolt head.

    Note: ORCA link frames sit at the joints, not at the very tip – sigma should be a few cm.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    bolt: Articulation = env.scene[bolt_cfg.name]
    tips = robot.data.body_pos_w[:, robot_cfg.body_ids]                 # (N, k, 3)
    head = bolt.data.body_pos_w[:, bolt_cfg.body_ids[0]].unsqueeze(1)   # (N, 1, 3)
    d = torch.linalg.norm(tips - head, dim=-1).mean(dim=-1)
    return torch.exp(-d / sigma)
