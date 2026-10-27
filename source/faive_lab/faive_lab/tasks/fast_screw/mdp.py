# Owner: X (Task X / FAST)
"""MDP terms for the FAST screw task: command (spin the nut in / out at a target speed), thread coupling,
observations and rewards.

Sign convention: nut spin about world +z (counter-clockwise seen from above) moves the nut UP (right-hand
thread). Screwing IN = negative spin speed.
"""
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


# --------------------------------------------------------------------------- command + thread
class ScrewCommand(CommandTerm):
    """Signed target nut speed [rad/s] (shape (N, 1)) and the software thread: every control step the
    nut_lift target is moved by d(angle) * pitch / 2pi, clamped to the free thread length."""

    cfg: "ScrewCommandCfg"

    def __init__(self, cfg: "ScrewCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.fixture: Articulation = env.scene[cfg.asset_name]
        self.spin_id = self.fixture.find_joints(cfg.spin_joint)[0][0]
        self.lift_id = self.fixture.find_joints(cfg.lift_joint)[0][0]
        lim = self.fixture.data.joint_pos_limits[0, self.lift_id]
        self.lift_lo, self.lift_hi = float(lim[0]), float(lim[1])
        self.target = torch.zeros(self.num_envs, 1, device=self.device)
        self.prev_angle = torch.zeros(self.num_envs, device=self.device)
        self.lift_target = torch.zeros(self.num_envs, device=self.device)
        self.lift_at_resample = torch.zeros(self.num_envs, device=self.device)
        for k in ("nut_speed", "target_speed", "speed_error", "progress_mm", "direction_ok", "nut_height_mm"):
            self.metrics[k] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.target

    def _spin(self):
        return self.fixture.data.joint_pos[:, self.spin_id], self.fixture.data.joint_vel[:, self.spin_id]

    def _lift(self):
        return self.fixture.data.joint_pos[:, self.lift_id]

    def _update_metrics(self):
        _, vel = self._spin()
        tgt = self.target[:, 0]
        h = self._lift()
        self.metrics["nut_speed"] = vel
        self.metrics["target_speed"] = tgt
        self.metrics["speed_error"] = torch.abs(vel - tgt)
        # progress in the commanded direction since the last resample (screw in -> nut down -> positive)
        self.metrics["progress_mm"] = torch.sign(tgt) * (h - self.lift_at_resample) * 1000.0
        self.metrics["direction_ok"] = (torch.sign(vel) == torch.sign(tgt)).float()
        self.metrics["nut_height_mm"] = h * 1000.0

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        lo, hi = self.cfg.speed_range
        speed = lo + torch.rand(n, device=self.device) * (hi - lo)
        if self.cfg.alternate_direction:
            sign = -torch.sign(self.target[env_ids, 0])
            sign[sign == 0] = -1.0                       # first command after reset: screw in
        else:
            sign = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
        self.target[env_ids, 0] = sign * speed
        # (re)sync the software thread with the true joint state (also right after a reset event)
        self.prev_angle[env_ids] = self._spin()[0][env_ids]
        self.lift_target[env_ids] = self._lift()[env_ids]
        self.lift_at_resample[env_ids] = self._lift()[env_ids]

    def _update_command(self):
        """Software thread: nut height follows its spin angle."""
        a, _ = self._spin()
        d = a - self.prev_angle
        d = torch.atan2(torch.sin(d), torch.cos(d))       # robust to angle wrapping
        self.prev_angle = a
        self.lift_target = torch.clamp(self.lift_target + d * self.cfg.thread_pitch / (2 * torch.pi),
                                       self.lift_lo, self.lift_hi)
        self.fixture.set_joint_position_target(self.lift_target.unsqueeze(-1), joint_ids=[self.lift_id])


@configclass
class ScrewCommandCfg(CommandTermCfg):
    class_type: type = ScrewCommand
    asset_name: str = MISSING
    spin_joint: str = "nut_spin"
    lift_joint: str = "nut_lift"
    speed_range: tuple[float, float] = (5.0, 15.0)
    """|target nut speed| in rad/s, uniform. 10 rad/s ≈ 1.6 rev/s ≈ 2 mm/s on M8."""
    alternate_direction: bool = True
    """True: in, out, in, ... at every resample (the demo); False: random sign."""
    thread_pitch: float = 0.00125
    """m per revolution (M8 coarse). Must match world.WorldGeometry.thread_pitch."""


# --------------------------------------------------------------------------- observations
def joint_vel_of(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Velocity of selected joints (privileged for the nut: needs a sensor on the real fixture)."""
    a: Articulation = env.scene[asset_cfg.name]
    return a.data.joint_vel[:, asset_cfg.joint_ids]


def joint_pos_of(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    a: Articulation = env.scene[asset_cfg.name]
    return a.data.joint_pos[:, asset_cfg.joint_ids]


# --------------------------------------------------------------------------- rewards
def spin_speed_tracking_exp(env: ManagerBasedRLEnv, command_name: str, sigma: float,
                            asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """exp(-(w - w*)^2 / sigma^2): spin the nut at the commanded signed speed."""
    a: Articulation = env.scene[asset_cfg.name]
    w = a.data.joint_vel[:, asset_cfg.joint_ids[0]]
    w_star = env.command_manager.get_command(command_name)[:, 0]
    return torch.exp(-torch.square(w - w_star) / sigma**2)


def spin_direction_progress(env: ManagerBasedRLEnv, command_name: str, max_speed: float,
                            asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Dense shaping: signed speed in the commanded direction, scaled and clipped to [-1, 1]."""
    a: Articulation = env.scene[asset_cfg.name]
    w = a.data.joint_vel[:, asset_cfg.joint_ids[0]]
    sign = torch.sign(env.command_manager.get_command(command_name)[:, 0])
    return torch.clamp(sign * w / max_speed, -1.0, 1.0)


def bodies_near_body(env: ManagerBasedRLEnv, sigma: float, robot_cfg: SceneEntityCfg,
                     target_cfg: SceneEntityCfg) -> torch.Tensor:
    """exp(-mean distance / sigma) between finger bodies and the nut. ORCA link frames sit at the
    joints, not at the very tip – keep sigma at a few cm."""
    robot: Articulation = env.scene[robot_cfg.name]
    tgt: Articulation = env.scene[target_cfg.name]
    tips = robot.data.body_pos_w[:, robot_cfg.body_ids]
    p = tgt.data.body_pos_w[:, target_cfg.body_ids[0]].unsqueeze(1)
    return torch.exp(-torch.linalg.norm(tips - p, dim=-1).mean(dim=-1) / sigma)
