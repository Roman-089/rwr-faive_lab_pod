# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Automatic domain randomization (ADR, OpenAI arXiv 1910.07113) for the two deployment gaps the perturbed eval tests:
action delay and a per-episode joint-encoder offset (docs/ablation/sim2real.md).

One global level ``env.adr_level`` in [0, 1] scales both ranges: the per-episode delay is drawn from
``[0, round(L * max_delay)]`` physics substeps and the encoder offset from ``±L * bias_rad``. ``adr_level`` (a
curriculum term) raises L by ``step`` every ``interval_steps`` env steps while the share of ended episodes that ended in
a drop stays below ``low``, and lowers it above ``high``; so the randomization grows only as fast as the policy copes.
A new episode samples with the current L, running episodes keep their values. The env's physics DR (gains, friction,
masses, gravity) is unchanged.
"""

from __future__ import annotations

import math
import re
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from .mdp.sim2real import (
    DelayedEMAJointPositionToLimitsAction,
    DelayedEMAJointPositionToLimitsActionCfg,
    joint_pos_limit_normalized_biased,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def _level(env) -> float:
    return getattr(env, "adr_level", 0.0)


class ADRDelayedEMAJointPositionToLimitsAction(DelayedEMAJointPositionToLimitsAction):
    """Delayed EMA joint targets whose per-episode delay is drawn from ``[0, round(L * delay_substeps_range[1])]``.
    ``delay_substeps_range[1]`` is the delay at L = 1 and sizes the target buffer."""

    def __init__(self, cfg: ADRDelayedEMAJointPositionToLimitsActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._delay.zero_()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        n = self.num_envs if env_ids is None else len(env_ids)
        hi = int(round(_level(self._env) * self.cfg.delay_substeps_range[1]))
        self._delay[ids] = torch.randint(0, hi + 1, (n,), device=self.device)


@configclass
class ADRDelayedEMAJointPositionToLimitsActionCfg(DelayedEMAJointPositionToLimitsActionCfg):
    class_type: type = ADRDelayedEMAJointPositionToLimitsAction


def trapezoid_step(pos, vel, goal, amax, vmax, dt):
    """One step of an acceleration-limited trajectory generator (the servo's motion profile) toward ``goal``.

    The velocity follows the braking curve sign(e) * sqrt(2 amax |e|) (clipped to ``vmax``), changing by at most
    ``amax * dt`` per step, so a step of size s peaks at sqrt(amax * s) and stops at the goal without overshoot (to
    within one step). Returns the new (pos, vel)."""
    err = goal - pos
    v_des = torch.sign(err) * torch.sqrt(2.0 * amax * err.abs())
    v_des = torch.maximum(torch.minimum(v_des, vmax), -vmax)
    dv = amax * dt
    vel = vel + torch.maximum(torch.minimum(v_des - vel, dv), -dv)
    # do not step past the goal: there the profile ends at rest
    step = vel * dt
    arrive = step.abs() >= err.abs()
    pos = torch.where(arrive, goal, pos + step)
    vel = torch.where(arrive, torch.zeros_like(vel), vel)
    return pos, vel


def _per_joint(table: dict[str, float], names: list[str], what: str) -> list[float]:
    out = []
    for n in names:
        hits = [v for k, v in table.items() if re.fullmatch(k, n)]
        if len(hits) != 1:
            raise ValueError(f"joint {n}: {len(hits)} entries in {what}")
        out.append(hits[0])
    return out


class ServoProfileAction(DelayedEMAJointPositionToLimitsAction):
    """Delayed EMA targets through a model of the real ORCA v2 servo before the joint PD:

    delayed target -> acceleration-limited (trapezoidal) profile -> slope (the joint covers ``slope`` of the commanded
    displacement from the episode's start pose) -> play (backlash of width ``play``: the joint target only moves once
    the profile has taken up the play) -> PD (made stiff by ``use_orca_v2_servo_model``).

    Fitted on the real hand (deploy session, 2026-10-31, chirps + 190 Hz steps): the acceleration limit is the Feetech
    profile register orca_core sets (22.78 rad/s^2 motor side / each joint's motor-per-joint ratio); the speed register
    never binds; the steady error is friction play (hysteresis 2.4-5.8 deg) plus a slope below 1. Per episode and joint
    the acceleration limit, the play width and the slope are scaled by U(``*_scale_range``)."""

    cfg: ServoProfileActionCfg

    def __init__(self, cfg: ServoProfileActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        names = list(self._joint_names)
        t = lambda v: torch.tensor(v, device=self.device)  # noqa: E731
        self._amax_nominal = t(_per_joint(cfg.amax_rad_s2, names, "amax_rad_s2"))
        self._play_nominal = t([math.radians(w) for w in _per_joint(cfg.play_deg, names, "play_deg")])
        self._slope_nominal = t(_per_joint(cfg.slope, names, "slope"))
        self._amax = self._amax_nominal.repeat(self.num_envs, 1)
        self._play = self._play_nominal.repeat(self.num_envs, 1)
        self._slope = self._slope_nominal.repeat(self.num_envs, 1)
        self._vmax = torch.full_like(self._amax, cfg.vmax_rad_s)
        self._prof_pos = self._prev_applied_actions.clone()
        self._prof_vel = torch.zeros_like(self._prof_pos)
        self._center = self._prof_pos.clone()
        self._joint_target = self._prof_pos.clone()
        self._dt = env.physics_dt

    def _scale(self, nominal, rng, n):
        lo, hi = rng
        return nominal * (lo + (hi - lo) * torch.rand(n, self.action_dim, device=self.device))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        n = self.num_envs if env_ids is None else len(env_ids)
        self._amax[ids] = self._scale(self._amax_nominal, self.cfg.amax_scale_range, n)
        self._play[ids] = self._scale(self._play_nominal, self.cfg.play_scale_range, n)
        self._slope[ids] = self._scale(self._slope_nominal, self.cfg.slope_scale_range, n)
        start = self._prev_applied_actions[ids]  # at rest at the reset pose
        self._prof_pos[ids] = start
        self._prof_vel[ids] = 0.0
        self._center[ids] = start
        self._joint_target[ids] = start

    def apply_actions(self):
        age = torch.clamp(self._delay - self._substep + self._decimation - 1, min=0) // self._decimation
        goal = self._target_buffer[age, self._env_idx]
        self._prof_pos, self._prof_vel = trapezoid_step(
            self._prof_pos, self._prof_vel, goal, self._amax, self._vmax, self._dt)
        reach = self._center + self._slope * (self._prof_pos - self._center)
        half = 0.5 * self._play
        self._joint_target = torch.maximum(torch.minimum(self._joint_target, reach + half), reach - half)
        self._asset.set_joint_position_target(self._joint_target, joint_ids=self._joint_ids)
        self._substep += 1


@configclass
class ServoProfileActionCfg(DelayedEMAJointPositionToLimitsActionCfg):
    class_type: type = ServoProfileAction

    amax_rad_s2: dict[str, float] = MISSING
    """Joint-side acceleration limit [rad/s^2] per joint-name regex (exactly one match per joint)."""
    play_deg: dict[str, float] = MISSING
    """Play (backlash) width [deg] per joint-name regex."""
    slope: dict[str, float] = MISSING
    """Fraction of the commanded displacement the joint covers, per joint-name regex."""
    amax_scale_range: tuple[float, float] = (0.7, 1.3)
    play_scale_range: tuple[float, float] = (0.5, 1.5)
    slope_scale_range: tuple[float, float] = (0.9, 1.1)
    vmax_rad_s: float = 100.0
    """Speed limit [rad/s]; the measured one (75-207 rad/s joint side) never binds."""


class ADRServoProfileAction(ADRDelayedEMAJointPositionToLimitsAction, ServoProfileAction):
    """``ServoProfileAction`` whose per-episode delay follows the ADR level like ``ADRDelayedEMAJointPositionToLimitsAction``
    (MRO: the ADR delay is sampled after the servo state is reset; the servo's ``apply_actions`` is used)."""


@configclass
class ADRServoProfileActionCfg(ServoProfileActionCfg):
    class_type: type = ADRServoProfileAction


# Real ORCA v2 right hand, deploy session 2026-10-31 (actuator_model_orca_v2_right_2026-10-31.json): measured on wrist,
# thumb_*, index_abd / mcp / pip; middle / ring / pinky copy the index finger except amax (their own motor ratios,
# from the automatic calibration, unverified).
ORCA_V2_SERVO_AMAX_RAD_S2 = {
    "right_wrist": 14.8, "right_thumb_cmc": 27.1, "right_thumb_abd": 38.0, "right_thumb_mcp": 23.9,
    "right_thumb_dip": 41.1, "right_index_abd": 25.8, "right_index_mcp": 24.4, "right_index_pip": 28.9,
    "right_middle_abd": 28.2, "right_ring_abd": 28.4, "right_pinky_abd": 26.1,
    "right_middle_mcp": 23.2, "right_ring_mcp": 23.9, "right_pinky_mcp": 25.4,
    "right_middle_pip": 28.1, "right_ring_pip": 26.6, "right_pinky_pip": 30.0,
}  # fmt: skip
ORCA_V2_SERVO_PLAY_DEG = {  # hysteresis width of the slow chirp; index_mcp (steps only) like thumb_mcp
    "right_wrist": 3.0, "right_thumb_cmc": 2.6, "right_thumb_abd": 5.8, "right_thumb_mcp": 2.7, "right_thumb_dip": 3.3,
    "right_(index|middle|ring|pinky)_abd": 2.4, "right_(index|middle|ring|pinky)_mcp": 2.7,
    "right_(index|middle|ring|pinky)_pip": 3.8,
}  # fmt: skip
ORCA_V2_SERVO_SLOPE = {  # index_pip 0.70, thumb_dip 0.80, thumb_abd 0.66 up / 0.95 down, index_abd asymmetric, others 0.87-0.95
    "right_wrist": 0.91, "right_thumb_cmc": 0.91, "right_thumb_abd": 0.80, "right_thumb_mcp": 0.91,
    "right_thumb_dip": 0.80, "right_(index|middle|ring|pinky)_abd": 0.85, "right_(index|middle|ring|pinky)_mcp": 0.91,
    "right_(index|middle|ring|pinky)_pip": 0.70,
}  # fmt: skip


class joint_pos_limit_normalized_adr_biased(joint_pos_limit_normalized_biased):
    """``joint_pos_limit_normalized_biased`` with the offset range scaled by the ADR level (``bias_rad`` at L = 1)."""

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = slice(None) if env_ids is None else env_ids
        n = self.num_envs if env_ids is None else len(env_ids)
        scale = self.bias_rad * _level(self._env)
        self.bias[ids] = (torch.rand(n, self.asset.num_joints, device=self.device) * 2 - 1) * scale


def adr_level(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    interval_steps: int = 480,
    step: float = 0.05,
    low: float = 0.25,
    high: float = 0.40,
    min_episodes: int = 1000,
    drop_term: str = "object_dropped",
) -> float:
    """Curriculum term: update ``env.adr_level`` from the drop share of the episodes that ended (see module doc).
    Called on every reset; returns the level for logging (``Curriculum/adr``)."""
    if not hasattr(env, "_adr_last"):
        env.adr_level = _level(env)
        env._adr_drops, env._adr_episodes, env._adr_last = 0, 0, env.common_step_counter
    if len(env_ids) > 0:
        env._adr_drops += int(env.termination_manager.get_term(drop_term)[env_ids].sum())
        env._adr_episodes += len(env_ids)
    if env.common_step_counter - env._adr_last >= interval_steps and env._adr_episodes >= min_episodes:
        share = env._adr_drops / env._adr_episodes
        if share < low:
            env.adr_level = min(1.0, env.adr_level + step)
        elif share > high:
            env.adr_level = max(0.0, env.adr_level - step)
        env._adr_drops, env._adr_episodes, env._adr_last = 0, 0, env.common_step_counter
    return env.adr_level

# Lag of the real servo's position loop behind the profile (2 zeta / wn of the deploy session's fit: fingers wn
# 140-300 rad/s, zeta ~1; wrist wn 20 rad/s). The task's implicit PD is heavily overdamped (zeta 3-160 at the joints'
# tiny inertias), i.e. a first-order lag Kd / Kp; its damping is set so that lag matches.
ORCA_V2_SERVO_LOOP_LAG_S = {"right_wrist": 0.10, "fingers": 0.01}


def use_orca_v2_servo_model(env_cfg, delay_substeps_range: tuple[int, int], adr: bool, nominal: bool = False):
    """Swap the env's joint action for the real servo model (delay -> acceleration-limited profile -> slope -> play ->
    stiff loop): ``(ADR)ServoProfileActionCfg`` with the measured per-joint values, and drive damping = stiffness * lag
    (``ORCA_V2_SERVO_LOOP_LAG_S``) for every actuator group. ``nominal``: no per-episode scaling (eval). Used by the
    -servo tasks and by the eval's --servo."""
    a = env_cfg.actions.joint_pos
    cls = ADRServoProfileActionCfg if adr else ServoProfileActionCfg
    one = (1.0, 1.0)
    env_cfg.actions.joint_pos = cls(
        asset_name=a.asset_name, joint_names=a.joint_names, alpha=a.alpha, rescale_to_limits=a.rescale_to_limits,
        delay_substeps_range=delay_substeps_range, amax_rad_s2=ORCA_V2_SERVO_AMAX_RAD_S2,
        play_deg=ORCA_V2_SERVO_PLAY_DEG, slope=ORCA_V2_SERVO_SLOPE,
        **({"amax_scale_range": one, "play_scale_range": one, "slope_scale_range": one} if nominal else {}),
    )
    for act in env_cfg.scene.robot.actuators.values():
        stiff = act.stiffness
        if not isinstance(stiff, dict):
            raise TypeError(f"expected per-joint stiffness dicts, got {type(stiff)}")
        act.damping = {k: kp * ORCA_V2_SERVO_LOOP_LAG_S["right_wrist" if "wrist" in k else "fingers"]
                       for k, kp in stiff.items()}
