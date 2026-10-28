# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learned servo model for the ORCA joints (Phase 2, sim2real; docs/ablation/actuator_net.md).

An actuator net after Hwangbo et al. 2019 (arXiv 1901.08652) and Yuryev & Hughes 2026 (arXiv 2603.04351), fitted to
chirp recordings of commanded vs measured joint position by scripts/actuator_net/train.py (no force sensor needed).
Per physics step and joint, the same update as scripts/actuator_net/servo_model.py:

    q*_L   = joint target L physics steps ago            servo / bus latency, per joint type
    y      = clip(y, q*_L - b, q*_L + b)                 servo play (stops short at reversals), per joint type
    delta  = net(history of q*_L - q and dq/dt)          learned correction, |delta| < delta_max, shared net
    target = y + delta                                   -> PhysX implicit PD drive (stiffness k, damping k * tau)

It subclasses ImplicitActuator and rewrites the PD target instead of returning a torque: an explicit torque is not
stable on these links at 120 Hz (2e-6..7e-5 kg m^2 about the finger joints; explicit damping d is stable only for
dt * d / inertia < 2, i.e. d < 0.02 Nms/rad). The PhysX drive keeps it stable for any gains, and the learned part acts
like a torque k * delta. Stiffness, effort limit, friction and armature stay those of the implicit cfg it replaces
(orca_v2.py): free motion does not identify a force scale. Damping becomes stiffness * tau with the fitted time
constant tau, so the free motion stays as fitted when orca_v2.py's stiffness changes; the gain randomization event
(randomize_actuator_gains) still scales stiffness and damping, as for the implicit actuator.

Assumes the env's physics dt of 1/120 s (delays and history taps are counted in physics steps).
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.utils.types import ArticulationActions

logger = logging.getLogger(__name__)

SERVO_NET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "actuator_net_weights", "servo_net.pt")
"""Weights written by ``scripts/actuator_net/train.py --deploy`` (TorchScript + per joint type metadata)."""

ORCA_V2_JOINT_TYPES = {
    "right_wrist": "wrist",
    "right_thumb_cmc": "thumb_cmc",
    "right_thumb_abd": "thumb_abd",
    "right_thumb_mcp": "thumb_mcp",
    "right_thumb_dip": "thumb_dip",
    "right_(index|middle|ring|pinky)_abd": "index_abd",
    "right_(index|middle|ring|pinky)_mcp": "index_mcp",
    "right_(index|middle|ring|pinky)_pip": "index_pip",
}
"""Joint name regex -> recorded joint type (RWR26 Task 1, 7.1: the index chirps stand for all four fingers)."""


class ServoActuatorNet(ImplicitActuator):
    """Implicit PD drive whose target is shaped by a learned servo model (delay, play, residual net)."""

    cfg: ServoActuatorNetCfg

    def __init__(self, cfg: ServoActuatorNetCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        extra = {"meta.json": ""}
        self.network = torch.jit.load(cfg.network_file, map_location=self._device, _extra_files=extra).eval()
        meta = json.loads(extra["meta.json"])
        types = _resolve_joint_types(self.joint_names, cfg.joint_types)
        missing = sorted(set(types) - set(meta["joint_types"]))
        if missing:
            raise ValueError(
                f"ServoActuatorNet: {cfg.network_file} has no fit for joint types {missing}"
                f" (trained: {sorted(meta['joint_types'])}). Record those joints and retrain."
            )
        per_joint = [meta["joint_types"][t] for t in types]

        def column(key: str, dtype=torch.float) -> torch.Tensor:
            return torch.tensor([p[key] for p in per_joint], dtype=dtype, device=self._device)

        self._base_delay = column("delay_steps", torch.long)
        self._base_play = column("play")
        # free motion identifies the time constant tau = damping / stiffness, not the force scale: keep the stiffness
        self.damping = self.stiffness * column("tau")
        train_k = column("stiffness")
        if not torch.allclose(self.stiffness[0], train_k, rtol=0.01):
            logger.warning(
                "ServoActuatorNet: stiffness differs from the one the net was fitted with (%s vs %s); the free motion"
                " only changes through friction / stiffness and inertia / stiffness, but consider retraining.",
                self.stiffness[0].tolist(),
                train_k.tolist(),
            )

        n, j = self._num_envs, self.num_joints
        self._history_len = int(meta["history_len"])
        self._max_delay = int(meta["max_delay_steps"]) + cfg.delay_jitter_steps
        self._delay = self._base_delay.repeat(n, 1)
        self._play = self._base_play.repeat(n, 1)
        self._targets = torch.zeros(n, j, self._max_delay + 1, device=self._device)
        """Past joint targets, index 0 = this physics step."""
        self._play_state = torch.zeros(n, j, device=self._device)
        self._err_hist = torch.zeros(n, j, self._history_len, device=self._device)
        self._vel_hist = torch.zeros(n, j, self._history_len, device=self._device)
        self._fresh = torch.ones(n, dtype=torch.bool, device=self._device)
        """Envs reset since their last compute(): buffers are filled from the joint state on the next compute()."""
        print(
            f"[INFO] ServoActuatorNet on {j} joints: {os.path.basename(cfg.network_file)} ({meta['created']},"
            f" {len(meta['trained_on'])} chirps), delays {sorted(set(self._base_delay.tolist()))} steps"
        )

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | slice):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self._num_envs, device=self._device)[env_ids]
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device)
        self._fresh[env_ids] = True
        if self.cfg.delay_jitter_steps > 0:
            jitter = torch.randint(
                -self.cfg.delay_jitter_steps,
                self.cfg.delay_jitter_steps + 1,
                (len(env_ids), self.num_joints),
                device=self._device,
            )
            self._delay[env_ids] = (self._base_delay + jitter).clamp(0, self._max_delay)
        lo, hi = self.cfg.play_scale_range
        if (lo, hi) != (1.0, 1.0):
            scale = lo + (hi - lo) * torch.rand(len(env_ids), self.num_joints, device=self._device)
            self._play[env_ids] = self._base_play * scale

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # after a reset: at rest at the current position, every past target = that position (no stale targets)
        fresh = self._fresh[:, None]
        self._targets = torch.where(fresh[..., None], joint_pos[..., None], self._targets)
        self._play_state = torch.where(fresh, joint_pos, self._play_state)
        self._err_hist = torch.where(fresh[..., None], 0.0, self._err_hist)
        self._vel_hist = torch.where(fresh[..., None], 0.0, self._vel_hist)
        self._fresh[:] = False

        # delayed target, play, history (new tensors: joint_positions may be a view of the articulation's buffer)
        self._targets = torch.cat([control_action.joint_positions[..., None], self._targets[..., :-1]], dim=2)
        q_delayed = self._targets.gather(2, self._delay[..., None]).squeeze(2)
        self._play_state = torch.clamp(self._play_state, q_delayed - self._play, q_delayed + self._play)
        self._err_hist = torch.cat([(q_delayed - joint_pos)[..., None], self._err_hist[..., :-1]], dim=2)
        self._vel_hist = torch.cat([joint_vel[..., None], self._vel_hist[..., :-1]], dim=2)
        with torch.no_grad():
            delta = self.network(
                self._err_hist.view(-1, self._history_len), self._vel_hist.view(-1, self._history_len)
            ).view(self._num_envs, self.num_joints)
        target = self._play_state + self.cfg.residual_scale * delta

        control_action.joint_positions = target
        # approximate drive torque for rewards / logging, as ImplicitActuator does (PhysX applies the drive itself)
        error_vel = control_action.joint_velocities - joint_vel
        self.computed_effort = (
            self.stiffness * (target - joint_pos) + self.damping * error_vel + control_action.joint_efforts
        )
        self.applied_effort = self._clip_effort(self.computed_effort)
        return control_action


@configclass
class ServoActuatorNetCfg(ImplicitActuatorCfg):
    """Configuration of :class:`ServoActuatorNet`. Build it with :func:`servo_net_actuators` from the implicit cfg."""

    class_type: type = ServoActuatorNet

    network_file: str = SERVO_NET_FILE
    """TorchScript file with the net and the per joint type fit (delay_steps, tau, play) in ``meta.json``."""

    joint_types: dict[str, str] = ORCA_V2_JOINT_TYPES
    """Joint name regex -> joint type of the fit. Every joint of the group must match exactly one entry."""

    residual_scale: float = 1.0
    """Scale of the learned correction; 0 leaves the physical prior (delay + play + PD time constant) alone."""

    delay_jitter_steps: int = 0
    """Per-episode, per-joint uniform jitter of the fitted delay (+- physics steps) as domain randomization."""

    play_scale_range: tuple[float, float] = (1.0, 1.0)
    """Per-episode, per-joint uniform scale of the fitted play as domain randomization. The play depends on the hand's
    tendon tension (v1 vs v1b thumb_cmc: +-3.3 vs ~0.5 deg), so e.g. (0.0, 1.5) covers re-tensioned hands."""


def servo_net_actuators(
    implicit: ImplicitActuatorCfg, joints: str = ".*", **cfg_kwargs
) -> dict[str, ImplicitActuatorCfg]:
    """Actuator groups replacing ``implicit``: joints whose name fully matches ``joints`` get the servo net (stiffness,
    effort limit, friction, armature, velocity limit copied from ``implicit``), the others keep ``implicit``.

    ``implicit.joint_names_expr`` must list plain joint names (as orca_v2.py does). Extra keyword arguments go to
    :class:`ServoActuatorNetCfg` (e.g. ``residual_scale``, ``delay_jitter_steps``, ``network_file``).
    """
    names = list(implicit.joint_names_expr)
    selected = [n for n in names if re.fullmatch(joints, n)]
    rest = [n for n in names if n not in selected]
    if not selected:
        raise ValueError(f"servo_net_actuators: no joint of {names} matches '{joints}'")
    groups: dict[str, ImplicitActuatorCfg] = {
        "servo_net": ServoActuatorNetCfg(
            joint_names_expr=selected,
            stiffness=_subset(implicit.stiffness, selected),
            damping=_subset(implicit.damping, selected),  # replaced by stiffness * fitted tau
            friction=_subset(implicit.friction, selected),
            armature=_subset(implicit.armature, selected),
            effort_limit_sim=_subset(implicit.effort_limit_sim, selected),
            velocity_limit_sim=_subset(implicit.velocity_limit_sim, selected),
            **cfg_kwargs,
        )
    }
    if rest:
        groups["fingers"] = implicit.replace(
            joint_names_expr=rest,
            **{
                key: _subset(getattr(implicit, key), rest)
                for key in ("stiffness", "damping", "friction", "armature", "effort_limit_sim", "velocity_limit_sim")
            },
        )
    return groups


def _resolve_joint_types(joint_names: list[str], patterns: dict[str, str]) -> list[str]:
    types = []
    for name in joint_names:
        hits = [t for pattern, t in patterns.items() if re.fullmatch(pattern, name)]
        if len(hits) != 1:
            raise ValueError(f"ServoActuatorNet: joint '{name}' matches {len(hits)} entries of joint_types {patterns}")
        types.append(hits[0])
    return types


def _subset(values, names: list[str]):
    """Per-joint dict restricted to the keys that match one of ``names`` (Isaac Lab requires every key to match)."""
    if not isinstance(values, dict):
        return values
    return {k: v for k, v in values.items() if any(re.fullmatch(k, n) for n in names)}
