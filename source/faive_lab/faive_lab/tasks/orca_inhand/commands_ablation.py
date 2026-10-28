# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command ablations (docs/ablation/rewards.md, "one direction, max speed").

The task is graded on how fast and how long the real hand turns the cube without a drop; the commanded speed and the
direction switches are not graded. HORA / LEAP / RotateIt train one rotation direction and reward speed. This command
always asks for the same direction at the full configured speed, so the policy sees a constant command (deploy:
``goal_ang_vel`` = that value, ``goal_flip_period_s`` = 0). The observation layout is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils import configclass

from .mdp.commands.commands_cfg import ContinuousRotationCommandCfg
from .mdp.commands.continuous_command import ContinuousRotationCommand


class FixedDirectionRotationCommand(ContinuousRotationCommand):
    """Angular velocity command = ``direction`` * ``angular_vel_range`` (no random speed, no random sign)."""

    cfg: FixedDirectionRotationCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        vel = torch.tensor(self.cfg.angular_vel_range, device=self.device)
        self.angular_vel_command[env_ids] = self.cfg.direction * vel


@configclass
class FixedDirectionRotationCommandCfg(ContinuousRotationCommandCfg):
    class_type: type = FixedDirectionRotationCommand

    direction: float = 1.0
    """+1: positive rotation about the axes in ``angular_vel_range`` (+z), -1: negative."""
