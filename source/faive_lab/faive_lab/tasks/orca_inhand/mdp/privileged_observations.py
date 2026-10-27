# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Privileged observations for the critic only (asymmetric actor-critic). Never feed these to the policy:
they are not available on the real hand."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class domain_randomization_params(ManagerTermBase):
    """The physics parameters the domain randomization sampled for each env (DeXtreme-style critic input).

    Layout: cube mass (1) | cube static friction, dynamic friction, restitution (3) |
    joint stiffness scale (n_joints) | joint damping scale (n_joints) | scene gravity (3).

    The per-env values only change on reset, so they are cached and refreshed in reset(), which the observation
    manager calls after the reset events have run. Gravity is global and changes on an interval event, so it is read
    every step.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.params.get("robot_cfg", SceneEntityCfg("robot")).name]
        self.object: RigidObject = env.scene[cfg.params.get("object_cfg", SceneEntityCfg("object")).name]
        n_joints = self.robot.num_joints
        self._per_env = torch.zeros(env.num_envs, 4 + 2 * n_joints, device=env.device)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        cpu_ids = env_ids.cpu()
        mass = self.object.root_physx_view.get_masses()[cpu_ids, :1]  # (n, 1), CPU
        material = self.object.root_physx_view.get_material_properties()[cpu_ids, 0]  # (n, 3), first shape
        stiffness = self.robot.data.joint_stiffness[env_ids] / self.robot.data.default_joint_stiffness[env_ids]
        damping = self.robot.data.joint_damping[env_ids] / self.robot.data.default_joint_damping[env_ids]
        self._per_env[env_ids] = torch.cat(
            [mass.to(self.device), material.to(self.device), stiffness, damping], dim=-1
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        g = sim_utils.SimulationContext.instance().physics_sim_view.get_gravity()
        gravity = torch.tensor([g.x, g.y, g.z], device=self.device).expand(self.num_envs, 3)
        return torch.cat([self._per_env, gravity], dim=-1)
