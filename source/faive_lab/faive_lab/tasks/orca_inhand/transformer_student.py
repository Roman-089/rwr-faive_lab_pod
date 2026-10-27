# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Transformer student with a reconstruction loss, after the ORCA Proprioceptive Transformer (arXiv 2605.21330).

Student input: the env's noisy ``policy`` group with ``history_length = H``. Isaac Lab flattens the history per term
(term 1 steps t-H+1..t, term 2 steps ..., ...), so the flat vector is cut back into H per-step tokens of all terms.
Each token is linearly projected to ``d_model``, a sinusoidal position embedding is added, a learnable [CLS] token is
prepended, and a transformer encoder runs over the sequence. The [CLS] output feeds
    - the action head (MLP), trained by behavior cloning on the teacher's actions, and
    - a reconstruction head predicting the privileged state from the ``teacher`` group: noise-free joint positions,
      noise-free joint velocities and the cube position (paper weights 0.3 / 0.2 / 0.5).

rsl_rl builds the policy and the algorithm with ``eval(class_name)`` inside ``rsl_rl.runners.distillation_runner``, so
``register()`` (called from this package's ``__init__``) puts the two classes into that module's namespace.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from tensordict import TensorDict

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlDistillationAlgorithmCfg, RslRlDistillationStudentTeacherCfg
from rsl_rl.algorithms import Distillation
from rsl_rl.modules import StudentTeacher
from rsl_rl.networks import MLP

# Layout of the 74-dim policy obs (orca_v2_repose_env_cfg.KinematicObsGroupCfg):
# joint_pos 17 | joint_vel 17 | goal_command 6 | last_action 17 | last_joint_pos_cmd 17
POLICY_TERM_DIMS = (17, 17, 6, 17, 17)
# Reconstruction targets = the first 37 dims of the 81-dim teacher obs: joint_pos 17 | joint_vel 17 | object_pos 3
RECON_SLICES = {"joint_pos": (0, 17), "joint_vel": (17, 34), "object_pos": (34, 37)}


class StudentTeacherTransformer(StudentTeacher):
    """StudentTeacher whose student is a transformer over the observation history (+ reconstruction head)."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        history_length: int = 10,
        d_model: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        self.history_length = history_length
        num_student_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        assert num_student_obs == history_length * sum(POLICY_TERM_DIMS), (
            f"student obs {num_student_obs} != history {history_length} x {sum(POLICY_TERM_DIMS)}"
        )
        self.student = _TransformerStudent(
            sum(POLICY_TERM_DIMS), num_actions, history_length, d_model, num_layers, num_heads, dim_feedforward,
            recon_dim=RECON_SLICES["object_pos"][1],
        )
        print(f"Student transformer: {sum(p.numel() for p in self.student.parameters()) / 1e6:.2f} M params")

    def _tokens(self, flat: torch.Tensor) -> torch.Tensor:
        """(N, H * 74) term-major flat history -> (N, H, 74) per-step tokens."""
        h, parts, i = self.history_length, [], 0
        for d in POLICY_TERM_DIMS:
            parts.append(flat[:, i:i + h * d].reshape(-1, h, d))
            i += h * d
        return torch.cat(parts, dim=-1)

    def _student_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._tokens(self.student_obs_normalizer(self.get_student_obs(obs)))

    def _update_distribution(self, tokens: torch.Tensor) -> None:
        mean, _ = self.student(tokens)
        std = self.std.expand_as(mean) if self.noise_std_type == "scalar" else torch.exp(self.log_std).expand_as(mean)
        self.distribution = torch.distributions.Normal(mean, std)

    def act(self, obs: TensorDict) -> torch.Tensor:
        self._update_distribution(self._student_obs(obs))
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self.student(self._student_obs(obs))[0]

    def act_and_reconstruct(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        return self.student(self._student_obs(obs))


class _TransformerStudent(nn.Module):
    def __init__(self, token_dim, num_actions, history_length, d_model, num_layers, num_heads, dim_feedforward,
                 recon_dim):
        super().__init__()
        self.embed = nn.Linear(token_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        pos = torch.arange(history_length + 1).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(history_length + 1, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pos_emb", pe.unsqueeze(0), persistent=False)
        layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.action_head = MLP(d_model, num_actions, [256, 128], "elu")
        self.recon_head = MLP(d_model, recon_dim, [256], "elu")

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([self.cls.expand(tokens.shape[0], -1, -1), self.embed(tokens)], dim=1) + self.pos_emb
        z = self.norm(self.encoder(x)[:, 0])
        return self.action_head(z), self.recon_head(z)


class DistillationRecon(Distillation):
    """rsl_rl Distillation (student acts, teacher labels) + weighted MSE reconstruction of the privileged state."""

    def __init__(self, policy, recon_weights: dict[str, float] | None = None, **kwargs):
        super().__init__(policy, **kwargs)
        self.recon_weights = recon_weights or {"object_pos": 0.5, "joint_pos": 0.3, "joint_vel": 0.2}

    def update(self) -> dict[str, float]:
        self.num_updates += 1
        mean_behavior, mean_recon, cnt, loss = 0.0, 0.0, 0, 0
        for _ in range(self.num_learning_epochs):
            for obs, _, privileged_actions, _ in self.storage.generator():
                actions, recon = self.policy.act_and_reconstruct(obs)
                behavior_loss = self.loss_fn(actions, privileged_actions)
                target = self.policy.get_teacher_obs(obs)
                recon_loss = sum(
                    w * nn.functional.mse_loss(recon[:, a:b], target[:, a:b])
                    for k, w in self.recon_weights.items()
                    for a, b in [RECON_SLICES[k]]
                )
                loss = loss + behavior_loss + recon_loss
                mean_behavior += behavior_loss.item()
                mean_recon += recon_loss.item()
                cnt += 1
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    loss = 0
        self.storage.clear()
        return {"behavior": mean_behavior / cnt, "reconstruction": mean_recon / cnt}


@configclass
class RslRlStudentTeacherTransformerCfg(RslRlDistillationStudentTeacherCfg):
    class_name: str = "StudentTeacherTransformer"
    history_length: int = 10
    d_model: int = 256
    num_layers: int = 3
    num_heads: int = 4
    dim_feedforward: int = 1024


@configclass
class RslRlDistillationReconAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    class_name: str = "DistillationRecon"
    recon_weights: dict = {"object_pos": 0.5, "joint_pos": 0.3, "joint_vel": 0.2}


def register():
    import rsl_rl.runners.distillation_runner as runner_module

    runner_module.StudentTeacherTransformer = StudentTeacherTransformer
    runner_module.DistillationRecon = DistillationRecon
