# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rapid Motor Adaptation (RMA) after HORA (Qi et al., CoRL 2022, arXiv 2210.04887; MIT code github.com/HaozhiQi/hora).

Stage 1 (PPO, ``ActorCriticRMA``): the actor sees the deployable noisy ``policy`` obs (74) plus an 8-dim extrinsics
latent z = tanh(env_mlp(priv)), where ``priv`` = noise-free cube position + sampled DR parameters. The critic is B's
privileged critic. Not deployable (needs ``priv``).

Stage 2 (distillation, ``StudentTeacherRMA`` + ``DistillationRMA``): the stage-1 actor and normalizer are frozen; an
adaptation module (HORA's ``ProprioAdaptTConv``: per-step MLP, three 1D convs over 30 steps, linear -> 8) regresses z
from the last 30 policy observations (MSE on z, as in HORA). Deployable: actor(obs_t, tanh(adapt(history))).

Compared with the transformer student (``transformer_student.py``), the bottleneck is an 8-dim physics latent instead
of a full action / state reconstruction, which is the cheaper end of the history-encoder design space.

rsl_rl builds the policy and the algorithm with ``eval(class_name)`` inside the runner modules, so ``register()``
(called from this package's ``__init__``) puts the classes into those namespaces.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlPpoActorCriticCfg,
)
from rsl_rl.algorithms import Distillation
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP, EmpiricalNormalization

from .transformer_student import POLICY_TERM_DIMS

TOKEN_DIM = sum(POLICY_TERM_DIMS)


class ActorCriticRMA(ActorCritic):
    """Stage 1: ActorCritic whose actor input is [policy obs, tanh(env_mlp(priv obs))]."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        priv_mlp_dims: list[int] = [256, 128, 8],
        actor_hidden_dims: list[int] = [512, 512, 256, 128],
        activation: str = "elu",
        **kwargs,
    ) -> None:
        super().__init__(
            obs, obs_groups, num_actions, actor_hidden_dims=actor_hidden_dims, activation=activation, **kwargs
        )
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_priv = sum(obs[g].shape[-1] for g in obs_groups["priv"])
        self.env_mlp = MLP(num_priv, priv_mlp_dims[-1], priv_mlp_dims[:-1], activation)
        self.priv_obs_normalizer = EmpiricalNormalization(num_priv)
        self.actor = MLP(num_actor_obs + priv_mlp_dims[-1], num_actions, actor_hidden_dims, activation)
        print(f"RMA env MLP: {self.env_mlp}\nRMA actor MLP: {self.actor}")

    def get_priv_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[g] for g in self.obs_groups["priv"]], dim=-1)

    def latent(self, obs: TensorDict) -> torch.Tensor:
        return torch.tanh(self.env_mlp(self.priv_obs_normalizer(self.get_priv_obs(obs))))

    def _actor_input(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([self.actor_obs_normalizer(self.get_actor_obs(obs)), self.latent(obs)], dim=-1)

    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        self._update_distribution(self._actor_input(obs))
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self.actor(self._actor_input(obs))

    def update_normalization(self, obs: TensorDict) -> None:
        super().update_normalization(obs)
        self.priv_obs_normalizer.update(self.get_priv_obs(obs))


class ProprioAdaptTConv(nn.Module):
    """HORA's adaptation module (hora/algo/models/models.py), with our 74-dim per-step token instead of 32."""

    def __init__(self, token_dim: int, latent_dim: int = 8):
        super().__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(token_dim, 32), nn.ReLU(inplace=True), nn.Linear(32, 32), nn.ReLU(inplace=True)
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(32, 32, 9, stride=2), nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, 5, stride=1), nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, 5, stride=1), nn.ReLU(inplace=True),
        )  # 30 steps -> 11 -> 7 -> 3
        self.low_dim_proj = nn.Linear(32 * 3, latent_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # (N, 30, token_dim) -> (N, latent_dim)
        x = self.channel_transform(tokens).permute(0, 2, 1)
        return self.low_dim_proj(self.temporal_aggregation(x).flatten(1))


class StudentTeacherRMA(nn.Module):
    """Stage 2: frozen stage-1 actor + env_mlp (teacher latent) and a trainable adaptation module (``student``).

    Groups: ``policy`` = noisy policy obs with ``history_length = 30`` (term-major flat), ``teacher`` = the ``priv``
    group of stage 1. The current observation is the last step of the history.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        history_length: int = 30,
        actor_hidden_dims: list[int] = [512, 512, 256, 128],
        priv_mlp_dims: list[int] = [256, 128, 8],
        activation: str = "elu",
        init_noise_std: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.obs_groups = obs_groups
        self.history_length = history_length
        self.loaded_teacher = False
        num_student_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        assert num_student_obs == history_length * TOKEN_DIM, f"student obs {num_student_obs} != {history_length} x 74"
        num_priv = sum(obs[g].shape[-1] for g in obs_groups["teacher"])
        latent_dim = priv_mlp_dims[-1]
        # frozen, loaded from the stage-1 PPO checkpoint (same names as in ActorCriticRMA)
        self.actor = MLP(TOKEN_DIM + latent_dim, num_actions, actor_hidden_dims, activation)
        self.actor_obs_normalizer = EmpiricalNormalization(TOKEN_DIM)
        self.env_mlp = MLP(num_priv, latent_dim, priv_mlp_dims[:-1], activation)
        self.priv_obs_normalizer = EmpiricalNormalization(num_priv)
        for m in (self.actor, self.actor_obs_normalizer, self.env_mlp, self.priv_obs_normalizer):
            m.requires_grad_(False)
        # trainable
        self.student = ProprioAdaptTConv(TOKEN_DIM, latent_dim)
        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)), requires_grad=False)
        self.distribution = None
        print(f"RMA adaptation module: {sum(p.numel() for p in self.student.parameters()) / 1e3:.1f} k params")

    # --- observation handling
    def _tokens(self, obs: TensorDict) -> torch.Tensor:
        """Term-major flat history -> normalized (N, H, 74) per-step tokens."""
        flat = torch.cat([obs[g] for g in self.obs_groups["policy"]], dim=-1)
        h, parts, i = self.history_length, [], 0
        for d in POLICY_TERM_DIMS:
            parts.append(flat[:, i:i + h * d].reshape(-1, h, d))
            i += h * d
        return self.actor_obs_normalizer(torch.cat(parts, dim=-1))

    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[g] for g in self.obs_groups["teacher"]], dim=-1)

    def latents(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(tokens, student latent, teacher latent)."""
        tokens = self._tokens(obs)
        z_student = torch.tanh(self.student(tokens))
        with torch.no_grad():
            z_teacher = torch.tanh(self.env_mlp(self.priv_obs_normalizer(self.get_teacher_obs(obs))))
        return tokens, z_student, z_teacher

    # --- rsl_rl StudentTeacher interface
    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        tokens, z, _ = self.latents(obs)
        return self.actor(torch.cat([tokens[:, -1], z], dim=-1))

    def act(self, obs: TensorDict) -> torch.Tensor:
        mean = self.act_inference(obs)
        self.distribution = torch.distributions.Normal(mean, torch.exp(self.log_std).expand_as(mean))
        return self.distribution.sample()

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        with torch.no_grad():
            tokens, _, z_teacher = self.latents(obs)
            return self.actor(torch.cat([tokens[:, -1], z_teacher], dim=-1))

    def reset(self, dones=None, hidden_states=(None, None)) -> None:
        pass

    def get_hidden_states(self):
        return None, None

    def detach_hidden_states(self, dones=None) -> None:
        pass

    def update_normalization(self, obs: TensorDict) -> None:
        pass  # normalizers are the frozen stage-1 statistics

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def train(self, mode: bool = True):
        super().train(mode)
        for m in (self.actor, self.actor_obs_normalizer, self.env_mlp, self.priv_obs_normalizer):
            m.eval()
        return self

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        if any(k.startswith("env_mlp.") for k in state_dict) and not any(k.startswith("student.") for k in state_dict):
            # stage-1 PPO checkpoint: take actor, env_mlp and both normalizers, drop the critic
            keep = ("actor.", "actor_obs_normalizer.", "env_mlp.", "priv_obs_normalizer.")
            super().load_state_dict({k: v for k, v in state_dict.items() if k.startswith(keep)}, strict=False)
            self.loaded_teacher = True
            return False
        super().load_state_dict(state_dict, strict=strict)
        self.loaded_teacher = True
        return True


class DistillationRMA(Distillation):
    """rsl_rl Distillation with HORA's stage-2 loss: MSE between the adaptation latent and the teacher latent."""

    def update(self) -> dict[str, float]:
        self.num_updates += 1
        mean_latent, mean_behavior, cnt, loss = 0.0, 0.0, 0, 0
        for _ in range(self.num_learning_epochs):
            for obs, _, privileged_actions, _ in self.storage.generator():
                tokens, z_student, z_teacher = self.policy.latents(obs)
                latent_loss = nn.functional.mse_loss(z_student, z_teacher)
                with torch.no_grad():
                    actions = self.policy.actor(torch.cat([tokens[:, -1], z_student], dim=-1))
                    mean_behavior += self.loss_fn(actions, privileged_actions).item()
                loss = loss + latent_loss
                mean_latent += latent_loss.item()
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
        return {"latent": mean_latent / cnt, "behavior": mean_behavior / cnt}


@configclass
class RslRlPpoActorCriticRMACfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticRMA"
    priv_mlp_dims: list = [256, 128, 8]


@configclass
class RslRlStudentTeacherRMACfg(RslRlDistillationStudentTeacherCfg):
    class_name: str = "StudentTeacherRMA"
    history_length: int = 30
    actor_hidden_dims: list = [512, 512, 256, 128]
    priv_mlp_dims: list = [256, 128, 8]


@configclass
class RslRlDistillationRMAAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    class_name: str = "DistillationRMA"


def register():
    import rsl_rl.runners.distillation_runner as distillation_runner
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    on_policy_runner.ActorCriticRMA = ActorCriticRMA
    distillation_runner.StudentTeacherRMA = StudentTeacherRMA
    distillation_runner.DistillationRMA = DistillationRMA
