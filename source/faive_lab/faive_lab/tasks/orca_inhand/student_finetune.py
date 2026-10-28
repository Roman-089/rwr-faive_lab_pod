# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO fine-tuning of a distilled student (teacher PPO -> Distillation -> PPO). Literature and defaults:
docs/ablation/student_finetune.md.

``ActorCriticFinetune`` (asymmetric):
    actor   = the distilled student (MLP on the 10-step noisy ``policy`` history), loaded from the Distillation
              checkpoint (``student.*`` -> ``actor.*``, ``student_obs_normalizer.*`` -> ``actor_obs_normalizer.*``)
    critic  = the teacher's trained privileged critic on the ``teacher`` group, loaded from the teacher PPO checkpoint
    teacher = the teacher's actor + normalizer, frozen, only for the regularizer (never in the inference path)
    std     = the teacher's final per-joint std (log_std), frozen by default
``ActorCriticResidualFinetune``: the student is frozen and a zero-initialized residual head is trained on top
(ResiP / Policy Decorator); ``actor`` is the combined module, so ``act_inference`` and the JIT/ONNX exporter see
student + residual.

``PPOFinetune`` = rsl_rl PPO with
    - critic warm-up: for ``critic_warmup_iterations`` only the value loss is optimized (actor, std untouched)
    - separate actor / critic LRs; the adaptive KL schedule moves only the actor LR, clamped to [min, max], and the
      actor LR ramps up linearly over ``actor_lr_warmup_iterations`` after the warm-up
    - kickstarting term beta * KL(pi_teacher || pi_student) with the shared std, i.e.
      beta * sum_j (mu_s - mu_t)_j^2 / (2 sigma_j^2), beta linearly annealed to 0 after the warm-up
    - logs: behavior MSE to the teacher (same quantity as Distillation's ``behavior``), explained variance, std, LRs,
      beta, KL

The student checkpoint comes in through the normal ``--resume True --load_run <student run> --checkpoint model_X.pt``
path; ``load_state_dict`` recognizes the Distillation keys and pulls the critic, the teacher actor and the std from
``teacher_checkpoint`` (cfg field, Hydra: ``agent.policy.teacher_checkpoint=...``). A fine-tune checkpoint has
``teacher_actor.*`` keys and loads strictly (resume, eval_policy.py, play.py).

rsl_rl builds the policy and the algorithm with ``eval(class_name)`` inside the runner modules, so ``register()``
(called from this package's ``__init__``) puts the classes into those namespaces.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP, EmpiricalNormalization

# relative to the repo root (train.py / eval_policy.py run from there); 7.4 RPM teacher, see ABLATION_RESULTS.md
DEFAULT_TEACHER_CHECKPOINT = "logs/rsl_rl/orca_v2_repose/2026-10-27_23-55-26_teacher/model_9999.pt"


def _sub_state_dict(state_dict: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


class ActorCriticFinetune(ActorCritic):
    """Asymmetric actor-critic initialized from a distilled student (actor) and its teacher (critic, std, regularizer).

    Groups: ``policy`` = student input (noisy policy obs, history as in distillation), ``critic`` = ``teacher`` =
    the noise-free 81-dim teacher group.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        teacher_checkpoint: str = DEFAULT_TEACHER_CHECKPOINT,
        teacher_hidden_dims: list[int] = [512, 512, 256, 128],
        init_std_from_teacher: bool = True,
        init_std_scale: float = 1.0,
        learn_std: bool = False,
        std_min: float = 0.05,
        std_max: float = 1.0,
        update_actor_obs_normalization: bool = False,
        **kwargs,
    ) -> None:
        assert not kwargs.get("state_dependent_std", False), "state-dependent std is not supported"
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        self.teacher_checkpoint = teacher_checkpoint
        self.init_std_from_teacher = init_std_from_teacher
        self.init_std_scale = init_std_scale
        self.learn_std = learn_std
        self.std_min, self.std_max = std_min, std_max
        self.update_actor_obs_normalization = update_actor_obs_normalization
        # frozen teacher actor for the regularizer (same names as the teacher's ActorCritic, prefixed)
        num_teacher_obs = sum(obs[g].shape[-1] for g in obs_groups["teacher"])
        self.teacher_actor = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, kwargs.get("activation", "elu"))
        self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        self.teacher_actor.requires_grad_(False)
        self.teacher_obs_normalizer.requires_grad_(False)
        self._std_param().requires_grad_(learn_std)
        # completed PPO updates; drives warm-up and the beta schedule and survives save / resume
        self.register_buffer("ft_iter", torch.tensor(0, dtype=torch.long))

    # --- parameters
    def _std_param(self) -> nn.Parameter:
        return self.log_std if self.noise_std_type == "log" else self.std

    def _student_actor(self) -> nn.Module:
        """Module that receives the student's ``student.*`` weights."""
        return self.actor

    def parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """(trainable actor-side parameters incl. a learned std, critic parameters)."""
        actor = [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("critic.")]
        return actor, [p for p in self.critic.parameters() if p.requires_grad]

    # --- distribution / teacher
    def _update_distribution(self, obs: torch.Tensor) -> None:
        mean = self.actor(obs)
        std = torch.exp(self.log_std) if self.noise_std_type == "log" else self.std
        if self.learn_std:
            std = std.clamp(self.std_min, self.std_max)
        self.distribution = torch.distributions.Normal(mean, std.expand_as(mean))

    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[g] for g in self.obs_groups["teacher"]], dim=-1)

    def teacher_mean(self, obs: TensorDict) -> torch.Tensor:
        with torch.no_grad():
            return self.teacher_actor(self.teacher_obs_normalizer(self.get_teacher_obs(obs)))

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher_actor.eval()
        self.teacher_obs_normalizer.eval()
        if not self.update_actor_obs_normalization:
            self.actor_obs_normalizer.eval()  # EmpiricalNormalization.update is a no-op in eval mode
        return self

    # --- checkpoints
    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        if any(k.startswith("teacher_actor.") for k in state_dict):
            # own checkpoint: resume, eval_policy.py, play.py
            super().load_state_dict(state_dict, strict=strict)
            return True
        if any(k.startswith("student.") for k in state_dict):
            self._init_from_distillation(state_dict)
            return False  # fresh run: iteration 0, optimizer not loaded
        raise ValueError(
            "ActorCriticFinetune expects a Distillation checkpoint (student.* keys) or its own checkpoint "
            f"(teacher_actor.* keys); got keys like {list(state_dict)[:4]}"
        )

    def _init_from_distillation(self, student_sd: dict) -> None:
        if any(k.startswith("memory_s.") for k in student_sd):
            raise NotImplementedError("recurrent student: see docs/ablation/student_finetune.md (LSTM / transformer)")
        device = self._std_param().device
        # actor <- student
        self._student_actor().load_state_dict(_sub_state_dict(student_sd, "student."))
        norm_sd = _sub_state_dict(student_sd, "student_obs_normalizer.")
        if bool(norm_sd) != self.actor_obs_normalization:
            raise ValueError("actor_obs_normalization must match the student's student_obs_normalization")
        if norm_sd:
            self.actor_obs_normalizer.load_state_dict(norm_sd)
        # critic, teacher actor, std <- teacher PPO checkpoint
        path = os.path.abspath(os.path.expanduser(self.teacher_checkpoint))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"teacher checkpoint {path} not found; set agent.policy.teacher_checkpoint=<path>")
        teacher_sd = torch.load(path, map_location=device, weights_only=False)["model_state_dict"]
        self.critic.load_state_dict(_sub_state_dict(teacher_sd, "critic."))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.load_state_dict(_sub_state_dict(teacher_sd, "critic_obs_normalizer."))
        self.teacher_actor.load_state_dict(_sub_state_dict(teacher_sd, "actor."))
        self.teacher_obs_normalizer.load_state_dict(_sub_state_dict(teacher_sd, "actor_obs_normalizer."))
        # the student checkpoint carries the teacher actor it was distilled from: it must be this teacher
        distilled_from = _sub_state_dict(student_sd, "teacher.")
        if distilled_from:
            diff = max((distilled_from[k].to(device) - v).abs().max().item()
                       for k, v in self.teacher_actor.state_dict().items())
            if diff > 1e-6:
                print(f"[WARNING] student was distilled from a different teacher (max |dw| = {diff:.3g}) than {path}")
        with torch.no_grad():
            if self.init_std_from_teacher:
                teacher_std = torch.exp(teacher_sd["log_std"]) if "log_std" in teacher_sd else teacher_sd["std"]
                std = (teacher_std * self.init_std_scale).to(device)
                if self.noise_std_type == "log":
                    self.log_std.copy_(torch.log(std))
                else:
                    self.std.copy_(std)
            self.ft_iter.zero_()
        print(f"[student fine-tune] actor <- student, critic / teacher actor <- {path}, "
              f"std {self.current_std().mean():.3f} ({'learned' if self.learn_std else 'frozen'})")

    def current_std(self) -> torch.Tensor:
        return (torch.exp(self.log_std) if self.noise_std_type == "log" else self.std).detach()


class ResidualActor(nn.Module):
    """a = base(x) + scale * residual([x, base(x)]); base = frozen student, residual's last layer zero-initialized."""

    def __init__(self, base: nn.Module, num_obs: int, num_actions: int, hidden_dims: list[int], activation: str,
                 scale: float) -> None:
        super().__init__()
        self.base = base.requires_grad_(False)
        self.residual = MLP(num_obs + num_actions, num_actions, hidden_dims, activation)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.base(x)
        return a + self.scale * self.residual(torch.cat([x, a], dim=-1))

    def __getitem__(self, i: int) -> nn.Module:
        return self.base[i]  # the Isaac Lab ONNX exporter sizes its dummy input with actor[0].in_features


class ActorCriticResidualFinetune(ActorCriticFinetune):
    """Fallback: frozen student + small residual head (ResiP: zero-init last layer, action scale 0.1)."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        residual_hidden_dims: list[int] = [256, 128],
        residual_scale: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        self.actor = ResidualActor(self.actor, num_actor_obs, num_actions, residual_hidden_dims,
                                   kwargs.get("activation", "elu"), residual_scale)
        print(f"Residual head: {self.actor.residual}")

    def _student_actor(self) -> nn.Module:
        return self.actor.base


class PPOFinetune(PPO):
    """rsl_rl PPO with critic warm-up, split actor / critic LR and an annealed teacher (kickstarting) term."""

    policy: ActorCriticFinetune

    def __init__(
        self,
        policy: ActorCriticFinetune,
        critic_learning_rate: float = 1e-4,
        min_learning_rate: float = 1e-6,
        max_learning_rate: float = 1e-4,
        critic_warmup_iterations: int = 150,
        actor_lr_warmup_iterations: int = 25,
        teacher_coef: float = 1.0,
        teacher_coef_anneal_iterations: int = 1000,
        teacher_loss_type: str = "kl",
        **kwargs,
    ) -> None:
        super().__init__(policy, **kwargs)
        assert self.rnd is None and self.symmetry is None, "RND / symmetry are not supported by PPOFinetune"
        assert teacher_loss_type in ("kl", "mse")
        self.critic_learning_rate = critic_learning_rate
        self.min_learning_rate, self.max_learning_rate = min_learning_rate, max_learning_rate
        self.critic_warmup_iterations = critic_warmup_iterations
        self.actor_lr_warmup_iterations = actor_lr_warmup_iterations
        self.teacher_coef = teacher_coef
        self.teacher_coef_anneal_iterations = teacher_coef_anneal_iterations
        self.teacher_loss_type = teacher_loss_type
        actor_params, critic_params = policy.parameter_groups()
        # "base_lr" = the adaptive schedule's actor LR, "lr" = base_lr * ramp; both are saved with the optimizer
        self.optimizer = optim.Adam([
            {"params": actor_params, "lr": 0.0, "base_lr": self.learning_rate, "name": "actor"},
            {"params": critic_params, "lr": critic_learning_rate, "name": "critic"},
        ])
        self._trainable = actor_params + critic_params

    def actor_lr_ramp(self, it: int) -> float:
        """0 during the critic warm-up, then linear to 1 over ``actor_lr_warmup_iterations`` (Adam's first steps are
        ~lr * sign(grad) whatever the loss scale, so a cold start at full LR moves the student in one jump)."""
        if it < self.critic_warmup_iterations:
            return 0.0
        if self.actor_lr_warmup_iterations <= 0:
            return 1.0
        return min(1.0, (it - self.critic_warmup_iterations + 1) / self.actor_lr_warmup_iterations)

    def teacher_coef_at(self, it: int) -> float:
        """beta: constant during the warm-up, then linear to 0 over ``teacher_coef_anneal_iterations``."""
        if self.teacher_coef_anneal_iterations <= 0:
            return self.teacher_coef
        t = max(0, it - self.critic_warmup_iterations) / self.teacher_coef_anneal_iterations
        return self.teacher_coef * max(0.0, 1.0 - t)

    def update(self) -> dict[str, float]:
        it = int(self.policy.ft_iter)
        actor_frozen = it < self.critic_warmup_iterations
        beta = self.teacher_coef_at(it)
        ramp = self.actor_lr_ramp(it)
        actor_group = self.optimizer.param_groups[0]
        self.learning_rate = actor_group["base_lr"]  # restored by optimizer.load_state_dict on resume
        actor_group["lr"] = self.learning_rate * ramp
        with torch.no_grad():  # how well the rollout's values explained the returns (before this update)
            values, returns = self.storage.values.flatten(), self.storage.returns.flatten()
            explained_var = (1.0 - torch.var(returns - values) / (torch.var(returns) + 1e-8)).item()

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        stats = {k: 0.0 for k in ("value_function", "surrogate", "entropy", "teacher", "behavior_mse", "kl")}
        for (obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_log_prob_batch,
             old_mu_batch, old_sigma_batch, hidden_states_batch, masks_batch) in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch, sigma_batch = self.policy.action_mean, self.policy.action_std
            entropy_batch = self.policy.entropy

            with torch.inference_mode():
                kl_mean = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    dim=-1,
                ).mean()
                if self.is_multi_gpu:  # same KL, hence the same LR, on every rank
                    torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                    kl_mean /= self.gpu_world_size
            # adaptive actor LR, clamped (rsl_rl's own schedule grows x1.5 up to 1e-2); frozen during the warm-up
            if not actor_frozen and self.desired_kl is not None and self.schedule == "adaptive":
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(self.min_learning_rate, self.learning_rate / 1.5)
                elif 0.0 < kl_mean < self.desired_kl / 2.0:
                    self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.5)
                actor_group["base_lr"], actor_group["lr"] = self.learning_rate, self.learning_rate * ramp

            ratio = torch.exp(log_prob_batch - torch.squeeze(old_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.max(
                    (value_batch - returns_batch).pow(2), (value_clipped - returns_batch).pow(2)
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # teacher term on the student's own states (kickstarting); the teacher reads the privileged group
            diff = mu_batch - self.policy.teacher_mean(obs_batch)
            behavior_mse = diff.pow(2).mean()
            if self.teacher_loss_type == "kl":  # KL(N(mu_t, s) || N(mu_s, s)), s = the shared std
                teacher_loss = (diff.pow(2) / (2.0 * sigma_batch.detach().pow(2))).sum(dim=-1).mean()
            else:
                teacher_loss = behavior_mse

            if actor_frozen:
                loss = self.value_loss_coef * value_loss  # actor and std get no gradient (Adam skips grad=None)
            else:
                loss = (surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
                        + beta * teacher_loss)

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self._trainable, self.max_grad_norm)
            self.optimizer.step()

            stats["value_function"] += value_loss.item()
            stats["surrogate"] += surrogate_loss.item()
            stats["entropy"] += entropy_batch.mean().item()
            stats["teacher"] += teacher_loss.item()
            stats["behavior_mse"] += behavior_mse.item()
            stats["kl"] += kl_mean.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        stats = {k: v / num_updates for k, v in stats.items()}
        self.policy.ft_iter += 1
        self.storage.clear()
        stats.update(
            explained_variance=explained_var,
            teacher_coef=beta,
            actor_frozen=float(actor_frozen),
            actor_lr=actor_group["lr"],
            actor_lr_ramp=ramp,
            critic_lr=self.optimizer.param_groups[1]["lr"],
            std=self.policy.current_std().mean().item(),
        )
        return stats


@configclass
class RslRlPpoActorCriticFinetuneCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticFinetune"
    teacher_checkpoint: str = DEFAULT_TEACHER_CHECKPOINT
    teacher_hidden_dims: list = [512, 512, 256, 128]
    init_std_from_teacher: bool = True
    init_std_scale: float = 1.0
    learn_std: bool = False
    std_min: float = 0.05
    std_max: float = 1.0
    update_actor_obs_normalization: bool = False


@configclass
class RslRlPpoActorCriticResidualFinetuneCfg(RslRlPpoActorCriticFinetuneCfg):
    class_name: str = "ActorCriticResidualFinetune"
    residual_hidden_dims: list = [256, 128]
    residual_scale: float = 0.1


@configclass
class RslRlPpoFinetuneAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "PPOFinetune"
    critic_learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    max_learning_rate: float = 1e-4
    critic_warmup_iterations: int = 150
    actor_lr_warmup_iterations: int = 25
    teacher_coef: float = 1.0
    teacher_coef_anneal_iterations: int = 1000
    teacher_loss_type: str = "kl"


def register():
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    on_policy_runner.ActorCriticFinetune = ActorCriticFinetune
    on_policy_runner.ActorCriticResidualFinetune = ActorCriticResidualFinetune
    on_policy_runner.PPOFinetune = PPOFinetune
