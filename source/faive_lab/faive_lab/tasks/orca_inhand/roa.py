"""Regularized Online Adaptation (ROA), ported from the extreme-parkour-style rsl_rl in ~/go2-flyball-parkour.

Reference (Cheng et al., "Extreme Parkour with Legged Robots", 2023; ROA from Fu, Cheng, Pathak, "Deep Whole-Body
Control", CoRL 2022), files in ~/go2-flyball-parkour:
  rsl_rl/rsl_rl/modules/actor_critic.py   StateHistoryEncoder (l. 41-86), Actor (l. 88-207: priv_encoder l. 110-121,
                                          history_encoder l. 123, backbone l. 142-157, infer_priv_latent l. 197-199,
                                          infer_hist_latent l. 201-203), ActorCriticRMA (l. 209-)
  rsl_rl/rsl_rl/algorithms/ppo.py         priv_reg_loss in the PPO loss (l. 229-235, 288-291), hist_encoder_optimizer
                                          (l. 127), update_dagger (l. 322-348), update_counter (l. 400-401)
  rsl_rl/rsl_rl/runners/on_policy_runner.py  hist_encoding = it % dagger_update_freq == 0 (l. 209), rollout with it
                                          (l. 219), update() then update_dagger() (l. 259-261)
  legged_gym/legged_gym/envs/base/legged_robot_config.py  priv_encoder_dims [64, 20] (l. 621), dagger_update_freq 20
                                          (l. 648), priv_reg_coef_schedual [0, 0.1, 5000, 3000] (l. 649)

Mapping onto rsl_rl 3.1.2 (installed here; its runner has no hist_encoding / dagger hooks):
  - ``policy`` obs = our 74 noisy policy terms with ``history_length`` H = 20 (Isaac Lab term-major flat history);
    the current proprio is the last step, the history encoder sees all H steps (reference: ``num_prop`` + ``num_hist``).
  - ``priv`` obs = roa_obs.roa_dr_params (reference: ``num_priv_latent``); no ``priv_explicit`` / scan / estimator
    (num_priv_explicit = 0, num_scan = 0).
  - Actor backbone on [proprio_t, latent], latent = priv_encoder(priv) (training) or history_encoder(history) (DAgger
    rollouts, deployment). Encoders exactly as in the reference (ELU after every layer, no tanh).
  - PPOROA.update(): rsl_rl 3.1.2's PPO update without RND / symmetry (unused here), plus the reference's
    ``priv_reg_coef * ||z_priv - sg(z_hist)||_2`` term; on DAgger iterations (every ``dagger_update_freq``-th rollout,
    which then also *acted* with the history latent) followed by ``update_dagger()`` = the reference's history-encoder
    regression ``||sg(z_priv) - z_hist||_2`` with its own Adam. The counter that drives the coefficient schedule
    increments in both, as in the reference.
  - Kept from our pipeline: rsl_rl's empirical normalizer on the 74-dim per-step proprio (the reference scales obs in
    the env), rsl_rl's Gaussian with log-std (reference: plain std).
rsl_rl builds the classes with ``eval(class_name)`` inside the runner module: ``register()`` puts them there.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP, EmpiricalNormalization

from .transformer_student import POLICY_TERM_DIMS

TOKEN_DIM = sum(POLICY_TERM_DIMS)


class StateHistoryEncoder(nn.Module):
    """Reference actor_critic.py l. 41-86, unchanged (channel_size 10, tsteps 10 / 20 / 50)."""

    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        super().__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps
        channel_size = 10
        self.encoder = nn.Sequential(nn.Linear(input_size, 3 * channel_size), self.activation_fn)
        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(3 * channel_size, 2 * channel_size, kernel_size=8, stride=4), self.activation_fn,
                nn.Conv1d(2 * channel_size, channel_size, kernel_size=5, stride=1), self.activation_fn,
                nn.Conv1d(channel_size, channel_size, kernel_size=5, stride=1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(3 * channel_size, 2 * channel_size, kernel_size=4, stride=2), self.activation_fn,
                nn.Conv1d(2 * channel_size, channel_size, kernel_size=2, stride=1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(3 * channel_size, 2 * channel_size, kernel_size=6, stride=2), self.activation_fn,
                nn.Conv1d(2 * channel_size, channel_size, kernel_size=4, stride=2), self.activation_fn,
                nn.Flatten())
        else:
            raise ValueError("tsteps must be 10, 20 or 50")
        self.linear_output = nn.Sequential(nn.Linear(channel_size * 3, output_size), self.activation_fn)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # (nd, T, n_proprio) -> (nd, output_size)
        nd = obs.shape[0]
        projection = self.encoder(obs.reshape([nd * self.tsteps, -1]))
        output = self.conv_layers(projection.reshape([nd, self.tsteps, -1]).permute((0, 2, 1)))
        return self.linear_output(output)


def mlp_with_final_activation(in_dim: int, dims: list[int], activation: nn.Module) -> nn.Sequential:
    """Reference priv_encoder (actor_critic.py l. 110-121): Linear + activation for every layer, the last included."""
    layers, d = [], in_dim
    for h in dims:
        layers += [nn.Linear(d, h), activation]
        d = h
    return nn.Sequential(*layers)


def term_major_to_tokens(flat: torch.Tensor, history_length: int) -> torch.Tensor:
    """Isaac Lab's term-major flat history -> (N, H, 74) per-step tokens, oldest first."""
    parts, i = [], 0
    for d in POLICY_TERM_DIMS:
        parts.append(flat[:, i:i + history_length * d].reshape(-1, history_length, d))
        i += history_length * d
    return torch.cat(parts, dim=-1)


class ActorCriticROA(ActorCritic):
    """rsl_rl ActorCritic with the reference Actor: backbone([proprio_t, z]), z = priv_encoder(priv) or
    history_encoder(H steps of proprio). Critic: rsl_rl's MLP on the ``critic`` set (our privileged critic + priv)."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        history_length: int = 20,
        priv_encoder_dims: list[int] = [64, 20],
        actor_hidden_dims: list[int] = [512, 512, 256, 128],
        activation: str = "elu",
        tanh_encoder_output: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(obs, obs_groups, num_actions, actor_hidden_dims=actor_hidden_dims, activation=activation,
                         **kwargs)
        num_hist = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        if num_hist != history_length * TOKEN_DIM:
            raise ValueError(f"policy obs {num_hist} != history_length {history_length} x {TOKEN_DIM}")
        num_priv = sum(obs[g].shape[-1] for g in obs_groups["priv"])
        act_fn = nn.ELU() if activation == "elu" else nn.ReLU()
        self.history_length = history_length
        self.priv_encoder = mlp_with_final_activation(num_priv, priv_encoder_dims, act_fn)
        latent_dim = priv_encoder_dims[-1] if priv_encoder_dims else num_priv
        self.history_encoder = StateHistoryEncoder(act_fn, TOKEN_DIM, history_length, latent_dim)
        # the base class sized actor / normalizer for the whole history; the backbone sees one step + the latent
        self.actor = MLP(TOKEN_DIM + latent_dim, num_actions, actor_hidden_dims, activation)
        if tanh_encoder_output:  # reference: tanh on the backbone output
            self.actor = nn.Sequential(*self.actor, nn.Tanh())
        self.actor_obs_normalizer = EmpiricalNormalization(TOKEN_DIM) if self.actor_obs_normalization else nn.Identity()
        self.hist_encoding = False  # set by PPOROA for the rollout of a DAgger iteration
        print(f"ROA backbone: {self.actor}\nROA priv_encoder: {self.priv_encoder}\n"
              f"ROA history_encoder: {sum(p.numel() for p in self.history_encoder.parameters()) / 1e3:.1f} k params")

    # --- reference Actor methods
    def tokens(self, obs: TensorDict) -> torch.Tensor:
        flat = torch.cat([obs[g] for g in self.obs_groups["policy"]], dim=-1)
        return self.actor_obs_normalizer(term_major_to_tokens(flat, self.history_length))

    def infer_priv_latent(self, obs: TensorDict) -> torch.Tensor:
        return self.priv_encoder(torch.cat([obs[g] for g in self.obs_groups["priv"]], dim=-1))

    def infer_hist_latent(self, obs: TensorDict) -> torch.Tensor:
        return self.history_encoder(self.tokens(obs))

    def backbone(self, obs: TensorDict, hist_encoding: bool) -> torch.Tensor:
        tokens = self.tokens(obs)
        latent = self.history_encoder(tokens) if hist_encoding else self.infer_priv_latent(obs)
        return self.actor(torch.cat([tokens[:, -1], latent], dim=-1))

    # --- rsl_rl ActorCritic interface
    def act(self, obs: TensorDict, hist_encoding: bool | None = None, **kwargs) -> torch.Tensor:
        h = self.hist_encoding if hist_encoding is None else hist_encoding
        mean = self.backbone(obs, h)
        std = torch.exp(self.log_std) if self.noise_std_type == "log" else self.std
        self.distribution = torch.distributions.Normal(mean, std.expand_as(mean))
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """Deployment: history latent (reference play: hist_encoding=True)."""
        return self.backbone(obs, hist_encoding=True)

    def act_inference_privileged(self, obs: TensorDict) -> torch.Tensor:
        return self.backbone(obs, hist_encoding=False)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            flat = torch.cat([obs[g] for g in self.obs_groups["policy"]], dim=-1)
            self.actor_obs_normalizer.update(term_major_to_tokens(flat, self.history_length)[:, -1])
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))


class PPOROA(PPO):
    """rsl_rl 3.1.2 PPO with the reference's ROA terms (see module docstring)."""

    def __init__(self, policy, *args, dagger_update_freq: int = 20,
                 priv_reg_coef_schedual: list[float] = [0.0, 0.1, 2500, 2000], **kwargs):
        super().__init__(policy, *args, **kwargs)
        if self.rnd is not None or self.symmetry is not None:
            raise NotImplementedError("PPOROA: RND / symmetry are not ported")
        self.dagger_update_freq = dagger_update_freq
        self.priv_reg_coef_schedual = list(priv_reg_coef_schedual)
        self.counter = 0  # reference update_counter(): +1 per update() and per update_dagger()
        self.rollout_it = 0  # runner iteration (reference: hist_encoding = it % dagger_update_freq == 0)
        self.hist_encoder_optimizer = torch.optim.Adam(self.policy.history_encoder.parameters(), lr=self.learning_rate)
        self.policy.hist_encoding = True  # rollout 0 is a DAgger rollout, as in the reference (0 % 20 == 0)

    def priv_reg_coef(self) -> float:
        start, end, start_it, ramp = self.priv_reg_coef_schedual
        stage = min(max(self.counter - start_it, 0) / ramp, 1)
        return stage * (end - start) + start

    def update(self) -> dict[str, float]:
        dagger_iteration = self.policy.hist_encoding
        mean_value_loss = mean_surrogate_loss = mean_entropy = mean_priv_reg = 0.0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        priv_reg_coef = self.priv_reg_coef()
        for (obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch,
             old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, _hidden, masks_batch) in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
            # reference ppo.py l. 220: recompute with the privileged latent (hist_encoding=False)
            self.policy.act(obs_batch, hist_encoding=False)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch)
            mu_batch, sigma_batch = self.policy.action_mean, self.policy.action_std
            entropy_batch = self.policy.entropy
            # reference ppo.py l. 229-235
            priv_latent_batch = self.policy.infer_priv_latent(obs_batch)
            with torch.no_grad():
                hist_latent_batch = self.policy.infer_hist_latent(obs_batch)
            priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()

            if self.desired_kl is not None and self.schedule == "adaptive":  # rsl_rl 3.1.2 adaptive lr
                with torch.inference_mode():
                    kl = torch.sum(torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                                   + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                                   / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif self.desired_kl / 2.0 > kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_loss = torch.max((value_batch - returns_batch).pow(2),
                                       (value_clipped - returns_batch).pow(2)).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            # reference ppo.py l. 288-291
            loss = (surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
                    + priv_reg_coef * priv_reg_loss)

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_priv_reg += priv_reg_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        loss_dict = {"value_function": mean_value_loss / num_updates, "surrogate": mean_surrogate_loss / num_updates,
                     "entropy": mean_entropy / num_updates, "priv_reg": mean_priv_reg / num_updates,
                     "priv_reg_coef": priv_reg_coef}
        self.counter += 1  # reference update_counter() at the end of update()
        if dagger_iteration:  # reference runner l. 259-261
            loss_dict["hist_latent"] = self.update_dagger()
        self.storage.clear()
        self.rollout_it += 1
        self.policy.hist_encoding = self.rollout_it % self.dagger_update_freq == 0
        return loss_dict

    def update_dagger(self) -> float:
        """Reference ppo.py l. 322-348: regress the history latent on the (detached) privileged latent."""
        mean_hist_latent_loss = 0.0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            obs_batch = batch[0]
            with torch.no_grad():
                priv_latent_batch = self.policy.infer_priv_latent(obs_batch)
            hist_latent_batch = self.policy.infer_hist_latent(obs_batch)
            hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
            self.hist_encoder_optimizer.zero_grad()
            hist_latent_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.history_encoder.parameters(), self.max_grad_norm)
            self.hist_encoder_optimizer.step()
            mean_hist_latent_loss += hist_latent_loss.item()
        self.counter += 1  # reference update_counter() at the end of update_dagger()
        return mean_hist_latent_loss / (self.num_learning_epochs * self.num_mini_batches)


def register():
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    on_policy_runner.ActorCriticROA = ActorCriticROA
    on_policy_runner.PPOROA = PPOROA
