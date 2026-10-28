# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runner cfgs for the architecture ablations (teacher, recurrent policy, distillation). Kept out of agents/ (owned by
S0); PPO settings are the baseline's."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlDistillationStudentTeacherRecurrentCfg,
    RslRlPpoActorCriticRecurrentCfg,
)

from .agents.rsl_rl_ppo_cfg import PPORunnerCfgOrcaV2
from .rma import RslRlDistillationRMAAlgorithmCfg, RslRlPpoActorCriticRMACfg, RslRlStudentTeacherRMACfg
from .student_finetune import (
    RslRlPpoActorCriticFinetuneCfg,
    RslRlPpoActorCriticResidualFinetuneCfg,
    RslRlPpoFinetuneAlgorithmCfg,
)
from .transformer_student import RslRlDistillationReconAlgorithmCfg, RslRlStudentTeacherTransformerCfg


@configclass
class PPORunnerCfgOrcaV2Teacher(PPORunnerCfgOrcaV2):
    """Baseline PPO; actor and critic both read the privileged ``teacher`` observation group."""

    obs_groups = {"policy": ["teacher"], "critic": ["teacher"]}


@configclass
class PPORunnerCfgOrcaV2LSTM(PPORunnerCfgOrcaV2):
    """Recurrent actor-critic trained directly with PPO: LSTM (256, 1 layer) in front of each MLP head.

    Same env, rewards and PPO settings as the baseline; truncated BPTT over the 24-step rollout.
    """

    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.5,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )


@configclass
class DistillationRunnerCfgOrcaV2(RslRlDistillationRunnerCfg):
    """Teacher -> student distillation (DAgger-style: the student acts, the frozen teacher labels every state).

    The teacher checkpoint is given with ``--load_run <teacher run dir name> --checkpoint model_X.pt``.
    Student = MLP on the env's noisy ``policy`` group (use the -h10 task id for a 10-step history).
    """

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 250
    experiment_name = "orca_v2_repose"
    # train.py loads the checkpoint only for resume=True or algorithm "Distillation"; DistillationRecon needs resume
    resume = True
    obs_groups = {"policy": ["policy"], "teacher": ["teacher"]}
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type="log",
        student_obs_normalization=True,
        teacher_obs_normalization=True,
        student_hidden_dims=[512, 512, 256, 128],
        teacher_hidden_dims=[512, 512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=5e-4,
        gradient_length=15,
        max_grad_norm=1.0,
    )


@configclass
class DistillationRunnerCfgOrcaV2LSTM(DistillationRunnerCfgOrcaV2):
    """Student = LSTM (256, 1 layer) + MLP on the single noisy ``policy`` observation (paper: 7.0 RPM, motor encoders)."""

    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        init_noise_std=0.1,
        noise_std_type="log",
        student_obs_normalization=True,
        teacher_obs_normalization=True,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 512, 256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        teacher_recurrent=False,
    )


@configclass
class DistillationRunnerCfgOrcaV2Transformer(DistillationRunnerCfgOrcaV2):
    """Student = transformer over a 10-step history + reconstruction loss (PT paper: 9.3 RPM with motor encoders).

    Paper: 3 layers, 4 heads, d_model 256, FF 1024; recon weights 0.5 cube pos / 0.3 joint pos / 0.2 joint vel. The
    paper distils offline with lr 1e-5; this is online (DAgger-style) with a higher lr.
    """

    policy = RslRlStudentTeacherTransformerCfg(
        init_noise_std=0.1,
        noise_std_type="log",
        student_obs_normalization=True,
        teacher_obs_normalization=True,
        student_hidden_dims=[256],  # unused (the transformer replaces the MLP student)
        teacher_hidden_dims=[512, 512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationReconAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1e-4,
        gradient_length=15,
        max_grad_norm=1.0,
        optimizer="adamw",
    )


@configclass
class PPORunnerCfgOrcaV2RMA(PPORunnerCfgOrcaV2):
    """RMA stage 1 (HORA): baseline PPO, actor = [policy obs, tanh(env_mlp(priv))], critic = B's privileged critic."""

    obs_groups = {"policy": ["policy"], "critic": ["critic"], "priv": ["priv"]}
    policy = RslRlPpoActorCriticRMACfg(
        init_noise_std=1.5,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 256, 128],
        activation="elu",
        priv_mlp_dims=[256, 128, 8],
    )


@configclass
class DistillationRunnerCfgOrcaV2RMA(DistillationRunnerCfgOrcaV2):
    """RMA stage 2: train the adaptation module on the stage-1 latent (pass stage 1 with --load_run / --checkpoint)."""

    obs_groups = {"policy": ["policy"], "teacher": ["priv"]}
    policy = RslRlStudentTeacherRMACfg(
        init_noise_std=0.1,
        noise_std_type="log",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 512, 256, 128],
        teacher_hidden_dims=[512, 512, 256, 128],
        activation="elu",
        history_length=30,
        actor_hidden_dims=[512, 512, 256, 128],
        priv_mlp_dims=[256, 128, 8],
    )
    algorithm = RslRlDistillationRMAAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=5e-4,
        gradient_length=15,
        max_grad_norm=1.0,
    )


@configclass
class PPORunnerCfgOrcaV2StudentFinetune(PPORunnerCfgOrcaV2):
    """PPO fine-tuning of the distilled MLP-h10 student (student_finetune.py, docs/ablation/student_finetune.md).

    Pass the student with ``--resume True --load_run <student run dir> --checkpoint model_X.pt``; the critic, the
    teacher actor (regularizer) and the std come from ``policy.teacher_checkpoint``. Actor = student on the 10-step
    noisy history, critic = the teacher's critic on the ``teacher`` group. Deviations from the baseline PPO: clip 0.1,
    entropy 0, actor LR from the teacher's final LR (1e-5) with the adaptive schedule clamped to [1e-6, 1e-4] at half
    the KL target and ramped in over 25 iterations, critic LR 1e-4, 150 critic-only iterations, kickstarting
    beta 1 -> 0 over the next 1000.
    """

    max_iterations = 3000
    save_interval = 50
    resume = True  # train.py loads --load_run / --checkpoint only for resume=True
    obs_groups = {"policy": ["policy"], "critic": ["teacher"], "teacher": ["teacher"]}
    policy = RslRlPpoActorCriticFinetuneCfg(
        init_noise_std=0.3,  # unused: the std is the teacher's (init_std_from_teacher)
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 256, 128],
        activation="elu",
        teacher_hidden_dims=[512, 512, 256, 128],
        init_std_from_teacher=True,
        init_std_scale=1.0,
        learn_std=False,
    )
    algorithm = RslRlPpoFinetuneAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
        critic_learning_rate=1e-4,
        min_learning_rate=1e-6,
        max_learning_rate=1e-4,
        critic_warmup_iterations=150,
        actor_lr_warmup_iterations=25,
        teacher_coef=1.0,
        teacher_coef_anneal_iterations=1000,
        teacher_loss_type="kl",
    )


@configclass
class PPORunnerCfgOrcaV2StudentResidual(PPORunnerCfgOrcaV2StudentFinetune):
    """Fallback: the student stays frozen, PPO trains a zero-initialized residual head (ResiP; scale 0.1) on top.

    Same checkpoint arguments. The actor LR can be higher than for full fine-tuning (ResiP: 3e-4) because the
    residual's output is bounded by ``residual_scale``.
    """

    policy = RslRlPpoActorCriticResidualFinetuneCfg(
        init_noise_std=0.3,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 256, 128],
        activation="elu",
        teacher_hidden_dims=[512, 512, 256, 128],
        init_std_from_teacher=True,
        init_std_scale=1.0,
        learn_std=False,
        residual_hidden_dims=[256, 128],
        residual_scale=0.1,
    )
    algorithm = RslRlPpoFinetuneAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
        critic_learning_rate=1e-4,
        min_learning_rate=1e-6,
        max_learning_rate=3e-4,
        critic_warmup_iterations=150,
        actor_lr_warmup_iterations=25,
        teacher_coef=0.0,  # the residual is bounded and starts at zero: no anchor needed
        teacher_coef_anneal_iterations=0,
        teacher_loss_type="kl",
    )
