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
