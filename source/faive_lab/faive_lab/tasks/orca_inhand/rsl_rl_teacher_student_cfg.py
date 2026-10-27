# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runner cfgs for the architecture ablations (teacher, recurrent policy, distillation). Kept out of agents/ (owned by
S0); PPO settings are the baseline's."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticRecurrentCfg

from .agents.rsl_rl_ppo_cfg import PPORunnerCfgOrcaV2


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
