import gymnasium as gym

from . import agents

gym.register(
    id="orca-v2-inhand-repose-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.orca_v2_repose_env_cfg:OrcaV2ReposeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfgOrcaV2",
    },
)

# Subtask 2 (exteroception): proprio task + cube pose in the policy observation (OBS_DIM 81).
gym.register(
    id="orca-v2-inhand-repose-cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.orca_v2_repose_cube_env_cfg:OrcaV2ReposeCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfgOrcaV2Cube",
    },
)

# Ablations of the proprio task (same policy observation, same PPO cfg, same log folder): see
# orca_v2_repose_ablation_env_cfg.py.
for _task_id, _cfg in (
    ("orca-v2-inhand-repose-asym-v0", "OrcaV2ReposeAsymEnvCfg"),
    ("orca-v2-inhand-repose-asym-dr-v0", "OrcaV2ReposeAsymDREnvCfg"),
    ("orca-v2-inhand-repose-nodr-v0", "OrcaV2ReposeNoDREnvCfg"),
    ("orca-v2-inhand-repose-delay-v0", "OrcaV2ReposeDelayEnvCfg"),
    ("orca-v2-inhand-repose-jointbias-v0", "OrcaV2ReposeJointBiasEnvCfg"),
    ("orca-v2-inhand-repose-droppen-v0", "OrcaV2ReposeDropPenaltyEnvCfg"),
    ("orca-v2-inhand-repose-hist3-v0", "OrcaV2ReposeHist3EnvCfg"),
    ("orca-v2-inhand-repose-handgrav-v0", "OrcaV2ReposeHandGravityEnvCfg"),
    ("orca-v2-inhand-repose-wristfix-v0", "OrcaV2ReposeWristFixedEnvCfg"),
):
    gym.register(
        id=_task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.orca_v2_repose_ablation_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfgOrcaV2",
        },
    )

# Privileged teacher (sim only): same env + a noise-free "teacher" obs group read by actor and critic.
gym.register(
    id="orca-v2-inhand-repose-teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.orca_v2_repose_ablation_env_cfg:OrcaV2ReposeTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_teacher_student_cfg:PPORunnerCfgOrcaV2Teacher",
    },
)

# Recurrent (LSTM) actor-critic on the baseline env.
gym.register(
    id="orca-v2-inhand-repose-lstm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.orca_v2_repose_env_cfg:OrcaV2ReposeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_teacher_student_cfg:PPORunnerCfgOrcaV2LSTM",
    },
)
