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
    ("orca-v2-inhand-repose-hist10-v0", "OrcaV2ReposeHist10EnvCfg"),
    ("orca-v2-inhand-repose-asym-hist3-v0", "OrcaV2ReposeAsymHist3EnvCfg"),
    ("orca-v2-inhand-repose-asym-hist10-v0", "OrcaV2ReposeAsymHist10EnvCfg"),
    ("orca-v2-inhand-repose-handgrav-v0", "OrcaV2ReposeHandGravityEnvCfg"),
    ("orca-v2-inhand-repose-wristfix-v0", "OrcaV2ReposeWristFixedEnvCfg"),
    ("orca-v2-inhand-repose-asym-droppen-v0", "OrcaV2ReposeAsymDropPenaltyEnvCfg"),
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

# Transformer student classes must be visible to rsl_rl's DistillationRunner (it resolves class names with eval).
from .transformer_student import register as _register_transformer_student  # noqa: E402

_register_transformer_student()

# Students distilled from the teacher (pass the teacher with --load_run / --checkpoint).
for _task_id, _env_cfg, _runner_cfg in (
    ("orca-v2-inhand-repose-student-mlp-h10-v0", "OrcaV2ReposeTeacherHist10EnvCfg", "DistillationRunnerCfgOrcaV2"),
    ("orca-v2-inhand-repose-student-lstm-v0", "OrcaV2ReposeTeacherEnvCfg", "DistillationRunnerCfgOrcaV2LSTM"),
    ("orca-v2-inhand-repose-student-pt-v0", "OrcaV2ReposeTeacherHist10EnvCfg", "DistillationRunnerCfgOrcaV2Transformer"),
):
    gym.register(
        id=_task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.orca_v2_repose_ablation_env_cfg:{_env_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_teacher_student_cfg:{_runner_cfg}",
        },
    )

# LSTM actor-critic on top of ablation B (asymmetric critic).
gym.register(
    id="orca-v2-inhand-repose-asym-lstm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.orca_v2_repose_ablation_env_cfg:OrcaV2ReposeAsymEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_teacher_student_cfg:PPORunnerCfgOrcaV2LSTM",
    },
)

# Rapid Motor Adaptation (HORA): stage 1 PPO with an extrinsics encoder, stage 2 adaptation module (see rma.py).
from .rma import register as _register_rma  # noqa: E402

_register_rma()

for _task_id, _env_cfg, _runner_cfg in (
    ("orca-v2-inhand-repose-rma-teacher-v0", "OrcaV2ReposeRMATeacherEnvCfg", "PPORunnerCfgOrcaV2RMA"),
    ("orca-v2-inhand-repose-rma-student-v0", "OrcaV2ReposeRMAStudentEnvCfg", "DistillationRunnerCfgOrcaV2RMA"),
):
    gym.register(
        id=_task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.orca_v2_repose_ablation_env_cfg:{_env_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_teacher_student_cfg:{_runner_cfg}",
        },
    )
