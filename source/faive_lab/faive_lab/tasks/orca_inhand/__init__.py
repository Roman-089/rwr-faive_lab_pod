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
    ("orca-v2-inhand-repose-asym-nodr-v0", "OrcaV2ReposeAsymNoDREnvCfg"),
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
    ("orca-v2-inhand-repose-asym-rotlin-v0", "OrcaV2ReposeAsymRotLinEnvCfg"),
    ("orca-v2-inhand-repose-asym-rotsoft-v0", "OrcaV2ReposeAsymRotSoftEnvCfg"),
    ("orca-v2-inhand-repose-asym-relact-v0", "OrcaV2ReposeAsymRelActEnvCfg"),
    ("orca-v2-inhand-repose-asym-rotlin-relact-v0", "OrcaV2ReposeAsymRotLinRelActEnvCfg"),
    ("orca-v2-inhand-repose-asym-rotlin-hist3-v0", "OrcaV2ReposeAsymRotLinHist3EnvCfg"),
    ("orca-v2-inhand-repose-asym-rotlin-work-v0", "OrcaV2ReposeAsymRotLinWorkEnvCfg"),
    ("orca-v2-inhand-repose-asym-rotlin-droppen-v0", "OrcaV2ReposeAsymRotLinDropPenEnvCfg"),
    ("orca-v2-inhand-repose-asym-acthold-v0", "OrcaV2ReposeAsymActHoldEnvCfg"),
    ("orca-v2-inhand-repose-asym-actnet-v0", "OrcaV2ReposeAsymActNetEnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-v0", "OrcaV2ReposeAsymOneDirEnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-v0", "OrcaV2ReposeAsymOneDirDropPenEnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-ramp-v0", "OrcaV2ReposeAsymOneDirDropPenRampEnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-work-v0", "OrcaV2ReposeAsymOneDirDropPenWorkEnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-wristpen-v0", "OrcaV2ReposeAsymOneDirDropPenWristPenEnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-wristfix-v0", "OrcaV2ReposeAsymOneDirDropPenWristFixEnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-v0", "OrcaV2ReposeAsymOneDirDropPenUprightEnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-wristfix-v0", "OrcaV2ReposeAsymOneDirDropPenUprightWristFixEnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-v0", "OrcaV2ReposeAsymOneDirDropPenUprightWristPenEnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-adr-v0", "OrcaV2ReposeAsymOneDirDropPenADREnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-adr-v0", "OrcaV2ReposeAsymOneDirDropPenUprightADREnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-adr-v0",
     "OrcaV2ReposeAsymOneDirDropPenUprightWristPenADREnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-servo-v0", "OrcaV2ReposeAsymOneDirDropPenUprightServoEnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-servo-wristpen-v0",
     "OrcaV2ReposeAsymOneDirDropPenUprightServoWristPenEnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-servo-wristpen-realfric-v0",
     "OrcaV2ReposeAsymOneDirDropPenUprightServoWristPenRealFrictionEnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-v0",
     "OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionEnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric2-v0",
     "OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFriction2EnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-region-v0",
     "OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionRegionEnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-region200-v0",
     "OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionRegion200EnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-region200c-v0",
     "OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionRegion200CEnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-region200cs-v0",
     "OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionRegion200CSEnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-mount10-v0",
     "OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionMount10EnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-hist3-v0", "OrcaV2ReposeAsymOneDirDropPenUprightHist3EnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-v0",
     "OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1EnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-hist3-v0",
     "OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1Hist3EnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-hist3-servofast-v0",
     "OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1Hist3ServoFastEnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-servofast-v0",
     "OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1ServoFastEnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-hist3-v0",
     "OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionHist3EnvCfg"),
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
from .student_recurrent import register as _register_student_recurrent  # noqa: E402

_register_student_recurrent()

# Students distilled from the teacher (pass the teacher with --load_run / --checkpoint).
for _task_id, _env_cfg, _runner_cfg in (
    ("orca-v2-inhand-repose-student-mlp-h10-v0", "OrcaV2ReposeTeacherHist10EnvCfg", "DistillationRunnerCfgOrcaV2"),
    ("orca-v2-inhand-repose-student-lstm-v0", "OrcaV2ReposeTeacherEnvCfg", "DistillationRunnerCfgOrcaV2LSTM"),
    ("orca-v2-inhand-repose-student-pt-v0", "OrcaV2ReposeTeacherHist10EnvCfg", "DistillationRunnerCfgOrcaV2Transformer"),
    # the same pipeline on the -rotlin reward (teacher: PPO on the privileged group, as -teacher-v0)
    ("orca-v2-inhand-repose-teacher-rotlin-v0", "OrcaV2ReposeTeacherRotLinEnvCfg", "PPORunnerCfgOrcaV2Teacher"),
    ("orca-v2-inhand-repose-student-pt-rotlin-v0", "OrcaV2ReposeTeacherRotLinHist10EnvCfg",
     "DistillationRunnerCfgOrcaV2Transformer"),
    ("orca-v2-inhand-repose-student-mlp-h10-rotlin-v0", "OrcaV2ReposeTeacherRotLinHist10EnvCfg",
     "DistillationRunnerCfgOrcaV2"),
    ("orca-v2-inhand-repose-student-lstm-rotlin-v0", "OrcaV2ReposeTeacherRotLinEnvCfg", "DistillationRunnerCfgOrcaV2LSTM"),
    # and on the demo reward (-onedir + -droppen; the eval commands +z only for "-onedir" tasks)
    ("orca-v2-inhand-repose-teacher-onedir-droppen-v0", "OrcaV2ReposeTeacherOneDirDropPenEnvCfg", "PPORunnerCfgOrcaV2Teacher"),
    ("orca-v2-inhand-repose-student-pt-onedir-droppen-v0", "OrcaV2ReposeTeacherOneDirDropPenHist10EnvCfg",
     "DistillationRunnerCfgOrcaV2Transformer"),
    ("orca-v2-inhand-repose-student-mlp-h10-onedir-droppen-v0", "OrcaV2ReposeTeacherOneDirDropPenHist10EnvCfg",
     "DistillationRunnerCfgOrcaV2"),
    ("orca-v2-inhand-repose-student-lstm-onedir-droppen-v0", "OrcaV2ReposeTeacherOneDirDropPenEnvCfg",
     "DistillationRunnerCfgOrcaV2LSTM"),
    # demo reward + wrist velocity penalty
    ("orca-v2-inhand-repose-teacher-onedir-droppen-wristpen-v0", "OrcaV2ReposeTeacherOneDirDropPenWristPenEnvCfg",
     "PPORunnerCfgOrcaV2Teacher"),
    ("orca-v2-inhand-repose-student-pt-onedir-droppen-wristpen-v0", "OrcaV2ReposeTeacherOneDirDropPenWristPenHist10EnvCfg",
     "DistillationRunnerCfgOrcaV2Transformer"),
    # demo reward + cube tilt penalty (cube position x3 via Hydra override)
    ("orca-v2-inhand-repose-teacher-onedir-droppen-upright-v0", "OrcaV2ReposeTeacherOneDirDropPenUprightEnvCfg",
     "PPORunnerCfgOrcaV2Teacher"),
    ("orca-v2-inhand-repose-student-pt-onedir-droppen-upright-v0", "OrcaV2ReposeTeacherOneDirDropPenUprightHist10EnvCfg",
     "DistillationRunnerCfgOrcaV2Transformer"),
    # demo reward + cube tilt penalty + wrist not actuated
    ("orca-v2-inhand-repose-teacher-onedir-droppen-upright-wristfix-v0", "OrcaV2ReposeTeacherOneDirDropPenUprightWristFixEnvCfg",
     "PPORunnerCfgOrcaV2Teacher"),
    ("orca-v2-inhand-repose-student-pt-onedir-droppen-upright-wristfix-v0",
     "OrcaV2ReposeTeacherOneDirDropPenUprightWristFixHist10EnvCfg", "DistillationRunnerCfgOrcaV2Transformer"),
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

# PPO fine-tuning of the distilled MLP-h10 student (student_finetune.py): pass the student with --resume True
# --load_run <student run> --checkpoint model_X.pt; the teacher comes from agent.policy.teacher_checkpoint.
from .student_finetune import register as _register_student_finetune  # noqa: E402

_register_student_finetune()

for _task_id, _runner_cfg in (
    ("orca-v2-inhand-repose-student-mlp-h10-ft-v0", "PPORunnerCfgOrcaV2StudentFinetune"),
    ("orca-v2-inhand-repose-student-mlp-h10-res-v0", "PPORunnerCfgOrcaV2StudentResidual"),
):
    gym.register(
        id=_task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.orca_v2_repose_ablation_env_cfg:OrcaV2ReposeTeacherHist10EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_teacher_student_cfg:{_runner_cfg}",
        },
    )

# Regularized Online Adaptation (ROA), ported from the extreme-parkour rsl_rl in ~/go2-flyball-parkour (roa.py,
# docs/ablation/roa.md): one-stage privileged encoder + history encoder with the priv_reg / DAgger terms.
from .roa import register as _register_roa  # noqa: E402

_register_roa()

# Recurrent actor-critic (rsl_rl ActorCriticRecurrent, LSTM 256 + MLP heads, PPORunnerCfgOrcaV2LSTM) on the staged path of
# the shipped line: 06 base (-onedir-droppen, tracking 45) -> calm (-wristl1) -> servo hardening (-servofast).
for _task_id, _cfg in (
    ("orca-v2-inhand-repose-asym-onedir-droppen-lstm-v0", "OrcaV2ReposeAsymOneDirDropPenEnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-lstm-v0",
     "OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1EnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-servofast-lstm-v0",
     "OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1ServoFastEnvCfg"),
):
    gym.register(
        id=_task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.orca_v2_repose_ablation_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_teacher_student_cfg:PPORunnerCfgOrcaV2LSTM",
        },
    )

for _task_id, _cfg in (
    ("orca-v2-inhand-repose-asym-onedir-droppen-roa-v0", "OrcaV2ReposeAsymOneDirDropPenROAEnvCfg"),
    ("orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-roa-v0",
     "OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1ROAEnvCfg"),
):
    gym.register(
        id=_task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.orca_v2_repose_ablation_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_roa_cfg:PPORunnerCfgOrcaV2ROA",
        },
    )

gym.register(
    id="orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-servofast-roa-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.orca_v2_repose_ablation_env_cfg:"
        "OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1ServoFastROAEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_roa_cfg:PPORunnerCfgOrcaV2ROA",
    },
)
