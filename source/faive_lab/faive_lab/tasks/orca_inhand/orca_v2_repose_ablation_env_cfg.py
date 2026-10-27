# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ablation variants of the proprio task ``orca-v2-inhand-repose-v0``, selected with ``--task``.

All variants INHERIT ``OrcaV2ReposeEnvCfg`` (S0) and change one thing; the policy observation (74-dim) is unchanged,
so every policy deploys like the baseline. Logs go to the same ``orca_v2_repose/`` folder and W&B project as the
baseline; tell runs apart with ``--run_name``.

  ablation A  orca-v2-inhand-repose-v0               baseline: critic sees the policy observation
  ablation B  orca-v2-inhand-repose-asym-v0          + critic group: noise-free joints + cube pose
  ablation C  orca-v2-inhand-repose-asym-dr-v0       B + cube lin/ang velocity + sampled DR parameters (DeXtreme-style)
  sanity      orca-v2-inhand-repose-nodr-v0          baseline without domain randomization and observation noise
  teacher     orca-v2-inhand-repose-teacher-v0       privileged teacher (actor + critic see teacher obs), for distillation

Sim2real wave (single change each vs the baseline; see HYPERPARAM_NOTES.md):
  delay       orca-v2-inhand-repose-delay-v0         joint targets 50-100 ms late (6-12 substeps, per episode)
  jointbias   orca-v2-inhand-repose-jointbias-v0     per-episode joint-position obs offset U(+-0.05 rad)
  droppen     orca-v2-inhand-repose-droppen-v0       -100 return penalty on a cube drop
  hist3       orca-v2-inhand-repose-hist3-v0         policy obs = last 3 steps (222-dim; deploy needs the history)
  hist10      orca-v2-inhand-repose-hist10-v0        policy obs = last 10 steps (740-dim)
  handgrav    orca-v2-inhand-repose-handgrav-v0      gravity on the hand links
  wristfix    orca-v2-inhand-repose-wristfix-v0      wrist not actuated, held at its default (72-dim obs)

Play / replay a B or C checkpoint with the same task id: the critic's input size differs, so the checkpoint only
loads into the matching variant.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .mdp.privileged_observations import domain_randomization_params
from .mdp.sim2real import DelayedEMAJointPositionToLimitsActionCfg, joint_pos_limit_normalized_biased
from .orca_v2_repose_env_cfg import ObservationsCfg, OrcaV2ReposeEnvCfg


@configclass
class PrivilegedDRCriticObsGroupCfg(ObservationsCfg.TeacherObsGroupCfg):
    """Teacher group + cube velocities + the DR parameters sampled for the env (appended at the end)."""

    object_lin_vel = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("object")})
    object_ang_vel = ObsTerm(func=mdp.root_ang_vel_w, params={"asset_cfg": SceneEntityCfg("object")})
    dr_params = ObsTerm(
        func=domain_randomization_params,
        params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("object")},
    )


@configclass
class OrcaV2ReposeAsymEnvCfg(OrcaV2ReposeEnvCfg):
    """Ablation B: asymmetric critic with the (so far unused) noise-free teacher observations."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.critic = ObservationsCfg.TeacherObsGroupCfg()


@configclass
class OrcaV2ReposeAsymDREnvCfg(OrcaV2ReposeEnvCfg):
    """Ablation C: asymmetric critic that also sees cube velocities and the sampled DR parameters."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.critic = PrivilegedDRCriticObsGroupCfg()


@configclass
class OrcaV2ReposeNoDREnvCfg(OrcaV2ReposeEnvCfg):
    """Sanity check: no physics randomization, no hand-mount randomization, no observation noise.

    Kept: the task's own reset randomization (cube start pose, small joint reset) and the rotation command.
    """

    def __post_init__(self):
        super().__post_init__()
        for term in (
            "robot_physics_material",
            "robot_scale_mass",
            "robot_joint_stiffness_and_damping",
            "robot_joint_pos_limits",
            "object_physics_material",
            "object_scale_mass",
            "reset_gravity",
            "reset_robot_base",
        ):
            setattr(self.events, term, None)
        self.observations.policy.enable_corruption = False


@configclass
class OrcaV2ReposeTeacherEnvCfg(OrcaV2ReposeEnvCfg):
    """Privileged teacher for teacher -> student distillation (arXiv 2605.21330). NOT deployable.

    Groups: ``policy`` = the deployable noisy 74-dim observation (the later student input, unused by PPO), ``teacher`` =
    noise-free joints + cube pose (81-dim). ``PPORunnerCfgOrcaV2Teacher`` maps the teacher group to both actor and critic,
    so one env cfg serves teacher training and distillation (which maps policy -> student, teacher -> teacher).
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.teacher = ObservationsCfg.TeacherObsGroupCfg()


@configclass
class OrcaV2ReposeDelayEnvCfg(OrcaV2ReposeEnvCfg):
    """Joint targets reach the joints 50-100 ms late (6-12 physics substeps), sampled per episode.

    Matches next-tick actuation on deploy (1 policy step) + 0-50 ms servo / bus lag (HYPERPARAM_NOTES 6b).
    """

    def __post_init__(self):
        super().__post_init__()
        a = self.actions.joint_pos
        self.actions.joint_pos = DelayedEMAJointPositionToLimitsActionCfg(
            asset_name=a.asset_name,
            joint_names=a.joint_names,
            alpha=a.alpha,
            rescale_to_limits=a.rescale_to_limits,
            delay_substeps_range=(6, 12),
        )


@configclass
class OrcaV2ReposeJointBiasEnvCfg(OrcaV2ReposeEnvCfg):
    """Per-episode joint-position offset U(+-0.05 rad) on the policy obs (motor-encoder zero / tendon pre-tension)."""

    def __post_init__(self):
        super().__post_init__()
        term = self.observations.policy.joint_pos
        term.func = joint_pos_limit_normalized_biased
        term.params = {"bias_rad": 0.05}


@configclass
class OrcaV2ReposeDropPenaltyEnvCfg(OrcaV2ReposeEnvCfg):
    """Explicit drop penalty (AnyRotate: 50, Aero Hand: 100). Weight is per step and scaled by dt = 0.05 -> -100."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.object_drop = RewTerm(
            func=mdp.is_terminated_term, weight=-2000.0, params={"term_keys": "object_dropped"}
        )


@configclass
class OrcaV2ReposeHist3EnvCfg(OrcaV2ReposeEnvCfg):
    """Policy obs stacks the last 3 steps (HORA / Faive use 3-5 steps for MLP actors)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 3


@configclass
class OrcaV2ReposeHandGravityEnvCfg(OrcaV2ReposeEnvCfg):
    """Gravity on the hand links (the real fingers sag; arXiv 2605.21330 sees gravity-induced deflection)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.rigid_props.disable_gravity = False


@configclass
class OrcaV2ReposeWristFixedEnvCfg(OrcaV2ReposeEnvCfg):
    """Wrist not actuated; its PD drive holds the default position (as on hardware in arXiv 2605.21330)."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.joint_names = ["(?!right_wrist).*"]


@configclass
class OrcaV2ReposeTeacherHist10EnvCfg(OrcaV2ReposeTeacherEnvCfg):
    """Teacher env whose noisy ``policy`` group stacks the last 10 steps (0.5 s): input of the MLP-h10 student."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 10


@configclass
class OrcaV2ReposeHist10EnvCfg(OrcaV2ReposeEnvCfg):
    """Policy obs stacks the last 10 steps (0.5 s; PT paper's window). HORA reports plain MLPs degrade with long raw
    history, so this is the counterpart to hist3."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 10
