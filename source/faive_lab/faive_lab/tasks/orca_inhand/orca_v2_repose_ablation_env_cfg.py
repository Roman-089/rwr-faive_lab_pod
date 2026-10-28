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
  asym-nodr   orca-v2-inhand-repose-asym-nodr-v0     sanity env + B's critic group (to evaluate B-family checkpoints without DR)
  teacher     orca-v2-inhand-repose-teacher-v0       privileged teacher (actor + critic see teacher obs), for distillation

Sim2real wave (single change each vs the baseline; see docs/ablation/):
  delay       orca-v2-inhand-repose-delay-v0         joint targets 50-100 ms late (6-12 substeps, per episode)
  jointbias   orca-v2-inhand-repose-jointbias-v0     per-episode joint-position obs offset U(+-0.05 rad)
  droppen     orca-v2-inhand-repose-droppen-v0       -100 return penalty on a cube drop
  hist3       orca-v2-inhand-repose-hist3-v0         policy obs = last 3 steps (222-dim; deploy needs the history)
  hist10      orca-v2-inhand-repose-hist10-v0        policy obs = last 10 steps (740-dim)
  asym-hist3  orca-v2-inhand-repose-asym-hist3-v0    B + 3-step history      (architecture study on top of B)
  asym-hist10 orca-v2-inhand-repose-asym-hist10-v0   B + 10-step history
  asym-lstm   orca-v2-inhand-repose-asym-lstm-v0     B + LSTM actor-critic
  handgrav    orca-v2-inhand-repose-handgrav-v0      gravity on the hand links
  wristfix    orca-v2-inhand-repose-wristfix-v0      wrist not actuated, held at its default (72-dim obs)
  asym-droppen orca-v2-inhand-repose-asym-droppen-v0 B + -100 return penalty on a cube drop
  rma-teacher orca-v2-inhand-repose-rma-teacher-v0   RMA stage 1 (HORA): B + actor sees tanh(env_mlp(cube pos, DR params))
  rma-student orca-v2-inhand-repose-rma-student-v0   RMA stage 2: adaptation module on 30-step history (deployable)
  asym-acthold orca-v2-inhand-repose-asym-acthold-v0 B + random action holding: joint targets not updated w.p.
                                                     p ~ U(0, 0.1) per step (lost / late servo packet; Phase 2)
  asym-actnet  orca-v2-inhand-repose-asym-actnet-v0  B + chirp-fitted servo model (delay, play, tau, learned target
                                                     correction) on all joints (actuator_net.py; Phase 2)

Reward wave (on top of B; docs/ablation/rewards.md, terms in rewards_ablation.py):
  asym-rotlin  orca-v2-inhand-repose-asym-rotlin-v0  direction + magnitude terms -> 90 * clipped linear progress
                                                     about the target axis (HORA/LEAP/DLR 2024) - 5 * off-axis |w|
  asym-relact  orca-v2-inhand-repose-asym-relact-v0  B with relative joint targets (+-10 % of the half-range per step, no EMA)
  asym-rotsoft orca-v2-inhand-repose-asym-rotsoft-v0 same terms, reweighted: magnitude sigma 0.2 -> 0.75, weight 30 -> 60;
                                                     direction 60 -> 30

Play / replay a B or C checkpoint with the same task id: the critic's input size differs, so the checkpoint only
loads into the matching variant.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp, rewards_ablation
from .actions_ablation import HoldingEMAJointPositionToLimitsActionCfg, RelativeJointPositionToLimitsActionCfg
from .actuator_net import servo_net_actuators
from .commands_ablation import FixedDirectionRotationCommandCfg
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
class OrcaV2ReposeAsymNoDREnvCfg(OrcaV2ReposeNoDREnvCfg):
    """Sanity env with B's critic group, so B-family checkpoints (81-dim critic) load: evaluate DR-trained policies
    without DR / obs noise (the mirror of evaluating ``sanity_nodr`` with DR)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.critic = ObservationsCfg.TeacherObsGroupCfg()


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

    Matches next-tick actuation on deploy (1 policy step) + 0-50 ms servo / bus lag (docs/ablation/sim2real.md, fixed action delay).
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


# --- Architecture variants on top of ablation B (asymmetric critic), which beat the baseline by far (docs/ablation/asymmetric_critic.md).


@configclass
class OrcaV2ReposeAsymHist3EnvCfg(OrcaV2ReposeAsymEnvCfg):
    """B + policy obs stacks the last 3 steps."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 3


@configclass
class OrcaV2ReposeAsymHist10EnvCfg(OrcaV2ReposeAsymEnvCfg):
    """B + policy obs stacks the last 10 steps."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 10


@configclass
class OrcaV2ReposeAsymDropPenaltyEnvCfg(OrcaV2ReposeAsymEnvCfg):
    """B + explicit drop penalty (-100 return, as in Aero Hand; AnyRotate uses 50). Drops are B's main failure mode."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.object_drop = RewTerm(
            func=mdp.is_terminated_term, weight=-2000.0, params={"term_keys": "object_dropped"}
        )


@configclass
class OrcaV2ReposeAsymRotLinEnvCfg(OrcaV2ReposeAsymEnvCfg):
    """B with a rotation reward linear in the speed about the target axis, clipped at the command (+ off-axis penalty).

    Baseline: direction 60 (speed-independent cosine) + magnitude 30 * exp(-|err| / 0.2), which is ~0.4 at our policies'
    6 RPM vs the commanded 14 RPM, so speed is barely rewarded. 90 = the same total weight; the max is unchanged.
    """

    def __post_init__(self):
        super().__post_init__()
        cmd = {"command_name": "continuous_rotation", "object_cfg": SceneEntityCfg("object")}
        self.rewards.angular_velocity_direction = None
        self.rewards.angular_velocity_magnitude = None
        self.rewards.rotation_progress = RewTerm(
            func=rewards_ablation.rotation_progress_clipped, weight=90.0, params={**cmd, "min_vel": -0.5}
        )
        self.rewards.off_axis_angular_velocity = RewTerm(
            func=rewards_ablation.off_axis_angular_velocity_l1, weight=-5.0, params=cmd
        )


@configclass
class OrcaV2ReposeAsymOneDirEnvCfg(OrcaV2ReposeAsymEnvCfg):
    """B for the demo metric (fast, long, no drop): always +z at 1.5 rad/s, reward linear in speed with no upper clip.

    +z is the faster direction in 11 of 12 evaluated policies (per-direction eval, 2026-10-28). Rotation terms as in
    -rotlin (weight 90, off-axis -5) but unclipped, so speed above 1.5 rad/s keeps paying. Deploy: goal_ang_vel=1.5,
    goal_flip_period_s=0. Eval: scripts/eval/eval_policy.py commands +z only for "-onedir" tasks.
    """

    def __post_init__(self):
        super().__post_init__()
        c = self.commands.continuous_rotation
        self.commands.continuous_rotation = FixedDirectionRotationCommandCfg(
            asset_name=c.asset_name,
            init_pos_offset=c.init_pos_offset,
            angular_vel_range=c.angular_vel_range,
            angular_vel_magnitude_threshold=c.angular_vel_magnitude_threshold,
            angular_vel_direction_threshold=c.angular_vel_direction_threshold,
            position_success_threshold=c.position_success_threshold,
            resampling_time_range=c.resampling_time_range,
            direction=1.0,
        )
        cmd = {"command_name": "continuous_rotation", "object_cfg": SceneEntityCfg("object")}
        self.rewards.angular_velocity_direction = None
        self.rewards.angular_velocity_magnitude = None
        self.rewards.rotation_progress = RewTerm(
            func=rewards_ablation.rotation_progress_linear,
            weight=90.0,
            params={**cmd, "ref_speed": 1.5, "min_vel": -0.5},
        )
        self.rewards.off_axis_angular_velocity = RewTerm(
            func=rewards_ablation.off_axis_angular_velocity_l1, weight=-5.0, params=cmd
        )


@configclass
class OrcaV2ReposeAsymRotSoftEnvCfg(OrcaV2ReposeAsymEnvCfg):
    """B with the baseline's rotation terms reweighted toward speed: wider magnitude kernel, more weight on it."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.angular_velocity_magnitude.params["sigma"] = 0.75
        self.rewards.angular_velocity_magnitude.weight = 60.0
        self.rewards.angular_velocity_direction.weight = 30.0


@configclass
class OrcaV2ReposeAsymRelActEnvCfg(OrcaV2ReposeAsymEnvCfg):
    """B with relative joint targets (HORA / LEAP): the action moves the previous target by a bounded delta instead of
    setting an absolute target behind EMA alpha 0.2. Deploy: the policy node must apply the same update."""

    def __post_init__(self):
        super().__post_init__()
        a = self.actions.joint_pos
        self.actions.joint_pos = RelativeJointPositionToLimitsActionCfg(
            asset_name=a.asset_name, joint_names=a.joint_names, rescale_to_limits=a.rescale_to_limits, delta_scale=0.1
        )


# --- Rapid Motor Adaptation (HORA, arXiv 2210.04887): see rma.py.


@configclass
class RMAPrivObsGroupCfg(ObsGroup):
    """Input of the RMA env encoder, as in HORA: noise-free cube position + the sampled physics parameters."""

    object_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
    dr_params = ObsTerm(
        func=domain_randomization_params,
        params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("object")},
    )
    enable_corruption = False
    concatenate_terms = True


@configclass
class OrcaV2ReposeRMATeacherEnvCfg(OrcaV2ReposeAsymEnvCfg):
    """RMA stage 1: B + ``priv`` group feeding the actor's extrinsics encoder. NOT deployable."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.priv = RMAPrivObsGroupCfg()


@configclass
class OrcaV2ReposeRMAStudentEnvCfg(OrcaV2ReposeRMATeacherEnvCfg):
    """RMA stage 2: policy obs stacks the last 30 steps (1.5 s, HORA's window) for the adaptation module."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 30


# --- Sim2real on top of B (Phase 2, docs/ablation/sim2real.md).


@configclass
class OrcaV2ReposeAsymActHoldEnvCfg(OrcaV2ReposeAsymEnvCfg):
    """B with random action holding (TeleDexter's action masking, arXiv 2607.11481): per episode p ~ U(0, 0.1); each
    policy step w.p. p the joint targets are not updated (lost / late servo packet), while ``last_joint_pos_cmd`` still
    shows the command the policy sent. Deploy: unchanged."""

    def __post_init__(self):
        super().__post_init__()
        a = self.actions.joint_pos
        self.actions.joint_pos = HoldingEMAJointPositionToLimitsActionCfg(
            asset_name=a.asset_name,
            joint_names=a.joint_names,
            alpha=a.alpha,
            rescale_to_limits=a.rescale_to_limits,
            hold_prob_max=0.1,
            hold_steps_range=(1, 1),
            per_joint=False,
        )


@configclass
class OrcaV2ReposeAsymActNetEnvCfg(OrcaV2ReposeAsymEnvCfg):
    """B with the chirp-fitted servo model on all 17 joints (actuator_net.py, docs/ablation/actuator_net.md): per joint
    type command delay + servo play + PD time constant + bounded learned target correction on the implicit PD drive.
    Stiffness / effort limit / friction stay orca_v2.py's. Policy obs unchanged. Not to be stacked with -delay (the
    servo delay is already in the actuator)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators = servo_net_actuators(self.scene.robot.actuators["fingers"])
