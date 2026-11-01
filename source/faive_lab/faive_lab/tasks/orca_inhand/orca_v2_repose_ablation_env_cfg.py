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

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import adr, mdp, rewards_ablation
from .actions_ablation import HoldingEMAJointPositionToLimitsActionCfg, RelativeJointPositionToLimitsActionCfg
from .actuator_net import servo_net_actuators
from .commands_ablation import FixedDirectionRotationCommandCfg
from .mdp.privileged_observations import domain_randomization_params
from .mdp.sim2real import DelayedEMAJointPositionToLimitsActionCfg, joint_pos_limit_normalized_biased
from .orca_v2_repose_env_cfg import ObservationsCfg, OrcaV2ReposeEnvCfg
from .roa_obs import roa_dr_params
from .tracker_obs import TrackedCubePose


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
        use_rotlin_reward(self)


def use_rotlin_reward(env_cfg: OrcaV2ReposeEnvCfg):
    """Replace the direction + magnitude terms by -rotlin's clipped linear progress (90) and off-axis penalty (-5)."""
    cmd = {"command_name": "continuous_rotation", "object_cfg": SceneEntityCfg("object")}
    env_cfg.rewards.angular_velocity_direction = None
    env_cfg.rewards.angular_velocity_magnitude = None
    env_cfg.rewards.rotation_progress = RewTerm(
        func=rewards_ablation.rotation_progress_clipped, weight=90.0, params={**cmd, "min_vel": -0.5}
    )
    env_cfg.rewards.off_axis_angular_velocity = RewTerm(
        func=rewards_ablation.off_axis_angular_velocity_l1, weight=-5.0, params=cmd
    )


@configclass
class OrcaV2ReposeTeacherRotLinEnvCfg(OrcaV2ReposeTeacherEnvCfg):
    """Privileged teacher on the -rotlin reward (the reward-wave winner), for re-distilling the students."""

    def __post_init__(self):
        super().__post_init__()
        use_rotlin_reward(self)


@configclass
class OrcaV2ReposeTeacherRotLinHist10EnvCfg(OrcaV2ReposeTeacherRotLinEnvCfg):
    """-teacher-rotlin env with the 10-step noisy ``policy`` history: input of the MLP-h10 / transformer students."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 10


def use_onedir_reward(env_cfg: OrcaV2ReposeEnvCfg):
    """-onedir: fixed +z command and a linear speed reward clipped at 3 rad/s (see OrcaV2ReposeAsymOneDirEnvCfg)."""
    c = env_cfg.commands.continuous_rotation
    env_cfg.commands.continuous_rotation = FixedDirectionRotationCommandCfg(
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
    env_cfg.rewards.angular_velocity_direction = None
    env_cfg.rewards.angular_velocity_magnitude = None
    env_cfg.rewards.rotation_progress = RewTerm(
        func=rewards_ablation.rotation_progress_linear,
        weight=90.0,
        params={**cmd, "ref_speed": 1.5, "min_vel": -0.5, "max_vel": 3.0},
    )
    env_cfg.rewards.off_axis_angular_velocity = RewTerm(
        func=rewards_ablation.off_axis_angular_velocity_l1, weight=-5.0, params=cmd
    )


@configclass
class OrcaV2ReposeAsymOneDirEnvCfg(OrcaV2ReposeAsymEnvCfg):
    """B for the demo metric (fast, long, no drop): always +z at 1.5 rad/s, reward linear in speed clipped at 3 rad/s.

    +z is the faster direction in 11 of 12 evaluated policies (per-direction eval, 2026-10-28). Rotation terms as in
    -rotlin (weight 90, off-axis -5) but clipped at 3 rad/s (2x the command, ~3x our best speed) instead of the command,
    so speed above 1.5 rad/s keeps paying. Unclipped, the policy flung the cube (see rotation_progress_linear).
    Deploy: goal_ang_vel=1.5, goal_flip_period_s=0. Eval: scripts/eval/eval_policy.py commands +z only for "-onedir".
    """

    def __post_init__(self):
        super().__post_init__()
        use_onedir_reward(self)


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



@configclass
class OrcaV2ReposeAsymRotLinRelActEnvCfg(OrcaV2ReposeAsymRotLinEnvCfg):
    """-rotlin (the reward-wave winner, 10.4 RPM) with relative joint targets as in -relact."""

    def __post_init__(self):
        super().__post_init__()
        a = self.actions.joint_pos
        self.actions.joint_pos = RelativeJointPositionToLimitsActionCfg(
            asset_name=a.asset_name, joint_names=a.joint_names, rescale_to_limits=a.rescale_to_limits, delta_scale=0.1
        )


@configclass
class OrcaV2ReposeAsymRotLinDropPenEnvCfg(OrcaV2ReposeAsymRotLinEnvCfg):
    """-rotlin + the -droppen drop penalty (-100 return): rotlin turns faster but drops more than B (DFSR 84 vs 88 %)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.object_drop = RewTerm(
            func=mdp.is_terminated_term, weight=-2000.0, params={"term_keys": "object_dropped"}
        )


@configclass
class OrcaV2ReposeAsymOneDirDropPenEnvCfg(OrcaV2ReposeAsymOneDirEnvCfg):
    """-onedir (+z, speed reward clipped at 3 rad/s) + the -droppen drop penalty (-100 return): -onedir plateaus at ~50 %
    of episodes ending in a drop while the speed reward keeps rising."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.object_drop = RewTerm(
            func=mdp.is_terminated_term, weight=-2000.0, params={"term_keys": "object_dropped"}
        )


def add_drop_penalty(env_cfg: OrcaV2ReposeEnvCfg):
    """-droppen: -2000 per-step weight on the drop termination (-100 return at dt = 1/20 s)."""
    env_cfg.rewards.object_drop = RewTerm(
        func=mdp.is_terminated_term, weight=-2000.0, params={"term_keys": "object_dropped"}
    )


@configclass
class OrcaV2ReposeTeacherOneDirDropPenEnvCfg(OrcaV2ReposeTeacherEnvCfg):
    """Privileged teacher on the demo reward (-onedir + -droppen): on rotlin the teacher beat B by 1.5 RPM and 11 points
    drop-free, so distilling this one is the path to a deployable demo policy. NOT deployable itself."""

    def __post_init__(self):
        super().__post_init__()
        use_onedir_reward(self)
        add_drop_penalty(self)


@configclass
class OrcaV2ReposeTeacherOneDirDropPenHist10EnvCfg(OrcaV2ReposeTeacherOneDirDropPenEnvCfg):
    """-teacher-onedir-droppen env with the 10-step noisy ``policy`` history (MLP-h10 / transformer students)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 10


@configclass
class OrcaV2ReposeAsymOneDirDropPenRampEnvCfg(OrcaV2ReposeAsymOneDirDropPenEnvCfg):
    """-onedir-droppen with a spin-up ramp on the speed clip: 0.5 -> 3 rad/s over the first 3 s of each episode. A third of
    onedir-droppen's drops happen in the first 3 s, while the hand accelerates the cube from rest; the drop penalty
    did not reduce them. Same cap after 3 s, so steady-state speed should be unchanged."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.rotation_progress.params.update(ramp_s=3.0, ramp_start_vel=0.5)


@configclass
class OrcaV2ReposeAsymOneDirDropPenWorkEnvCfg(OrcaV2ReposeAsymOneDirDropPenEnvCfg):
    """-onedir-droppen + HORA's work penalty (-0.15 * sum |tau * qdot|). On rotlin the same term kept the speed
    (10.0 vs 10.3-10.4 RPM) and raised the perturbed drop-free rate (mean of the last 3 checkpoints 56 vs 39 %); onedir is
    our fastest config but only 19 % drop-free under the perturbed eval."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.joint_work = RewTerm(func=rewards_ablation.joint_work_l1, weight=-0.15)


def add_wrist_vel_penalty(env_cfg: OrcaV2ReposeEnvCfg, weight: float = -4.0):
    """L2 penalty on the wrist joint velocity. The onedir policies swing the wrist (forearm -> carpals, eval rollouts
    2026-10-30): range 1.2-1.3 rad, mean |qdot| ~2 rad/s, mean qdot^2 6.4, vs 0.36-0.48 rad / 0.15-0.23 rad/s / 0.06 for
    the rotlin teacher and student. -4 costs ~25 per second at that motion (rotation reward ~95 per second) and ~0.2 at
    rotlin-like wrist motion."""
    env_cfg.rewards.wrist_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=weight, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["right_wrist"])}
    )


def add_wrist_vel_l1_penalty(env_cfg: OrcaV2ReposeEnvCfg, weight: float = -40.0):
    """L1 penalty on the wrist joint velocity, on top of the L2 one. The L2 term vanishes at small speeds: at the 06_calm
    wrist (~0.15 rad/s) -16 x qdot^2 costs < 1 per second against ~90 of rotation reward, and doubling it moved the wrist
    0.152 -> 0.148 rad/s (512 envs). |qdot| keeps a constant pull: -40 costs ~6 per second at 0.15 rad/s, ~3 at the 07
    line's 0.07."""
    env_cfg.rewards.wrist_vel_l1 = RewTerm(
        func=mdp.joint_vel_l1, weight=weight, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["right_wrist"])}
    )


@configclass
class OrcaV2ReposeAsymOneDirDropPenWristPenEnvCfg(OrcaV2ReposeAsymOneDirDropPenEnvCfg):
    """-onedir-droppen + wrist velocity penalty (deployable PPO version, comparable to the three onedir seeds)."""

    def __post_init__(self):
        super().__post_init__()
        add_wrist_vel_penalty(self)


@configclass
class OrcaV2ReposeAsymOneDirDropPenWristFixEnvCfg(OrcaV2ReposeAsymOneDirDropPenEnvCfg):
    """-onedir-droppen with the wrist not actuated (its PD drive holds the default position, 16-dim action), as
    -wristfix-v0 for the old reward: the hard version of the wrist penalty, and the hardware setting of the PT paper
    (arXiv 2605.21330: wrist unsensed, fixed at 0)."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.joint_names = ["(?!right_wrist).*"]


# Region of the cube's rotation axis on the palm, chosen by the user (2026-10-31) with
# docs/ablation/tools/palm_axis_region.html; env frame, metres. Contains the commanded cube position (0, 0) at 17.9 mm.
CUBE_AXIS_REGION = dict(center=(-0.0116, 0.0136), radius=0.0275)


def add_cube_region_penalty(env_cfg: OrcaV2ReposeEnvCfg, weight: float = 50.0, sigma: float = 0.010,
                            region: dict = CUBE_AXIS_REGION):
    """Term falling exponentially once the cube's rotation axis leaves ``region``
    (``rewards_ablation.cube_outside_region`` = exp(-excess / sigma) - 1): 0 inside, -50 x (1 - e^-1) = -32 per second
    10 mm outside, slope 5 per mm at the boundary, saturating at -50 per second (rotation reward ~90 per second)."""
    env_cfg.rewards.cube_region = RewTerm(
        func=rewards_ablation.cube_outside_region, weight=weight,
        params={"center": region["center"], "radius": region["radius"], "sigma": sigma,
                "object_cfg": SceneEntityCfg("object")},
    )


def add_cube_tilt_penalty(env_cfg: OrcaV2ReposeEnvCfg, weight: float = -200.0):
    """Penalty on the cube tilt (``rewards_ablation.cube_tilt``): -200 costs ~38 per second at the +z policies' mean
    tilt (36 deg, 1 - cos = 0.19) against ~110 per second of rotation reward, and ~1 per second at rotlin's 6 deg."""
    env_cfg.rewards.cube_tilt = RewTerm(
        func=rewards_ablation.cube_tilt, weight=weight, params={"object_cfg": SceneEntityCfg("object")}
    )


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightEnvCfg(OrcaV2ReposeAsymOneDirDropPenEnvCfg):
    """-onedir-droppen + cube tilt penalty: the +z policies tumble the cube instead of turning it flat about z."""

    def __post_init__(self):
        super().__post_init__()
        add_cube_tilt_penalty(self)


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightWristFixEnvCfg(OrcaV2ReposeAsymOneDirDropPenUprightEnvCfg):
    """-onedir-droppen-upright with the wrist not actuated (as -wristfix): flat z-rotation without the wrist swing."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.joint_names = ["(?!right_wrist).*"]


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightWristPenEnvCfg(OrcaV2ReposeAsymOneDirDropPenUprightEnvCfg):
    """-onedir-droppen-upright + wrist velocity penalty: the soft alternative to the fixed wrist."""

    def __post_init__(self):
        super().__post_init__()
        add_wrist_vel_penalty(self)


@configclass
class OrcaV2ReposeTeacherOneDirDropPenWristPenEnvCfg(OrcaV2ReposeTeacherOneDirDropPenEnvCfg):
    """Privileged onedir + drop-penalty teacher with the wrist velocity penalty: 742zn06f swings the wrist ~10x more
    than the rotlin policies, which may not exist on the real hand and may explain the poor MuJoCo transfer."""

    def __post_init__(self):
        super().__post_init__()
        add_wrist_vel_penalty(self)


@configclass
class OrcaV2ReposeTeacherOneDirDropPenWristPenHist10EnvCfg(OrcaV2ReposeTeacherOneDirDropPenWristPenEnvCfg):
    """-teacher-onedir-droppen-wristpen env with the 10-step noisy ``policy`` history (transformer / MLP-h10 students)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 10


@configclass
class OrcaV2ReposeTeacherOneDirDropPenUprightEnvCfg(OrcaV2ReposeTeacherOneDirDropPenEnvCfg):
    """Privileged +z teacher with the cube tilt penalty (flat z-rotation instead of 742zn06f's tumbling). Cube
    position ×3 via ``env.rewards.object_position_tracking.weight=45.0``."""

    def __post_init__(self):
        super().__post_init__()
        add_cube_tilt_penalty(self)


@configclass
class OrcaV2ReposeTeacherOneDirDropPenUprightHist10EnvCfg(OrcaV2ReposeTeacherOneDirDropPenUprightEnvCfg):
    """-teacher-onedir-droppen-upright env with the 10-step noisy ``policy`` history (transformer student)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 10


@configclass
class OrcaV2ReposeTeacherOneDirDropPenUprightWristFixEnvCfg(OrcaV2ReposeTeacherOneDirDropPenEnvCfg):
    """Privileged +z teacher with the cube tilt penalty and the wrist not actuated: the flat-rotation, no-wrist-swing
    version of 742zn06f (which tumbles the cube and swings the wrist)."""

    def __post_init__(self):
        super().__post_init__()
        add_cube_tilt_penalty(self)
        self.actions.joint_pos.joint_names = ["(?!right_wrist).*"]


@configclass
class OrcaV2ReposeTeacherOneDirDropPenUprightWristFixHist10EnvCfg(OrcaV2ReposeTeacherOneDirDropPenUprightWristFixEnvCfg):
    """-teacher-onedir-droppen-upright-wristfix env with the 10-step noisy ``policy`` history (transformer student)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 10


@configclass
class ADRCurriculumCfg:
    """ADR level for action delay and encoder offset (``adr.adr_level``)."""

    adr = CurrTerm(func=adr.adr_level, params={"interval_steps": 480, "step": 0.05, "low": 0.25, "high": 0.40})


@configclass
class OrcaV2ReposeAsymOneDirDropPenADREnvCfg(OrcaV2ReposeAsymOneDirDropPenEnvCfg):
    """-onedir-droppen with ADR (arXiv 1910.07113) on the two deployment gaps of the perturbed eval: per-episode action
    delay U[0, L * 15] physics substeps (eval: 6-12) and joint-encoder offset U(±L * 0.08 rad) (eval: ±0.05), L in [0, 1]
    raised while the drop share of ended episodes stays below 25 % (adr.py). Critic obs unchanged, so a checkpoint of the
    same config without ADR can be resumed (e.g. the position ×3 policy)."""

    def __post_init__(self):
        super().__post_init__()
        a = self.actions.joint_pos
        self.actions.joint_pos = adr.ADRDelayedEMAJointPositionToLimitsActionCfg(
            asset_name=a.asset_name, joint_names=a.joint_names, alpha=a.alpha,
            rescale_to_limits=a.rescale_to_limits, delay_substeps_range=(0, 15),
        )
        self.observations.policy.joint_pos.func = adr.joint_pos_limit_normalized_adr_biased
        self.observations.policy.joint_pos.params = {"bias_rad": 0.08}
        self.curriculum = ADRCurriculumCfg()


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightADREnvCfg(OrcaV2ReposeAsymOneDirDropPenADREnvCfg):
    """-onedir-droppen-adr + the cube tilt penalty: ADR fine-tune of an -onedir-droppen-upright policy (same obs and
    critic, so its checkpoint resumes directly)."""

    def __post_init__(self):
        super().__post_init__()
        add_cube_tilt_penalty(self)


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightServoEnvCfg(OrcaV2ReposeAsymOneDirDropPenUprightEnvCfg):
    """-onedir-droppen-upright with the real ORCA v2 servo model as DR (adr.ServoProfileAction, fitted on the real hand
    2026-10-31): pure action delay 0-67 ms (0-8 substeps; measured 0-40 ms + 0-10 ms ROS, the rest of the 60-90 ms
    dead time is the profile's ramp), per-joint acceleration-limited profile (x0.8-1.2) and play (x0.5-1.5) in front of
    the task's own PD, no slope (adr.DEFAULT_SERVO_PARTS); joint-encoder offset +-0.08 rad. Policy obs and critic unchanged,
    so the deployable +z policies (07_tilt, ...) resume directly."""

    def __post_init__(self):
        super().__post_init__()
        adr.use_orca_v2_servo_model(self, delay_substeps_range=(0, 8), adr=False)
        self.actions.joint_pos.amax_scale_range = (0.8, 1.2)
        self.observations.policy.joint_pos.func = adr.joint_pos_limit_normalized_biased
        self.observations.policy.joint_pos.params = {"bias_rad": 0.08}


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightServoWristPenEnvCfg(OrcaV2ReposeAsymOneDirDropPenUprightServoEnvCfg):
    """-upright-servo + the wrist velocity penalty (-4): on the real hand the policies that keep the wrist still (07,
    02b: 2-7 deg/s mean wrist speed) ran the full minute, those that move it (05: 51 deg/s) were stopped."""

    def __post_init__(self):
        super().__post_init__()
        add_wrist_vel_penalty(self)


ORCA_V2_PALM_BODIES = ["right_forearm", "right_tower", "right_carpals", "right_thumb_tp"]


def use_real_hand_friction(env_cfg: OrcaV2ReposeEnvCfg):
    """Friction of the real ORCA v2 + cube, measured 2026-10-31 by slip angle (mu = tan): PLA cube on the TPU palm
    18-25 deg (mu 0.32-0.47, n = 5), on the soft silicone finger pads 30-40 deg (mu 0.58-0.84). PhysX averages the two
    materials, so cube 0.3-0.5 with palm links 0.3-0.5 gives palm contacts 0.3-0.5, and finger links 0.8-1.3 give finger
    contacts 0.55-0.9 (the task had 0.55-1.5 everywhere)."""
    rng = lambda lo, hi: {"static_friction_range": (lo, hi), "dynamic_friction_range": (lo, hi)}  # noqa: E731
    env_cfg.events.robot_physics_material.params.update(rng(0.8, 1.3))
    env_cfg.events.object_physics_material.params.update(rng(0.3, 0.5))
    # after robot_physics_material (same startup mode, defined later): overrides only the palm links' shapes
    env_cfg.events.palm_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=ORCA_V2_PALM_BODIES), **rng(0.3, 0.5),
                "restitution_range": (0.0, 0.0), "num_buckets": 250},
    )


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightServoWristPenRealFrictionEnvCfg(
    OrcaV2ReposeAsymOneDirDropPenUprightServoWristPenEnvCfg
):
    """-upright-servo-wristpen with the real hand's friction (``use_real_hand_friction``): a slippery TPU palm, grippy
    silicone fingers. On the real hand every policy that tips the palm loses the cube."""

    def __post_init__(self):
        super().__post_init__()
        use_real_hand_friction(self)


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionEnvCfg(OrcaV2ReposeAsymOneDirDropPenUprightWristPenEnvCfg):
    """-upright-wristpen with the real hand's friction and the deployment gaps as plain DR, no servo profile: action
    delay 0-100 ms (0-12 substeps; the real dead time is 60-90 ms), joint-encoder offset +-0.08 rad. At low friction
    the task's own actuators rank the real-hand policies the hand's way (07 first, wrist speed separating 07 / 02b from
    01 / 05), while the servo profile in closed loop put 07 last."""

    def __post_init__(self):
        super().__post_init__()
        use_real_hand_friction(self)
        a = self.actions.joint_pos
        self.actions.joint_pos = DelayedEMAJointPositionToLimitsActionCfg(
            asset_name=a.asset_name, joint_names=a.joint_names, alpha=a.alpha,
            rescale_to_limits=a.rescale_to_limits, delay_substeps_range=(0, 12),
        )
        self.observations.policy.joint_pos.func = joint_pos_limit_normalized_biased
        self.observations.policy.joint_pos.params = {"bias_rad": 0.08}


ORCA_V2_TPU_PALM_BODIES = ["right_forearm", "right_tower", "right_carpals"]
ORCA_V2_PLA_BODIES = [  # finger links without silicone skin (all collision hulls printed PLA)
    "right_index_ap", "right_middle_ap", "right_pinky_ap", "right_ring_ap", "right_thumb_tp", "right_thumb_ap",
]


def use_real_hand_friction_v2(env_cfg: OrcaV2ReposeEnvCfg):
    """``use_real_hand_friction`` with the PLA skeleton as its own material: contacts with the PLA cube of the TPU palm
    0.3-0.5 (measured 0.32-0.47), the silicone-skinned finger links 0.55-0.9 (measured 0.58-0.84), and the skinless PLA
    links 0.1-0.25 (PLA on PLA, "extreme low", not measured). With PhysX's average combine and the cube at 0.2-0.3 the
    link ranges are 2 * contact - cube: PLA 0.0-0.2, palm 0.4-0.7, skin links 0.9-1.5. Per link: the skinned links' PLA
    skeleton hulls get the skin value (the hand's 53 colliders are instanced, no per-collider materials)."""
    rng = lambda lo, hi: {"static_friction_range": (lo, hi), "dynamic_friction_range": (lo, hi)}  # noqa: E731
    env_cfg.events.robot_physics_material.params.update(rng(0.9, 1.5))
    env_cfg.events.object_physics_material.params.update(rng(0.2, 0.3))
    # after robot_physics_material (same startup mode, defined later): override only these links' shapes
    for name, bodies, r in (("palm_physics_material", ORCA_V2_TPU_PALM_BODIES, (0.4, 0.7)),
                            ("pla_physics_material", ORCA_V2_PLA_BODIES, (0.0, 0.2))):
        setattr(env_cfg.events, name, EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=bodies), **rng(*r),
                    "restitution_range": (0.0, 0.0), "num_buckets": 250},
        ))


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFriction2EnvCfg(
    OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionEnvCfg
):
    """-upright-wristpen-realfric with the PLA skeleton links as a third friction class (``use_real_hand_friction_v2``)."""

    def __post_init__(self):
        super().__post_init__()
        use_real_hand_friction_v2(self)


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionRegionEnvCfg(
    OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionEnvCfg
):
    """-upright-wristpen-realfric (the real-hand recipe of 07ft_s0_m4600) + the user's cube rotation-axis region
    (``add_cube_region_penalty``)."""

    def __post_init__(self):
        super().__post_init__()
        add_cube_region_penalty(self)


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionRegion200EnvCfg(
    OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionEnvCfg
):
    """-realfric-region with the region penalty ×4 (weight 200): at weight 50 the term is ~3 % of the rotation reward
    per episode and 300 fine-tune iterations left the in-region rate at the base's 94 %. Same exponential shape, now
    −126 per second at 10 mm outside and −200 saturated (rotation ~90 per second)."""

    def __post_init__(self):
        super().__post_init__()
        add_cube_region_penalty(self, weight=200.0)


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionRegion200CEnvCfg(
    OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionRegion200EnvCfg
):
    """-realfric-region200 with the position-tracking reward pulled to the region's centre instead of (0, 0). The cube
    of 07ft_s0_m4600 sits on average at (-6.9, -6.0) mm, 20 mm from the centre (-11.6, 13.6) mm, and 95 % of its
    out-of-region steps leave the circle towards -y, where the old target (0, 0) lies 17.9 mm from the centre:
    the tracking reward and the region term pull against each other. Command and goal observation unchanged
    (``rewards_ablation.track_position_exp_xy``), so the unchanged deploy node still fits."""

    def __post_init__(self):
        super().__post_init__()
        t = self.rewards.object_position_tracking
        t.func = rewards_ablation.track_position_exp_xy
        t.params["target_xy"] = CUBE_AXIS_REGION["center"]


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionMount10EnvCfg(
    OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionEnvCfg
):
    """-realfric + mount pose DR: hand base roll and pitch ±10 deg per reset (default roll ±2.9 deg, no pitch). Roll ±10
    deg alone (Hydra override) kept the calm wrist at the same speed; pitch changes the palm's angle to gravity, the main
    risk of the flat-hand + -30 deg wrist relay. The base cfg has no pitch key, so Hydra cannot add one."""

    def __post_init__(self):
        super().__post_init__()
        pr = self.events.reset_robot_base.params["pose_range"]
        pr["roll"] = [-0.1745, 0.1745]
        pr["pitch"] = [-0.1745, 0.1745]


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightHist3EnvCfg(OrcaV2ReposeAsymOneDirDropPenUprightEnvCfg):
    """07's training task (-onedir-droppen-upright) with a 3-step observation history in the actor (222 inputs; the
    critic keeps its privileged one-step obs). Stage 1 of a history version of the real-hand recipe: at the measured
    friction all memoryless fine-tunes plateau at 89-92 % drop-free (512 envs); a short history lets the actor infer
    slip / contact state that one step of joint data cannot show. Deploy: policy_history.pt + the node's reset line."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 3


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionHist3EnvCfg(
    OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionEnvCfg
):
    """-realfric (recipe of 07ft_s0_m4600) with the 3-step actor history: stage 2, fine-tuned from -upright-hist3."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 3


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1EnvCfg(
    OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionEnvCfg
):
    """-realfric + L1 wrist velocity penalty (``add_wrist_vel_l1_penalty``, -40) for the calm-wrist stage of the fast 06
    line (06_calm: 13.5 RPM, 98.8 % in region, wrist 0.15 rad/s vs the ~0.1 ship rule)."""

    def __post_init__(self):
        super().__post_init__()
        add_wrist_vel_l1_penalty(self)


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1ServoFastEnvCfg(
    OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1EnvCfg
):
    """-realfric-wristl1 (the rank-1 recipe) with the servo model as DR, its acceleration limit randomised over 1.5-3x the
    fitted value: the profile alone did nearly all of the servo model's damage, rank 1 recovers at ~2x and A at ~3x
    (512-env scan, 2026-11-01), and the fitted 1x never predicted the hand in closed loop. A hedge toward servos a bit
    slower than ideal without training against the doubtful 1x. Delay 0-67 ms and play as in -upright-servo."""

    def __post_init__(self):
        super().__post_init__()
        adr.use_orca_v2_servo_model(self, delay_substeps_range=(0, 8), adr=False)
        self.actions.joint_pos.amax_scale_range = (1.5, 3.0)


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1Hist3EnvCfg(
    OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1EnvCfg
):
    """-realfric-wristl1 (the rank-1 calm recipe) with the 3-step actor history: the history actor after its real-friction
    stage is the fastest policy so far (15.0 RPM at the measured friction, 512 envs) but unreliable (87.7 % drop-free after
    2 s vs 98.8 % for rank 1), the state the 06 line was in before the calm recipe."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 3


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1Hist3ServoFastEnvCfg(
    OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1Hist3EnvCfg
):
    """-realfric-wristl1-hist3 with the servo model as DR (acceleration limit 1.5-3x, as in -servofast): the history actor
    is the fastest line (15.8 RPM calmed, 512 envs) and the shipped rank 1 is servo-hardened."""

    def __post_init__(self):
        super().__post_init__()
        adr.use_orca_v2_servo_model(self, delay_substeps_range=(0, 8), adr=False)
        self.actions.joint_pos.amax_scale_range = (1.5, 3.0)


@configclass
class ROAPrivObsGroupCfg(ObsGroup):
    """``priv`` input of ROA's privileged encoder (roa.py): the env's randomisation parameters (cube friction / mass,
    finger and palm friction, joint gain scales, servo model, mount angle), each normalised to [-1, 1]
    (roa_obs.roa_dr_params, 13 values), noise-free."""

    dr = ObsTerm(
        func=roa_dr_params,
        params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("object"), "action_name": "joint_pos"},
    )
    enable_corruption = False
    concatenate_terms = True


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1ServoFastROAEnvCfg(
    OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1ServoFastEnvCfg
):
    """-servofast (calm recipe + servo DR) for ROA (roa.py, docs/ablation/roa.md): the policy group keeps the last 20
    steps (1 s) for the history encoder, the ``priv`` group feeds the privileged encoder, and the critic sees B's
    privileged group + ``priv`` (runner cfg rsl_rl_roa_cfg.PPORunnerCfgOrcaV2ROA)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 20
        self.observations.priv = ROAPrivObsGroupCfg()


@configclass
class OrcaV2ReposeAsymOneDirDropPenROAEnvCfg(OrcaV2ReposeAsymOneDirDropPenEnvCfg):
    """ROA stage 1: the 06 base task (-onedir-droppen; run with tracking 45) with the 20-step policy history and the
    ``priv`` group, from scratch. Every good policy came from this base + fine-tunes, while the calm recipe from scratch
    stayed at 64-84 % drop-free, so ROA follows the same stages (base -> -wristl1 calm -> -servofast)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 20
        self.observations.priv = ROAPrivObsGroupCfg()


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1ROAEnvCfg(
    OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1EnvCfg
):
    """ROA stage 2: the calm recipe (-wristl1; run with tracking 45 / tilt -800 / wrist -16), no servo model yet (servo DR
    during calming made a third of the envs stall, servofast_from_base)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 20
        self.observations.priv = ROAPrivObsGroupCfg()


def center_cube_spawn(env_cfg: OrcaV2ReposeEnvCfg, center: tuple[float, float] = CUBE_AXIS_REGION["center"]):
    """Spawn the cube around ``center`` (env frame, the same ±5 mm) instead of (0, 0) through the reset event's pose range
    (offsets from the default root state), so the command's target and the goal observation stay unchanged."""
    pr = env_cfg.events.reset_object.params["pose_range"]
    pr["x"] = [center[0] - 0.005, center[0] + 0.005]
    pr["y"] = [center[1] - 0.005, center[1] + 0.005]


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionRegion200CSEnvCfg(
    OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionRegion200CEnvCfg
):
    """-region200c with the cube also spawned at the region centre: from (0, 0) it starts 17.9 mm off centre, 7.5 mm
    inside the circle on the side where 95 % of the excursions leave (the user places the real cube by hand)."""

    def __post_init__(self):
        super().__post_init__()
        center_cube_spawn(self)


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightWristPenADREnvCfg(OrcaV2ReposeAsymOneDirDropPenUprightADREnvCfg):
    """-onedir-droppen-upright-adr + the wrist velocity penalty: ADR fine-tune of an -onedir-droppen-upright-wristpen
    policy (same obs and critic)."""

    def __post_init__(self):
        super().__post_init__()
        add_wrist_vel_penalty(self)


@configclass
class OrcaV2ReposeAsymRotLinHist3EnvCfg(OrcaV2ReposeAsymHist3EnvCfg):
    """-asym-hist3 (3-step obs history MLP) on the -rotlin reward: the memory comparison so far was on the old reward."""

    def __post_init__(self):
        super().__post_init__()
        use_rotlin_reward(self)


@configclass
class OrcaV2ReposeAsymRotLinWorkEnvCfg(OrcaV2ReposeAsymRotLinEnvCfg):
    """-rotlin + HORA's work penalty (-0.15 * sum |tau * qdot|): rotlin is fast but wobbly and weak under the perturbed
    eval; the penalty targets jerky, high-effort gaits. At -1.5 (job 29851888) the term was 2x the rotation reward at
    the start and, with no drop penalty, the policy learned to drop the cube to end the episode (100 % drops,
    rotation 1.7 vs 25 for rotlin at iteration 500): -0.15 starts at ~1/5 of the rotation reward."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.joint_work = RewTerm(func=rewards_ablation.joint_work_l1, weight=-0.15)

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


def add_cube_tracker_obs(env_cfg: OrcaV2ReposeEnvCfg, **params):
    """Subtask 2 (exteroception, docs/ablation/exteroception.md): append the tracked cube pose to the policy group (74 ->
    82: + object position 3, symmetry-canonical quaternion 4, tracked flag 1; position and quaternion in the order of the
    team's -cube-v0 scaffold, the flag last). Tracker lag, noise, calibration offset, smoothing and lost detections are
    randomised per episode (tracker_obs.TrackedCubePose); ``params`` override its defaults (e.g. force_lost=True)."""
    env_cfg.observations.policy.cube_tracker = ObsTerm(
        func=TrackedCubePose,
        params={"asset_cfg": SceneEntityCfg("object"), "command_name": "continuous_rotation", **params},
    )


@configclass
class OrcaV2ReposeAsymOneDirDropPenCubeEnvCfg(OrcaV2ReposeAsymOneDirDropPenEnvCfg):
    """Subtask 2, stage 1 from scratch: the 06 base task (run with tracking 45) + the tracked cube pose."""

    def __post_init__(self):
        super().__post_init__()
        add_cube_tracker_obs(self)


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightCubeEnvCfg(OrcaV2ReposeAsymOneDirDropPenUprightEnvCfg):
    """Subtask 2 eval task (-upright, as the LOWFRIC evals) + the tracked cube pose; tracker off via
    env.observations.policy.cube_tracker.params.force_lost=true."""

    def __post_init__(self):
        super().__post_init__()
        add_cube_tracker_obs(self)


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1CubeEnvCfg(
    OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1EnvCfg
):
    """Subtask 2, calm stage: -realfric-wristl1 (run with tracking 45 / tilt -800 / wrist -16) + the tracked cube pose."""

    def __post_init__(self):
        super().__post_init__()
        add_cube_tracker_obs(self)


@configclass
class OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1ServoFastCubeEnvCfg(
    OrcaV2ReposeAsymOneDirDropPenUprightWristPenRealFrictionWristL1ServoFastEnvCfg
):
    """Subtask 2, servo stage: -servofast (amax 1.5-3x; servowide via env.actions.joint_pos.amax_scale_range=[1.0,3.0])
    + the tracked cube pose. Also the fine-tune task of the rank-1 family (checkpoint widened with
    scripts/export/widen_actor_input.py)."""

    def __post_init__(self):
        super().__post_init__()
        add_cube_tracker_obs(self)
