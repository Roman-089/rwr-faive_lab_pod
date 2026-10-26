# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Subtask 2 (exteroception) – task ``orca-v2-inhand-repose-cube-v0``.

Owner: E (exteroception). See OWNERSHIP in rwr-fast/rwr-fast.

This config INHERITS everything (scene, actions, rewards, domain randomization, proprio
observations) from ``OrcaV2ReposeEnvCfg`` (owner S0, Subtask 1) and only appends the cube pose
to the ``policy`` observation group. Improvements S0 makes to the proprio task therefore flow into
this task automatically - do not copy or edit S0's config here.

Observation order (policy group, concatenated):
  joint_pos (17) | joint_vel (17) | goal_command (6) | last_action (17) | last_joint_pos_cmd (17)  -> 74, as S0
  object_pos (3) | object_quat (4)                                                                -> +7 = 81
The policy node for this task (rwr_faive_system, E's copy) must use exactly this order and OBS_DIM = 81.

TODO (E):
  - match the noise to the measured tracker jitter (mm / deg)
  - model tracker lag (smoothing filter) and lost detections (what the policy sees when there is no pose)
  - keep the frame identical to what the policy node computes (hand_root -> cube, converted to the sim frame)
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

from . import mdp
from .mdp.noise_cfg import GaussianNoiseAndNormalizeQuatCfg as QuatNoise
from .orca_v2_repose_env_cfg import ObservationsCfg, OrcaV2ReposeEnvCfg


@configclass
class CubePolicyObsGroupCfg(ObservationsCfg.KinematicObsGroupCfg):
    """S0's proprio observations + cube pose (dataclass inheritance appends these terms at the end)."""

    # same terms as the commented-out ones in orca_v2_repose_env_cfg.py; noise values are placeholders
    object_pos = ObsTerm(
        func=mdp.root_pos_w, noise=Gnoise(std=0.01), params={"asset_cfg": SceneEntityCfg("object")}
    )
    object_quat = ObsTerm(
        func=mdp.root_quat_w,
        noise=QuatNoise(std=0.05),
        params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False},
    )


@configclass
class OrcaV2ReposeCubeEnvCfg(OrcaV2ReposeEnvCfg):
    """Cube rotation with the cube pose from the tracker in the policy observation."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy = CubePolicyObsGroupCfg()
