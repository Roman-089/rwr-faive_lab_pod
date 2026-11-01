"""Privileged parameter vector for ROA (roa.py): the per-env randomisation parameters, each normalised to [-1, 1] from
its range in the env cfg (0 when the range is degenerate, i.e. not randomised).

Layout (13 values):
  0-2   cube static friction, dynamic friction, restitution     (object_physics_material, reset)
  3     cube mass scale                                         (object_scale_mass, reset)
  4     mean finger / skin friction                             (robot_physics_material, startup; non-palm shapes)
  5     mean palm / PLA link friction                           (palm_physics_material [+ pla_physics_material], startup)
  6-7   mean joint stiffness scale, mean damping scale           (robot_joint_stiffness_and_damping, reset)
  8-10  servo model: delay substeps, mean acceleration-limit scale, mean play scale (ServoProfileAction, reset)
  11-12 mount roll, pitch                                       (reset_robot_base, reset; pitch 0 unless randomised)
Not included: the joint-encoder offset (17 values: an unobservable calibration error the history cannot recover, so
ROA's regulariser would have to suppress it again), and the cube pose / velocity (state, not parameters; a later
variant could add a cube-state estimator, extreme-parkour's ``priv_explicit``).
Cube mass, cube material and the joint gain scales come from the existing ``domain_randomization_params`` term
(mdp/privileged_observations.py), reused as a sub-term; the rest is read from the event terms and the action term.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

from .mdp.privileged_observations import domain_randomization_params

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

ROA_PRIV_DIM = 13
PALM_EVENTS = ("palm_physics_material", "pla_physics_material")


def _norm(x: torch.Tensor, rng) -> torch.Tensor:
    """Map x from [lo, hi] to [-1, 1]; 0 for a degenerate (non-randomised) range."""
    lo, hi = float(rng[0]), float(rng[1])
    if hi - lo < 1e-9:
        return torch.zeros_like(x)
    return 2.0 * (x - lo) / (hi - lo) - 1.0


class roa_dr_params(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.params.get("robot_cfg", SceneEntityCfg("robot")).name]
        self.object = env.scene[cfg.params.get("object_cfg", SceneEntityCfg("object")).name]
        self.action_name = cfg.params.get("action_name", "joint_pos")
        self.dr = domain_randomization_params(cfg, env)  # cube mass | cube material (3) | stiffness, damping scales
        self._friction = torch.zeros(env.num_envs, 2, device=env.device)
        self._friction_read = False
        self._palm_shapes = None

    def reset(self, env_ids: torch.Tensor | None = None):
        self.dr.reset(env_ids)

    # --- helpers
    def _event_params(self, env, name):
        cfg = getattr(env.cfg.events, name, None)
        return None if cfg is None else cfg.params

    def _palm_shape_ids(self, env) -> list[int]:
        """Shape indices of the bodies the palm (/ PLA) material events randomise (Isaac Lab's per-body shape layout)."""
        ids = []
        for mode, names in env.event_manager._mode_term_names.items():
            for name, tcfg in zip(names, env.event_manager._mode_term_cfgs[mode]):
                term = tcfg.func
                if name in PALM_EVENTS and getattr(term, "num_shapes_per_body", None) is not None:
                    for b in term.asset_cfg.body_ids:
                        start = sum(term.num_shapes_per_body[:b])
                        ids.extend(range(start, start + term.num_shapes_per_body[b]))
        return sorted(set(ids))

    def _read_friction(self, env):
        """Startup materials: mean static friction over the finger (non-palm) and the palm / PLA shapes, read once."""
        mu = self.robot.root_physx_view.get_material_properties()[..., 0]  # (num_envs, shapes), CPU
        palm = self._palm_shape_ids(env)
        mask = torch.zeros(mu.shape[1], dtype=torch.bool)
        mask[palm] = True
        finger_mu = mu[:, ~mask].mean(-1)
        palm_mu = mu[:, mask].mean(-1) if mask.any() else torch.zeros_like(finger_mu)
        p_rob = self._event_params(env, "robot_physics_material") or {}
        p_palm = self._event_params(env, "palm_physics_material") or {}
        self._friction[:, 0] = _norm(finger_mu, p_rob.get("static_friction_range", (0, 0))).to(self.device)
        self._friction[:, 1] = _norm(palm_mu, p_palm.get("static_friction_range", (0, 0))).to(self.device)
        self._friction_read = True

    def __call__(self, env: ManagerBasedEnv, robot_cfg=None, object_cfg=None, action_name="joint_pos") -> torch.Tensor:
        if not self._friction_read:
            self._read_friction(env)
        nj = self.robot.num_joints
        per = self.dr._per_env  # mass (1) | material (3) | stiffness (nj) | damping (nj)
        p_omat = self._event_params(env, "object_physics_material") or {}
        p_omass = self._event_params(env, "object_scale_mass") or {}
        p_gain = self._event_params(env, "robot_joint_stiffness_and_damping") or {}
        mass_scale = per[:, 0] / self.object.data.default_mass[:, 0].to(self.device)
        cube = torch.stack([
            _norm(per[:, 1], p_omat.get("static_friction_range", (0, 0))),
            _norm(per[:, 2], p_omat.get("dynamic_friction_range", (0, 0))),
            _norm(per[:, 3], p_omat.get("restitution_range", (0, 0))),
            _norm(mass_scale, p_omass.get("mass_distribution_params", (1, 1))),
        ], dim=-1)
        gains = torch.stack([
            _norm(per[:, 4:4 + nj].mean(-1), p_gain.get("stiffness_distribution_params", (1, 1))),
            _norm(per[:, 4 + nj:4 + 2 * nj].mean(-1), p_gain.get("damping_distribution_params", (1, 1))),
        ], dim=-1)
        act = env.action_manager.get_term(self.action_name)
        acfg = act.cfg
        zero = torch.zeros(self.num_envs, device=self.device)
        delay = _norm(act._delay.float(), acfg.delay_substeps_range) if hasattr(act, "_delay") else zero
        amax = _norm((act._amax / act._amax_nominal).mean(-1), acfg.amax_scale_range) if hasattr(act, "_amax") else zero
        play = _norm((act._play / torch.clamp(act._play_nominal, min=1e-9)).mean(-1), acfg.play_scale_range) \
            if hasattr(act, "_play") else zero
        servo = torch.stack([delay, amax, play], dim=-1)
        q_rel = math_utils.quat_mul(math_utils.quat_conjugate(self.robot.data.default_root_state[:, 3:7]),
                                    self.robot.data.root_quat_w)
        roll, pitch, _ = math_utils.euler_xyz_from_quat(q_rel)
        roll, pitch = math_utils.wrap_to_pi(roll), math_utils.wrap_to_pi(pitch)
        pose = (self._event_params(env, "reset_robot_base") or {}).get("pose_range", {})
        mount = torch.stack([_norm(roll, pose.get("roll", (0, 0))), _norm(pitch, pose.get("pitch", (0, 0)))], dim=-1)
        return torch.cat([cube, self._friction, gains, servo, mount], dim=-1)
