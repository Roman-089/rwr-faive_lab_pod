"""Subtask 2: the cube pose as the camera tracker delivers it, for the policy observation (docs/ablation/exteroception.md).

The real tracker (rwr_faive_system cube_detection_node.py) finds the 8 cube corners with Mask R-CNN, solves PnP and
smooths the result (Kalman filter on the position, slerp with alpha 0.25 on the orientation), at most 20 Hz; the policy
node reads the last pose from TF and treats a pose older than 0.5 s as "not tracked". ``TrackedCubePose`` imitates that
chain per env, with its parameters randomised per episode:

  true pose -> latency (0-3 steps) -> calibration offset (per episode) + per-step noise -> detection or miss
  (independent misses + bursts of lost tracking, e.g. fingers covering the cube) -> smoothing (position EMA, orientation
  slerp) -> held while there is no detection -> symmetry-canonical quaternion + tracked flag (last detection <= 0.5 s ago).

Output (8): object position in the env frame (3, as ``mdp.root_pos_w``), object quaternion (w, x, y, z) (4) and the
tracked flag (1). Before any detection (``force_lost``, the tracker-off eval) the pose is the command's target position
with the identity orientation and the flag is 0.

Cube symmetry: a plain cube looks the same under 24 rotations, so the corner-based tracker cannot tell them apart and its
orientation can jump by 90 deg when the corner labels flip. Smoothing compares each measurement with the symmetric
equivalent closest to the current estimate, and the observation is the equivalent closest to the identity (w >= 0). The
policy node must apply the same ``canonical_cube_quat`` to the tracker's quaternion.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of (w, x, y, z) quaternions, broadcasting over leading dims."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw), dim=-1)


def quat_from_rotvec(v: torch.Tensor) -> torch.Tensor:
    """(w, x, y, z) from rotation vectors (..., 3)."""
    angle = v.norm(dim=-1, keepdim=True)
    axis = v / angle.clamp_min(1e-9)
    return torch.cat((torch.cos(angle / 2), axis * torch.sin(angle / 2)), dim=-1)


def cube_symmetry_quats(device=None) -> torch.Tensor:
    """The 24 proper rotations that map a cube onto itself, as (24, 4) (w, x, y, z) quaternions."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = torch.zeros(3, 3, dtype=torch.float64)
            for row, (col, s) in enumerate(zip(perm, signs)):
                m[row, col] = s
            if torch.det(m) > 0:
                mats.append(m)
    quats = []
    for m in mats:  # rotation matrix -> quaternion (robust branch on the largest diagonal term)
        t = m.trace()
        if t > 0:
            s = math.sqrt(t + 1.0) * 2
            q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        else:
            i = int(torch.argmax(torch.diagonal(m)))
            j, k = (i + 1) % 3, (i + 2) % 3
            s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
            q = [0.0] * 4
            q[0] = (m[k, j] - m[j, k]) / s
            q[1 + i] = 0.25 * s
            q[1 + j] = (m[j, i] + m[i, j]) / s
            q[1 + k] = (m[k, i] + m[i, k]) / s
        quats.append([float(x) for x in q])
    return torch.tensor(quats, dtype=torch.float32, device=device)


def closest_symmetric(q: torch.Tensor, ref: torch.Tensor, sym: torch.Tensor) -> torch.Tensor:
    """The equivalent q * g (g in the cube's symmetry group, body frame) closest to ``ref``, with the sign that faces it."""
    cand = quat_mul(q.unsqueeze(-2), sym)  # (..., 24, 4)
    dots = (cand * ref.unsqueeze(-2)).sum(-1)
    best = dots.abs().argmax(-1, keepdim=True)
    out = torch.gather(cand, -2, best.unsqueeze(-1).expand(*best.shape, 4)).squeeze(-2)
    sign = torch.sign(torch.gather(dots, -1, best))
    return out * torch.where(sign == 0, torch.ones_like(sign), sign)


def canonical_cube_quat(q: torch.Tensor, sym: torch.Tensor) -> torch.Tensor:
    """Symmetry-canonical orientation: the equivalent closest to the identity, w >= 0."""
    ident = torch.zeros_like(q)
    ident[..., 0] = 1.0
    return closest_symmetric(q, ident, sym)


def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical interpolation from q0 to q1 (same hemisphere assumed) by fraction t (..., 1)."""
    dot = (q0 * q1).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot.abs())
    sin_t = torch.sin(theta)
    small = sin_t < 1e-4
    w0 = torch.where(small, 1.0 - t, torch.sin((1.0 - t) * theta) / sin_t.clamp_min(1e-9))
    w1 = torch.where(small, t, torch.sin(t * theta) / sin_t.clamp_min(1e-9))
    out = w0 * q0 + w1 * q1
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-9)


TRACKER_DEFAULTS = {
    "delay_steps": (0, 3), "pos_noise_m": (0.0, 0.003), "rot_noise_deg": (0.0, 5.0), "pos_offset_m": 0.004,
    "rot_offset_deg": 3.0, "p_miss": (0.0, 0.3), "p_lose": (0.0, 0.05), "p_recover": (0.1, 0.5),
    "alpha_pos": (0.3, 1.0), "alpha_rot": (0.2, 0.6), "stale_steps": 10, "force_lost": False,
}


class TrackedCubePose(ManagerTermBase):
    """Cube pose as seen through the camera tracker (module docstring). Parameters (``params`` of the ObsTerm), each
    (lo, hi) range sampled per env and episode unless noted:

    delay_steps: latency in control steps (int range); pos_noise_m / rot_noise_deg: per-step Gaussian std;
    pos_offset_m / rot_offset_deg: max per-episode calibration offset (scalar, uniform in +-); p_miss: per-step chance
    that a frame gives no pose; p_lose / p_recover: per-step chance to enter / leave a lost burst; alpha_pos / alpha_rot:
    smoothing gains (1 = no smoothing); stale_steps: steps without a detection before the flag drops (scalar);
    force_lost: no detections at all (tracker-off eval).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.p = {**TRACKER_DEFAULTS, **{k: v for k, v in cfg.params.items() if k in TRACKER_DEFAULTS}}
        self.object = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("object")).name]
        self.command_name = cfg.params.get("command_name", "continuous_rotation")
        n, dev = env.num_envs, env.device
        self.max_delay = int(self.p["delay_steps"][1])
        self.sym = cube_symmetry_quats(dev)
        self.buf_pos = torch.zeros(self.max_delay + 1, n, 3, device=dev)
        self.buf_quat = torch.zeros(self.max_delay + 1, n, 4, device=dev)
        self.filt_pos = torch.zeros(n, 3, device=dev)
        self.filt_quat = torch.zeros(n, 4, device=dev)
        self.filt_quat[:, 0] = 1.0
        self.since = torch.full((n,), 10**6, dtype=torch.long, device=dev)
        self.lost = torch.zeros(n, dtype=torch.bool, device=dev)
        self.needs_init = torch.ones(n, dtype=torch.bool, device=dev)
        self.delay = torch.zeros(n, dtype=torch.long, device=dev)
        self.par = {k: torch.zeros(n, device=dev) for k in
                    ("pos_noise_m", "rot_noise_deg", "p_miss", "p_lose", "p_recover", "alpha_pos", "alpha_rot")}
        self.off_pos = torch.zeros(n, 3, device=dev)
        self.off_quat = torch.zeros(n, 4, device=dev)
        self.off_quat[:, 0] = 1.0
        self.out = torch.zeros(n, 8, device=dev)
        self.last_step = -1
        self.reset()

    def _sample(self, key, ids):
        lo, hi = self.p[key]
        return torch.empty(len(ids), device=self.filt_pos.device).uniform_(lo, hi)

    def reset(self, env_ids=None):
        ids = torch.arange(self.filt_pos.shape[0], device=self.filt_pos.device) if env_ids is None else \
            torch.as_tensor(env_ids, device=self.filt_pos.device)
        if len(ids) == 0:
            return
        p = self.p
        lo, hi = p["delay_steps"]
        self.delay[ids] = torch.randint(int(lo), int(hi) + 1, (len(ids),), device=ids.device)
        for k in self.par:
            self.par[k][ids] = self._sample(k, ids)
        po = float(p["pos_offset_m"])
        self.off_pos[ids] = torch.empty(len(ids), 3, device=ids.device).uniform_(-po, po)
        ro = math.radians(float(p["rot_offset_deg"]))
        self.off_quat[ids] = quat_from_rotvec(torch.empty(len(ids), 3, device=ids.device).uniform_(-ro, ro))
        self.lost[ids] = False
        self.since[ids] = 10**6
        self.needs_init[ids] = True

    def __call__(  # every parameter listed: the manager checks params against this signature; values come from self.p
        self, env: ManagerBasedRLEnv, asset_cfg=None, command_name=None, delay_steps=None, pos_noise_m=None,
        rot_noise_deg=None, pos_offset_m=None, rot_offset_deg=None, p_miss=None, p_lose=None, p_recover=None,
        alpha_pos=None, alpha_rot=None, stale_steps=None, force_lost=None,
    ) -> torch.Tensor:
        step = int(env.common_step_counter)
        if step == self.last_step and not bool(self.needs_init.any()):
            return self.out  # compute() called twice in one step: don't advance the tracker
        self.last_step = step
        p = self.p
        true_pos = self.object.data.root_pos_w - env.scene.env_origins
        true_quat = self.object.data.root_quat_w
        init = self.needs_init
        nominal_pos = env.command_manager.get_command(self.command_name)[:, :3]
        if bool(init.any()):  # fresh episode: the delay line starts full of the current pose
            self.buf_pos[:, init] = true_pos[init]
            self.buf_quat[:, init] = true_quat[init]
            self.filt_pos[init] = nominal_pos[init]
            self.filt_quat[init] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=true_pos.device)
        self.buf_pos = torch.roll(self.buf_pos, 1, dims=0)
        self.buf_quat = torch.roll(self.buf_quat, 1, dims=0)
        self.buf_pos[0] = true_pos
        self.buf_quat[0] = true_quat
        ar = torch.arange(true_pos.shape[0], device=true_pos.device)
        meas_pos = self.buf_pos[self.delay, ar] + self.off_pos
        meas_pos = meas_pos + torch.randn_like(meas_pos) * self.par["pos_noise_m"].unsqueeze(-1)
        rot_sigma = torch.deg2rad(self.par["rot_noise_deg"]).unsqueeze(-1)
        noise_q = quat_from_rotvec(torch.randn_like(meas_pos) * rot_sigma)
        meas_quat = quat_mul(noise_q, quat_mul(self.off_quat, self.buf_quat[self.delay, ar]))
        # detection: bursts of lost tracking plus independent misses
        u = torch.rand_like(self.par["p_lose"])
        self.lost = torch.where(self.lost, u >= self.par["p_recover"], u < self.par["p_lose"])
        detected = ~self.lost & (torch.rand_like(u) >= self.par["p_miss"])
        if p["force_lost"]:
            detected[:] = False
        first = detected & (self.since >= 10**6)  # first detection of the episode: no smoothing from the nominal pose
        meas_quat = closest_symmetric(meas_quat, self.filt_quat, self.sym)
        a_p = torch.where(first, 1.0, self.par["alpha_pos"]).unsqueeze(-1)
        a_q = torch.where(first, 1.0, self.par["alpha_rot"]).unsqueeze(-1)
        d = detected.unsqueeze(-1)
        self.filt_pos = torch.where(d, self.filt_pos + a_p * (meas_pos - self.filt_pos), self.filt_pos)
        self.filt_quat = torch.where(d, quat_slerp(self.filt_quat, meas_quat, a_q), self.filt_quat)
        self.since = torch.where(detected, torch.zeros_like(self.since), (self.since + 1).clamp_max(10**6))
        self.needs_init[:] = False
        flag = (self.since <= int(p["stale_steps"])).float().unsqueeze(-1)
        self.out = torch.cat((self.filt_pos, canonical_cube_quat(self.filt_quat, self.sym), flag), dim=-1)
        return self.out
