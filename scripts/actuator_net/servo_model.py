"""Differentiable 1-DoF model of an ORCA joint driven by the servo actuator net (pure torch; no Isaac Lab needed).

Per physics tick (dt = 1/120 s) and joint, the actuator does (ServoActuatorNet.compute in
source/faive_lab/faive_lab/tasks/orca_inhand/actuator_net.py, which must stay identical to rollout() below):

    q*_L   = joint target L ticks ago                   command delay, fitted per joint type
    y      = clip(y, q*_L - b, q*_L + b)                servo play (backlash / stiction band), fitted per type
    e, v   = q*_L - q, dq/dt                            pushed into history buffers (column 0 = now)
    delta  = ServoNet(e history, v history)             learned, bounded |delta| < delta_max
    target = y + delta                                  -> PhysX implicit PD drive, stiffness k, damping k * tau

Delay, play and tau are the physical prior (fit_prior); the net learns what they miss. The play acts on the target,
upstream of the drive, so the joint stays as stiff as the PD inside the band (a friction-locked servo), and it has no
feedback from the joint state. On v1/v1b chirps the measured joint stops ~2 deg short of every reversal and then
follows with that offset, which is what makes the chirp gain look like a flat -1.5 dB.

The stiff part stays in PhysX's implicit drive because an explicit torque cannot be stable here: the finger links
have 2e-6..7e-5 kg m^2 about their joints, and an explicit damping d is only stable for dt * d / I < 2, i.e.
d < 0.02 Nms/rad, while the drives need 0.25..2 Nms/rad (index_mcp: dt * d / I = 100 fitted, 300 in orca_v2.py).
pd_step() replicates the implicit update (implicit Euler of I qdd = k (target - q) - d qd - Coulomb friction) so the
net can be trained by backpropagating through it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from chirp_data import PHYSICS_DT


def pd_step(q, v, target, k, d, inertia, friction, dt: float = PHYSICS_DT):
    """One physics tick of a joint under an implicit PD drive with Coulomb joint friction (all args broadcast).

    Implicit Euler in position and velocity, as PhysX treats articulation drives: stable for any k, d, inertia > 0.
    Friction acts as a constraint: the joint stays at rest if holding it there needs at most ``friction``.
    """
    drive = k * (target - q)
    hold = inertia * v / dt + drive  # force that keeps v_new = 0
    v_new = (inertia * v + dt * (drive - friction * torch.sign(hold))) / (inertia + dt * d + dt * dt * k)
    v_new = torch.where(hold.abs() <= friction, torch.zeros_like(v_new), v_new)
    return q + dt * v_new, v_new


class ServoNet(nn.Module):
    """MLP on the history of (delayed-target error, joint velocity) -> bounded correction of the PD target (rad).

    Like Isaac Lab's ActuatorNetMLP (Hwangbo et al. 2019), ``input_idx`` picks taps of a per-tick history buffer;
    input normalization is stored in the module so the TorchScript file is self-contained (as in arXiv 2603.04351).
    """

    def __init__(self, input_idx: list[int], hidden: tuple[int, ...] = (32, 32), delta_max: float = 0.087):
        super().__init__()
        self.register_buffer("input_idx", torch.tensor(input_idx, dtype=torch.long))
        n_in = 2 * len(input_idx)
        self.register_buffer("in_mean", torch.zeros(n_in))
        self.register_buffer("in_std", torch.ones(n_in))
        self.delta_max = float(delta_max)
        layers: list[nn.Module] = []
        for n_out in hidden:
            layers += [nn.Linear(n_in, n_out), nn.Softsign()]
            n_in = n_out
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(n_in, 1)
        # start as the physical prior alone (delta = 0)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def features(self, err_hist: torch.Tensor, vel_hist: torch.Tensor) -> torch.Tensor:
        return torch.cat([err_hist.index_select(1, self.input_idx), vel_hist.index_select(1, self.input_idx)], dim=1)

    def forward(self, err_hist: torch.Tensor, vel_hist: torch.Tensor) -> torch.Tensor:
        """err_hist, vel_hist: (B, H) with column 0 = current tick. Returns delta (B,) in rad."""
        x = (self.features(err_hist, vel_hist) - self.in_mean) / self.in_std
        y = self.head(self.body(x)).squeeze(-1)
        return self.delta_max * torch.tanh(y / self.delta_max)


@dataclass
class JointParams:
    """Simulated joint + fitted servo prior, one entry per sequence in the batch (tensors of shape (B,))."""

    k: torch.Tensor
    tau: torch.Tensor
    """PD time constant damping / stiffness (s): what free motion identifies (the force scale k does not)."""
    inertia: torch.Tensor
    friction: torch.Tensor
    delay: torch.Tensor
    """Command delay in physics ticks (long)."""
    play: torch.Tensor
    """Half-width b of the servo play band (rad)."""

    @property
    def d(self) -> torch.Tensor:
        return self.k * self.tau


@dataclass
class RolloutState:
    q: torch.Tensor
    v: torch.Tensor
    targets: torch.Tensor
    """(B, max_delay + 1) past joint targets, column 0 = now."""
    play: torch.Tensor
    """Output y of the play operator (rad)."""
    err_hist: torch.Tensor
    vel_hist: torch.Tensor

    @staticmethod
    def at_rest(q0: torch.Tensor, max_delay: int, history_len: int) -> RolloutState:
        """Joint at rest at q0 with every past target = q0 (what the actuator assumes after a reset)."""
        b = q0.shape[0]
        return RolloutState(
            q=q0.clone(),
            v=torch.zeros_like(q0),
            targets=q0[:, None].repeat(1, max_delay + 1),
            play=q0.clone(),
            err_hist=q0.new_zeros(b, history_len),
            vel_hist=q0.new_zeros(b, history_len),
        )

    def detach(self) -> RolloutState:
        fields = (self.q, self.v, self.targets, self.play, self.err_hist, self.vel_hist)
        return RolloutState(*(t.detach() for t in fields))


def rollout(
    cmd: torch.Tensor, state: RolloutState, p: JointParams, net: ServoNet | None = None, residual_scale: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, RolloutState]:
    """Simulate (B, T) per-tick joint targets. Returns q (B, T + 1) incl. the start state, delta (B, T), end state."""
    qs, deltas = [state.q], []
    q, v, targets, y = state.q, state.v, state.targets, state.play
    err_hist, vel_hist = state.err_hist, state.vel_hist
    delay = p.delay[:, None]
    for n in range(cmd.shape[1]):
        targets = torch.cat([cmd[:, n : n + 1], targets[:, :-1]], dim=1)
        q_l = targets.gather(1, delay).squeeze(1)
        y = torch.minimum(torch.maximum(y, q_l - p.play), q_l + p.play)
        err_hist = torch.cat([(q_l - q)[:, None], err_hist[:, :-1]], dim=1)
        vel_hist = torch.cat([v[:, None], vel_hist[:, :-1]], dim=1)
        delta = residual_scale * net(err_hist, vel_hist) if net is not None else torch.zeros_like(q)
        q, v = pd_step(q, v, y + delta, p.k, p.d, p.inertia, p.friction)
        qs.append(q)
        deltas.append(delta)
    return torch.stack(qs, 1), torch.stack(deltas, 1), RolloutState(q, v, targets, y, err_hist, vel_hist)


def sample_at(q: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of q (B, T + 1) at fractional tick positions pos (B, M)."""
    i0 = pos.floor().long().clamp(0, q.shape[1] - 2)
    w = (pos - i0).clamp(0.0, 1.0)
    return q.gather(1, i0) * (1 - w) + q.gather(1, i0 + 1) * w


@dataclass
class SequenceBatch:
    """Recordings padded to a common length on the physics grid."""

    cmd: torch.Tensor  # (B, T) target per tick (rad)
    pos: torch.Tensor  # (B, M) tick position of each measured sample
    meas: torch.Tensor  # (B, M) measured position (rad)
    mask: torch.Tensor  # (B, M) 1 for real samples, 0 for padding

    @staticmethod
    def from_recordings(recordings, dt: float = PHYSICS_DT) -> SequenceBatch:
        from chirp_data import physics_grid

        grids = [physics_grid(r, dt) for r in recordings]
        n_t = max(len(c) for c, _ in grids)
        n_m = max(len(r.measured) for r in recordings)
        cmd, pos, meas, mask = (np.zeros((len(recordings), n)) for n in (n_t, n_m, n_m, n_m))
        for i, (r, (c, s)) in enumerate(zip(recordings, grids)):
            cmd[i, : len(c)], cmd[i, len(c) :] = c, c[-1]
            m = len(r.measured)
            pos[i, :m], meas[i, :m], mask[i, :m] = np.minimum(s, n_t - 1e-6), r.measured, 1.0
        f = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
        return SequenceBatch(f(cmd), f(pos), f(meas), f(mask))

    def repeat(self, n: int) -> SequenceBatch:
        return SequenceBatch(*(t.repeat(n, 1) for t in (self.cmd, self.pos, self.meas, self.mask)))

    @property
    def q0(self) -> torch.Tensor:
        return self.meas[:, 0]


def params_for(recordings, sim: dict, prior: dict | None = None) -> JointParams:
    """JointParams for each recording: simulated joint from ``sim`` (read_orca_v2_params) and delay / tau / play from
    ``prior`` (fit_prior); without a prior, the plain implicit PD of orca_v2.py (tau = damping / stiffness, no delay,
    no play)."""
    rows = []
    for r in recordings:
        s = sim[r.joint_type]
        if prior is None:
            delay, tau, play = 0, s["damping"] / s["stiffness"], 0.0
        else:
            delay, tau, play = (prior[r.joint_type][x] for x in ("delay_steps", "tau", "play"))
        rows.append((s["stiffness"], tau, s["inertia"] + s["armature"], s["friction"], delay, play))
    k, tau, inertia, friction, delay, play = zip(*rows)
    f = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    return JointParams(f(k), f(tau), f(inertia), f(friction), torch.tensor(delay, dtype=torch.long), f(play))


@torch.no_grad()
def _prior_mse(batch: SequenceBatch, base: JointParams, cand: list[tuple[int, float, float]]) -> torch.Tensor:
    """Free-run MSE of each (delay, tau, play) candidate, averaged over the recordings in ``batch``."""
    n, r = len(cand), batch.cmd.shape[0]
    col = lambda j, dtype=torch.float32: torch.tensor([c[j] for c in cand], dtype=dtype).repeat_interleave(r)  # noqa
    p = JointParams(
        base.k.repeat(n), col(1), base.inertia.repeat(n), base.friction.repeat(n), col(0, torch.long), col(2)
    )
    b = batch.repeat(n)  # row c * r + i = candidate c on recording i, as in p
    q, _, _ = rollout(b.cmd, RolloutState.at_rest(b.q0, max(c[0] for c in cand), 1), p)
    err = ((sample_at(q, b.pos) - b.meas) ** 2 * b.mask).sum(1) / b.mask.sum(1)
    return err.view(n, r).mean(1)


def fit_prior(
    recordings, sim: dict, delays=range(0, 25), taus=None, plays=None
) -> dict[str, dict[str, float]]:
    """Per joint type: command delay (ticks), PD time constant tau (s) and play half-width (rad) that minimise the
    free-run position RMSE over the recordings of that type.

    Exhaustive over a coarse (delay, tau, play) grid, then a finer (tau, play) grid around the best at delay +-1.
    The joint's stiffness, friction and inertia stay those of orca_v2.py (``sim``).
    """
    taus = np.geomspace(0.003, 0.3, 32) if taus is None else np.asarray(taus)
    plays = np.deg2rad(np.arange(0.0, 4.01, 0.4)) if plays is None else np.asarray(plays)
    prior = {}
    for jt in sorted({r.joint_type for r in recordings}):
        recs = [r for r in recordings if r.joint_type == jt]
        batch, base = SequenceBatch.from_recordings(recs), params_for(recs, sim)
        best = (np.inf, None)
        for d in delays:  # one batch per delay keeps memory small
            cand = [(d, t, b) for t in taus for b in plays]
            mse = _prior_mse(batch, base, cand)
            i = int(torch.argmin(mse))
            best = min(best, (float(mse[i]), cand[i]), key=lambda x: x[0])
        d0, t0, b0 = best[1]
        fine_b = np.clip(b0 + np.deg2rad(np.arange(-0.4, 0.41, 0.1)), 0.0, None)
        cand = [
            (d, t, b)
            for d in range(max(d0 - 1, 0), d0 + 2)
            for t in t0 * np.geomspace(0.87, 1.15, 9)
            for b in np.unique(fine_b)
        ]
        mse = _prior_mse(batch, base, cand)
        i = int(torch.argmin(mse))
        d, t, b = cand[i]
        prior[jt] = {
            "delay_steps": int(d),
            "tau": float(t),
            "play": float(b),
            "rmse_deg": float(np.rad2deg(np.sqrt(mse[i]))),
        }
    return prior


# --- export -------------------------------------------------------------------------------------------------------

META_FILE = "meta.json"
"""Name of the metadata JSON stored inside the TorchScript file (torch.jit ``_extra_files``)."""


def save_model(path, net: ServoNet, meta: dict) -> None:
    """TorchScript of the net + metadata (per joint type delay_steps / tau / play, history_len, ...) in one file."""
    import json
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(torch.jit.script(net.cpu().eval()), str(path), _extra_files={META_FILE: json.dumps(meta, indent=1)})


def load_model(path, device: str = "cpu"):
    """Returns (scripted net, metadata dict)."""
    import json

    extra = {META_FILE: ""}
    net = torch.jit.load(str(path), map_location=device, _extra_files=extra).eval()
    return net, json.loads(extra[META_FILE])


def prior_from_meta(meta: dict) -> dict[str, dict[str, float]]:
    return {jt: {k: v[k] for k in ("delay_steps", "tau", "play")} for jt, v in meta["joint_types"].items()}
