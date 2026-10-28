"""Fit the servo actuator net to chirp recordings (CPU, ~10 min) and export it for Isaac Lab.

No Isaac Sim needed. Folder arguments that do not exist relative to the cwd are looked up in
scripts/motor_param_characterization/. Results (metrics.csv, Bode plots, servo_net.pt) go to
scripts/actuator_net/results/<run_name>/; ``--deploy`` also writes the weights the env loads,
source/faive_lab/faive_lab/tasks/orca_inhand/actuator_net_weights/servo_net.pt.

  # held-out test on the reference data: fit on the ORCA v1 hand (2025-06), test on the v1b hand (2025-10)
  python scripts/actuator_net/train.py --train 202506_collected_realworld_data \
      --test 202510_orcav1b_realworld_data --run_name v1_to_v1b

  # deployed weights: all reference data
  python scripts/actuator_net/train.py --train 202506_collected_realworld_data 202510_orcav1b_realworld_data \
      --run_name v1_v1b_all --deploy

  # ORCA v2 (one chirp per joint type): train on the first 75 % of each chirp, test on the last 25 % (1.9-5 Hz)
  python scripts/actuator_net/train.py --train <logs/chirp_v2/<timestamp>> --holdout_tail 0.25 --run_name v2 --deploy

Steps:
  1. Physical prior per joint type (servo_model.fit_prior): command delay, servo play and PD time constant that
     minimise the free-run position error, with orca_v2.py's stiffness / friction and the USD link inertia.
  2. Residual net shared by all joint types (one servo model, as the "robot-agnostic" model of arXiv 2603.04351):
     trained by backpropagating the position error through the 120 Hz replica of the implicit PD joint
     (truncated BPTT over 1 s segments of the full free-run rollouts), with a small penalty on the residual.
  3. Export (TorchScript + metadata) and evaluation against the plain implicit PD (evaluate.py).
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from chirp_data import ORCA_V2_PY, PHYSICS_DT, find_recordings, read_orca_v2_params
from evaluate import RESULTS_ROOT, evaluate, summary_table
from servo_model import (
    JointParams,
    RolloutState,
    SequenceBatch,
    ServoNet,
    fit_prior,
    params_for,
    rollout,
    sample_at,
    save_model,
)

DEPLOY_PATH = Path(__file__).resolve().parents[2] / (
    "source/faive_lab/faive_lab/tasks/orca_inhand/actuator_net_weights/servo_net.pt"
)
INPUT_IDX = [0, 1, 2, 3, 4, 6, 8, 12, 18, 24]
"""History taps in physics steps (0-200 ms). The command delay is explicit, so the net only needs the servo's own
response time, not the latency on top (Hwangbo et al.: history longer than delays + response time)."""
MAX_DELAY_STEPS = 24
"""Largest command delay searched (200 ms); also the delay buffer length in the actuator."""


def truncate(rec, keep: float):
    """First ``keep`` fraction of a recording (by time)."""
    m = rec.time <= keep * rec.time[-1]
    return replace(rec, time=rec.time[m], commanded=rec.commanded[m], measured=rec.measured[m])


@torch.no_grad()
def set_normalization(net: ServoNet, batch: SequenceBatch, p: JointParams) -> None:
    """Input mean / std from a rollout of the prior alone (the net's input distribution at the start of training)."""
    q, _, _ = rollout(batch.cmd, RolloutState.at_rest(batch.q0, int(p.delay.max()), 1), p)
    n_t = batch.cmd.shape[1]
    padded = torch.cat([batch.q0[:, None].expand(-1, MAX_DELAY_STEPS), batch.cmd], dim=1)
    idx = MAX_DELAY_STEPS + torch.arange(n_t)[None, :] - p.delay[:, None]
    err = padded.gather(1, idx) - q[:, :-1]
    vel = torch.diff(q, dim=1, prepend=q[:, :1])[:, :-1] / PHYSICS_DT
    m = len(net.input_idx)
    net.in_mean.copy_(torch.tensor([err.mean().item()] * m + [vel.mean().item()] * m))
    net.in_std.copy_(torch.tensor([err.std().item()] * m + [vel.std().item()] * m))


def train_net(recordings, sim: dict, prior: dict, args) -> ServoNet:
    torch.manual_seed(args.seed)
    batch = SequenceBatch.from_recordings(recordings)
    p = params_for(recordings, sim, prior)
    net = ServoNet(args.input_idx, tuple(args.hidden), np.deg2rad(args.delta_max_deg))
    set_normalization(net, batch, p)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=0.1 * args.lr)
    n_t, seg = batch.cmd.shape[1], args.segment
    history_len = max(args.input_idx) + 1
    for epoch in range(args.epochs):
        t0, sq, cnt, reg = time.time(), 0.0, 0.0, 0.0
        state = RolloutState.at_rest(batch.q0, int(p.delay.max()), history_len)
        for n0 in range(0, n_t, seg):
            n1 = min(n0 + seg, n_t)
            q, delta, state = rollout(batch.cmd[:, n0:n1], state, p, net)
            w = batch.mask * (batch.pos >= n0) * (batch.pos < n1)
            se = ((sample_at(q, batch.pos - n0) - batch.meas) ** 2 * w).sum()
            penalty = (delta**2).mean()
            loss = se / w.sum().clamp(min=1.0) + args.residual_penalty * penalty
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            state = state.detach()
            sq, cnt, reg = sq + se.item(), cnt + w.sum().item(), reg + penalty.item() * (n1 - n0) / n_t
        sched.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(
                f"[epoch {epoch:3d}] train RMSE {np.rad2deg(np.sqrt(sq / cnt)):.3f} deg | residual RMS"
                f" {np.rad2deg(np.sqrt(reg)):.2f} deg | {time.time() - t0:.1f} s"
            )
    return net.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", nargs="+", required=True, help="folders / CSVs of chirp recordings to fit")
    parser.add_argument("--test", nargs="*", default=[], help="held-out folders / CSVs (evaluation only)")
    parser.add_argument("--holdout_tail", type=float, default=0.0, help="hold out the last fraction of every chirp")
    parser.add_argument("--naming", default="auto", choices=("auto", "v1b", "v2"), help="joint naming of the files")
    parser.add_argument("--exclude", nargs="*", default=["p1200"], help="skip files whose name contains these "
                        "(default: the v1 index_mcp chirp with a non-default servo P gain)")
    parser.add_argument("--run_name", default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--deploy", action="store_true", help=f"also write the weights to {DEPLOY_PATH}")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--segment", type=int, default=120, help="truncated BPTT segment (physics steps)")
    parser.add_argument("--hidden", type=int, nargs="+", default=[32, 32])
    parser.add_argument("--input_idx", type=int, nargs="+", default=INPUT_IDX, help="history taps (physics steps)")
    parser.add_argument("--delta_max_deg", type=float, default=5.0, help="bound of the learned target correction")
    parser.add_argument("--residual_penalty", type=float, default=0.01, help="weight of mean(delta^2) in the loss")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = RESULTS_ROOT / args.run_name
    sim = read_orca_v2_params()
    train_recs = find_recordings(args.train, args.naming, tuple(args.exclude))
    test_recs = find_recordings(args.test, args.naming, tuple(args.exclude)) if args.test else []
    fit_recs = [truncate(r, 1.0 - args.holdout_tail) for r in train_recs] if args.holdout_tail > 0 else train_recs
    print(f"[INFO] {len(fit_recs)} training recordings, {len(test_recs)} test recordings -> {out_dir}")

    t0 = time.time()
    prior = fit_prior(fit_recs, sim, delays=range(0, MAX_DELAY_STEPS + 1))
    print(f"[INFO] prior fitted in {time.time() - t0:.0f} s")
    for jt, v in prior.items():
        print(
            f"  {jt:10s} delay {v['delay_steps']:2d} steps ({1e3 * v['delay_steps'] * PHYSICS_DT:3.0f} ms)"
            f"  tau {1e3 * v['tau']:5.1f} ms  play +-{np.rad2deg(v['play']):.1f} deg  RMSE {v['rmse_deg']:.2f} deg"
        )
    net = train_net(fit_recs, sim, prior, args)

    meta = {
        "format": 1,
        "physics_dt": PHYSICS_DT,
        "history_len": max(args.input_idx) + 1,
        "input_idx": args.input_idx,
        "max_delay_steps": MAX_DELAY_STEPS,
        "delta_max": float(net.delta_max),
        "joint_types": {
            jt: {**v, "stiffness": sim[jt]["stiffness"], "damping": sim[jt]["stiffness"] * v["tau"],
                 "friction": sim[jt]["friction"], "inertia": sim[jt]["inertia"]}
            for jt, v in prior.items()
        },
        "trained_on": [str(r.path.relative_to(r.path.parents[1])) for r in train_recs],
        "holdout_tail": args.holdout_tail,
        "orca_v2_py": str(ORCA_V2_PY.name),
        "args": vars(args),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    save_model(out_dir / "servo_net.pt", net, meta)
    if args.deploy:
        DEPLOY_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(out_dir / "servo_net.pt", DEPLOY_PATH)
        print(f"[INFO] deployed to {DEPLOY_PATH}")

    with torch.no_grad():
        if test_recs:
            df = evaluate(test_recs, sim, prior, net, out_dir / "test")
            print("\nHeld-out recordings (" + ", ".join(args.test) + "):\n" + summary_table(df))
        if args.holdout_tail > 0:
            df = evaluate(train_recs, sim, prior, net, out_dir / "tail", tail=args.holdout_tail)
            print(f"\nHeld-out last {args.holdout_tail:.0%} of each chirp (RMSE on it; FRF over the whole chirp):")
            print(summary_table(df))
        df = evaluate(train_recs, sim, prior, net, out_dir / "train", plots=not test_recs)
        print("\nTraining recordings (in-sample):\n" + summary_table(df))
    (out_dir / "prior.json").write_text(json.dumps(meta["joint_types"], indent=1))


if __name__ == "__main__":
    main()
