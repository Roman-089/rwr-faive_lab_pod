"""Compare the servo actuator net with the current implicit PD on chirp recordings: position RMSE and Bode gain/phase.

Three models are simulated open loop from the start of each recording, with the same 1-DoF replica of the PhysX joint
(servo_model.pd_step, link inertia from the v2 USD, stiffness / friction / effort limit from orca_v2.py):

  pd_orca_v2     the current sim: implicit PD with orca_v2.py's stiffness and damping, no delay
  prior          the net's physical prior alone: fitted command delay + servo play + PD time constant
  actuator_net   prior + learned residual (what ServoActuatorNet runs in Isaac Lab)

The FRF uses scripts/motor_param_characterization/signal_processing.py (mean removal, Tukey window, FFT ratio), i.e.
the same Bode plot as optimization.py / plot_chirp_utils.py.

Evaluate a trained model on other recordings (e.g. the v1/v1b model on new ORCA v2 chirps):

  python scripts/actuator_net/evaluate.py --model <servo_net.pt> --data <v2_dir> --run_name v1b_model_on_v2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from chirp_data import CHIRP_DATA_ROOT, Recording, find_recordings, read_orca_v2_params  # noqa: E402
from servo_model import (  # noqa: E402
    RolloutState,
    SequenceBatch,
    load_model,
    params_for,
    prior_from_meta,
    rollout,
    sample_at,
)

sys.path.insert(0, str(CHIRP_DATA_ROOT))
from signal_processing import compute_frequency_response_from_dataframe  # noqa: E402

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
FRF_BAND = (0.1, 4.5)
"""Hz; the v1 example fit (motor_param_characterization/example_optimization_output) stops at 4.5 Hz as well."""

MODELS = ("pd_orca_v2", "prior", "actuator_net")
LABELS = {
    "real": "measured",
    "pd_orca_v2": "implicit PD (orca_v2.py)",
    "prior": "delay + play + PD (fit)",
    "actuator_net": "actuator net",
}
# measured in secondary ink, models in the first three slots of the validated categorical palette (dataviz skill)
COLORS = {"real": "#52514e", "pd_orca_v2": "#2a78d6", "prior": "#eb6834", "actuator_net": "#1baf7a"}


@torch.no_grad()
def simulate(recordings: list[Recording], sim: dict, prior: dict | None, net=None) -> list[np.ndarray]:
    """Simulated joint position (rad) at each recording's measured timestamps."""
    batch = SequenceBatch.from_recordings(recordings)
    p = params_for(recordings, sim, prior)
    history_len = int(net.input_idx.max()) + 1 if net is not None else 1
    q, _, _ = rollout(batch.cmd, RolloutState.at_rest(batch.q0, int(p.delay.max()), history_len), p, net)
    q = sample_at(q, batch.pos).numpy()
    return [q[i, : len(r.measured)] for i, r in enumerate(recordings)]


def frf(rec: Recording, position: np.ndarray):
    df = pd.DataFrame(
        {
            "timestamp": rec.time,
            "commanded_position": np.rad2deg(rec.commanded),
            "measured_position": np.rad2deg(position),
        }
    )
    _, response = compute_frequency_response_from_dataframe(df, min_frequency=FRF_BAND[0], max_frequency=FRF_BAND[1])
    return response


def evaluate(
    recordings: list[Recording],
    sim: dict,
    prior: dict,
    net,
    out_dir: Path,
    tail: float = 0.0,
    plots: bool = True,
) -> pd.DataFrame:
    """Per recording and model: RMSE (deg; only over the last ``tail`` fraction if > 0) and mean |gain error| (dB) /
    |phase error| (deg) of the FRF against the measured one over FRF_BAND. Saves metrics.csv and bode_*.png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sims = {
        "pd_orca_v2": simulate(recordings, sim, None),
        "prior": simulate(recordings, sim, prior),
        "actuator_net": simulate(recordings, sim, prior, net),
    }
    rows, plotted = [], set()
    for i, rec in enumerate(recordings):
        real = frf(rec, rec.measured)
        sel = rec.time >= (1.0 - tail) * rec.time[-1] if tail > 0 else np.ones_like(rec.time, bool)
        curves = {"real": (rec.measured, real)}
        for m in MODELS:
            pos = sims[m][i]
            h = frf(rec, pos)
            curves[m] = (pos, h)
            rows.append(
                {
                    "joint_type": rec.joint_type,
                    "recording": rec.name,
                    "model": m,
                    "rmse_deg": float(np.rad2deg(np.sqrt(np.mean((pos[sel] - rec.measured[sel]) ** 2)))),
                    "gain_err_db": float(np.mean(np.abs(h.magnitude_db - real.magnitude_db))),
                    "phase_err_deg": float(np.mean(np.abs(np.rad2deg(np.angle(h.response / real.response))))),
                }
            )
        if plots and rec.joint_type not in plotted:  # one figure per joint type keeps the results folder small
            plotted.add(rec.joint_type)
            plot_recording(rec, curves, out_dir / f"bode_{rec.joint_type}.png", tail)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False, float_format="%.4f")
    return df


def summary_table(df: pd.DataFrame) -> str:
    """Markdown table: per joint type (mean over its recordings), RMSE / gain / phase error for each model."""
    agg = df.groupby(["joint_type", "model"])[["rmse_deg", "gain_err_db", "phase_err_deg"]].mean().unstack("model")
    lines = [
        "| joint type | n | RMSE deg: PD / prior / net | gain err dB: PD / prior / net"
        " | phase err deg: PD / prior / net |",
        "|---|---|---|---|---|",
    ]
    counts = df[df.model == MODELS[0]].groupby("joint_type").size()
    order = [jt for jt in ("wrist", "thumb_cmc", "thumb_abd", "thumb_mcp", "thumb_dip", "index_abd", "index_mcp",
                           "index_pip") if jt in agg.index]
    for jt in order + ["mean"]:
        row = agg.mean() if jt == "mean" else agg.loc[jt]
        cells = [" / ".join(f"{row[(metric, m)]:.{d}f}" for m in MODELS) for metric, d in
                 (("rmse_deg", 2), ("gain_err_db", 2), ("phase_err_deg", 1))]
        lines.append(f"| {jt} | {counts.get(jt, len(df) // len(MODELS))} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_recording(rec: Recording, curves: dict, path: Path, tail: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    (ax_mag, ax_ph), (ax_lo, ax_hi) = axes
    t_end = rec.time[-1]
    for key, (pos, h) in curves.items():
        style = {"color": COLORS[key], "lw": 2.4 if key == "real" else 1.4, "label": LABELS[key]}
        ax_mag.semilogx(h.frequency, h.magnitude_db, **style)
        ax_ph.semilogx(h.frequency, np.rad2deg(np.unwrap(np.angle(h.response))), **style)
        for ax, (lo, hi) in ((ax_lo, (2.0, 8.0)), (ax_hi, (t_end - 3.0, t_end))):
            m = (rec.time >= lo) & (rec.time <= hi)
            ax.plot(rec.time[m], np.rad2deg(pos[m]), **style)
    for ax, (lo, hi) in ((ax_lo, (2.0, 8.0)), (ax_hi, (t_end - 3.0, t_end))):
        m = (rec.time >= lo) & (rec.time <= hi)
        ax.plot(rec.time[m], np.rad2deg(rec.commanded[m]), color="#0b0b0b", lw=0.8, ls=":", label="command")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("position [deg]")
    ax_lo.set_title("low frequency (play: stops short at reversals)", fontsize=9)
    ax_hi.set_title("end of chirp (~5 Hz)" + (" - held out" if tail > 0 else ""), fontsize=9)
    ax_mag.set_ylabel("gain [dB]")
    ax_ph.set_ylabel("phase [deg]")
    for ax in (ax_mag, ax_ph):
        ax.set_xlabel("frequency [Hz]")
    for ax in axes.flat:
        ax.grid(True, color="#d9d8d4", lw=0.6)
        ax.tick_params(labelsize=8)
    ax_mag.legend(fontsize=8, loc="lower left")
    fig.suptitle(f"{rec.joint_type}: {rec.name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=72)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="servo_net.pt written by train.py")
    parser.add_argument("--data", nargs="+", required=True, help="folders / CSVs of chirp recordings")
    parser.add_argument("--naming", default="auto", choices=("auto", "v1b", "v2"))
    parser.add_argument("--exclude", nargs="*", default=["p1200"], help="skip files whose name contains these")
    parser.add_argument("--run_name", default="eval")
    args = parser.parse_args()

    net, meta = load_model(args.model)
    recordings = find_recordings(args.data, args.naming, tuple(args.exclude))
    out_dir = RESULTS_ROOT / args.run_name
    df = evaluate(recordings, read_orca_v2_params(), prior_from_meta(meta), net, out_dir)
    print(summary_table(df))
    print(f"[INFO] metrics and Bode plots in {out_dir}")


if __name__ == "__main__":
    main()
