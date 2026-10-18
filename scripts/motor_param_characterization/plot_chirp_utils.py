#!/usr/bin/env python3
"""
Functions to visualize chirp raw data and frequency responses.
Plot just one sweep, or multiple to compare them together.

python plot_chirp_utils.py <path_to_csv1> <path_to_csv2> ... --labels <label1> <label2> ... --show
e.g. python plot_chirp_utils.py 202506_collected_realworld_data/chirp_index_abd_20250606_170437_ctrl40Hz_20.0s_0.1-5.0Hz_10deg.csv --labels real --show
"""

from __future__ import annotations

# Standard library
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local
from signal_processing import (
    ChirpProcessingResult,
    FrequencyResponse,
    compute_frequency_response_from_dataframe,
)


plt.style.use("dark_background")


@dataclass(slots=True)
class ChirpDataset:
    """Container combining processed chirp data and its frequency response."""

    label: str
    source: Path
    processed: ChirpProcessingResult
    frequency_response: FrequencyResponse


def plot_chirp_frequency_response(
    datasets: Sequence[ChirpDataset],
    *,
    title: str = "Chirp signal and frequency response",
    show_window: bool = True,
) -> plt.Figure:
    """Create a figure comparing multiple chirp experiments."""

    if not datasets:
        raise ValueError("At least one dataset is required")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_time, ax_window), (ax_mag, ax_phase) = axes
    fig.suptitle(title, color="white", fontsize=18)

    colors = plt.get_cmap("tab10")
    for idx, dataset in enumerate(datasets):
        color = colors(idx % 10)
        proc = dataset.processed
        frf = dataset.frequency_response

        ax_time.plot(
            proc.time,
            proc.commanded,
            linestyle="-",
            color=color,
            label=f"{dataset.label} – commanded",
        )
        ax_time.plot(
            proc.time,
            proc.measured,
            linestyle="--",
            color=color,
            label=f"{dataset.label} – measured",
        )

        ax_window.plot(
            proc.time,
            proc.commanded_windowed,
            linestyle="-",
            color=color,
            label=f"{dataset.label} – commanded",
        )
        ax_window.plot(
            proc.time,
            proc.measured_windowed,
            linestyle="--",
            color=color,
            label=f"{dataset.label} – measured",
        )
        if show_window:
            ax_window.plot(
                proc.time,
                proc.window * np.max(np.abs(proc.commanded_windowed)),
                linestyle=":",
                color=color,
                alpha=0.6,
            )

        ax_mag.semilogx(
            frf.frequency,
            frf.magnitude_db,
            color=color,
            label=dataset.label,
        )
        ax_phase.semilogx(
            frf.frequency,
            frf.phase_deg,
            color=color,
            label=dataset.label,
        )

    ax_time.set_title("Chirp response")
    ax_time.set_xlabel("Time [s]")
    ax_time.set_ylabel("Position")
    ax_time.grid(True, color="gray", alpha=0.3)

    ax_window.set_title("Mean removed & Tukey window applied")
    ax_window.set_xlabel("Time [s]")
    ax_window.set_ylabel("Position")
    ax_window.grid(True, color="gray", alpha=0.3)

    ax_mag.set_title("Bode magnitude")
    ax_mag.set_xlabel("Frequency [Hz]")
    ax_mag.set_ylabel("Magnitude [dB]")
    ax_mag.grid(True, color="gray", alpha=0.3)

    ax_phase.set_title("Bode phase")
    ax_phase.set_xlabel("Frequency [Hz]")
    ax_phase.set_ylabel("Phase [deg]")
    ax_phase.grid(True, color="gray", alpha=0.3)

    for ax in (ax_time, ax_window, ax_mag, ax_phase):
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        ax.set_facecolor("black")

    ax_time.legend(loc="best")
    ax_window.legend(loc="best")
    ax_mag.legend(loc="best")
    ax_phase.legend(loc="best")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.patch.set_facecolor("black")
    return fig


def sanitise_filename_component(text: str) -> str:
    """Return a filesystem-friendly label."""

    safe = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in text)
    return safe.strip("-") or "dataset"


def _build_datasets(
    csv_paths: List[Path],
    labels: List[str],
    *,
    min_freq: float,
    max_freq: float,
    tukey_alpha: float,
) -> List[ChirpDataset]:
    datasets: List[ChirpDataset] = []
    for path, label in zip(csv_paths, labels):
        df = pd.read_csv(path)
        processed, frf = compute_frequency_response_from_dataframe(
            df,
            min_frequency=min_freq,
            max_frequency=max_freq,
            tukey_alpha=tukey_alpha,
        )
        datasets.append(
            ChirpDataset(
                label=label,
                source=path,
                processed=processed,
                frequency_response=frf,
            )
        )
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot chirp CSV data and overlay the frequency response."
    )
    parser.add_argument("csv_paths", nargs="+", help="Path(s) to the chirp CSV file(s).")
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional labels for each dataset (defaults to filename stems).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("logs"),
        help="Directory where plots and FRF CSVs will be stored.",
    )
    parser.add_argument(
        "--min_freq",
        type=float,
        default=0.1,
        help="Lower frequency bound for the FRF computation (Hz).",
    )
    parser.add_argument(
        "--max_freq",
        type=float,
        default=5.0,
        help="Upper frequency bound for the FRF computation (Hz).",
    )
    parser.add_argument(
        "--tukey_alpha",
        type=float,
        default=0.1,
        help="Alpha parameter of the Tukey window.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively instead of saving only.",
    )

    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csv_paths]
    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    if args.labels and len(args.labels) != len(csv_paths):
        raise ValueError("Number of labels must match the number of CSV files")

    labels = [p.stem for p in csv_paths]
    if args.labels:
        labels = args.labels

    datasets = _build_datasets(
        csv_paths,
        labels,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        tukey_alpha=args.tukey_alpha,
    )

    figure_title = "Chirp signal and frequency response"
    fig = plot_chirp_frequency_response(datasets, title=figure_title)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if len(datasets) == 1:
        dataset = datasets[0]
        prefix = dataset.label
        figure_name = f"{prefix}_{dataset.source.stem}.png"
    else:
        prefix = "chirp"
        combined = "_vs_".join(
            sanitise_filename_component(dataset.label) for dataset in datasets
        )
        figure_name = f"{prefix}_{combined}_comparison.png"

    figure_path = args.output_dir / figure_name
    fig.savefig(figure_path, facecolor=fig.get_facecolor())
    print(f"[INFO]: Plot saved to {figure_path}")

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        plt.close(fig)

    for dataset in datasets:
        prefix = dataset.label
        freq_filename = f"{prefix}_{dataset.source.stem}_freq_response.csv"
        freq_path = args.output_dir / freq_filename
        frf = dataset.frequency_response
        pd.DataFrame(
            {
                "frequency": frf.frequency,
                "magnitude_db": frf.magnitude_db,
                "phase_deg": frf.phase_deg,
            }
        ).to_csv(freq_path, index=False)
        print(f"[INFO]: Frequency response data saved to {freq_path}")


if __name__ == "__main__":
    main()

