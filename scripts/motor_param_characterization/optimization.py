#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
optimization.py – parameter tuning **in the frequency domain**

Run: 
python optimization.py \
        --ground_truth <your_data_dir>/chirp_index_mcp_<timestamp>_ctrl20Hz_20.0s_0.1-5.0Hz_10deg.csv \
        --joint_name right_index_mcp

Note: the CSVs in 202506_collected_realworld_data/ (v1) and 202510_orcav1b_realworld_data/ (v1b) were recorded
at 40 Hz on older hands and are for reference only. The sim chirp runs at 20 Hz, so fit against your own 20 Hz v2 data.

Options:
` --min_freq / --max_freq` let you shrink the fitting band.
`--nfev` sets the number of function evaluations (default: 4). GP calculation time rises quickly on each iteration so don't make this too large- 4 seems to be enough.
`--log_file` specifies the CSV file to log optimization progress (default: auto-generated with timestamp).

The optimization progress is automatically logged to a CSV file containing:
- Elapsed time since start
- Best parameters found so far
- Smallest error achieved
- Number of points evaluated
- Batch statistics for each iteration

Plotting:
- Progress plots are automatically generated after each iteration and saved to the same directory as the log file
- The plot file has the same name as the log file but with a .png extension
- Plots include: error evolution, parameter evolution, points evaluated, and timing statistics

"""
from __future__ import annotations

import argparse
import datetime
import numpy as np
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from skopt import Optimizer
from skopt.space import Real, Categorical

from plot_optimization_progress import plot_optimization_progress
import plot_chirp_utils
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use non-interactive backend for headless operation

from signal_processing import FrequencyResponse, compute_frequency_response_from_dataframe


def compute_frf(df: pd.DataFrame, min_freq: float, max_freq: float) -> FrequencyResponse:
    """Wrapper around :func:`compute_frequency_response_from_dataframe`."""

    _, frf = compute_frequency_response_from_dataframe(
        df,
        min_frequency=min_freq,
        max_frequency=max_freq,
    )
    return frf


def run_characterization_parallel(
    joint: str, params_list: Sequence[Sequence[float]]
) -> list[pd.DataFrame]:
    """Run characterization for multiple parameter sets in one call."""
    LOG_DIR = Path("logs/characterization")
    stiffness, damping, friction, armature, effort_limit = zip(*params_list)
    cmd = ["python", "chirp_characterization.py", f"--joint_name={joint}"]
    cmd += ["--stiffness"] + [str(x) for x in stiffness]
    cmd += ["--damping"] + [str(x) for x in damping]
    cmd += ["--friction"] + [str(x) for x in friction]
    cmd += ["--armature"] + [str(x) for x in armature]
    cmd += ["--effort_limit"] + [str(x) for x in effort_limit]
    cmd += ["--headless"]

    print("────────────────────────────────────────────────────────")
    print("Running characterization script")

    env = os.environ.copy()
    env.setdefault("ISAACLAB_DISABLE_SHUTDOWN_ON_STOP", "1")
    res = subprocess.run(cmd, env=env, text=True, capture_output=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise RuntimeError("characterization.py failed")

    dfs = []
    for s, d, f, a, e in params_list:
        csv_name = f"chirp_data{s:.6f}_{d:.6f}_{f:.6f}_{a:.6f}_{e:.6f}.csv"
        path = LOG_DIR / csv_name
        if not path.exists():
            raise FileNotFoundError(f"Expected log {path} not found")
        dfs.append(pd.read_csv(path))

    print("────────────────────────────────────────────────────────")
    print("Characterization completed successfully.")
    return dfs


def make_freq_weights(freqs: np.ndarray, mode: str = "per_hz") -> np.ndarray:
    """Return frequency weights.

    mode:
        "per_bin" – equal weight per sample (current default)
        "per_hz"  – equal weight per physical Hz
        "per_log" – equal weight per log-decade (≈ per octave)
    """
    if mode == "per_bin":
        w = np.ones_like(freqs, dtype=float)

    elif mode == "per_hz":
        Δf = np.diff(freqs)
        Δf = np.append(Δf, Δf[-1])        # pad so vector lengths match 
        w = Δf

    elif mode == "per_log":
        logf = np.log10(freqs)
        Δlogf = np.diff(logf)
        Δlogf = np.append(Δlogf, Δlogf[-1])
        w = Δlogf
    
    elif mode == "inverse_hz":
        w = 1.0 / freqs
        w /= np.sum(w)  # normalise so Σw = 1

    else:
        raise ValueError(f"unknown weighting mode: {mode}")

    # (optional but convenient) normalise so Σw = 1
    return w / np.sum(w)


# ─── helper ──────────────────────────────────────────────────────────────────────
def _interp_complex(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    real = np.interp(x_new, x, y.real)
    imag = np.interp(x_new, x, y.imag)
    return real + 1j * imag


def make_residual(
    gt_frf: FrequencyResponse,
    *,
    joint: str,
    min_freq: float,
    max_freq: float,
) -> callable:
    """Return a function that evaluates a batch of parameters."""

    freq_weights = make_freq_weights(gt_frf.frequency, mode="inverse_hz")

    def residual(params_list: Sequence[Sequence[float]]) -> list[float]:
        """Evaluate each parameter set and return squared errors."""

        dfs = run_characterization_parallel(joint, params_list)
        errors = []
        print("Computing residuals for characterization results...")
        for sim_df in dfs:
            frf_sim = compute_frf(sim_df, min_freq=min_freq, max_freq=max_freq)

            H_sim_i = _interp_complex(gt_frf.frequency, frf_sim.frequency, frf_sim.response)

            e_abs = H_sim_i - gt_frf.response
            residual = np.vstack([e_abs.real, e_abs.imag])
            err = np.sum(freq_weights * np.abs(residual) ** 2)
            errors.append(err)

        errors_array = np.array(errors)
        print(
            f"Error statistics (n={len(errors)}): "
            f"mean={errors_array.mean():.4f}, std={errors_array.std():.4f}, "
            f"min={errors_array.min():.4f}, max={errors_array.max():.4f}"
        )
        return errors

    return residual


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--joint_name", default="right_index_mcp")
    p.add_argument("--ground_truth", type=str, help="CSV file with ground-truth FRF data", required=True)
    p.add_argument("--min_freq", type=float, default=0.1)
    p.add_argument("--max_freq", type=float, default=5.0)
    p.add_argument("--nfev", type=int, default=4)
    p.add_argument("--log_file", type=str, help="CSV file to log optimization progress", default=None)
    args = p.parse_args()

    # Create output directory for this optimization run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"logs/optimizations/{args.joint_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")

    # Set up logging
    log_data = []
    if args.log_file is None:
        # Generate default log filename with timestamp
        args.log_file = output_dir / "log.csv"
    else:
        # Ensure log file is in the output directory
        args.log_file = output_dir / Path(args.log_file).name
    
    # compute ground‑truth FRF once
    df_gt = pd.read_csv(args.ground_truth)
    gt_frf = compute_frf(df_gt, min_freq=args.min_freq, max_freq=args.max_freq)

    error_function = make_residual(
        gt_frf,
        joint=args.joint_name,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
    )

    num_envs = 100
    space = [
        Real(0.1, 15.0, name="stiffness"),
        Real(0.0, 3.0, name="damping"),
        Real(0.0, 10.0, name="friction"),
        # keep armature to zero for now to reduce search space
        Categorical([0.0], name="armature"),
        Real(0.5, 2.0, name="effort_limit"),
    ]
    optimizer = Optimizer(space, base_estimator="GP", acq_func="EI", n_initial_points=num_envs)
    best_x = None
    best_err = float("inf")
    # collect all the running bests to compare to GT
    best_x_per_iter = []
    total_points_evaluated = 0
    start_time = time.time()

    print("\n================== STARTING OPTIMIZATION ==================")
    print(f"Logging progress to: {args.log_file}")
    
    for i in range(args.nfev):
        iteration_start_time = time.time()
        print("Finding new parameters... this may take a while.")
        params_batch = optimizer.ask(n_points=num_envs)
        errors = error_function(params_batch)
        optimizer.tell(params_batch, errors)
        
        # Update statistics
        total_points_evaluated += len(params_batch)
        batch_best = int(np.argmin(errors))
        if errors[batch_best] < best_err:
            best_err = errors[batch_best]
            best_x = params_batch[batch_best]
            best_x_per_iter.append(best_x)

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        iteration_time = time.time() - iteration_start_time
        
        # Log iteration data
        log_entry = {
            'iteration': i + 1,
            'elapsed_time_seconds': elapsed_time,
            'iteration_time_seconds': iteration_time,
            'total_points_evaluated': total_points_evaluated,
            'best_error': best_err,
            'best_stiffness': best_x[0] if best_x is not None else None,
            'best_damping': best_x[1] if best_x is not None else None,
            'best_friction': best_x[2] if best_x is not None else None,
            'best_armature': best_x[3] if best_x is not None else None,
            'best_effort_limit': best_x[4] if best_x is not None else None,
            'batch_mean_error': np.mean(errors),
            'batch_std_error': np.std(errors),
            'batch_min_error': np.min(errors),
            'batch_max_error': np.max(errors),
            'timestamp': datetime.datetime.now().isoformat()
        }
        log_data.append(log_entry)
        
        # Save log after each iteration
        log_df = pd.DataFrame(log_data)
        log_df.to_csv(args.log_file, index=False)
        print(f"Saved log to {args.log_file}")

        # Generate plot after each iteration
        try:
            plot_optimization_progress(str(args.log_file), show_plot=False)
            print(f"Plot updated after iteration {i + 1}")
        except Exception as e:
            print(f"Warning: Failed to generate plot: {e}")

        print(f"Iteration {i + 1}/{args.nfev} - current best error: {best_err:.4f} - best parameters: {best_x}")
        print(f"  Elapsed time: {elapsed_time:.1f}s, Points evaluated: {total_points_evaluated}")

        print("\n================== BEST‑FIT PARAMETERS SO FAR ==================")
        print(f"stiffness    = {best_x[0]:.4f}")
        print(f"damping      = {best_x[1]:.4f}")
        print(f"friction     = {best_x[2]:.4f}")
        print(f"armature     = {best_x[3]:.4f}")
        print(f"effort_limit = {best_x[4]:.4f}")
        print(f"optimization time: {time.time() - start_time:.1f}s")
        print(f"Optimization log saved to: {args.log_file}")

        # Compare the best for each iteration to ground truth
        if len(best_x_per_iter) > 0:
            best_params_log_path_list = []
            for best_x_iter in best_x_per_iter:
                best_params_log_path_list.append(f"logs/characterization/chirp_data{best_x_iter[0]:.6f}_{best_x_iter[1]:.6f}_{best_x_iter[2]:.6f}_{best_x_iter[3]:.6f}_{best_x_iter[4]:.6f}.csv")
            datasets = plot_chirp_utils._build_datasets(
                best_params_log_path_list + [args.ground_truth],
                labels=[f"best_{j+1}" for j in range(len(best_x_per_iter))] + ["ground_truth"],
                min_freq=args.min_freq,
                max_freq=args.max_freq,
                tukey_alpha=0.1,
            )
            fig = plot_chirp_utils.plot_chirp_frequency_response(datasets, title="Comparison between best fits to the GT response")
            figure_path = output_dir / "frf_comparison.png"
            fig.savefig(figure_path, facecolor=fig.get_facecolor())
            print(f"Frequency response comparison plot saved to: {figure_path}")
            plt.close(fig)

    # Print final message
    print("\nOptimization completed successfully.")
    print(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
