#!/usr/bin/env python3

"""
plot_optimization_progress.py - Analyze and visualize optimization progress

Usage:
    python plot_optimization_progress.py example_optimization_output/optimization_log.csv
"""

import argparse
import math
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_optimization_progress(log_file: str, show_plot: bool = True):
    """Plot optimization progress from CSV log file."""
    
    # Read the log file
    df = pd.read_csv(log_file)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Optimization Progress - {Path(log_file).name}', fontsize=14)
    
    # Plot 1: Best error over time
    axes[0, 0].plot(df['elapsed_time_seconds'] / 60, df['best_error'], 'b-o', markersize=4)
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 0].set_xlabel('Time (minutes)')
    axes[0, 0].set_ylabel('Best Error')
    axes[0, 0].set_title('Best Error vs Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Best error over iterations
    axes[0, 1].plot(df['iteration'], df['total_points_evaluated'], 'g-o', markersize=4)
    axes[0, 1].set_ylim(bottom=0)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Samples Evaluated')
    axes[0, 1].set_title('Samples Evaluated vs Iteration')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Parameter evolution with automatic scaling
    param_configs = [
        ('best_stiffness', 'Stiffness', 'r'),
        ('best_damping', 'Damping', 'b'),
        ('best_friction', 'Friction', 'g'),
        ('best_armature', 'Armature', 'm'),
        ('best_effort_limit', 'Effort Limit', 'c'),
    ]

    def compute_scale(series: pd.Series) -> float:
        """
        Compute how much to scale in order to make the maximum absolute value around 10.
        """
        max_val = series.abs().max()
        if pd.isna(max_val) or max_val == 0:
            return 1.0
        exponent = math.floor(math.log10(10 / max_val))
        return 10 ** exponent

    def format_scale(scale: float) -> str:
        if math.isclose(scale, round(scale)):
            return str(int(round(scale)))
        return f"{scale:g}"

    for column, label, color in param_configs:
        scale = compute_scale(df[column])
        scaled_values = df[column] * scale
        axes[1, 0].plot(
            df['iteration'],
            scaled_values,
            color=color,
            linewidth=2,
            label=f"{label} (x{format_scale(scale)})",
        )

    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Scaled Parameter Value')
    axes[1, 0].set_title('Best Parameter Evolution')
    axes[1, 0].set_ylim(bottom=0, top=10)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Accumulated and per-iteration timing
    ax4a = axes[1, 1]
    ax4b = ax4a.twinx()

    accumulated_time_minutes = df['iteration_time_seconds'].cumsum() / 60

    line1 = ax4a.plot(df['iteration'], accumulated_time_minutes, 'b-o', markersize=4, label='Accumulated Time')
    ax4a.set_xlabel('Iteration')
    ax4a.set_ylabel('Accumulated Time (minutes)', color='b')
    ax4a.set_ylim(bottom=0)
    ax4a.tick_params(axis='y', labelcolor='b')

    line2 = ax4b.plot(df['iteration'], df['iteration_time_seconds'] / 60, 'r-s', markersize=4, label='Iteration Time')
    ax4b.set_ylabel('Iteration Time (minutes)', color='r')
    ax4b.set_ylim(bottom=0)
    ax4b.tick_params(axis='y', labelcolor='r')

    axes[1, 1].set_title('Accumulated & Iteration Time')
    
    # Add legend for both y-axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4a.legend(lines, labels, loc='upper left')
    
    ax4a.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = Path(log_file).with_suffix('.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    
    if show_plot:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze optimization progress log")
    parser.add_argument("log_file", help="Path to the optimization log CSV file")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()
    
    if not Path(args.log_file).exists():
        print(f"Error: Log file {args.log_file} not found")
        return
    
    if not args.no_plot:
        try:
            plot_optimization_progress(args.log_file)
        except ImportError:
            print("Warning: matplotlib not available, skipping plots")
        except Exception as e:
            print(f"Error creating plots: {e}")


if __name__ == "__main__":
    main()
