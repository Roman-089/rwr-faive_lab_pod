# Chirp characterization of Isaac Sim joints to match real characterizations

This directory contains scripts for optimizing robotic joint parameters using frequency response function (FRF) data. The main script `optimization.py` uses Bayesian optimization to find the best actuator parameters that match real-world frequency response data.

## Overview

The optimization process works by:
1. Taking real-world chirp signal data as ground truth
2. Running parallel simulations with different parameter sets
3. Computing frequency response functions for both real and simulated data
4. Using Bayesian optimization to minimize the error between real and simulated frequency responses

## Real-world data collection

Collect ORCA joint data with [faive_system/experiments/orca/chirp_characterization.py](https://github.com/srl-ethz/faive_system/blob/main/experiments/orca/chirp_characterization.py). Make sure the same chirp signal parameters and control frequency (`CTRL_FREQ = 20` Hz) are used as in chirp_characterization.py of this repository.

The real-world data should be in CSV format with the following columns:
- `timestamp`: Time in seconds
- `commanded_position`: Commanded joint position in degrees
- `measured_position`: Measured joint position in degrees

Reference recordings from older hands are kept in `202506_collected_realworld_data/` (ORCA v1) and
`202510_orcav1b_realworld_data/` (ORCA v1b). They were recorded at 40 Hz, so don't use them as ground truth
for the 20 Hz v2 sim; record your own data on your v2 hand.

## Scripts Overview
See the documentation at the top of each file for the usage.

### `optimization.py`
Main optimization script that orchestrates the entire process.

### `chirp_characterization.py`
Runs parallel simulations with different parameter sets. Called by `optimization.py`, but you can test run it on its own (`python chirp_characterization.py`).

### `signal_processing.py`
This implements the FFT call, so that it doesn't have to be repeated across files and a unified call can be used.

### `plot_chirp_utils.py`
Plot the raw response to the chirp and its FRF. Multiple results can be compared.

## Example output (ORCA v1, 40 Hz)
see `example_optimization_output/` for more

![](example_optimization_output/frf_comparison.png)
