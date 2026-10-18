"""Utility functions for processing chirp experiments and computing FRFs.

The scripts in :mod:`scripts.motor_param_characterization` perform the same
pre-processing steps in several places.  This module centralises the logic so
that the chirp trimming, windowing and FFT configuration stay consistent across
the tools.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal
from scipy.fft import fft, fftfreq


@dataclass(slots=True)
class ChirpProcessingResult:
    """Intermediate signals produced while preparing chirp data."""

    time: np.ndarray
    commanded: np.ndarray
    measured: np.ndarray
    window: np.ndarray
    commanded_windowed: np.ndarray
    measured_windowed: np.ndarray


@dataclass(slots=True)
class FrequencyResponse:
    """Frequency response representation as complex transfer function values."""

    frequency: np.ndarray
    response: np.ndarray

    @property
    def magnitude_db(self) -> np.ndarray:
        return 20.0 * np.log10(np.abs(self.response))

    @property
    def phase_deg(self) -> np.ndarray:
        return np.rad2deg(np.angle(self.response))


def _ensure_array(data: ArrayLike, *, dtype=float) -> np.ndarray:
    array = np.asarray(data, dtype=dtype)
    if array.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    return array


def prepare_chirp_signals(
    *,
    time: ArrayLike,
    commanded: ArrayLike,
    measured: ArrayLike,
    trim_start: float = 0.1,
    align: bool = False,
    tukey_alpha: float = 0.1,
) -> ChirpProcessingResult:
    """Apply the standard trimming, alignment and windowing steps."""

    timestamps = _ensure_array(time)
    cmd = _ensure_array(commanded)
    meas = _ensure_array(measured)

    if not (len(timestamps) == len(cmd) == len(meas)):
        raise ValueError("time, commanded and measured must have the same length")

    trim_mask = timestamps >= (timestamps[0] + trim_start)
    trimmed_time = timestamps[trim_mask]
    if trimmed_time.size == 0:
        raise ValueError("Trimming removed every sample; decrease trim_start")

    trimmed_time = trimmed_time - trimmed_time[0]
    cmd_trimmed = cmd[trim_mask]
    meas_trimmed = meas[trim_mask]

    if align:
        correlation = signal.correlate(meas_trimmed, cmd_trimmed, mode="full")
        delay_samples = int(np.argmax(correlation) - (len(cmd_trimmed) - 1))

        if delay_samples > 0:
            meas_aligned = meas_trimmed[delay_samples:]
            cmd_aligned = cmd_trimmed[: len(meas_aligned)]
        elif delay_samples < 0:
            cmd_aligned = cmd_trimmed[-delay_samples:]
            meas_aligned = meas_trimmed[: len(cmd_aligned)]
        else:
            cmd_aligned = cmd_trimmed
            meas_aligned = meas_trimmed

        time_aligned = trimmed_time[: len(cmd_aligned)]
    else:
        cmd_aligned = cmd_trimmed
        meas_aligned = meas_trimmed
        time_aligned = trimmed_time

    cmd_centered = cmd_aligned - np.mean(cmd_aligned)
    meas_centered = meas_aligned - np.mean(meas_aligned)

    window = signal.windows.tukey(len(cmd_centered), alpha=tukey_alpha)
    cmd_windowed = cmd_centered * window
    meas_windowed = meas_centered * window

    return ChirpProcessingResult(
        time=time_aligned,
        commanded=cmd_aligned,
        measured=meas_aligned,
        window=window,
        commanded_windowed=cmd_windowed,
        measured_windowed=meas_windowed,
    )


def compute_frequency_response(
    processed: ChirpProcessingResult,
    *,
    min_frequency: float = 0.1,
    max_frequency: float = 5.0,
    eps: float = 1e-12,
) -> FrequencyResponse:
    """Compute the Frequency Response Function from processed signals."""

    dt = float(np.mean(np.diff(processed.time)))
    if dt <= 0.0:
        raise ValueError("Non-positive time step encountered")

    cmd_fft = fft(processed.commanded_windowed)
    meas_fft = fft(processed.measured_windowed)

    freqs = fftfreq(processed.commanded_windowed.size, dt)
    mask = (
        (freqs > 0.0)
        & (freqs >= min_frequency)
        & (freqs <= max_frequency)
    )

    valid_cmd = np.abs(cmd_fft) > eps
    mask &= valid_cmd

    return FrequencyResponse(
        frequency=freqs[mask],
        response=meas_fft[mask] / cmd_fft[mask],
    )


def compute_frequency_response_from_dataframe(
    df,
    *,
    trim_start: float = 0.1,
    align: bool = False,
    min_frequency: float = 0.1,
    max_frequency: float = 5.0,
    tukey_alpha: float = 0.1,
) -> tuple[ChirpProcessingResult, FrequencyResponse]:
    """Convenience wrapper using a Pandas DataFrame."""

    processed = prepare_chirp_signals(
        time=df["timestamp"],
        commanded=df["commanded_position"],
        measured=df["measured_position"],
        trim_start=trim_start,
        align=align,
        tukey_alpha=tukey_alpha,
    )
    frf = compute_frequency_response(
        processed,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
    )
    return processed, frf

