"""Surface EMG research feature extraction using SciPy."""
from __future__ import annotations

import numpy as np
from scipy import signal


def _validate(raw_mv: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    values = np.asarray(raw_mv, dtype=float)
    if values.ndim != 1 or len(values) < 32:
        raise ValueError("EMG trace must be a one-dimensional signal with at least 32 samples")
    if sample_rate_hz <= 2 * 450:
        raise ValueError("Sample rate must exceed 900 Hz for a 20–450 Hz bandpass")
    return values


def filter_emg(raw_mv: np.ndarray, sample_rate_hz: float, notch_hz: float = 50.0) -> np.ndarray:
    """20–450 Hz Butterworth bandpass followed by a 50/60 Hz notch filter."""
    values = _validate(raw_mv, sample_rate_hz)
    sos = signal.butter(4, (20, 450), btype="bandpass", fs=sample_rate_hz, output="sos")
    bandpassed = signal.sosfiltfilt(sos, values)
    b, a = signal.iirnotch(notch_hz, Q=30, fs=sample_rate_hz)
    return signal.filtfilt(b, a, bandpassed)


def extract_features(raw_mv: np.ndarray, sample_rate_hz: float, notch_hz: float = 50.0) -> dict[str, float | np.ndarray]:
    filtered = filter_emg(raw_mv, sample_rate_hz, notch_hz)
    rectified = np.abs(filtered)
    rms_mv = float(np.sqrt(np.mean(rectified ** 2)))
    frequencies, power = signal.welch(filtered, fs=sample_rate_hz, nperseg=min(1024, len(filtered)))
    cumulative = np.cumsum(power)
    mdf_hz = float(frequencies[np.searchsorted(cumulative, cumulative[-1] / 2)])
    return {"filtered_mv": filtered, "rectified_mv": rectified, "rms_mv": rms_mv, "median_frequency_hz": mdf_hz}
