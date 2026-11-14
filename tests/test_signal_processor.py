import numpy as np
from signal_processor import extract_features


def test_extract_features_returns_positive_values():
    fs = 1000
    t = np.arange(fs * 3) / fs
    raw = 0.5 * np.sin(2 * np.pi * 100 * t) + 0.02 * np.random.default_rng(7).normal(size=len(t))
    result = extract_features(raw, fs)
    assert result["rms_mv"] > 0
    assert 20 < result["median_frequency_hz"] < 450
    assert len(result["rectified_mv"]) == len(raw)
