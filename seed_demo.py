"""Generate reproducible, clearly synthetic EMG records for dashboard exploration."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from scipy import signal

from database import DEFAULT_DB, add_astronaut, add_baseline, add_virtual_decision, ingest_session, initialize_database
from decision_engine import assess_muscle, virtual_ems_agent
from signal_processor import extract_features

MUSCLES = ("Soleus/Calves", "Quadriceps", "Erector Spinae")
PROFILES = {
    "AST-01": ("Alex Rivera — synthetic optimal profile", {m: (1.0, 1.0) for m in MUSCLES}),
    "AST-02": ("Jordan Chen — synthetic calf-fatigue profile", {"Soleus/Calves": (1.0, 0.82), "Quadriceps": (1.0, 1.0), "Erector Spinae": (1.0, 1.0)}),
    "AST-03": ("Samira Okafor — synthetic quadriceps-risk profile", {"Soleus/Calves": (1.0, 1.0), "Quadriceps": (0.70, 1.0), "Erector Spinae": (1.0, 1.0)}),
}


def synthetic_emg(rng: np.random.Generator, fs: int, seconds: float, amplitude: float, frequency_scale: float) -> np.ndarray:
    n = int(fs * seconds)
    white = rng.normal(0, amplitude, n)
    # Shape noise to emulate broad-band EMG and shift spectral content for the demo.
    b, a = signal.butter(3, min(0.42 * frequency_scale, 0.46))
    return signal.filtfilt(b, a, white) + 0.03 * np.sin(2 * np.pi * 50 * np.arange(n) / fs)


def seed(db_path: Path, reset: bool = False) -> None:
    if reset and db_path.exists():
        db_path.unlink()
    initialize_database(db_path)
    rng, fs = np.random.default_rng(42), 1000
    start = datetime.now(UTC).replace(microsecond=0) - timedelta(days=20)
    for astronaut_id, (name, trends) in PROFILES.items():
        add_astronaut(db_path, astronaut_id, name, 20)
        for muscle in MUSCLES:
            baseline = synthetic_emg(rng, fs, 2.5, 0.75, 1.0)
            f = extract_features(baseline, fs)
            add_baseline(db_path, astronaut_id, muscle, mvc_mv=2.5, rms_mv=f["rms_mv"], mdf_hz=f["median_frequency_hz"], acquired_at=start.isoformat())
            amp_target, freq_target = trends[muscle]
            for day in range(21):
                progress = day / 20
                raw = synthetic_emg(rng, fs, 2.5, 0.75 * (1 + (amp_target - 1) * progress), 1 + (freq_target - 1) * progress)
                features = extract_features(raw, fs)
                captured_at = (start + timedelta(days=day)).isoformat()
                session_id = ingest_session(db_path, astronaut_id, muscle, captured_at, fs, raw,
                                            features["rms_mv"], features["median_frequency_hz"])
                assessment = assess_muscle(features["rms_mv"], features["median_frequency_hz"],
                                           f["rms_mv"], f["median_frequency_hz"], day + 1)
                decision = virtual_ems_agent(assessment)
                add_virtual_decision(db_path, astronaut_id, session_id, muscle, captured_at, assessment.status,
                                     assessment.model_score, assessment.confidence, assessment.uncertainty,
                                     decision.action, decision.explanation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    seed(args.db, args.reset)
    print(f"Synthetic research demo seeded at {args.db}")
