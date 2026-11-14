"""SQLite persistence for astronaut-scoped EMG research records."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

DEFAULT_DB = Path("data/emg_monitor.db")


@contextmanager
def connection(db_path: str | Path = DEFAULT_DB) -> Iterator[sqlite3.Connection]:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize_database(db_path: str | Path = DEFAULT_DB) -> None:
    with connection(db_path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS astronauts (
            astronaut_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            mission_day INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS baselines (
            astronaut_id TEXT NOT NULL REFERENCES astronauts(astronaut_id),
            muscle_group TEXT NOT NULL,
            mvc_mv REAL NOT NULL,
            baseline_rms_mv REAL NOT NULL,
            baseline_mdf_hz REAL NOT NULL,
            acquired_at TEXT NOT NULL,
            PRIMARY KEY (astronaut_id, muscle_group)
        );
        CREATE TABLE IF NOT EXISTS sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            astronaut_id TEXT NOT NULL REFERENCES astronauts(astronaut_id),
            muscle_group TEXT NOT NULL,
            captured_at TEXT NOT NULL,
            sample_rate_hz REAL NOT NULL,
            task_label TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS emg_samples (
            session_id INTEGER NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            sample_index INTEGER NOT NULL,
            voltage_mv REAL NOT NULL,
            PRIMARY KEY (session_id, sample_index)
        );
        CREATE TABLE IF NOT EXISTS features (
            session_id INTEGER PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
            rms_mv REAL NOT NULL,
            median_frequency_hz REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS decision_history (
            decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
            astronaut_id TEXT NOT NULL REFERENCES astronauts(astronaut_id),
            session_id INTEGER NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            muscle_group TEXT NOT NULL,
            created_at TEXT NOT NULL,
            condition_class TEXT NOT NULL,
            model_score REAL NOT NULL,
            confidence REAL NOT NULL,
            uncertainty TEXT NOT NULL,
            agent_action TEXT NOT NULL,
            explanation TEXT NOT NULL,
            simulation_only INTEGER NOT NULL CHECK(simulation_only = 1)
        );
        CREATE INDEX IF NOT EXISTS ix_sessions_astronaut_time ON sessions(astronaut_id, captured_at);
        CREATE INDEX IF NOT EXISTS ix_decisions_astronaut_time ON decision_history(astronaut_id, created_at);
        """)


def add_astronaut(db_path: str | Path, astronaut_id: str, display_name: str, mission_day: int = 0) -> None:
    with connection(db_path) as conn:
        conn.execute("""INSERT INTO astronauts (astronaut_id, display_name, mission_day) VALUES (?, ?, ?)
                        ON CONFLICT(astronaut_id) DO UPDATE SET display_name=excluded.display_name, mission_day=excluded.mission_day""",
                     (astronaut_id, display_name, mission_day))


def add_baseline(db_path: str | Path, astronaut_id: str, muscle_group: str, mvc_mv: float, rms_mv: float, mdf_hz: float, acquired_at: str) -> None:
    with connection(db_path) as conn:
        conn.execute("""INSERT INTO baselines VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(astronaut_id, muscle_group) DO UPDATE SET
                        mvc_mv=excluded.mvc_mv, baseline_rms_mv=excluded.baseline_rms_mv,
                        baseline_mdf_hz=excluded.baseline_mdf_hz, acquired_at=excluded.acquired_at""",
                     (astronaut_id, muscle_group, mvc_mv, rms_mv, mdf_hz, acquired_at))


def ingest_session(db_path: str | Path, astronaut_id: str, muscle_group: str, captured_at: str,
                   sample_rate_hz: float, raw_mv: np.ndarray, rms_mv: float, mdf_hz: float,
                   task_label: str = "standardized monitoring task") -> int:
    """Persist raw voltage trace and its derived features under one astronaut only."""
    with connection(db_path) as conn:
        cur = conn.execute("INSERT INTO sessions (astronaut_id, muscle_group, captured_at, sample_rate_hz, task_label) VALUES (?, ?, ?, ?, ?)",
                           (astronaut_id, muscle_group, captured_at, sample_rate_hz, task_label))
        session_id = cur.lastrowid
        conn.executemany("INSERT INTO emg_samples VALUES (?, ?, ?)",
                         ((session_id, i, float(v)) for i, v in enumerate(raw_mv)))
        conn.execute("INSERT INTO features VALUES (?, ?, ?)", (session_id, float(rms_mv), float(mdf_hz)))
        return int(session_id)


def list_astronauts(db_path: str | Path = DEFAULT_DB) -> list[dict]:
    with connection(db_path) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM astronauts ORDER BY astronaut_id")]


def get_baselines(db_path: str | Path, astronaut_id: str) -> list[dict]:
    with connection(db_path) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM baselines WHERE astronaut_id = ? ORDER BY muscle_group", (astronaut_id,))]


def get_feature_history(db_path: str | Path, astronaut_id: str) -> list[dict]:
    with connection(db_path) as conn:
        return [dict(r) for r in conn.execute("""
            SELECT s.session_id, s.muscle_group, s.captured_at, s.task_label, s.sample_rate_hz, f.rms_mv, f.median_frequency_hz
            FROM sessions s JOIN features f ON f.session_id = s.session_id
            WHERE s.astronaut_id = ? ORDER BY s.captured_at
        """, (astronaut_id,))]


def add_virtual_decision(db_path: str | Path, astronaut_id: str, session_id: int, muscle_group: str,
                         created_at: str, condition_class: str, model_score: float, confidence: float,
                         uncertainty: str, agent_action: str, explanation: str) -> None:
    """Record a software-only simulation decision. Hardware fields are intentionally absent."""
    with connection(db_path) as conn:
        conn.execute("""INSERT INTO decision_history
            (astronaut_id, session_id, muscle_group, created_at, condition_class, model_score,
             confidence, uncertainty, agent_action, explanation, simulation_only)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
            (astronaut_id, session_id, muscle_group, created_at, condition_class, model_score,
             confidence, uncertainty, agent_action, explanation))


def get_decision_history(db_path: str | Path, astronaut_id: str) -> list[dict]:
    with connection(db_path) as conn:
        return [dict(r) for r in conn.execute("""
            SELECT decision_id, session_id, muscle_group, created_at, condition_class, model_score,
                   confidence, uncertainty, agent_action, explanation, simulation_only
            FROM decision_history WHERE astronaut_id = ? ORDER BY created_at DESC, decision_id DESC
        """, (astronaut_id,))]


def get_recent_raw_signal(db_path: str | Path, astronaut_id: str, muscle_group: str) -> tuple[np.ndarray, float] | None:
    with connection(db_path) as conn:
        row = conn.execute("""SELECT session_id, sample_rate_hz FROM sessions WHERE astronaut_id=? AND muscle_group=?
                              ORDER BY captured_at DESC LIMIT 1""", (astronaut_id, muscle_group)).fetchone()
        if not row:
            return None
        values = conn.execute("SELECT voltage_mv FROM emg_samples WHERE session_id=? ORDER BY sample_index", (row["session_id"],)).fetchall()
        return np.array([v[0] for v in values]), float(row["sample_rate_hz"])
