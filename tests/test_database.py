import numpy as np
from database import add_astronaut, add_baseline, add_virtual_decision, get_decision_history, get_feature_history, ingest_session, initialize_database


def test_astronaut_query_is_isolated(tmp_path):
    db = tmp_path / "test.db"
    initialize_database(db)
    for astronaut in ("AST-A", "AST-B"):
        add_astronaut(db, astronaut, astronaut)
        add_baseline(db, astronaut, "Quadriceps", 2, 1, 100, "2026-01-01T00:00:00Z")
        ingest_session(db, astronaut, "Quadriceps", "2026-01-02T00:00:00Z", 1000, np.ones(40), 1, 100)
    assert len(get_feature_history(db, "AST-A")) == 1
    assert get_feature_history(db, "AST-A")[0]["session_id"] != get_feature_history(db, "AST-B")[0]["session_id"]


def test_virtual_decision_is_marked_as_simulation(tmp_path):
    db = tmp_path / "test.db"
    initialize_database(db)
    add_astronaut(db, "AST-A", "A")
    add_baseline(db, "AST-A", "Quadriceps", 2, 1, 100, "2026-01-01T00:00:00Z")
    session_id = ingest_session(db, "AST-A", "Quadriceps", "2026-01-02T00:00:00Z", 1000, np.ones(40), 1, 100)
    add_virtual_decision(db, "AST-A", session_id, "Quadriceps", "2026-01-02T00:00:00Z", "Stable", .1, 100, "enough data", "VIRTUAL READY", "software only")
    assert get_decision_history(db, "AST-A")[0]["simulation_only"] == 1
