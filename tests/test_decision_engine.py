from decision_engine import assess_muscle, ems_safety_gate, virtual_ems_agent


def test_elevated_change_flag_is_explainable():
    result = assess_muscle(0.7, 80, 1.0, 100, history_count=7)
    assert result.status == "Elevated change flag"
    assert result.model_score >= 0.67
    assert result.confidence == 100


def test_ems_is_never_enabled_or_parameterized():
    response = ems_safety_gate()
    assert response["enabled"] is False
    assert "SIMULATION ONLY" in response["message"]


def test_agent_returns_virtual_only_state():
    decision = virtual_ems_agent(assess_muscle(1.0, 100, 1.0, 100, history_count=7))
    assert decision.simulation_only is True
    assert decision.action == "VIRTUAL READY"
