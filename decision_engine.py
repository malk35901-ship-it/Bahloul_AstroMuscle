"""Transparent hybrid model + agent for SOFTWARE-ONLY virtual EMS simulation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MuscleAssessment:
    status: str
    rationale: str
    rms_change_pct: float
    mdf_change_pct: float
    model_score: float
    confidence: float
    uncertainty: str


@dataclass(frozen=True)
class VirtualDecision:
    action: str
    explanation: str
    simulation_only: bool = True


def assess_muscle(current_rms: float, current_mdf: float, baseline_rms: float, baseline_mdf: float,
                  history_count: int = 1, comparable_task: bool = True) -> MuscleAssessment:
    """A baseline-relative, explainable score; it is not a trained or clinical model."""
    rms_change = (current_rms / baseline_rms - 1) * 100
    mdf_change = (current_mdf / baseline_mdf - 1) * 100
    model_score = min(1.0, max(0.0, 0.65 * max(0.0, -rms_change) / 30 + 0.35 * max(0.0, -mdf_change) / 15))
    evidence = min(history_count, 7) / 7
    confidence = round(100 * evidence * (1.0 if comparable_task else 0.45), 1)
    uncertainty = "Limited longitudinal evidence" if history_count < 7 else "Seven or more observations available"
    if not comparable_task:
        uncertainty += "; task label does not match baseline context"
    if model_score >= 0.67:
        status = "Elevated change flag"
        rationale = "Combined baseline-relative RMS/MDF change crosses the dashboard's high research threshold."
    elif model_score >= 0.30:
        status = "Watch"
        rationale = "A moderate baseline-relative signal change is present; compare a repeat standardized session."
    else:
        status = "Stable"
        rationale = "No dashboard research threshold is crossed in the available comparable data."
    return MuscleAssessment(status, rationale, rms_change, mdf_change, round(model_score, 3), confidence, uncertainty)


def virtual_ems_agent(assessment: MuscleAssessment) -> VirtualDecision:
    """Policy agent emits only a virtual state, never commands, parameters, or hardware access."""
    if assessment.status == "Elevated change flag":
        return VirtualDecision("VIRTUAL HOLD + REVIEW", "Agent holds the virtual suit/bed simulation pending review of data quality and context. No physical output exists.")
    if assessment.status == "Watch":
        return VirtualDecision("VIRTUAL OBSERVE", "Agent keeps the virtual suit/bed in observation state and requests another comparable simulated session. No physical output exists.")
    return VirtualDecision("VIRTUAL READY", "Agent marks the virtual suit/bed as ready in software only; no stimulation, settings, or device control are produced.")


def ems_safety_gate() -> dict[str, str | bool]:
    return {"enabled": False, "message": "SIMULATION ONLY — virtual EMS decisions cannot activate a physical bed, suit, electrodes, or any hardware."}
