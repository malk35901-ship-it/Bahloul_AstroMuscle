"""Protocol planning for the Countermeasure Planning Agent (simulated stimulator model)."""
from __future__ import annotations

AGENT_NAME = "Countermeasure Planning Agent (CPA)"
AGENT_ROLE = (
    "Reads each crew member's baseline-relative EMG assessment, decides per muscle group whether "
    "neuromuscular electrical stimulation (NMES) is warranted, and plans the stimulation protocol: "
    "current, voltage from electrode-skin impedance, waveform timing, and stop rule."
)

ELEVATED = "Elevated change flag"
COMPLIANCE_V = 120.0                      # modelled constant-current stimulator voltage ceiling
MAX_CURRENT_MA = {"Soleus/Calves": 35.0, "Quadriceps": 50.0, "Erector Spinae": 40.0}
IMPEDANCE_KOHM = {"Soleus/Calves": 1.4, "Quadriceps": 1.1, "Erector Spinae": 1.6}  # assumed values
FREQUENCY_HZ = 50
PULSE_WIDTH_US = 300
ON_S, OFF_S, RAMP_S = 5, 10, 1
SESSION_MIN = 20
SESSIONS_PER_WEEK = 3


def plan_protocol(muscle: str, status: str, score: float) -> dict | None:
    """Return a simulated NMES protocol only for an elevated change flag, else None."""
    if status != ELEVATED:
        return None
    severity = min(1.0, max(0.0, (score - 0.67) / 0.33))
    current_ma = round(MAX_CURRENT_MA[muscle] * (0.70 + 0.30 * severity), 1)
    impedance = IMPEDANCE_KOHM[muscle]
    voltage = current_ma * impedance                       # Ohm's law: mA x kOhm = V
    charge_uc = current_ma * PULSE_WIDTH_US / 1000         # mA x us = nC; /1000 -> uC
    return {
        "muscle": muscle,
        "current_ma": current_ma,
        "impedance_kohm": impedance,
        "voltage_v": round(voltage, 1),
        "margin_v": round(COMPLIANCE_V - voltage, 1),
        "charge_uc": round(charge_uc, 2),
        "severity": round(severity, 2),
    }


def protocol_rows(p: dict) -> list[tuple[str, str]]:
    return [
        ("Stimulus current", f"{p['current_ma']:.1f} mA (constant current)"),
        ("Electrode-skin impedance", f"{p['impedance_kohm']:.1f} kΩ (assumed)"),
        ("Required voltage (V = I·R)", f"{p['voltage_v']:.1f} V"),
        ("Compliance margin", f"{p['margin_v']:.1f} V below {COMPLIANCE_V:.0f} V ceiling"),
        ("Waveform", "Biphasic, symmetric, charge-balanced"),
        ("Pulse frequency / width", f"{FREQUENCY_HZ} Hz / {PULSE_WIDTH_US} µs"),
        ("Charge per phase", f"{p['charge_uc']:.2f} µC"),
        ("Duty cycle", f"{ON_S} s on / {OFF_S} s off, {RAMP_S} s ramp"),
        ("Session length", f"{SESSION_MIN} min, {SESSIONS_PER_WEEK} per week"),
        ("Stop rule", "Discontinue when change score < 0.30 on two consecutive sessions"),
    ]
