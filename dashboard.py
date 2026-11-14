"""Astra EMS neuromuscular monitoring console (Streamlit)."""
from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import signal

from countermeasures import COUNTERMEASURES
from database import (DEFAULT_DB, get_baselines, get_decision_history, get_feature_history,
                      get_recent_raw_signal, list_astronauts)
from decision_engine import assess_muscle, virtual_ems_agent
from ems_protocol import AGENT_NAME, AGENT_ROLE, plan_protocol, protocol_rows
from signal_processor import extract_features

st.set_page_config(page_title="Astra EMS", page_icon="◈", layout="wide")

MUSCLES = ("Soleus/Calves", "Quadriceps", "Erector Spinae")
STATUS_COLOR = {"Stable": "#2fbf9b", "Watch": "#f2b84b", "Elevated change flag": "#e5564d"}
STATUS_RANK = {"Stable": 0, "Watch": 1, "Elevated change flag": 2}
EMS_DECISION = {
    "Elevated change flag": "EMS session indicated",
    "Watch": "EMS deferred, repeat measurement",
    "Stable": "EMS not indicated",
}
NEXT_ACTION = {
    "Watch": "No stimulation planned; repeat a comparable session within 24 h.",
    "Stable": "No stimulation planned; continue scheduled monitoring.",
}
SERIES_COLOR = {"Soleus/Calves": "#4cc3ff", "Quadriceps": "#b48cff", "Erector Spinae": "#ffb86b"}

st.markdown("""
<style>
.stApp{background:#0b1018;color:#dbe4f0}
[data-testid="stSidebar"]{background:#0e1520}
.block-container{padding-top:1.6rem;max-width:1500px}
.card{background:#111826;border:1px solid #1f2b3d;border-radius:10px;padding:14px 16px;height:100%}
.k{font-size:.7rem;letter-spacing:.09em;text-transform:uppercase;color:#7f8ea3}
.v{font-size:1.75rem;font-weight:600;color:#eef3fa;line-height:1.25}
.row{display:flex;justify-content:space-between;font-size:.85rem;color:#a9b6c8;padding:3px 0;border-top:1px solid #1b2536}
.row b{color:#e6edf7;font-weight:500}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.72rem;font-weight:600}
.sec{font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;color:#7f8ea3;margin:26px 0 8px 0;
     border-bottom:1px solid #1f2b3d;padding-bottom:6px}
.ttl{font-size:1.7rem;font-weight:650;color:#f1f5fb;margin:0}
.sub{color:#7f8ea3;font-size:.9rem;margin-top:2px}
.ex{font-size:.88rem;color:#c3cfdf;padding:6px 0;border-top:1px solid #1b2536}
.ex span{display:block;color:#7f8ea3;font-size:.78rem}
</style>
""", unsafe_allow_html=True)


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def pill(status: str) -> str:
    c = STATUS_COLOR[status]
    return f'<span class="pill" style="background:{rgba(c, .16)};color:{c};border:1px solid {rgba(c, .5)}">{status}</span>'


def show_plot(fig: go.Figure) -> None:
    if "width" in inspect.signature(st.plotly_chart).parameters:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def show_table(df: pd.DataFrame) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


def section(title: str) -> None:
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)


# ---------- data ----------
if not Path(DEFAULT_DB).exists():
    st.error("Database not found. Run `python seed_demo.py` in the project folder first.")
    st.stop()
try:
    ASTRONAUTS = list_astronauts(DEFAULT_DB)
except sqlite3.OperationalError:
    ASTRONAUTS = []
if not ASTRONAUTS:
    st.error("The database has no astronaut records. Run `python seed_demo.py`.")
    st.stop()


def assess_all(astronaut_id: str) -> dict:
    baselines = {b["muscle_group"]: b for b in get_baselines(DEFAULT_DB, astronaut_id)}
    hist = pd.DataFrame(get_feature_history(DEFAULT_DB, astronaut_id))
    out = {}
    for muscle in MUSCLES:
        if hist.empty or muscle not in baselines:
            continue
        h = hist[hist.muscle_group == muscle].sort_values("captured_at")
        if h.empty:
            continue
        last, base = h.iloc[-1], baselines[muscle]
        a = assess_muscle(float(last.rms_mv), float(last.median_frequency_hz),
                          base["baseline_rms_mv"], base["baseline_mdf_hz"], len(h))
        out[muscle] = {"a": a, "agent": virtual_ems_agent(a), "last": last, "base": base, "n": len(h)}
    return out


# ---------- body map ----------
def smooth(points, dx=0.0) -> str:
    pts = [(x + dx, y) for x, y in points]
    mid = lambda p, q: ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    start = mid(pts[-1], pts[0])
    d = f"M {start[0]:.1f},{start[1]:.1f}"
    for i, p in enumerate(pts):
        e = mid(p, pts[(i + 1) % len(pts)])
        d += f" Q {p[0]:.1f},{p[1]:.1f} {e[0]:.1f},{e[1]:.1f}"
    return d + " Z"


def mirror(points):
    return [(100 - x, y) for x, y in points][::-1]


TORSO = [(36, 30), (50, 27), (64, 30), (68, 44), (65, 70), (60, 94), (40, 94), (35, 70), (32, 44)]
NECK = [(46, 21), (54, 21), (54, 30), (46, 30)]
ARM = [(31, 31), (24, 34), (20, 50), (17, 78), (19, 87), (25, 87), (28, 64), (33, 46)]
THIGH = [(40, 94), (50, 94), (50, 120), (47, 143), (37, 143), (36, 118)]
SHIN = [(37, 146), (47, 146), (46, 168), (44, 190), (38, 190), (36, 168)]
FOOT = [(36, 192), (45, 192), (47, 199), (32, 199)]
SPINE = [(44, 34), (49, 34), (49, 90), (44, 90), (42, 62)]


def body_figure(state: dict) -> go.Figure:
    fig = go.Figure()
    neutral_fill, neutral_line = "#161f2e", "#33435c"

    def part(points, dx, color=None):
        fig.add_shape(type="path", path=smooth(points, dx),
                      fillcolor=rgba(color, .55) if color else neutral_fill,
                      line=dict(color=color or neutral_line, width=1.6 if color else 1.2))

    def view(dx, thigh=None, shin=None, spine=None):
        fig.add_shape(type="circle", x0=41 + dx, x1=59 + dx, y0=5, y1=23,
                      fillcolor=neutral_fill, line=dict(color=neutral_line, width=1.2))
        part(NECK, dx); part(TORSO, dx)
        for pts in (ARM, mirror(ARM)):
            part(pts, dx)
        for pts in (THIGH, mirror(THIGH)):
            part(pts, dx, thigh)
        for pts in (SHIN, mirror(SHIN)):
            part(pts, dx, shin)
        for pts in (FOOT, mirror(FOOT)):
            part(pts, dx)
        if spine:
            for pts in (SPINE, mirror(SPINE)):
                part(pts, dx, spine)

    def color_of(muscle):
        return STATUS_COLOR[state[muscle]["a"].status] if muscle in state else None

    ap_dx, post_dx = 0, 150
    view(ap_dx, thigh=color_of("Quadriceps"))
    view(post_dx, shin=color_of("Soleus/Calves"), spine=color_of("Erector Spinae"))

    def label(muscle, tx, ty, lx, ly, dx):
        if muscle not in state:
            return
        a, c = state[muscle]["a"], STATUS_COLOR[state[muscle]["a"].status]
        fig.add_annotation(x=tx + dx, y=ty, ax=lx + dx, ay=ly, axref="x", ayref="y", xref="x", yref="y",
                           text=f"<b>{muscle}</b><br>{a.status} · score {a.model_score:.2f}",
                           showarrow=True, arrowcolor=c, arrowwidth=1.2, arrowhead=0, align="left",
                           font=dict(size=12, color="#e6edf7"), bgcolor="rgba(15,22,34,.92)",
                           bordercolor=c, borderwidth=1, borderpad=5, xanchor="left")
        fig.add_trace(go.Scatter(x=[tx + dx], y=[ty], mode="markers", marker=dict(size=22, opacity=0),
                                 hoverinfo="text", showlegend=False,
                                 hovertext=f"{muscle}<br>RMS {a.rms_change_pct:+.1f}% · MDF {a.mdf_change_pct:+.1f}%"))

    label("Quadriceps", 56, 118, 90, 112, ap_dx)
    label("Soleus/Calves", 56, 168, 90, 172, post_dx)
    label("Erector Spinae", 54, 60, 90, 52, post_dx)
    for dx, text in ((ap_dx, "ANTERIOR"), (post_dx, "POSTERIOR")):
        fig.add_annotation(x=50 + dx, y=206, text=text, showarrow=False, font=dict(size=11, color="#7f8ea3"))
    fig.update_xaxes(range=[-8, 300], visible=False)
    fig.update_yaxes(range=[212, -2], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=470, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", hoverlabel=dict(bgcolor="#111826"))
    return fig


# ---------- signal + trend figures ----------
def signal_figure(raw, fs, baseline_mdf, feats) -> go.Figure:
    n = min(int(0.5 * fs), len(raw))
    t = np.arange(n) / fs * 1000
    filt = feats["filtered_mv"]
    fig = make_subplots(rows=1, cols=2, column_widths=[.62, .38],
                        subplot_titles=("Surface EMG, first 500 ms", "Power spectral density (Welch)"))
    fig.add_trace(go.Scatter(x=t, y=raw[:n], name="Raw", line=dict(color="#6b7a90", width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=filt[:n], name="Filtered", line=dict(color="#4cc3ff", width=1.4)), 1, 1)
    f, p = signal.welch(filt, fs=fs, nperseg=min(1024, len(filt)))
    fig.add_trace(go.Scatter(x=f, y=p, line=dict(color="#4cc3ff"), fill="tozeroy",
                             fillcolor="rgba(76,195,255,.15)", showlegend=False), 1, 2)
    fig.add_vline(x=feats["median_frequency_hz"], line=dict(color="#f2b84b", dash="dash"), row=1, col=2,
                  annotation_text="MDF", annotation_position="top right", annotation_font_color="#f2b84b")
    fig.add_vline(x=baseline_mdf, line=dict(color="#2fbf9b", dash="dot"), row=1, col=2,
                  annotation_text="Baseline", annotation_position="top left", annotation_font_color="#2fbf9b")
    fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude (mV)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", range=[0, 500], row=1, col=2)
    fig.update_yaxes(title_text="PSD (mV²/Hz)", row=1, col=2)
    fig.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f1623",
                      legend=dict(orientation="h", y=-0.22))
    return fig


def trend_figure(astronaut_id: str) -> go.Figure:
    baselines = {b["muscle_group"]: b for b in get_baselines(DEFAULT_DB, astronaut_id)}
    hist = pd.DataFrame(get_feature_history(DEFAULT_DB, astronaut_id))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.1,
                        subplot_titles=("RMS amplitude change vs. baseline (%)",
                                        "Median frequency change vs. baseline (%)"))
    for m in MUSCLES:
        if hist.empty or m not in baselines:
            continue
        h = hist[hist.muscle_group == m].sort_values("captured_at")
        x = pd.to_datetime(h.captured_at)
        b = baselines[m]
        c = SERIES_COLOR[m]
        fig.add_trace(go.Scatter(x=x, y=(h.rms_mv / b["baseline_rms_mv"] - 1) * 100, name=m,
                                 line=dict(color=c, width=2), legendgroup=m), 1, 1)
        fig.add_trace(go.Scatter(x=x, y=(h.median_frequency_hz / b["baseline_mdf_hz"] - 1) * 100, name=m,
                                 line=dict(color=c, width=2), legendgroup=m, showlegend=False), 2, 1)
    for r in (1, 2):
        fig.add_hline(y=0, line=dict(color="#4a5a73", dash="dot"), row=r, col=1)
    fig.update_yaxes(title_text="Δ RMS (%)", row=1, col=1)
    fig.update_yaxes(title_text="Δ MDF (%)", row=2, col=1)
    fig.update_xaxes(title_text="Session date (UTC)", row=2, col=1)
    fig.update_layout(template="plotly_dark", height=470, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f1623", legend=dict(orientation="h", y=-0.14))
    return fig


# ---------- page ----------
with st.sidebar:
    st.markdown("### Crew member")
    selected = st.selectbox("Astronaut", ASTRONAUTS, format_func=lambda a: f"{a['astronaut_id']} · {a['display_name'].split(' — ')[0]}")
    muscle_view = st.selectbox("Signal view", MUSCLES)

aid = selected["astronaut_id"]
state = assess_all(aid)
if not state:
    st.warning("No feature records for this astronaut. Run `python seed_demo.py`.")
    st.stop()

n_sessions = max(s["n"] for s in state.values())
fs_hz = float(next(iter(state.values()))["last"].sample_rate_hz)
worst = max(state, key=lambda m: (STATUS_RANK[state[m]["a"].status], state[m]["a"].model_score))
worst_a = state[worst]["a"]

st.markdown(f'<p class="ttl">Astra EMS · Neuromuscular Monitoring Console</p>'
            f'<p class="sub">{selected["display_name"]} · Mission day {selected["mission_day"]} · '
            f'{n_sessions} sessions per muscle · {fs_hz:.0f} Hz sampling · Dataset: synthetic sEMG</p>',
            unsafe_allow_html=True)

section("Condition classification")
cols = st.columns(3)
for col, m in zip(cols, MUSCLES):
    if m not in state:
        continue
    a = state[m]["a"]
    col.markdown(f"""<div class="card"><div class="k">{m}</div>{pill(a.status)}
    <div class="v">{a.model_score:.2f}<span style="font-size:.8rem;color:#7f8ea3"> / 1.00 change score</span></div>
    <div class="row"><span>RMS vs. baseline</span><b>{a.rms_change_pct:+.1f}%</b></div>
    <div class="row"><span>MDF vs. baseline</span><b>{a.mdf_change_pct:+.1f}%</b></div>
    <div class="row"><span>Evidence coverage</span><b>{a.confidence:.0f}%</b></div>
    <div class="row"><span>Uncertainty</span><b>{a.uncertainty}</b></div></div>""", unsafe_allow_html=True)

section("Anatomical map and AI decision")
left, right = st.columns([1.35, 1])
with left:
    show_plot(body_figure(state))
with right:
    targets = [m for m in MUSCLES if m in state and state[m]["a"].status == worst_a.status]
    detail = "; ".join(f"{m}: RMS {state[m]['a'].rms_change_pct:+.1f}%, MDF {state[m]['a'].mdf_change_pct:+.1f}%"
                       for m in targets)
    c = STATUS_COLOR[worst_a.status]
    st.markdown(f"""<div class="card" style="border-color:{rgba(c, .6)}">
    <div class="k">{AGENT_NAME} · decision</div>
    <div class="v" style="color:{c}">{EMS_DECISION[worst_a.status]}</div>
    <div class="row"><span>Governing muscle group</span><b>{', '.join(targets)}</b></div>
    <div class="row"><span>Policy state</span><b>{state[worst]['agent'].action}</b></div>
    <div class="row"><span>Change score</span><b>{worst_a.model_score:.2f}</b></div>
    <div class="row"><span>Evidence coverage</span><b>{worst_a.confidence:.0f}%</b></div>
    <p style="font-size:.86rem;color:#a9b6c8;margin:10px 0 0 0"><b style="color:#e6edf7">Rationale.</b>
    {worst_a.rationale} {detail}.</p></div>""", unsafe_allow_html=True)

section("AI agent console")
st.markdown(f'<div class="card"><div class="k">{AGENT_NAME}</div>'
            f'<div style="font-size:.95rem;color:#c3cfdf;margin-top:6px">{AGENT_ROLE}</div></div>',
            unsafe_allow_html=True)
st.write("")
trace_col, plan_col = st.columns([1, 1.25])
gm = state[worst]
ga, g_last, g_base = gm["a"], gm["last"], gm["base"]
protocols = {m: plan_protocol(m, state[m]["a"].status, state[m]["a"].model_score) for m in MUSCLES if m in state}
active = {m: p for m, p in protocols.items() if p}
steps = [
    ("Observe", f"{worst}: RMS {float(g_last.rms_mv):.3f} mV, MDF {float(g_last.median_frequency_hz):.1f} Hz, "
                f"{gm['n']} sessions on record."),
    ("Compare", f"Baseline RMS {g_base['baseline_rms_mv']:.3f} mV, MDF {g_base['baseline_mdf_hz']:.1f} Hz; "
                f"ΔRMS {ga.rms_change_pct:+.1f}%, ΔMDF {ga.mdf_change_pct:+.1f}%."),
    ("Classify", f"Change score {ga.model_score:.2f} gives {ga.status}. {ga.uncertainty}."),
    ("Decide", f"{EMS_DECISION[ga.status]}."),
    ("Plan", f"Protocol generated for {', '.join(active)}." if active else NEXT_ACTION[ga.status]),
]
with trace_col:
    rows = "".join(f'<div class="row" style="display:block"><b style="color:#4cc3ff">{i}. {name}</b>'
                   f'<div style="margin-top:2px">{text}</div></div>' for i, (name, text) in enumerate(steps, 1))
    st.markdown(f'<div class="card"><div class="k">Reasoning trace · {worst}</div>{rows}</div>',
                unsafe_allow_html=True)
with plan_col:
    if active:
        for m, p in active.items():
            body = "".join(f'<div class="row"><span>{k}</span><b>{v}</b></div>' for k, v in protocol_rows(p))
            st.markdown(f'<div class="card" style="margin-bottom:10px"><div class="k">EMS protocol · {m}</div>'
                        f'<div class="v" style="color:#e5564d">{p["current_ma"]:.1f} mA</div>{body}</div>',
                        unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="card"><div class="k">EMS protocol</div>'
                    f'<div class="v">None generated</div>'
                    f'<p style="color:#a9b6c8;font-size:.9rem">{NEXT_ACTION[ga.status]}</p></div>',
                    unsafe_allow_html=True)

ordered = sorted((m for m in MUSCLES if m in state), key=lambda x: -STATUS_RANK[state[x]["a"].status])
crew_name = selected["display_name"].split(" — ")[0]
section(f"Post-wake exercise routine · {crew_name}")
first = ordered[0]
st.markdown(f'<div style="color:#a9b6c8;font-size:.92rem;margin-bottom:10px">After waking, begin with the '
            f'<b style="color:#e6edf7">{first}</b> block ({state[first]["a"].status}), then continue in the order below. '
            f'Stable groups receive a single maintenance exercise.</div>', unsafe_allow_html=True)
ex_cols = st.columns(len(ordered))
for step, (col, m) in enumerate(zip(ex_cols, ordered), 1):
    info = COUNTERMEASURES[m]
    status = state[m]["a"].status
    chosen = info["exercises"] if status != "Stable" else info["exercises"][:1]
    items = "".join(f'<div class="ex">{n}<span>{why}</span></div>' for n, why in chosen)
    priority = {"Stable": "Maintenance", "Watch": "Priority: moderate", "Elevated change flag": "Priority: high"}[status]
    col.markdown(f'<div class="card"><div class="k">Block {step} · {m}</div>{pill(status)} '
                 f'<span style="font-size:.78rem;color:#7f8ea3;margin-left:6px">{priority}</span>'
                 f'<div style="font-size:.82rem;color:#7f8ea3;margin:8px 0 4px 0">{info["target"]}</div>{items}</div>',
                 unsafe_allow_html=True)

section(f"Signal analysis · {muscle_view}")
sig = get_recent_raw_signal(DEFAULT_DB, aid, muscle_view)
if sig and muscle_view in state:
    raw, fs = sig
    feats = extract_features(raw, fs)
    show_plot(signal_figure(raw, fs, state[muscle_view]["base"]["baseline_mdf_hz"], feats))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMS (mV)", f"{feats['rms_mv']:.3f}", f"{state[muscle_view]['a'].rms_change_pct:+.1f}% vs baseline")
    m2.metric("Median frequency (Hz)", f"{feats['median_frequency_hz']:.1f}", f"{state[muscle_view]['a'].mdf_change_pct:+.1f}% vs baseline")
    m3.metric("Sampling rate (Hz)", f"{fs:.0f}")
    m4.metric("Epoch length (s)", f"{len(raw) / fs:.1f}")
else:
    st.info("No raw signal stored for this muscle group.")

section("Longitudinal trends")
show_plot(trend_figure(aid))

section("Decision history")
hist_rows = get_decision_history(DEFAULT_DB, aid)
if hist_rows:
    df = pd.DataFrame(hist_rows)
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d")
    df = df[["created_at", "muscle_group", "condition_class", "model_score", "confidence", "agent_action"]].head(45)
    df.columns = ["Date", "Muscle group", "Condition class", "Score", "Evidence (%)", "Agent state"]
    show_table(df)

with st.expander("Methods"):
    st.markdown("""
**Acquisition.** Single-channel surface EMG, 2.5 s epochs, 1 kHz sampling, one standardized task per session.

**Preprocessing.** 4th-order Butterworth band-pass (20–450 Hz, zero-phase), followed by a 50 Hz IIR notch (Q = 30).

**Features.** Root-mean-square amplitude of the rectified signal; median frequency (MDF) of the Welch power spectrum (segment length 1024).

**Change score.** `score = 0.65 · max(0, −ΔRMS%) / 30 + 0.35 · max(0, −ΔMDF%) / 15`, clipped to [0, 1], where Δ is relative to the individual baseline for each muscle group.

**Classes.** Stable (< 0.30), Watch (0.30–0.67), Elevated change flag (≥ 0.67).

**Evidence coverage.** `min(n, 7) / 7`, where n is the number of sessions on record; reduced to 45% when the task label does not match the baseline.

**Agent.** Rule-based policy (not a trained model): the change class sets the decision, and for an Elevated change flag the planner scales stimulus current within a per-muscle range by score severity. Voltage follows V = I·R from an assumed electrode-skin impedance, and charge per phase is I × pulse width. All stimulation values are outputs of the simulated stimulator model.
""")
