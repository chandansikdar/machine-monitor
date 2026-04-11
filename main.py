"""
main.py \u2014 Symbion Machine Analytics Platform
Electrical diagnostics for three-phase induction motor-driven equipment.
Run with:  streamlit run main.py
"""

from __future__ import annotations

import dataclasses
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from database import Database
from electrical_diagnostics import (
    AssessmentRecord,
    BaselineMetadata,
    BandRecord,
    BaselineState,
    CleaningReport,
    MotorSideResult,
    Zone4Result,
    SupplyAlarm,
    IUF_CRITICAL,
    IUF_WATCH,
    PF_DRIFT_ACTION,
    PF_DRIFT_ALERT,
    PF_DRIFT_WATCH,
    VUF_CRITICAL,
    VUF_WATCH,
    ZONE4_SIGNIFICANCE_PCT,
    ingest_baseline,
    run_assessment,
    assessment_summary,
)

try:
    from data_checker import run_data_quality_checks, format_quality_report_for_claude
    DQ_AVAILABLE = True
except ImportError:
    DQ_AVAILABLE = False
    def run_data_quality_checks(df, **kw):
        return {"issues": [], "summary": {"total": 0, "critical": 0, "warning": 0}, "passed": True, "score": 100}
    def format_quality_report_for_claude(r): return ""

load_dotenv()

# ---------------------------------------------------------------------------
# Required DataFrame columns
# ---------------------------------------------------------------------------

REQUIRED_COLS = [
    "phase_1_voltage", "phase_2_voltage", "phase_3_voltage",
    "phase_1_current", "phase_2_current", "phase_3_current",
    "phase_1_active_power", "phase_2_active_power", "phase_3_active_power",
]

# ---------------------------------------------------------------------------
# Machine type registry
# ---------------------------------------------------------------------------

MACHINE_TYPES = [
    "Centrifugal Pump",
    "Reciprocating Compressor",
    "Screw Compressor",
    "Chiller",
]

# Auto-derive application_type from machine type (controls Zone 4)
APP_TYPE_MAP = {
    "Centrifugal Pump":        "process_pump",
    "Reciprocating Compressor": "compressed_air",
    "Screw Compressor":         "compressed_air",
    "Chiller":                  "hvac_chiller",
}

# ---------------------------------------------------------------------------
# Baseline serialisation helpers
# ---------------------------------------------------------------------------

def baseline_to_dict(bm: BaselineMetadata) -> dict:
    return dataclasses.asdict(bm)


def baseline_from_dict(d: dict) -> BaselineMetadata:
    """Reconstruct BaselineMetadata from a plain dict (loaded from JSON)."""
    cr_d = d.get("cleaning_report")
    cleaning = CleaningReport(**cr_d) if cr_d else None

    bands = [BandRecord(**b) for b in (d.get("bands") or [])]

    bs_d = d.get("baseline_state")
    state = BaselineState(**bs_d) if bs_d else None

    return BaselineMetadata(
        timestamp_start=d.get("timestamp_start"),
        timestamp_end=d.get("timestamp_end"),
        p_baseline_avg_kw=d.get("p_baseline_avg_kw"),
        bands=bands,
        n_qualifying_bands=d.get("n_qualifying_bands", 0),
        cleaning_report=cleaning,
        baseline_state=state,
        user_filter_expr=d.get("user_filter_expr"),
        warnings=d.get("warnings") or [],
    )


# ---------------------------------------------------------------------------
# Electrical metadata helpers
# ---------------------------------------------------------------------------

_META_SENTINEL = "=== ELECTRICAL METADATA ==="

def parse_electrical_meta(description: str) -> dict:
    """Extract electrical nameplate values from the machine description text."""
    meta = {}
    if _META_SENTINEL not in description:
        return meta
    block = description.split(_META_SENTINEL, 1)[1].split("===")[0]
    for line in block.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        try:
            if k == "four_wire":
                meta[k] = v.lower() == "true"
            elif k == "measurement_at_panel":
                meta[k] = v.lower() == "true"
            elif k == "application_type":
                meta[k] = v
            else:
                meta[k] = float(v)
        except ValueError:
            meta[k] = v
    return meta


def build_meta(machine_info: dict) -> dict | None:
    """Build the metadata dict required by electrical_diagnostics functions.
    Returns None if required fields are missing.
    """
    desc = machine_info.get("description", "")
    em = parse_electrical_meta(desc)
    required = ["v_nominal_phase", "p_rated_shaft_kw", "pf_rated", "eta_rated", "i_rated"]
    missing = [r for r in required if r not in em]
    if missing:
        return None
    # Derive application_type from machine type if not explicitly set
    if "application_type" not in em:
        em["application_type"] = APP_TYPE_MAP.get(
            machine_info.get("machine_type", ""), "compressed_air"
        )
    em.setdefault("four_wire", True)
    em.setdefault("measurement_at_panel", True)
    return em


def serialise_meta_block(
    v_nominal: float, p_rated: float, pf_rated: float,
    eta_rated: float, i_rated: float,
    four_wire: bool, at_panel: bool, app_type: str,
) -> str:
    return (
        f"{_META_SENTINEL}\n"
        f"v_nominal_phase: {v_nominal}\n"
        f"p_rated_shaft_kw: {p_rated}\n"
        f"pf_rated: {pf_rated}\n"
        f"eta_rated: {eta_rated}\n"
        f"i_rated: {i_rated}\n"
        f"four_wire: {str(four_wire).lower()}\n"
        f"measurement_at_panel: {str(at_panel).lower()}\n"
        f"application_type: {app_type}\n"
    )


def replace_meta_block(description: str, new_block: str) -> str:
    """Replace the electrical metadata block in a description string."""
    if _META_SENTINEL in description:
        # Remove old block (up to the next === or end of string)
        before = description.split(_META_SENTINEL, 1)[0].rstrip()
        after_raw = description.split(_META_SENTINEL, 1)[1]
        # Find where the next section starts (another ===)
        if "===" in after_raw:
            after = after_raw.split("===", 1)[1]
            after = "===" + after
        else:
            after = ""
        return (before + "\n\n" + new_block + ("\n\n" + after.strip() if after.strip() else "")).strip()
    else:
        return (description.strip() + "\n\n" + new_block).strip()


# ---------------------------------------------------------------------------
# Column validation
# ---------------------------------------------------------------------------

def check_required_columns(df: pd.DataFrame) -> list[str]:
    """Return list of required columns that are missing from df."""
    return [c for c in REQUIRED_COLS if c not in df.columns]


# ---------------------------------------------------------------------------
# Assessment rendering
# ---------------------------------------------------------------------------

def _tier_badge(tier: str | None) -> str:
    colours = {"critical": "\U0001f534", "action": "\U0001f534",
               "alert": "\U0001f7e0", "watch": "\U0001f7e1", None: "\U0001f7e2"}
    labels  = {"critical": "Critical", "action": "Action",
               "alert": "Alert",    "watch": "Watch",  None: "Normal"}
    icon  = colours.get(tier, "\u26aa")
    label = labels.get(tier, str(tier).title() if tier else "Normal")
    return f"{icon} **{label}**"


def render_cleaning_report(report: CleaningReport):
    st.markdown("**Data cleaning**")
    steps = [
        ("Raw samples",         report.n_raw),
        ("After integrity gate",report.n_after_integrity),
        ("After running mask",  report.n_after_running_mask),
        ("After user filter",   report.n_after_user_filter),
        ("After load \u226540%",     report.n_after_load_precondition),
        ("After IQR rejection", report.n_cleaned),
    ]
    cols = st.columns(len(steps))
    for col, (label, count) in zip(cols, steps):
        col.metric(label, f"{count:,}")
    retained_pct = report.fraction_retained * 100
    colour = "green" if retained_pct >= 70 else "orange" if retained_pct >= 40 else "red"
    st.caption(f":{colour}[{retained_pct:.0f}% of raw samples retained for analysis]")


def render_supply_zone(alarm: SupplyAlarm):
    tier = alarm.tier
    vuf  = alarm.vuf_pct
    if tier is None:
        bg, bc = "#F0FFF4", "#177E40"
        icon, label = "\U0001f7e2", "Normal"
        body = f"VUF = **{vuf:.2f}%** \u2014 below {VUF_WATCH:.1f}% watch threshold."
    elif tier == "watch":
        bg, bc = "#FFFBF0", "#BA7517"
        icon, label = "\U0001f7e1", "Watch"
        body = (f"VUF = **{vuf:.2f}%** \u2014 above {VUF_WATCH:.1f}% watch threshold. "
                f"Log and monitor. No immediate action required.")
    else:
        bg, bc = "#FFF0F0", "#A32D2D"
        icon, label = "\U0001f534", "Critical"
        body = (f"VUF = **{vuf:.2f}%** \u2014 above {VUF_CRITICAL:.1f}% critical threshold. "
                f"Investigate upstream installation. Motor derating recommended (NEMA MG-1).")

    st.markdown(
        f'<div style="background:{bg};border-left:5px solid {bc};'
        f'padding:12px 16px;border-radius:4px;margin-bottom:8px">'
        f'<span style="font-weight:700;color:{bc}">{icon} Zone 1 \u2014 Supply Channel: {label}</span><br>'
        f'<span style="font-size:0.9em">{body}</span></div>',
        unsafe_allow_html=True,
    )


def render_motor_side(m: MotorSideResult):
    cell_colours = {1: "#F0FFF4", 2: "#FFF0F0", 3: "#FFFBF0", 4: "#FFF0F0"}
    cell_borders = {1: "#177E40",  2: "#A32D2D",  3: "#BA7517",  4: "#A32D2D"}
    cell_icons   = {1: "\U0001f7e2", 2: "\U0001f534", 3: "\U0001f7e1", 4: "\U0001f534"}

    bg = cell_colours.get(m.cell, "#F8F8F8")
    bc = cell_borders.get(m.cell, "#555")
    icon = cell_icons.get(m.cell, "\u26aa")
    tier_str = f" \u2014 {m.alarm_tier.title()} tier" if m.alarm_tier else ""

    # IUF line
    if m.iuf_tier:
        iuf_html = (f"IUF = <b>{m.iuf_mean_pct:.1f}%</b> "
                    f"(Phase {m.outlier_phase} {m.outlier_direction}, "
                    f"threshold {IUF_WATCH:.0f}%)")
    else:
        iuf_html = f"IUF = <b>{m.iuf_mean_pct:.1f}%</b> (below {IUF_WATCH:.0f}% threshold)"

    # PF drift line
    if m.pf_drift_suppressed:
        pf_html = f"PF drift \u2014 suppressed ({m.pf_drift_suppression_reason})"
    elif m.pf_drift_aggregated is not None:
        pf_html = (f"PF drift = <b>{m.pf_drift_aggregated:+.3f}</b> "
                   f"(watch \u2264 {PF_DRIFT_WATCH:.2f}, "
                   f"alert \u2264 {PF_DRIFT_ALERT:.2f}, "
                   f"action \u2264 {PF_DRIFT_ACTION:.2f})")
    else:
        pf_html = "PF drift \u2014 no data"

    # Inspection refs
    refs_html = ""
    if m.inspection_refs:
        refs_html = ("<br><span style='font-size:0.82em;color:#555'>"
                     "Inspection: " + " \u00b7 ".join(m.inspection_refs) + "</span>")

    st.markdown(
        f'<div style="background:{bg};border-left:5px solid {bc};'
        f'padding:12px 16px;border-radius:4px;margin-bottom:8px">'
        f'<span style="font-weight:700;color:{bc}">'
        f'{icon} Zones 2 & 3 \u2014 Motor-side: Cell {m.cell} \u2014 {m.cell_label}{tier_str}'
        f'</span><br>'
        f'<span style="font-size:0.9em">{iuf_html}</span><br>'
        f'<span style="font-size:0.9em">{pf_html}</span>'
        f'{refs_html}</div>',
        unsafe_allow_html=True,
    )

    # Per-band breakdown (collapsible)
    active_bands = [b for b in m.bands if not b.suppressed and b.pf_drift is not None]
    if active_bands:
        with st.expander(f"PF drift \u2014 {len(active_bands)} band(s) active", expanded=False):
            rows = []
            for b in active_bands:
                rows.append({
                    "Band centre (kW)":  f"{b.centre_kw:.1f}",
                    "Baseline PF":       f"{b.mean_pf_baseline:.4f}",
                    "Recent PF":         f"{b.mean_pf_recent:.4f}" if b.mean_pf_recent else "\u2014",
                    "Drift":             f"{b.pf_drift:+.4f}" if b.pf_drift is not None else "\u2014",
                    "n baseline":        b.n_baseline,
                    "n recent":          b.n_recent,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    suppressed_bands = [b for b in m.bands if b.suppressed]
    if suppressed_bands:
        st.caption(
            f"\u26a0\ufe0f {len(suppressed_bands)} band(s) suppressed (insufficient recent samples): "
            + ", ".join(f"{b.centre_kw:.1f} kW" for b in suppressed_bands)
        )


def render_zone4(z: Zone4Result):
    if z.suppressed:
        bg, bc = "#F4F4F4", "#888"
        icon = "\u23d0"
        header = "Zone 4 \u2014 Driven Equipment: Suppressed"
    elif z.finding_state is None:
        bg, bc = "#F0FFF4", "#177E40"
        icon = "\U0001f7e2"
        header = "Zone 4 \u2014 Driven Equipment: No Finding"
    elif z.finding_state == "positive":
        bg, bc = "#FFFBF0", "#BA7517"
        icon = "\U0001f7e1"
        header = f"Zone 4 \u2014 Driven Equipment: Power Increase ({z.delta_p_pct:+.1f}%)"
    else:
        bg, bc = "#EAF4FF", "#185FA5"
        icon = "\U0001f535"
        header = f"Zone 4 \u2014 Driven Equipment: Power Decrease ({z.delta_p_pct:+.1f}%)"

    st.markdown(
        f'<div style="background:{bg};border-left:5px solid {bc};'
        f'padding:12px 16px;border-radius:4px;margin-bottom:8px">'
        f'<span style="font-weight:700;color:{bc}">{icon} {header}</span><br>'
        f'<span style="font-size:0.9em">{z.message}</span></div>',
        unsafe_allow_html=True,
    )


def render_baseline_state(state: BaselineState):
    """Show baseline validation messages if any alerts were raised at ingestion."""
    if state is None:
        return
    messages = []
    if state.iuf_message:
        messages.append(("IUF check", state.iuf_tier, state.iuf_message))
    if state.pf_message:
        messages.append(("PF check", state.pf_tier, state.pf_message))
    if not messages:
        return
    with st.expander("\u26a0\ufe0f Baseline validation alerts", expanded=True):
        for label, tier, msg in messages:
            colour = "#A32D2D" if tier == "critical" else "#BA7517" if tier in ("alert","action","watch") else "#185FA5"
            bg = "#FFF0F0" if tier == "critical" else "#FFFBF0" if tier else "#EAF4FF"
            st.markdown(
                f'<div style="background:{bg};border-left:4px solid {colour};'
                f'padding:8px 12px;margin-bottom:6px;border-radius:3px;font-size:0.88em">'
                f'<b>{label}:</b> {msg}</div>',
                unsafe_allow_html=True,
            )


def render_assessment(record: AssessmentRecord):
    """Render a complete AssessmentRecord in the Streamlit UI."""
    if record.suppressed:
        st.error(f"Assessment suppressed: {record.suppression_reason}")
        return

    # Cleaning report
    if record.cleaning_report:
        render_cleaning_report(record.cleaning_report)
    st.markdown("---")

    # Zone 1
    if record.supply_alarm:
        render_supply_zone(record.supply_alarm)

    # Zones 2 & 3
    if record.motor_side:
        render_motor_side(record.motor_side)

    # Zone 4
    if record.zone4:
        render_zone4(record.zone4)


# ---------------------------------------------------------------------------
# Assessment charts
# ---------------------------------------------------------------------------

def build_assessment_charts(data: pd.DataFrame, record: AssessmentRecord) -> list:
    """Build control charts for VUF, P_total and PF_machine from the full dataset."""
    figs = []
    if data is None or data.empty:
        return figs

    missing = check_required_columns(data)
    if missing:
        return figs

    v1 = data["phase_1_voltage"]
    v2 = data["phase_2_voltage"]
    v3 = data["phase_3_voltage"]
    v_avg = (v1 + v2 + v3) / 3.0
    vuf = (
        pd.concat([
            (v1 - v_avg).abs(),
            (v2 - v_avg).abs(),
            (v3 - v_avg).abs(),
        ], axis=1).max(axis=1) / v_avg * 100.0
    )

    i1 = data["phase_1_current"]
    i2 = data["phase_2_current"]
    i3 = data["phase_3_current"]

    p_total_w = (data["phase_1_active_power"] + data["phase_2_active_power"]
                 + data["phase_3_active_power"])
    p_total_kw = p_total_w / 1000.0

    s_sum = v1 * i1 + v2 * i2 + v3 * i3
    pf_machine = (p_total_w / s_sum.replace(0, np.nan)).clip(0, 1)

    baseline = record.motor_side

    def _line_chart(x, y, title, y_label, h_lines=None, y_range=None):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color="#185FA5", width=1.2),
            name=y_label,
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>" + y_label + ": %{y:.3f}<extra></extra>",
        ))
        for val, colour, dash, name in (h_lines or []):
            fig.add_hline(
                y=val, line_color=colour, line_dash=dash, line_width=1.5,
                annotation_text=name, annotation_position="top right",
                annotation_font_size=10,
            )
        fig.update_layout(
            title=dict(text=title, font=dict(size=13)),
            xaxis_title="Time", yaxis_title=y_label,
            yaxis=dict(range=y_range) if y_range else {},
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=20, t=45, b=40),
            hovermode="x unified", font=dict(size=11), height=280,
        )
        return fig

    # VUF chart
    figs.append(_line_chart(
        data.index, vuf,
        "Zone 1 \u2014 Voltage Unbalance Factor (VUF)", "VUF (%)",
        h_lines=[
            (VUF_CRITICAL, "#C0392B", "solid",  f"Critical {VUF_CRITICAL:.1f}%"),
            (VUF_WATCH,    "#E67E22", "dash",   f"Watch {VUF_WATCH:.1f}%"),
        ],
        y_range=[0, max(float(vuf.max()) * 1.3, VUF_CRITICAL * 1.5)],
    ))

    # P_total chart — add baseline mean if available
    p_hlines = []
    if record.zone4 and record.zone4.p_baseline_avg_kw:
        p_hlines.append((
            record.zone4.p_baseline_avg_kw, "#054D5F", "dashdot",
            f"Baseline avg {record.zone4.p_baseline_avg_kw:.1f} kW",
        ))
    figs.append(_line_chart(
        data.index, p_total_kw,
        "Total Active Power (P_total)", "P_total (kW)",
        h_lines=p_hlines or None,
    ))

    # PF_machine chart — add per-band baselines if available
    pf_hlines = []
    if baseline and baseline.bands:
        for b in baseline.bands[:6]:  # cap at 6 to keep chart readable
            pf_hlines.append((
                b.mean_pf_baseline, "rgba(180,180,180,0.6)", "dot",
                f"Band {b.centre_kw:.0f} kW baseline PF",
            ))
    figs.append(_line_chart(
        data.index, pf_machine,
        "Machine Power Factor (PF_machine)", "PF_machine",
        h_lines=pf_hlines or None,
        y_range=[0, 1.05],
    ))

    return figs


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Machine Analytics",
    page_icon="\U0001f4ca",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="metric-container"] {
        background: rgba(128,128,128,0.05);
        border-radius: 8px;
        padding: 8px 12px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1rem !important;
    }
    .block-container { padding-top: 1.5rem; }
    section[data-testid="stSidebar"] .stButton { margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

for _k, _v in [
    ("last_assessment", None),
    ("last_data",       None),
    ("last_dq_report",  None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------

@st.cache_resource
def get_services(_v=1):
    return Database()

db = get_services()


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
        <style>
        section[data-testid="stSidebar"] .block-container { padding-top: 0.5rem; }
        section[data-testid="stSidebar"] hr { margin: 0.4rem 0; }
        </style>""", unsafe_allow_html=True)

    st.title("Machine Analytics")
    st.caption("Electrical Diagnostics \u2014 Symbion")

    # ── Register new machine ──────────────────────────────────────────────
    with st.expander("Register new machine", expanded=not db.get_machines()):
        machine_id = st.text_input(
            "Machine ID",
            placeholder="e.g. COMP-001, PUMP-A3",
            help="Your site reference tag.",
        )
        machine_type = st.selectbox(
            "Machine type",
            options=["-- Select --"] + MACHINE_TYPES,
            key="reg_machine_type",
        )
        machine_type = machine_type if machine_type != "-- Select --" else ""

        if machine_type:
            app_type = APP_TYPE_MAP.get(machine_type, "compressed_air")
            zone4_note = (
                "\u26a0\ufe0f Zone 4 detection suppressed for Chiller "
                "(temperature-dependent load \u2014 methodology pending)."
                if app_type == "hvac_chiller"
                else f"\u2705 Zone 4 detection active (application: {app_type})."
            )
            st.caption(zone4_note)

        if st.button("Register", type="primary", use_container_width=True,
                     disabled=not (machine_id and machine_type)):
            db.register_machine(machine_id.strip(), machine_type.strip(), "")
            st.success(f"**{machine_id}** registered. Enter electrical specs in the main area.")
            st.rerun()

    st.markdown("---")

    # ── Machine selector ──────────────────────────────────────────────────
    machines = db.get_machines()
    if not machines:
        st.info("Register a machine above to get started.")
        st.stop()

    machine_labels = {
        m["machine_id"]: f"{m['machine_id']}  ({m['machine_type']})"
        for m in machines
    }
    selected_id = st.selectbox(
        "Active machine",
        options=list(machine_labels.keys()),
        format_func=lambda x: machine_labels[x],
        key="active_machine_select",
    )

    # Clear session data when machine changes
    if st.session_state.get("_last_machine") != selected_id:
        st.session_state["last_assessment"] = None
        st.session_state["last_data"]       = None
        st.session_state["last_dq_report"]  = None
        st.session_state["_last_machine"]   = selected_id

    # ── Delete machine ────────────────────────────────────────────────────
    with st.expander("\U0001f5d1\ufe0f Delete this machine", expanded=False):
        st.warning(f"Permanently delete **{selected_id}** and all its data.")
        if st.checkbox(f"Confirm delete {selected_id}", key=f"del_confirm_{selected_id}"):
            if st.button("\U0001f5d1\ufe0f Delete permanently", type="primary",
                         key=f"del_btn_{selected_id}", use_container_width=True):
                db.delete_baseline(selected_id)
                db.delete_machine(selected_id)
                st.session_state["_last_machine"] = None
                st.rerun()

    st.markdown("---")

    # ── Data upload ───────────────────────────────────────────────────────
    st.subheader("Upload data")
    st.caption(
        "CSV or Excel with columns: "
        "`phase_1_voltage`, `phase_2_voltage`, `phase_3_voltage`, "
        "`phase_1_current`, `phase_2_current`, `phase_3_current`, "
        "`phase_1_active_power`, `phase_2_active_power`, `phase_3_active_power` "
        "+ a timestamp column."
    )
    uploaded_file = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])

    _existing = db.get_file_info(selected_id)
    _dup = bool(
        uploaded_file and any(
            f["file"].rsplit(".", 1)[0] == uploaded_file.name.rsplit(".", 1)[0]
            for f in _existing
        )
    )

    if st.button("Ingest", use_container_width=True, disabled=not uploaded_file):
        if _dup:
            st.warning(f"\u26a0\ufe0f **{uploaded_file.name}** already ingested. Delete it first to re-upload.")
        else:
            with st.spinner("Reading and storing data\u2026"):
                result = db.ingest_file(uploaded_file, selected_id)
            if result["success"]:
                st.success(f"\u2713 {result['rows']:,} rows ingested")
                # Reset session data
                st.session_state["last_assessment"] = None
                st.session_state["last_data"]       = None
                st.session_state["last_dq_report"]  = None
                st.rerun()
            else:
                st.error(result["error"])

    # File management
    file_info = db.get_file_info(selected_id)
    if file_info:
        st.caption(f"{len(file_info)} file(s) stored")
        with st.expander("Manage files", expanded=False):
            for fi in file_info:
                c1, c2 = st.columns([0.8, 0.2])
                c1.caption(f"\U0001f4c2 {fi['file']}  \u00b7  {fi['rows']:,} rows")
                if c2.button("Del", key=f"del_file_{fi['file']}"):
                    db.delete_file(selected_id, fi["file"])
                    st.session_state["last_data"] = None
                    st.rerun()


# ---------------------------------------------------------------------------
# MAIN AREA
# ---------------------------------------------------------------------------

machine_info = db.get_machine_info(selected_id)
if not machine_info:
    st.warning("Machine not found.")
    st.stop()

st.title(f"{machine_info['machine_type']}  \u00b7  {selected_id}")

# ---------------------------------------------------------------------------
# Electrical parameters expander
# ---------------------------------------------------------------------------

with st.expander("\u26a1 Electrical parameters (nameplate)", expanded=not build_meta(machine_info)):
    st.caption(
        "Required for diagnostics. Enter values from the motor nameplate. "
        "Saved immediately to the machine profile."
    )
    _desc = machine_info.get("description", "")
    _em   = parse_electrical_meta(_desc)

    _c1, _c2, _c3 = st.columns(3)
    _v_nom = _c1.number_input(
        "Phase-to-neutral voltage (V)",
        min_value=0.0, value=float(_em.get("v_nominal_phase", 230.0)),
        step=1.0, format="%.1f", key="ep_v_nom",
        help="e.g. 230 V for a 400/230 V system",
    )
    _p_rated = _c2.number_input(
        "Rated shaft power (kW)",
        min_value=0.0, value=float(_em.get("p_rated_shaft_kw", 0.0)),
        step=0.5, format="%.1f", key="ep_p_rated",
    )
    _i_rated = _c3.number_input(
        "Full-load current (A)",
        min_value=0.0, value=float(_em.get("i_rated", 0.0)),
        step=0.1, format="%.1f", key="ep_i_rated",
    )
    _c4, _c5, _c6 = st.columns(3)
    _pf_rated = _c4.number_input(
        "Rated full-load PF",
        min_value=0.0, max_value=1.0,
        value=float(_em.get("pf_rated", 0.87)),
        step=0.01, format="%.2f", key="ep_pf_rated",
    )
    _eta_rated = _c5.number_input(
        "Rated efficiency (0\u20131)",
        min_value=0.0, max_value=1.0,
        value=float(_em.get("eta_rated", 0.90)),
        step=0.01, format="%.2f", key="ep_eta_rated",
        help="e.g. 0.93 for 93%",
    )
    _at_panel = _c6.checkbox(
        "Voltage measured at panel",
        value=bool(_em.get("measurement_at_panel", True)),
        key="ep_at_panel",
    )
    _fw = st.checkbox(
        "Four-wire system (neutral present)",
        value=bool(_em.get("four_wire", True)),
        key="ep_four_wire",
    )

    if st.button("Save electrical parameters", key="save_ep_btn", use_container_width=True):
        _app_type = APP_TYPE_MAP.get(machine_info["machine_type"], "compressed_air")
        _new_block = serialise_meta_block(
            _v_nom, _p_rated, _pf_rated, _eta_rated, _i_rated, _fw, _at_panel, _app_type
        )
        _non_meta = _desc.split(_META_SENTINEL)[0].rstrip() if _META_SENTINEL in _desc else _desc
        # Preserve any === NOTES === block that may follow
        _new_desc = replace_meta_block(_desc, _new_block)
        db.register_machine(selected_id, machine_info["machine_type"], _new_desc)
        st.success("\u2713 Electrical parameters saved.")
        st.rerun()

    # Show status
    _meta_check = build_meta(machine_info)
    if _meta_check is None:
        st.warning("\u26a0\ufe0f Enter and save electrical parameters before running diagnostics.")
    else:
        st.success("\u2705 Electrical parameters complete \u2014 diagnostics ready.")

# ---------------------------------------------------------------------------
# Machine notes expander
# ---------------------------------------------------------------------------

with st.expander("\u270f\ufe0f Machine notes", expanded=False):
    st.caption("Installation details, commissioning date, service history notes.")
    _desc_full  = machine_info.get("description", "")
    _notes_only = _desc_full.split(_META_SENTINEL)[0].strip() if _META_SENTINEL in _desc_full else _desc_full
    _new_notes  = st.text_area(
        "Notes", value=_notes_only, height=120, label_visibility="collapsed",
        key="machine_notes_ta",
    )
    if st.button("Save notes", key="save_notes_btn"):
        if _META_SENTINEL in _desc_full:
            _meta_part = _META_SENTINEL + _desc_full.split(_META_SENTINEL, 1)[1]
            _updated = (_new_notes.strip() + "\n\n" + _meta_part).strip()
        else:
            _updated = _new_notes.strip()
        db.register_machine(selected_id, machine_info["machine_type"], _updated)
        st.success("\u2713 Notes saved.")
        st.rerun()

# ---------------------------------------------------------------------------
# Maintenance logs expander
# ---------------------------------------------------------------------------

with st.expander("\U0001f4cb Maintenance logs", expanded=False):
    st.caption("Text logs are included automatically in every assessment report.")
    _ml_title = st.text_input(
        "Entry title", placeholder="e.g. Bearing replacement 2024-03-01", key="ml_title"
    )
    _ml_body = st.text_area(
        "Log entry",
        placeholder=(
            "e.g. Replaced drive-end bearing (SKF 6308). Vibration was 4.2 mm/s before, "
            "dropped to 1.1 mm/s after. Seal inspected \u2014 no leaks found."
        ),
        height=120, key="ml_body", label_visibility="collapsed",
    )
    if st.button("Save log entry", disabled=not _ml_body.strip(), key="save_log_btn",
                 use_container_width=True):
        _fname = (_ml_title.strip() or "Manual entry") + ".txt"
        db.save_log(selected_id, _fname, "text", _ml_body.strip())
        st.success(f"\u2713 Saved as **{_fname}**")
        st.rerun()

    _logs = db.get_logs(selected_id)
    if _logs:
        st.markdown(f"**{len(_logs)} log(s) stored**")
        for _l in _logs:
            _lc1, _lc2 = st.columns([0.85, 0.15])
            _lc1.caption(f"\u270f\ufe0f {_l['filename']}  \u00b7  {str(_l['uploaded_at'])[:10]}")
            if _lc2.button("Del", key=f"del_log_{_l['filename']}"):
                db.delete_log(selected_id, _l["filename"])
                st.rerun()
    else:
        st.caption("No logs stored yet.")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

_file_info = db.get_file_info(selected_id)
_active_file = st.session_state.get(f"active_file_{selected_id}")
if not _active_file and _file_info:
    _active_file = _file_info[-1]["file"]
    st.session_state[f"active_file_{selected_id}"] = _active_file

data = db.get_data_from_file(selected_id, _active_file) if _active_file else None

# Auto-clear stale session data when the loaded dataset changes
_data_fp = (
    f"{selected_id}|{len(data)}|{str(data.index.min())}|{str(data.index.max())}"
    if data is not None and not data.empty else f"{selected_id}|empty"
)
if st.session_state.get("_data_fp") != _data_fp:
    st.session_state["_data_fp"]       = _data_fp
    st.session_state["last_assessment"] = None
    st.session_state["last_dq_report"]  = None
    st.session_state["last_data"]       = None


# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------

tab_data, tab_analysis, tab_history, tab_logs = st.tabs(
    ["Data", "Analysis", "History", "Maintenance Logs"]
)


# ================================================================== #
# TAB 1 \u2014 Data
# ================================================================== #

with tab_data:
    if data is None or data.empty:
        st.info("No data uploaded yet. Use the sidebar to ingest a CSV or Excel file.")
    else:
        numeric_cols = data.select_dtypes(include="number").columns.tolist()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows",       f"{len(data):,}")
        c2.metric("Parameters", len(numeric_cols))
        c3.metric("From",       str(data.index.min().date()))
        c4.metric("To",         str(data.index.max().date()))

        # Required column check
        missing_cols = check_required_columns(data)
        if missing_cols:
            st.error(
                f"\u274c **Required columns missing:** {', '.join(f'`{c}`' for c in missing_cols)}  \n"
                "Please ensure your CSV uses the exact column names listed in the sidebar upload hint."
            )
        else:
            st.success("\u2705 All required electrical measurement columns present.")

        # Data quality (auto-run)
        if DQ_AVAILABLE and st.session_state.get("last_dq_report") is None:
            with st.spinner("Running data quality checks\u2026"):
                _dq = run_data_quality_checks(data)
                st.session_state["last_dq_report"] = _dq

        _dq = st.session_state.get("last_dq_report") or {}
        if _dq:
            _score = _dq.get("score", 100)
            _issues = _dq.get("issues", [])
            _crits  = [x for x in _issues if x["severity"] == "critical"]
            _warns  = [x for x in _issues if x["severity"] == "warning"]
            _dq_label = (
                (f"  \u00b7  {len(_crits)} critical" if _crits else "") +
                (f"  \u00b7  {len(_warns)} warning(s)" if _warns else "") +
                ("  \u00b7  All checks passed" if not _issues else "")
            )
            with st.expander(f"Data quality \u2014 score {_score}/100{_dq_label}",
                             expanded=bool(_crits)):
                if not _issues:
                    st.success("All data quality checks passed.")
                else:
                    _sev_bc = {"critical": "#A32D2D", "warning": "#BA7517", "info": "#185FA5"}
                    _sev_bg = {"critical": "#FFF0F0", "warning": "#FFFBF0", "info": "#EAF4FF"}
                    for _iss in _issues:
                        _sv = _iss["severity"]
                        _icon = {"critical": "\u274c", "warning": "\u26a0\ufe0f", "info": "\u2139\ufe0f"}.get(_sv, "\u2022")
                        st.markdown(
                            f'<div style="background:{_sev_bg.get(_sv,"#f8f8f8")};'
                            f'border-left:4px solid {_sev_bc.get(_sv,"#555")};'
                            f'padding:8px 12px;margin-bottom:4px;border-radius:3px;">'
                            f'<b>{_icon} {_iss["check"]}</b> \u00b7 <code>{_iss["col"]}</code>'
                            f' \u00b7 <span style="color:#888;font-size:0.85em">'
                            f'{_iss["affected_pct"]}% affected</span><br>'
                            f'<span style="font-size:0.88em">{_iss["detail"]}</span></div>',
                            unsafe_allow_html=True,
                        )

        st.subheader("Recent readings")
        st.dataframe(data.tail(200), use_container_width=True, height=280)

        st.subheader("Descriptive statistics")
        if numeric_cols:
            st.dataframe(data[numeric_cols].describe().round(3), use_container_width=True)

        # Sensor overview charts
        if not missing_cols:
            st.subheader("Sensor overview")

            def _group_cols(cols):
                groups = {"Voltage (V)": [], "Current (A)": [], "Active Power (W)": [], "Other": []}
                for c in cols:
                    cl = c.lower()
                    if "voltage" in cl:    groups["Voltage (V)"].append(c)
                    elif "current" in cl:  groups["Current (A)"].append(c)
                    elif "power" in cl:    groups["Active Power (W)"].append(c)
                    else:                  groups["Other"].append(c)
                return {k: v for k, v in groups.items() if v}

            for grp_name, grp_cols in _group_cols(numeric_cols).items():
                fig = go.Figure()
                for col in grp_cols:
                    fig.add_trace(go.Scatter(
                        x=data.index, y=data[col], mode="lines",
                        name=col.replace("_", " ").title(),
                        line=dict(width=1.3),
                        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>" + col + ": %{y:.3f}<extra></extra>",
                    ))
                fig.update_layout(
                    title=dict(text=grp_name, font=dict(size=13)),
                    xaxis_title="Time", yaxis_title=grp_name,
                    height=260, plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=40, r=20, t=40, b=40),
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="top", y=-0.22,
                                xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
                    font=dict(size=11),
                )
                st.plotly_chart(fig, use_container_width=True)


# ================================================================== #
# TAB 2 \u2014 Analysis
# ================================================================== #

with tab_analysis:
    if data is None or data.empty:
        st.info("Upload data first (sidebar), then run diagnostics.")
    else:
        meta = build_meta(machine_info)
        missing_cols = check_required_columns(data)

        if meta is None:
            st.warning(
                "\u26a0\ufe0f **Electrical parameters not complete.** "
                "Fill in the \u26a1 Electrical parameters expander above and save before running diagnostics."
            )
        elif missing_cols:
            st.error(
                f"\u274c Required columns missing: {', '.join(f'`{c}`' for c in missing_cols)}. "
                "Check the Data tab for details."
            )
        else:
            left, right = st.columns([1, 3])

            with left:
                st.markdown("**Date range**")
                min_d, max_d = data.index.min().date(), data.index.max().date()
                if min_d < max_d:
                    date_range = st.date_input(
                        "Date range", value=[min_d, max_d],
                        min_value=min_d, max_value=max_d,
                        label_visibility="collapsed",
                    )
                    if not isinstance(date_range, (list, tuple)) or len(date_range) < 2:
                        date_range = (min_d, max_d)
                else:
                    date_range = (min_d, max_d)
                    st.caption(f"Single-day dataset: {min_d}")

                # Baseline section
                st.markdown("---")
                st.markdown("**Baseline**")
                _stored_bl_dict = db.get_baseline(selected_id)
                if _stored_bl_dict:
                    _stored_at = _stored_bl_dict.get("_stored_at", "")[:10]
                    _n_bands   = _stored_bl_dict.get("n_qualifying_bands", 0)
                    _bl_start  = str(_stored_bl_dict.get("timestamp_start", ""))[:10]
                    _bl_end    = str(_stored_bl_dict.get("timestamp_end",   ""))[:10]
                    _bl_warns  = _stored_bl_dict.get("warnings") or []
                    _bl_p_avg  = _stored_bl_dict.get("p_baseline_avg_kw")
                    st.success(
                        f"\u2705 Baseline ingested {_stored_at}  \n"
                        f"Period: {_bl_start} \u2192 {_bl_end}  \n"
                        f"PF bands: {_n_bands}  \u00b7  "
                        f"P\u209a\u2090\u209b\u2091\u2090\u2c7c\u2091: "
                        f"{_bl_p_avg:.1f} kW" if _bl_p_avg else
                        f"\u2705 Baseline ingested {_stored_at} \u2014 {_n_bands} PF band(s)"
                    )
                    for _w in _bl_warns:
                        st.caption(f"\u26a0\ufe0f {_w}")
                    if st.button("Delete baseline", key="del_baseline_btn",
                                 type="secondary", use_container_width=True):
                        db.delete_baseline(selected_id)
                        st.rerun()
                else:
                    st.info("No baseline ingested yet.")

                with st.expander("Baseline period", expanded=not bool(_stored_bl_dict)):
                    st.caption(
                        "Select a window of confirmed healthy operation as the reference. "
                        "Ideal: first weeks after commissioning or last major service."
                    )
                    _bl_cols = st.columns(2)
                    _total_days = max(1, (max_d - min_d).days)
                    _ideal_end  = min_d + __import__("datetime").timedelta(
                        days=max(14, int(_total_days * 0.20))
                    )
                    _bl_start_in = _bl_cols[0].date_input(
                        "Start", value=min_d, min_value=min_d, max_value=max_d,
                        key="bl_start",
                    )
                    _bl_end_in = _bl_cols[1].date_input(
                        "End", value=min(_ideal_end, max_d), min_value=min_d, max_value=max_d,
                        key="bl_end",
                    )
                    _bl_days = max(1, (_bl_end_in - _bl_start_in).days)
                    if _bl_days < 7:
                        st.warning(f"\u26a0\ufe0f {_bl_days} days is short. Minimum 7 days recommended.")
                    else:
                        st.caption(f"{_bl_days}-day baseline period.")

                    if st.button("\U0001f4e5 Ingest baseline", type="primary",
                                 use_container_width=True, key="ingest_bl_btn"):
                        _bl_start_ts = pd.Timestamp(_bl_start_in)
                        _bl_end_ts   = pd.Timestamp(_bl_end_in) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                        _raw_bl = data.loc[(_bl_start_ts <= data.index) & (data.index <= _bl_end_ts)]
                        if _raw_bl.empty:
                            st.error("No data in selected baseline period.")
                        else:
                            with st.spinner("Ingesting baseline\u2026"):
                                _raw_bl_reset = _raw_bl.reset_index()
                                _bm = ingest_baseline(_raw_bl_reset, meta)
                            if len(_bm.bands) < 3:
                                st.warning(
                                    f"\u26a0\ufe0f Only {len(_bm.bands)} PF band(s) produced. "
                                    "Consider extending the baseline window for more reliable PF drift detection."
                                )
                            _bm_dict = baseline_to_dict(_bm)
                            db.save_baseline(selected_id, _bm_dict)
                            for w in _bm.warnings:
                                st.caption(f"\u26a0\ufe0f {w}")
                            st.success(
                                f"\u2713 Baseline ingested: "
                                f"{_bm.n_qualifying_bands} PF band(s), "
                                f"{_bm.cleaning_report.n_cleaned:,} cleaned samples."
                            )
                            st.session_state["last_assessment"] = None
                            st.rerun()

                st.markdown("---")
                _run_disabled = not bool(_stored_bl_dict)
                if st.button(
                    "\u25b6 Run Assessment",
                    type="primary",
                    use_container_width=True,
                    disabled=_run_disabled,
                    key="run_assessment_btn",
                ):
                    if _run_disabled:
                        st.warning("Ingest a baseline first.")
                    else:
                        # Filter data to selected date range
                        _start_ts = pd.Timestamp(date_range[0])
                        _end_ts   = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                        _recent = data.loc[(_start_ts <= data.index) & (data.index <= _end_ts)]
                        if _recent.empty:
                            st.error("No data in selected date range.")
                        else:
                            _bm_loaded = baseline_from_dict(db.get_baseline(selected_id))
                            with st.spinner("Running electrical diagnostics\u2026"):
                                _raw_reset = _recent.reset_index()
                                _record = run_assessment(_raw_reset, _bm_loaded, meta)

                            # Serialise and store in history
                            _record_dict = dataclasses.asdict(_record)
                            db.save_analysis(
                                selected_id,
                                "Electrical Diagnostics",
                                {
                                    "record": _record_dict,
                                    "date_range": [str(date_range[0]), str(date_range[1])],
                                    "summary": assessment_summary(_record),
                                },
                            )
                            st.session_state["last_assessment"] = _record
                            st.session_state["last_data"]       = _recent
                            st.rerun()

                if _run_disabled:
                    st.caption("Ingest a baseline to enable assessment.")

            # ── Results ──────────────────────────────────────────────────────
            with right:
                record: AssessmentRecord | None = st.session_state.get("last_assessment")

                if record is None:
                    if _stored_bl_dict:
                        st.markdown(
                            "_Baseline ready. Select a date range and press "
                            "\u25b6 **Run Assessment**._"
                        )
                        # Show baseline state validation messages
                        _bm_disp = baseline_from_dict(_stored_bl_dict)
                        if _bm_disp.baseline_state:
                            render_baseline_state(_bm_disp.baseline_state)
                    else:
                        st.markdown(
                            "_Ingest a **baseline** (healthy reference period) first, "
                            "then run an assessment._"
                        )
                else:
                    render_assessment(record)

                    # Control charts
                    _chart_data = st.session_state.get("last_data") or data
                    if _chart_data is not None:
                        st.markdown("---")
                        st.subheader("Charts")
                        for fig in build_assessment_charts(_chart_data, record):
                            st.plotly_chart(fig, use_container_width=True)


# ================================================================== #
# TAB 3 \u2014 History
# ================================================================== #

with tab_history:
    history = db.get_analysis_history(selected_id)
    if not history:
        st.info("No assessment runs yet for this machine.")
    else:
        for rec in history:
            _ts   = str(rec["timestamp"])[:16]
            _atype = rec["analysis_type"]
            _ins  = rec["insights"]
            _summary = _ins.get("summary", "")
            with st.expander(f"{_atype}  \u2014  {_ts}", expanded=False):
                _dr = _ins.get("date_range")
                if _dr:
                    st.caption(f"Date range: {_dr[0]} \u2192 {_dr[1]}")
                if _summary:
                    st.text(_summary)
                else:
                    st.json(_ins)


# ================================================================== #
# TAB 4 \u2014 Maintenance Logs
# ================================================================== #

with tab_logs:
    stored_logs = db.get_logs(selected_id)
    if not stored_logs:
        st.info("No maintenance logs stored yet. Use the expander above to add log entries.")
    else:
        st.caption(f"{len(stored_logs)} log(s) \u2014 included automatically in every assessment.")
        for log in stored_logs:
            c1, c2 = st.columns([5, 1])
            with c1:
                with st.expander(f"{log['filename']}  \u2014  {str(log['uploaded_at'])[:10]}"):
                    st.text(log["content"][:3000] + ("\u2026" if len(log["content"]) > 3000 else ""))
            with c2:
                if st.button("Delete", key=f"tlog_del_{log['filename']}_{log['uploaded_at']}"):
                    db.delete_log(selected_id, log["filename"])
                    st.rerun()
