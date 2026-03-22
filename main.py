"""
main.py — Machine Continuous Monitoring Analytics
Run with:  streamlit run main.py
"""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from analyzer import Analyzer
from log_reader import read_log
from database import Database
from visualizer import Visualizer

load_dotenv()

import plotly.graph_objects as go
import numpy as np

def _build_compliance_chart(data: pd.DataFrame, schedule: dict) -> list:
    """
    Build a compliance chart showing:
    - The best available energy/power/current column as the main trace
    - Green shading for permitted schedule windows
    - Red shading for off-schedule periods
    """
    if data is None or data.empty:
        return []

    # Auto-detect best column to plot (same priority as energy calc)
    kwh_col   = next((c for c in data.columns if any(k in c.lower()
                      for k in ["kwh","kw_h","energy","consumption"])), None)
    power_col = next((c for c in data.columns if any(k in c.lower()
                      for k in ["kw","power"]) and c != kwh_col), None)
    current_col = next((c for c in data.columns if any(k in c.lower()
                        for k in ["current","amp"])), None)
    plot_col  = kwh_col or power_col or current_col
    if not plot_col:
        # Fall back to first numeric column
        numeric = data.select_dtypes(include="number").columns
        plot_col = numeric[0] if len(numeric) else None
    if not plot_col:
        return []

    work_days  = schedule.get("work_days", list(range(5)))
    hour_start = schedule.get("work_hour_start", 8)
    hour_end   = schedule.get("work_hour_end", 18)

    df = data.copy()
    df.index = pd.to_datetime(df.index)

    # Determine permitted mask
    in_schedule = (
        df.index.dayofweek.isin(work_days) &
        (df.index.hour >= hour_start) &
        (df.index.hour < hour_end)
    )

    # Build shading shapes — find contiguous blocks
    def _blocks(mask):
        blocks = []
        in_block = False
        for i, val in enumerate(mask):
            if val and not in_block:
                start = df.index[i]
                in_block = True
            elif not val and in_block:
                blocks.append((start, df.index[i]))
                in_block = False
        if in_block:
            blocks.append((start, df.index[-1]))
        return blocks

    scheduled_blocks   = _blocks(in_schedule)
    offschedule_blocks = _blocks(~in_schedule)

    shapes = []
    # Green bands for scheduled periods
    for s, e in scheduled_blocks[:200]:
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=s, x1=e, y0=0, y1=1,
            fillcolor="rgba(0,200,80,0.22)",
            line=dict(width=0.5, color="rgba(0,160,60,0.3)"),
            layer="below"
        ))
    # Red bands for off-schedule periods
    for s, e in offschedule_blocks[:200]:
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=s, x1=e, y0=0, y1=1,
            fillcolor="rgba(230,40,40,0.18)",
            line=dict(width=0.5, color="rgba(190,20,20,0.3)"),
            layer="below"
        ))

    # Main trace — colour points by schedule status
    colors = ["rgba(0,150,60,0.8)" if v else "rgba(200,40,40,0.8)"
              for v in in_schedule]

    col_label = plot_col.replace("_", " ").title()
    source    = ("kWh" if plot_col == kwh_col
                 else "Power (kW)" if plot_col == power_col
                 else "Motor Current (A)")

    fig = go.Figure()

    # Shaded area under line
    fig.add_trace(go.Scatter(
        x=df.index, y=df[plot_col],
        fill="tozeroy",
        fillcolor="rgba(100,100,200,0.08)",
        line=dict(color="rgba(100,100,200,0.5)", width=1),
        name=col_label,
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>" + col_label + ": %{y:.2f}<extra></extra>"
    ))

    # Legend entries matching actual shading colors
    day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    sched_label = f"Scheduled ({', '.join(day_names[d] for d in work_days)} {hour_start:02d}:00-{hour_end:02d}:00)"
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(
            color="rgba(0,200,80,0.85)",
            size=16,
            symbol="square",
            line=dict(color="rgba(0,150,60,1.0)", width=1.5)
        ),
        name=sched_label,
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(
            color="rgba(230,40,40,0.85)",
            size=16,
            symbol="square",
            line=dict(color="rgba(180,20,20,1.0)", width=1.5)
        ),
        name="Off-schedule",
    ))

    fig.update_layout(
        title=dict(
            text=f"{col_label} — Schedule Compliance ({source})",
            y=0.97,
            x=0,
            xanchor="left",
            font=dict(size=14),
        ),
        xaxis=dict(
            title=dict(text="Time", standoff=30),
        ),
        yaxis_title=col_label,
        shapes=shapes,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=90, b=110),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        font=dict(size=12),
    )

    return [fig]


# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="Machine Analytics",
    page_icon="",
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
        overflow: visible !important;
        white-space: normal !important;
        word-break: break-word !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
    }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# Helper — render a complete insights block
# (defined first so it can be called from anywhere below)
# ------------------------------------------------------------------ #

def render_insights(insights: dict, data: pd.DataFrame, viz: Visualizer,
                    analysis_type: str = ""):
    """Render KPIs, narrative, anomalies, key insights, and Plotly charts."""

    # ML tier badge
    tier       = insights.get("_ml_tier", 0)
    tier_label = insights.get("_ml_tier_label", "")
    if tier_label:
        st.caption(f"Analysis engine — Tier {tier}: {tier_label}")

    # Off-schedule runtime banner (Schedule Compliance only)
    if analysis_type == "Operational Schedule Compliance":
        off_pct = None
        on_pct  = None
        for kpi in insights.get("kpis", []):
            label = kpi.get("label", "").lower()
            val   = kpi.get("value", "")
            if "off" in label or "non-complian" in label:
                try:
                    off_pct = float(str(val).replace("%","").strip().split()[0])
                except Exception:
                    pass
            if "complian" in label and "off" not in label and "non" not in label:
                try:
                    on_pct = float(str(val).replace("%","").strip().split()[0])
                except Exception:
                    pass
        if off_pct is None:
            score_val = insights.get("health_score")
            if score_val is not None:
                on_pct  = score_val
                off_pct = 100 - score_val
        if off_pct is not None:
            on_pct = on_pct if on_pct is not None else round(100 - off_pct, 1)
            c1, c2, c3 = st.columns(3)
            c1.metric("Off-schedule", f"{off_pct:.1f}%")
            c2.metric("Scheduled", f"{on_pct:.1f}%")
            if data is not None and not data.empty:
                # Add one reading interval so end date is fully included
                interval_secs = 900  # default 15 min
                if len(data) > 1:
                    interval_secs = (data.index[1] - data.index[0]).total_seconds()
                total_hours = (
                    (data.index.max() - data.index.min()).total_seconds() + interval_secs
                ) / 3600
                off_hours = total_hours * off_pct / 100
                c3.metric("Off-schedule hours", f"{off_hours:,.1f} hrs")
            compliance_pct = min(100, max(0, on_pct))
            bar_colour = "green" if compliance_pct >= 95 else "orange" if compliance_pct >= 80 else "red"
            st.markdown(f"**Schedule compliance: :{bar_colour}[{compliance_pct:.1f}%]**")
            st.progress(compliance_pct / 100)
            st.markdown("---")
    else:
        score = insights.get("health_score")
        if score is not None:
            colour = "green" if score >= 80 else "orange" if score >= 60 else "red"
            st.markdown(f"### Health score: :{colour}[{score} / 100]")
            st.progress(int(score) / 100)

    kpis = insights.get("kpis", [])
    if analysis_type == "Operational Schedule Compliance":
        # Replace duplicate KPI row with energy saving potential
        if data is not None and not data.empty:
            # Find motor current column
            current_col = next(
                (c for c in data.columns if "current" in c.lower() or "amp" in c.lower()),
                None
            )
            voltage = 415  # standard 3-phase voltage (V) — adjust if needed
            pf      = 0.85  # typical power factor for induction motor

            # Get currency settings from schedule (set in schedule config)
            _schedule     = st.session_state.get("_last_schedule", {})
            currency_sym  = _schedule.get("currency_symbol", "$")
            rate_kwh      = _schedule.get("rate_per_kwh", 0.15)

            # Auto-detect energy source: kWh column > power column > current calculation
            kwh_col   = next(
                (c for c in data.columns if any(k in c.lower() for k in ["kwh", "kw_h", "energy", "consumption"])),
                None
            )
            power_col = next(
                (c for c in data.columns if any(k in c.lower() for k in ["kw", "power"]) and c != kwh_col),
                None
            )

            # Calculate time span
            interval_secs = 900
            if len(data) > 1:
                interval_secs = (data.index[1] - data.index[0]).total_seconds()
            total_hours   = ((data.index.max() - data.index.min()).total_seconds() + interval_secs) / 3600
            off_hours_val = total_hours * (off_pct / 100) if off_pct else 0

            # Format duration string
            total_days  = int(total_hours // 24)
            rem_hours   = int(total_hours % 24)
            if total_days > 0:
                duration_str = f"{total_days}d {rem_hours}h" if rem_hours else f"{total_days} days"
            else:
                duration_str = f"{int(total_hours)}h"

            e1, e2, e3, e4 = st.columns(4)

            calc_note = ""

            if kwh_col:
                # Use cumulative or interval kWh column directly
                total_kwh     = float(data[kwh_col].sum())
                off_kwh       = total_kwh * (off_pct / 100) if off_pct else 0
                avg_power     = total_kwh / total_hours if total_hours > 0 else 0
                cost_saved = off_kwh * rate_kwh
                e1.metric("Off-schedule energy",    f"{off_kwh:,.0f} kWh")
                e2.metric("Total energy (period)",  f"{total_kwh:,.1f} kWh")
                e3.metric("Cost saving potential",
                          f"{currency_sym}{cost_saved:,.0f}")
                calc_note = (
                    f"Scenario: Energy meter data available  \n"
                    f"Method: Direct summation of column `{kwh_col}`. No assumptions required.  \n"
                    f"---  \n"
                    f"Total energy in period ({duration_str}): **{total_kwh:,.1f} kWh**  \n"
                    f"Off-schedule energy = {total_kwh:,.1f} kWh x {off_pct:.1f}% = **{off_kwh:,.0f} kWh**  \n"
                    f"Cost saving = {off_kwh:,.0f} kWh x {currency_sym}{rate_kwh} = **{currency_sym}{cost_saved:,.0f}**"
                )

            elif power_col:
                # Use power (kW) column — integrate over time
                avg_power_kw  = float(data[power_col].mean())
                total_kwh     = avg_power_kw * total_hours
                off_kwh       = total_kwh * (off_pct / 100) if off_pct else 0
                cost_saved = off_kwh * rate_kwh
                e1.metric("Off-schedule energy",    f"{off_kwh:,.0f} kWh")
                e2.metric("Off-schedule hours",     f"{off_hours_val:,.1f} hrs")
                e3.metric("Cost saving potential",
                          f"{currency_sym}{cost_saved:,.0f}")
                calc_note = (
                    f"Scenario: Power (kW) data available, no energy meter  \n"
                    f"Method: Power x time from column `{power_col}`.  \n"
                    f"Assumption: Average power assumed constant. Actual energy may vary if load fluctuates.  \n"
                    f"---  \n"
                    f"Average power (full period): **{avg_power_kw:.1f} kW**  \n"
                    f"Off-schedule hours: **{off_hours_val:.1f} h**  \n"
                    f"Off-schedule energy = {avg_power_kw:.1f} kW x {off_hours_val:.1f} h = **{off_kwh:,.0f} kWh**  \n"
                    f"Cost saving = {off_kwh:,.0f} kWh x {currency_sym}{rate_kwh} = **{currency_sym}{cost_saved:,.0f}**"
                )

            elif current_col and off_pct is not None:
                # Current-based 3-phase calculation
                voltage_col = next(
                    (c for c in data.columns if any(k in c.lower()
                     for k in ["voltage", "volt", "_v", "volts"])),
                    None
                )
                pf_col = next(
                    (c for c in data.columns if any(k in c.lower()
                     for k in ["power_factor", "pf", "cos_phi", "cosphi", "cos phi"])),
                    None
                )
                # Build off-schedule + running mask
                _sched      = st.session_state.get("_last_schedule", {})
                _wdays      = _sched.get("work_days", list(range(5)))
                _hstart     = _sched.get("work_hour_start", 8)
                _hend       = _sched.get("work_hour_end", 18)
                _run_thresh = _sched.get("running_threshold", 0)
                voltage     = _sched.get("voltage", 415.0)
                pf          = _sched.get("power_factor", 0.85)
                _df         = data.copy()
                _df.index   = pd.to_datetime(_df.index)
                in_sched     = (
                    _df.index.dayofweek.isin(_wdays) &
                    (_df.index.hour >= _hstart) &
                    (_df.index.hour < _hend)
                )
                off_sched_mask = ~in_sched
                running_mask   = _df[current_col] > _run_thresh
                off_run_mask   = off_sched_mask & running_mask
                off_run_data   = _df[off_run_mask]
                if len(off_run_data) > 0:
                    avg_current   = float(off_run_data[current_col].mean())
                    off_run_hours = len(off_run_data) * interval_secs / 3600
                    current_src   = f"avg of {len(off_run_data):,} off-schedule running readings"
                else:
                    avg_current   = float(_df[off_sched_mask][current_col].mean()) if off_sched_mask.any() else float(_df[current_col].mean())
                    off_run_hours = off_hours_val
                    current_src   = "all off-schedule readings (running threshold not matched)"
                if voltage_col and pf_col:
                    avg_voltage = float(off_run_data[voltage_col].mean()) if len(off_run_data) > 0 else float(_df[voltage_col].mean())
                    avg_pf      = float(off_run_data[pf_col].mean()) if len(off_run_data) > 0 else float(_df[pf_col].mean())
                    assumptions = "No assumptions — V, I and PF all measured from off-schedule running periods."
                elif voltage_col:
                    avg_voltage = float(off_run_data[voltage_col].mean()) if len(off_run_data) > 0 else float(_df[voltage_col].mean())
                    avg_pf      = pf
                    assumptions = f"Voltage measured ({avg_voltage:.0f} V from off-schedule running). PF assumed = {pf}."
                elif pf_col:
                    avg_voltage = voltage
                    avg_pf      = float(off_run_data[pf_col].mean()) if len(off_run_data) > 0 else float(_df[pf_col].mean())
                    assumptions = f"PF measured ({avg_pf:.2f} from off-schedule running). Voltage assumed = {voltage} V."
                else:
                    avg_voltage = voltage
                    avg_pf      = pf
                    assumptions = f"Both assumed — {voltage} V, PF={pf}. Add voltage and power_factor columns for higher accuracy."
                power_kw   = (1.732 * avg_voltage * avg_current * avg_pf) / 1000
                off_kwh    = power_kw * off_run_hours
                cost_saved = off_kwh * rate_kwh
                e1.metric("Off-schedule energy",   f"{off_kwh:,.0f} kWh")
                e2.metric("Cost saving potential",  f"{currency_sym}{cost_saved:,.0f}")
                e3.metric("Period",                f"{duration_str}")
                # Build scenario label based on what was measured vs assumed
                if voltage_col and pf_col:
                    scenario_label = "Scenario: Current + voltage + power factor all measured"
                elif voltage_col:
                    scenario_label = "Scenario: Current + voltage measured | Power factor entered by user"
                elif pf_col:
                    scenario_label = "Scenario: Current + power factor measured | Voltage entered by user"
                else:
                    scenario_label = "Scenario: Current only | Voltage and power factor entered by user"
                v_source  = f"measured from `{voltage_col}`" if voltage_col else f"user input ({avg_voltage:.0f} V)"
                pf_source = f"measured from `{pf_col}`"     if pf_col      else f"user input ({avg_pf:.2f})"
                calc_note = (
                    f"{scenario_label}  \n"
                    f"Method: 3-phase power formula on off-schedule running readings only.  \n"
                    f"---  \n"
                    f"Off-schedule running readings (current > {_run_thresh} A): **{len(off_run_data):,}** ({current_src})  \n"
                    f"Avg current (off-schedule running): {avg_current:.1f} A — from `{current_col}`  \n"
                    f"Voltage: {avg_voltage:.0f} V — {v_source}  \n"
                    f"Power factor: {avg_pf:.2f} — {pf_source}  \n"
                    f"---  \n"
                    f"Power = 1.732 x {avg_voltage:.0f}V x {avg_current:.1f}A x {avg_pf:.2f} / 1000 = {power_kw:.1f} kW  \n"
                    f"Off-schedule running hours: {off_run_hours:.1f} h  \n"
                    f"Off-schedule energy = {power_kw:.1f} kW x {off_run_hours:.1f} h = **{off_kwh:,.0f} kWh**  \n"
                    f"Cost saving = {off_kwh:,.0f} kWh x {currency_sym}{rate_kwh} = **{currency_sym}{cost_saved:,.0f}**"
                )

            if calc_note:
                with st.expander("Calculation details", expanded=False):
                    st.markdown(
                        f'<div style="font-size:0.78rem;line-height:1.7;color:var(--text-color);">'
                        f'{calc_note.replace(chr(10), "<br>")}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            else:
                # No current column — show hours breakdown instead
                if kpis:
                    cols = st.columns(min(len(kpis), 4))
                    for i, kpi in enumerate(kpis[:4]):
                        status    = kpi.get("status", "normal")
                        delta_map = {"normal": None, "warning": "Warning", "critical": "Critical"}
                        delta_col = "inverse" if status == "critical" else "off" if status == "warning" else "normal"
                        cols[i].metric(kpi.get("label","—"), kpi.get("value","—"),
                                       delta=delta_map.get(status), delta_color=delta_col)
    elif kpis:
        n_cols = min(len(kpis), 4)
        cols   = st.columns(n_cols)
        for i, kpi in enumerate(kpis[:4]):
            status = kpi.get("status", "normal")
            value  = kpi.get("value", "—")
            label  = kpi.get("label", "—")
            status_colour = {"normal": "green", "warning": "orange", "critical": "red"}
            colour = status_colour.get(status, "gray")
            cols[i].metric(label, f":{colour}[{value}]")

    if insights.get("narrative"):
        st.info(insights["narrative"])

    anomalies = insights.get("anomalies", [])
    if anomalies:
        st.subheader("Anomalies detected")
        for a in anomalies:
            st.warning(f"**{a.get('parameter', '—')}** — {a.get('description', '')}")

    key_points = insights.get("insights", [])
    if key_points:
        st.subheader("Key insights")
        for point in key_points:
            st.markdown(f"- {point}")

    recs = insights.get("chart_recommendations", [])
    if data is not None:
        st.subheader("Charts")
        if analysis_type == "Operational Schedule Compliance":
            _schedule    = st.session_state.get("_last_schedule", {})
            _figs = _build_compliance_chart(data, _schedule)
            for fig in _figs:
                st.plotly_chart(fig, use_container_width=True)
        elif recs:
            for fig in viz.generate_charts(data, recs):
                st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------ #
# Session state
# ------------------------------------------------------------------ #

if "last_insights" not in st.session_state:
    st.session_state["last_insights"] = None
if "last_data" not in st.session_state:
    st.session_state["last_data"] = None
if "last_multi_results" not in st.session_state:
    st.session_state["last_multi_results"] = None


# ------------------------------------------------------------------ #
# Services (cached so they survive reruns)
# ------------------------------------------------------------------ #

@st.cache_resource
def get_services():
    return Database(), Visualizer()

db, viz = get_services()


# ================================================================== #
# SIDEBAR
# ================================================================== #

with st.sidebar:
    st.title("Machine Analytics")
    st.caption("Continuous monitoring - Powered by Claude")
    st.divider()

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Paste your key here, or add ANTHROPIC_API_KEY to a .env file.",
        )
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key

    st.divider()

    with st.expander("Register new machine", expanded=not db.get_machines()):
        machine_type = st.text_input(
            "Machine type", placeholder="e.g. Compressor, Pump, Furnace"
        )
        machine_id = st.text_input("Machine ID", placeholder="e.g. COMP-001")
        machine_desc = st.text_area(
            "Specs / notes",
            placeholder="Rated power: 75 kW\nFlow: 500 m³/h\nRPM: 1480",
            height=90,
        )

        # Optional thresholds
        with st.expander("Parameter thresholds (optional)", expanded=False):
            st.caption("Define warning and critical limits per parameter. Leave blank to let Claude decide automatically.")
            thresh_text = st.text_area(
                "Thresholds",
                placeholder=(
                    "vibration_mm_s: warning=2.8, critical=4.5\n"
                    "discharge_temp_C: warning=170, critical=185\n"
                    "motor_current_A: warning=46, critical=50\n"
                    "suction_pressure_bar: warning=0.93, critical=0.90"
                ),
                height=130,
                help="One parameter per line. Format: param_name: warning=X, critical=Y",
            )

        if st.button("Register", type="primary", use_container_width=True):
            if machine_type and machine_id:
                # Append thresholds to description if provided
                full_desc = machine_desc
                if thresh_text and thresh_text.strip():
                    full_desc = machine_desc + "\n\n=== PARAMETER THRESHOLDS ===\n" + thresh_text.strip()
                db.register_machine(machine_id.strip(), machine_type.strip(), full_desc)
                st.success(f"Machine **{machine_id}** registered.")
                st.rerun()
            else:
                st.error("Machine type and ID are required.")

    st.divider()

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
    )

    st.divider()

    st.subheader("Upload data")
    uploaded_file = st.file_uploader(
        "CSV or Excel",
        type=["csv", "xlsx", "xls"],
        help="Any column named timestamp/date/time is auto-detected as the time axis.",
    )
    if uploaded_file:
        if st.button("Ingest", use_container_width=True):
            with st.spinner("Reading and storing data…"):
                result = db.ingest_file(uploaded_file, selected_id)
            if result["success"]:
                st.success(f"✓ {result['rows']:,} rows ingested")
                st.caption("Columns: " + ", ".join(result["columns"]))
                st.rerun()
            else:
                st.error(result["error"])

    file_info = db.get_file_info(selected_id)
    if file_info:
        st.caption(f"{len(file_info)} file(s) stored for this machine")

    st.divider()
    st.subheader("Maintenance logs")
    log_file = st.file_uploader(
        "Upload log (CSV, PDF, image)",
        type=["csv", "xlsx", "pdf", "png", "jpg", "jpeg", "webp", "txt"],
        key="log_uploader",
        help="Handwritten or printed maintenance logs, service records, inspection reports.",
    )
    if log_file:
        if st.button("Read & store log", use_container_width=True):
            with st.spinner("Reading log file…"):
                result = read_log(log_file, api_key=os.getenv("ANTHROPIC_API_KEY", ""))
            if result["success"]:
                db.save_log(selected_id, log_file.name, result["method"], result["text"])
                st.success(f"Log stored — {result['method']} ({len(result['text'])} chars)")
                st.rerun()
            else:
                st.error(result["error"])

    stored_logs = db.get_logs(selected_id)
    if stored_logs:
        st.caption(f"{len(stored_logs)} log file(s) stored")


# ================================================================== #
# MAIN AREA
# ================================================================== #

machine_info = db.get_machine_info(selected_id)
st.title(f"{machine_info['machine_type']}  ·  {selected_id}")
if machine_info.get("description"):
    st.caption(machine_info["description"])

# Inline threshold editor
with st.expander("Edit parameter thresholds", expanded=False):
    st.caption("Define warning/critical limits. Leave blank for fully automatic (unsupervised) detection.")
    desc = machine_info.get("description", "")
    # Extract existing thresholds if present
    existing_thresh = ""
    if "=== PARAMETER THRESHOLDS ===" in desc:
        existing_thresh = desc.split("=== PARAMETER THRESHOLDS ===")[-1].strip()
    new_thresh = st.text_area(
        "Thresholds",
        value=existing_thresh,
        placeholder=(
            "vibration_mm_s: warning=2.8, critical=4.5\n"
            "discharge_temp_C: warning=170, critical=185\n"
            "motor_current_A: warning=46, critical=50"
        ),
        height=120,
        label_visibility="collapsed",
    )
    if st.button("Save thresholds", use_container_width=True):
        base_desc = desc.split("=== PARAMETER THRESHOLDS ===")[0].strip()
        if new_thresh.strip():
            updated_desc = base_desc + "\n\n=== PARAMETER THRESHOLDS ===\n" + new_thresh.strip()
        else:
            updated_desc = base_desc
        db.register_machine(selected_id, machine_info["machine_type"], updated_desc)
        st.success("Thresholds saved.")
        st.rerun()

data = db.get_data(selected_id)

tab_data, tab_analysis, tab_history, tab_logs = st.tabs(["Data", " Analysis", "History", "Maintenance Logs"])


# ------------------------------------------------------------------ #
# TAB 1 — Data preview
# ------------------------------------------------------------------ #

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

        st.subheader("Recent readings")
        st.dataframe(data.tail(200), use_container_width=True, height=280)

        st.subheader("Descriptive statistics")
        st.dataframe(data[numeric_cols].describe().round(3), use_container_width=True)

        if numeric_cols:
            st.subheader("Sensor overview")
            figs = viz.generate_charts(
                data,
                [{"type": "time_series",
                  "title": "All parameters",
                  "parameters": numeric_cols[:8]}],
            )
            st.plotly_chart(figs[0], use_container_width=True)


# ------------------------------------------------------------------ #
# TAB 2 — Analysis
# ------------------------------------------------------------------ #

with tab_analysis:
    if data is None or data.empty:
        st.info("Upload data first (sidebar), then run an analysis.")
    else:
        left, right = st.columns([1, 3])

        with left:
            st.markdown("**Select analyses to run**")
            select_all = st.checkbox("Select All", value=False, key="select_all")

            analysis_options = [
                ("Overall Health Assessment",       "Comprehensive health score and parameter review"),
                ("Trend & Drift Analysis",           "Gradual changes and degradation over time"),
                ("Anomaly Detection",                "Outliers, spikes, and out-of-range values"),
                ("Correlation Analysis",             "Relationships between parameters"),
                ("Parameter Distribution",           "Statistical spread and distribution shape"),
                ("Cross-Parameter Comparison",       "Imbalances between related parameters"),
                ("Operational Schedule Compliance",  "Running outside permitted hours or days"),
            ]

            selected_analyses = []
            for name, desc in analysis_options:
                checked = st.checkbox(
                    f"**{name}**",
                    value=select_all,
                    key=f"chk_{name}",
                    help=desc,
                )
                if checked:
                    selected_analyses.append(name)

            # Keep backward compat — use first selected for schedule UI trigger
            analysis_type = selected_analyses[0] if selected_analyses else ""

            min_d, max_d = data.index.min().date(), data.index.max().date()
            if min_d < max_d:
                date_range = st.date_input(
                    "Date range",
                    value=[min_d, max_d],
                    min_value=min_d,
                    max_value=max_d,
                )
                if not isinstance(date_range, (list, tuple)) or len(date_range) < 2:
                    date_range = (min_d, max_d)
            else:
                date_range = (min_d, max_d)

            # Schedule config — only shown when compliance is selected
            schedule = None
            if "Operational Schedule Compliance" in selected_analyses:
                with st.expander("Schedule configuration", expanded=True):
                    day_options = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
                    selected_days = st.multiselect(
                        "Permitted working days",
                        options=day_options,
                        default=["Mon","Tue","Wed","Thu","Fri"],
                    )
                    col_a, col_b = st.columns(2)
                    hour_start = col_a.number_input("Start hour (24h)", 0, 23, 8)
                    hour_end   = col_b.number_input("End hour (24h)",   0, 23, 18)
                    numeric_cols_list = data.select_dtypes(include="number").columns.tolist()
                    indicator_col = st.selectbox(
                        "Running indicator column",
                        options=[""] + numeric_cols_list,
                        help="Column whose value above threshold = machine is running. Leave blank to auto-detect.",
                    )
                    run_threshold = st.number_input(
                        "Running threshold (value above = running)", value=0.0
                    )
                    st.markdown("**Energy cost settings**")
                    CURRENCIES = [
                        "$ — USD (US Dollar)",
                        "€ — EUR (Euro)",
                        "£ — GBP (British Pound)",
                        "¥ — JPY (Japanese Yen)",
                        "¥ — CNY (Chinese Yuan)",
                        "₹ — INR (Indian Rupee)",
                        "CHF — CHF (Swiss Franc)",
                        "A$ — AUD (Australian Dollar)",
                        "C$ — CAD (Canadian Dollar)",
                        "S$ — SGD (Singapore Dollar)",
                        "HK$ — HKD (Hong Kong Dollar)",
                        "kr — SEK (Swedish Krona)",
                        "kr — NOK (Norwegian Krone)",
                        "kr — DKK (Danish Krone)",
                        "₩ — KRW (South Korean Won)",
                        "R — ZAR (South African Rand)",
                        "R$ — BRL (Brazilian Real)",
                        "₺ — TRY (Turkish Lira)",
                        "₽ — RUB (Russian Ruble)",
                        "﷼ — SAR (Saudi Riyal)",
                        "د.إ — AED (UAE Dirham)",
                        "₪ — ILS (Israeli Shekel)",
                        "Mex$ — MXN (Mexican Peso)",
                        "Rp — IDR (Indonesian Rupiah)",
                        "₱ — PHP (Philippine Peso)",
                        "฿ — THB (Thai Baht)",
                        "zł — PLN (Polish Zloty)",
                        "Kč — CZK (Czech Koruna)",
                        "Ft — HUF (Hungarian Forint)",
                        "RM — MYR (Malaysian Ringgit)",
                        "NZ$ — NZD (New Zealand Dollar)",
                        "CLP — CLP (Chilean Peso)",
                        "Col$ — COP (Colombian Peso)",
                        "S/. — PEN (Peruvian Sol)",
                        "Ksh — KES (Kenyan Shilling)",
                        "₦ — NGN (Nigerian Naira)",
                        "EGP — EGP (Egyptian Pound)",
                        "PKR — PKR (Pakistani Rupee)",
                        "৳ — BDT (Bangladeshi Taka)",
                        "Other — (type below)",
                    ]
                    selected_currency = st.selectbox(
                        "Currency",
                        options=CURRENCIES,
                        index=0,
                        help="Select your local currency"
                    )
                    if selected_currency.startswith("Other"):
                        currency_symbol = st.text_input(
                            "Enter currency symbol",
                            value="$",
                            help="Type your currency symbol"
                        )
                    else:
                        currency_symbol = selected_currency.split(" — ")[0].strip()

                    rate_per_kwh = st.number_input(
                        "Rate per kWh",
                        min_value=0.0,
                        value=0.15,
                        step=0.01,
                        format="%.4f",
                        help="Your electricity tariff per kWh"
                    )

                    # Detect what energy-related columns exist in the loaded data
                    _data_now = db.get_data(selected_id)
                    _cols     = list(_data_now.columns) if _data_now is not None else []

                    _has_kwh     = any(any(k in c.lower() for k in ["kwh","kw_h","energy","consumption"]) for c in _cols)
                    _has_power   = any(any(k in c.lower() for k in ["kw","power"]) for c in _cols) and not _has_kwh
                    _has_current = any(any(k in c.lower() for k in ["current","amp"]) for c in _cols)
                    _has_voltage = any(any(k in c.lower() for k in ["voltage","volt","_v","volts"]) for c in _cols)
                    _has_pf      = any(any(k in c.lower() for k in ["power_factor","pf","cos_phi","cosphi"]) for c in _cols)

                    user_voltage = 415.0
                    user_pf      = 0.85

                    if _has_kwh:
                        st.success("Energy column detected — no electrical parameters needed.")
                    elif _has_power:
                        st.success("Power (kW) column detected — no electrical parameters needed.")
                    elif _has_current:
                        st.markdown("**Electrical parameters**")
                        if _has_voltage and _has_pf:
                            st.success("Voltage and power factor columns detected — no manual input needed.")
                        elif _has_voltage:
                            st.info("Voltage column detected. Power factor not found in data — please enter below.")
                            user_pf = st.number_input(
                                "Power factor",
                                min_value=0.1, max_value=1.0, value=0.85, step=0.01, format="%.2f",
                                help="Motor power factor (0–1). Typical induction motor: 0.80–0.90."
                            )
                        elif _has_pf:
                            st.info("Power factor column detected. Voltage not found in data — please enter below.")
                            user_voltage = st.number_input(
                                "Supply voltage (V)",
                                min_value=1.0, value=415.0, step=1.0, format="%.0f",
                                help="Line-to-line voltage e.g. 415V (India/EU), 400V (EU), 480V (US)."
                            )
                        else:
                            st.warning("Neither voltage nor power factor found in data. Please enter both below.")
                            user_voltage = st.number_input(
                                "Supply voltage (V)",
                                min_value=1.0, value=415.0, step=1.0, format="%.0f",
                                help="Line-to-line voltage e.g. 415V (India/EU), 400V (EU), 480V (US)."
                            )
                            user_pf = st.number_input(
                                "Power factor",
                                min_value=0.1, max_value=1.0, value=0.85, step=0.01, format="%.2f",
                                help="Motor power factor (0–1). Typical induction motor: 0.80–0.90."
                            )
                    else:
                        st.warning("No energy, power, or current column detected in data. Energy calculation will not be available.")

                    day_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
                    schedule = {
                        "work_days":         [day_map[d] for d in selected_days],
                        "work_hour_start":   int(hour_start),
                        "work_hour_end":     int(hour_end),
                        "indicator_col":     indicator_col,
                        "running_threshold": float(run_threshold),
                        "currency_symbol":   currency_symbol,
                        "rate_per_kwh":      float(rate_per_kwh),
                        "voltage":           float(user_voltage),
                        "power_factor":      float(user_pf),
                    }

            extra_context = st.text_area(
                "Engineer notes (optional)",
                placeholder="e.g. Bearing replaced on 2024-03-01. Unusual vibration noticed last week.",
                height=100,
            )

            has_key    = bool(os.getenv("ANTHROPIC_API_KEY"))
            no_selection = len(selected_analyses) == 0
            analyze_clicked = st.button(
                "Analyze",
                type="primary",
                use_container_width=True,
                disabled=not has_key or no_selection,
            )
            if not has_key:
                st.caption("Add your API key in the sidebar first.")
            if no_selection and has_key:
                st.caption("Select at least one analysis type above.")

        with right:
            if analyze_clicked:
                logs_text    = db.get_logs_text(selected_id)
                analyzer_obj = Analyzer()
                all_results  = {}
                progress_bar = st.progress(0)
                status_text  = st.empty()

                # Filter data by date range for display consistency
                from analyzer import Analyzer as _A
                filtered_data = _A._filter_by_date(data, date_range)

                # Store schedule for use in energy display
                st.session_state["_last_schedule"] = schedule or {}

                for i, atype in enumerate(selected_analyses):
                    status_text.text(f"Running {atype} ({i+1} of {len(selected_analyses)})...")
                    result = analyzer_obj.analyze(
                        machine_info=machine_info,
                        data=data,
                        analysis_type=atype,
                        date_range=date_range,
                        extra_context=extra_context,
                        schedule=schedule if atype == "Operational Schedule Compliance" else None,
                        logs_text=logs_text,
                    )
                    if result["success"]:
                        all_results[atype] = result["insights"]
                        db.save_analysis(selected_id, atype, result["insights"])
                    else:
                        all_results[atype] = {"error": result["error"]}
                    progress_bar.progress((i + 1) / len(selected_analyses))

                status_text.empty()
                progress_bar.empty()
                st.session_state["last_multi_results"] = all_results
                st.session_state["last_data"]          = filtered_data
                st.rerun()

            # Display results — one section per analysis type
            multi_results = st.session_state.get("last_multi_results")
            if multi_results:
                for atype, insights in multi_results.items():
                    st.markdown("---")
                    st.subheader(atype)
                    if "error" in insights:
                        st.error(f"Failed: {insights['error']}")
                    else:
                        render_insights(insights, st.session_state.get("last_data"), viz,
                                        analysis_type=atype)
            else:
                st.markdown(
                    "_Select one or more analysis types and press **Analyze** to generate insights._"
                )


# ------------------------------------------------------------------ #
# TAB 3 — History
# ------------------------------------------------------------------ #

with tab_history:
    history = db.get_analysis_history(selected_id)
    if not history:
        st.info("No analysis runs yet for this machine.")
    else:
        for record in history:
            label = f"{record['analysis_type']}  —  {record['timestamp']}"
            with st.expander(label):
                ins = record["insights"]
                score = ins.get("health_score")
                if score is not None:
                    colour = "green" if score >= 80 else "orange" if score >= 60 else "red"
                    st.markdown(f"**Health score:** :{colour}[{score} / 100]")
                if ins.get("narrative"):
                    st.info(ins["narrative"])
                for point in ins.get("insights", []):
                    st.markdown(f"- {point}")


# ------------------------------------------------------------------ #
# TAB 4 — Maintenance Logs
# ------------------------------------------------------------------ #

with tab_logs:
    stored_logs = db.get_logs(selected_id)
    if not stored_logs:
        st.info("No maintenance logs uploaded yet. Use the sidebar to upload a log file.")
    else:
        st.caption(f"{len(stored_logs)} log file(s) — included automatically in every analysis")
        for log in stored_logs:
            col1, col2 = st.columns([5, 1])
            with col1:
                with st.expander(f"{log['filename']}  —  {log['uploaded_at']}  [{log['file_type']}]"):
                    st.text(log["content"][:3000] + ("..." if len(log["content"]) > 3000 else ""))
            with col2:
                if st.button("Delete", key=f"del_{log['filename']}_{log['uploaded_at']}"):
                    db.delete_log(selected_id, log["filename"])
                    st.rerun()
