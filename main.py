"""
main.py — Machine Continuous Monitoring Analytics
Run with:  streamlit run main.py
"""

import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from analyzer import Analyzer
from log_reader import read_log
try:
    from report_generator import generate_report
    REPORT_AVAILABLE = True
except ImportError:
    REPORT_AVAILABLE = False
try:
    from data_checker import run_data_quality_checks, format_quality_report_for_claude
    DQ_AVAILABLE = True
except ImportError:
    DQ_AVAILABLE = False
    def run_data_quality_checks(df, **kw): return {"issues":[],"summary":{"total":0,"critical":0,"warning":0,"info":0},"passed":True,"score":100}
    def format_quality_report_for_claude(r): return ""
try:
    from data_checker import check_timestamp_format
    TS_CHECK_AVAILABLE = True
except (ImportError, Exception):
    TS_CHECK_AVAILABLE = False
    def check_timestamp_format(path): return {"parseable":True,"col":None,"sample_raw":[],"sample_parsed":[],"n_failed":0,"n_total":0,"suggestion":"","corrected_df":None}
try:
    from data_corrector import fix_duplicate_timestamps, fix_missing_gaps, \
        fix_isolated_spikes, fix_physical_impossibles, fix_flatline, CORRECTION_SUGGESTIONS
    CORRECTOR_AVAILABLE = True
except Exception:
    CORRECTOR_AVAILABLE = False
    CORRECTION_SUGGESTIONS = {}
    def fix_duplicate_timestamps(df): return {"corrected_df":df,"changes":0,"description":"","method":""}
    def fix_missing_gaps(df,**kw): return {"corrected_df":df,"changes":0,"description":"","method":""}
    def fix_isolated_spikes(df,col,**kw): return {"corrected_df":df,"changes":0,"description":"","method":""}
    def fix_physical_impossibles(df,col,**kw): return {"corrected_df":df,"changes":0,"description":"","method":""}
    def fix_flatline(df,col,**kw): return {"corrected_df":df,"changes":0,"description":"","method":""}
try:
    from pump_physics import detect_phase, check_phase4_viability, run_all_phases, parse_nameplate
    PUMP_PHYSICS_AVAILABLE = True
except Exception:
    PUMP_PHYSICS_AVAILABLE = False
    def detect_phase(df): return {"phases":[],"highest_phase":0,"cols":{},"has_power":False,"has_flow":False,"has_pressure":False,"has_temp":False,"has_vibr":False,"has_speed":False,"has_3v":False,"has_3i":False}
    def check_phase4_viability(df,cols,**kw): return {"viable":False,"level":"not_viable","message":"","recommendation":""}
    def run_all_phases(df,desc,**kw): return {"phases":{},"all_findings":[],"summary":"","phase_info":{},"np_data":{}}
    def parse_nameplate(desc): return {}
try:
    from pump_curve_finder import (
        get_manufacturer_info, MANUFACTURER_URLS,
        search_and_extract_curves, validate_hq_points,
        format_source_reference_text
    )
    CURVE_FINDER_AVAILABLE = True
except Exception:
    CURVE_FINDER_AVAILABLE = False
    def get_manufacturer_info(m): return None
    MANUFACTURER_URLS = {}
    def search_and_extract_curves(*a,**k): return {"success":False,"method":"not_found","message":"Module unavailable","hq_points":[],"eta_points":[],"power_points":[],"source_ref":{},"needs_review":True,"raw_response":""}
    def validate_hq_points(*a,**k): return {"valid":False,"warnings":["Module unavailable"]}
    def format_source_reference_text(r): return ""
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

    # Assign score early so it's available for breakdown check
    score = insights.get("health_score")

    # Score breakdown (Overall Health Assessment only)
    if analysis_type not in ("Operational Schedule Compliance",) and score is not None:
        breakdown = insights.get("score_breakdown", [])
        if breakdown:
            with st.expander("Score explanation — what drove this score", expanded=True):
                impact_icon  = {"positive": "↑", "negative": "↓", "neutral": "—"}
                impact_color = {"positive": "green", "negative": "red", "neutral": "gray"}
                weight_style = {"high": "font-weight:600", "medium": "", "low": "color:#888"}

                rows_html = ""
                for f in breakdown:
                    impact  = f.get("impact", "neutral")
                    weight  = f.get("weight", "low")
                    icon    = impact_icon.get(impact, "—")
                    color   = impact_color.get(impact, "gray")
                    wstyle  = weight_style.get(weight, "")
                    factor  = f.get("factor", "")
                    detail  = f.get("detail", "")
                    rows_html += (
                        f'<tr>'
                        f'<td style="padding:6px 10px;color:{color};font-size:1.1em;text-align:center">{icon}</td>'
                        f'<td style="padding:6px 10px;{wstyle};font-size:0.88em">{factor}</td>'
                        f'<td style="padding:6px 10px;font-size:0.82em;color:#555">{detail}</td>'
                        f'<td style="padding:6px 10px;font-size:0.78em;color:#888;text-align:center">{weight.title()}</td>'
                        f'</tr>'
                    )

                st.markdown(
                    f'<table style="width:100%;border-collapse:collapse;border:1px solid #dde">'
                    f'<thead><tr style="background:#f0f4f8">'
                    f'<th style="padding:6px 10px;font-size:0.82em;width:40px"></th>'
                    f'<th style="padding:6px 10px;font-size:0.82em;text-align:left">Factor</th>'
                    f'<th style="padding:6px 10px;font-size:0.82em;text-align:left">Signal detail</th>'
                    f'<th style="padding:6px 10px;font-size:0.82em;width:80px">Weight</th>'
                    f'</tr></thead>'
                    f'<tbody>{rows_html}</tbody>'
                    f'</table>',
                    unsafe_allow_html=True
                )

    # Off-schedule runtime banner (Schedule Compliance only)
    if analysis_type not in ("Operational Schedule Compliance",) and False:
        pass
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
            _schedule = st.session_state.get("_last_schedule", {})
            for fig in _build_compliance_chart(data, _schedule):
                st.plotly_chart(fig, use_container_width=True)

        elif analysis_type == "Anomaly Detection":
            try:
                import plotly.graph_objects as _go
                import pandas as _pd
                from analyzer import _parse_thresholds
                _desc       = st.session_state.get("_machine_desc", "")
                _thresholds = _parse_thresholds(_desc) or {}
                _chart_data = data if (data is not None and not data.empty) else st.session_state.get("last_data")
                if _chart_data is not None and not _chart_data.empty:
                    # Ensure datetime index
                    _cd = _chart_data.copy()
                    if not isinstance(_cd.index, _pd.DatetimeIndex):
                        _cd.index = _pd.to_datetime(_cd.index)
                    _numeric = _cd.select_dtypes(include="number").columns.tolist()[:6]
                    st.caption("UCL/LCL = mean ±3σ (red).  UWL/LWL = mean ±2σ (amber dashed).  Red circles = |Z|>3 anomalies.  Purple dotted = engineering thresholds.")
                    for _col in _numeric:
                        _s  = _cd[_col].dropna()
                        _mu = float(_s.mean()); _sd = float(_s.std()) or 1e-9
                        _ucl = _mu+3*_sd; _uwl = _mu+2*_sd
                        _lwl = _mu-2*_sd; _lcl = _mu-3*_sd
                        _z   = (_s-_mu)/_sd; _anom = _s[_z.abs()>3]
                        _xi  = _cd.index  # DatetimeIndex
                        _fig = _go.Figure()
                        # Shaded zones using hrect (y-only, no x needed)
                        _fig.add_hrect(y0=_uwl, y1=_ucl, fillcolor="rgba(192,57,43,0.07)", line_width=0, layer="below")
                        _fig.add_hrect(y0=_lcl, y1=_lwl, fillcolor="rgba(192,57,43,0.07)", line_width=0, layer="below")
                        _fig.add_hrect(y0=_lwl, y1=_uwl, fillcolor="rgba(230,126,34,0.07)", line_width=0, layer="below")
                        # Limit lines as full-width scatter traces (same x type as data)
                        for _yv,_lc,_ld,_lw,_ln in [
                            (_ucl,"#C0392B","solid",1.5,f"UCL {_ucl:.3f} (mean+3σ)"),
                            (_uwl,"#E67E22","dash", 1.0,f"UWL {_uwl:.3f} (mean+2σ)"),
                            (_mu, "#2C3E50","solid",1.5,f"Mean {_mu:.3f}"),
                            (_lwl,"#E67E22","dash", 1.0,f"LWL {_lwl:.3f} (mean-2σ)"),
                            (_lcl,"#C0392B","solid",1.5,f"LCL {_lcl:.3f} (mean-3σ)"),
                        ]:
                            _fig.add_trace(_go.Scatter(
                                x=[_xi[0], _xi[-1]], y=[_yv, _yv],
                                mode="lines", line=dict(color=_lc, width=_lw, dash=_ld),
                                name=_ln, showlegend=True,
                                hoverinfo="skip",
                            ))
                        # Engineering thresholds
                        if _col in _thresholds:
                            for _tk,_tc in [("warning","#8E44AD"),("critical","#641E16")]:
                                _tv = _thresholds[_col].get(_tk)
                                if _tv is not None:
                                    _fig.add_trace(_go.Scatter(
                                        x=[_xi[0], _xi[-1]], y=[_tv, _tv],
                                        mode="lines", line=dict(color=_tc, width=1.5, dash="dot"),
                                        name=f"{_tk.title()} threshold {_tv}", showlegend=True,
                                        hoverinfo="skip",
                                    ))
                        # Data line
                        _fig.add_trace(_go.Scatter(
                            x=_xi, y=_cd[_col].values,
                            mode="lines", line=dict(color="#185FA5", width=1.2),
                            name=_col,
                            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>"+_col+": %{y:.3f}<extra></extra>",
                        ))
                        # Anomaly markers
                        if len(_anom):
                            _fig.add_trace(_go.Scatter(
                                x=_anom.index, y=_anom.values, mode="markers",
                                marker=dict(color="#C0392B", size=7, symbol="circle-open",
                                            line=dict(width=2, color="#C0392B")),
                                name=f"Anomalies (|Z|>3) n={len(_anom)}",
                            ))
                        _cl = _col.replace("_"," ").title()
                        _fig.update_layout(
                            title=dict(text=f"Control chart — {_cl}", font=dict(size=14)),
                            xaxis_title="Time", yaxis_title=_cl,
                            legend=dict(orientation="h", yanchor="top", y=-0.22,
                                        xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", borderwidth=0),
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=40, r=20, t=55, b=110),
                            hovermode="x unified", font=dict(size=12),
                        )
                        st.plotly_chart(_fig, use_container_width=True)
                else:
                    st.info("No data available for control charts.")
            except Exception as _e:
                import traceback
                st.error(f"Control chart error: {_e}\n{traceback.format_exc()}")

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


# ── Machine type registry ─────────────────────────────────────────────────────
MACHINE_TYPES = {
    "Rotating Machinery": [
        "Centrifugal Pump",
        "Axial / Reciprocating Compressor",
        "Centrifugal Compressor",
        "Axial Flow Fan / Blower",
        "Centrifugal Fan / Blower",
        "Electric Motor (standalone)",
        "Gearbox",
        "Turbine",
    ],
    "Static Equipment": [
        "Shell & Tube Heat Exchanger",
        "Plate Heat Exchanger",
        "Boiler / Steam Generator",
        "Cooling Tower",
        "Pressure Vessel",
    ],
    "Process Equipment": [
        "Extruder",
        "Conveyor",
        "Mixer / Agitator",
        "Centrifuge",
        "Hydraulic Power Unit",
    ],
    "Other": [
        "Other (specify in description)",
    ],
}

PHYSICS_MODULE_STATUS = {
    "Centrifugal Pump":                     ("available",    "Phases 1\u20135",  "🔧"),
    "Axial / Reciprocating Compressor":     ("planned",      "In development",    "🚧"),
    "Centrifugal Compressor":               ("planned",      "In development",    "🚧"),
    "Axial Flow Fan / Blower":              ("planned",      "In development",    "🚧"),
    "Centrifugal Fan / Blower":             ("planned",      "In development",    "🚧"),
    "Electric Motor (standalone)":          ("planned",      "In development",    "🚧"),
    "Gearbox":                              ("planned",      "In development",    "🚧"),
    "Shell & Tube Heat Exchanger":          ("future",       "Planned",           "📋"),
    "Plate Heat Exchanger":                 ("future",       "Planned",           "📋"),
    "Boiler / Steam Generator":             ("future",       "Planned",           "📋"),
}

DRIVE_TYPES = [
    # ── Induction motor drives (currently supported) ──────────────────
    "Induction motor — direct / rigid coupling",
    "Induction motor — direct / flexible coupling",
    "Induction motor — VFD (Variable Frequency Drive)",
    "Induction motor — belt drive (V-belt / flat belt)",
    "Induction motor — gearbox speed reducer",
    "Induction motor — gearbox speed increaser",
    # ── Other motor types (physics modules not yet available) ─────────
    "Permanent magnet motor — direct coupling  (no physics module yet)",
    "Synchronous reluctance motor — direct coupling  (no physics module yet)",
    "Wound rotor induction motor — direct coupling  (no physics module yet)",
    "DC motor — direct coupling  (no physics module yet)",
    # ── Non-motor drives ──────────────────────────────────────────────
    "Steam turbine drive  (no physics module yet)",
    "Hydraulic coupling",
    "Not applicable — static equipment",
    "Unknown",
]

# Drive types with full induction motor physics support
DRIVE_TYPES_SUPPORTED = [
    "Induction motor — direct / rigid coupling",
    "Induction motor — direct / flexible coupling",
    "Induction motor — VFD (Variable Frequency Drive)",
    "Induction motor — belt drive (V-belt / flat belt)",
    "Induction motor — gearbox speed reducer",
    "Induction motor — gearbox speed increaser",
]


@st.cache_resource
def get_services(_version=6):
    return Database(), Visualizer()

db, viz = get_services()


# ================================================================== #
# SIDEBAR
# ================================================================== #

def _extract_spec_text(uploaded_file, api_key: str) -> str:
    """
    Extract specification text from an uploaded PDF, image, or text file.
    Uses Claude vision for images/PDFs, plain read for text files.
    """
    import base64, io as _io2
    fname = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    # Plain text files
    if fname.endswith(".txt") or fname.endswith(".csv"):
        try:
            return file_bytes.decode("utf-8", errors="replace")
        except Exception:
            return file_bytes.decode("latin-1", errors="replace")

    # PDF or image — use Claude vision
    if not api_key:
        return "[No API key available — cannot extract text from file. Enter specifications manually.]"

    import anthropic as _ant
    _client = _ant.Anthropic(api_key=api_key)

    b64 = base64.standard_b64encode(file_bytes).decode("utf-8")
    if fname.endswith(".pdf"):
        media_type = "application/pdf"
        source_type = "base64"
        content_block = {
            "type": "document",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        }
    else:
        ext_map = {".png":"image/png", ".jpg":"image/jpeg",
                   ".jpeg":"image/jpeg", ".webp":"image/webp"}
        ext = "." + fname.rsplit(".",1)[-1]
        media_type = ext_map.get(ext, "image/jpeg")
        content_block = {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        }

    response = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                content_block,
                {
                    "type": "text",
                    "text": (
                        "This is a machine nameplate, datasheet, or specification document. "
                        "Extract all technical specifications, nameplate data, and relevant "
                        "parameters as structured key: value pairs, one per line. "
                        "Include: rated power, voltage, current, speed, efficiency, "
                        "flow, head, pressure, temperature, model number, serial number, "
                        "and any other technical parameters visible. "
                        "Format: Parameter name: value unit\n"
                        "Do not include page headers, footers, or marketing text. "
                        "Only technical data."
                    )
                }
            ]
        }]
    )
    return response.content[0].text if response.content else ""


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

        # ── Machine ID ────────────────────────────────────────────────
        machine_id = st.text_input(
            "Machine ID",
            placeholder="e.g. PUMP-A3, COMP-001",
            help="Your site reference tag for this machine."
        )

        # ── Machine type selector ───────────────────────────────────────
        st.markdown("**Machine type**")
        _reg_cat = st.selectbox(
            "Category",
            options=list(MACHINE_TYPES.keys()),
            key="reg_category",
            label_visibility="collapsed",
        )
        _type_options = ["-- Select machine type --"] + MACHINE_TYPES.get(_reg_cat, [])
        machine_type_sel = st.selectbox(
            "Type",
            options=_type_options,
            key="reg_machine_type",
            label_visibility="collapsed",
        )
        machine_type = machine_type_sel if machine_type_sel != "-- Select machine type --" else ""

        # Initialise drive_type early so physics badge can reference it
        # (drive selectbox is rendered below — this reads current session value)
        drive_type = st.session_state.get("reg_drive_type", "-- Select drive type --")
        if drive_type == "-- Select drive type --":
            drive_type = ""

        # Physics module status badge
        if machine_type:
            _phys = PHYSICS_MODULE_STATUS.get(machine_type)
            if _phys:
                _status, _detail, _icon = _phys
                if _status == "available":
                    # Show full availability only if drive type is also supported or not yet selected
                    if not drive_type or drive_type == "-- Select drive type --":
                        st.success(
                            f"{_icon} **Physics module available:** {machine_type} \u2014 {_detail}.  \n"
                            f"Select a drive type below to confirm full physics support."
                        )
                    elif drive_type in DRIVE_TYPES_SUPPORTED:
                        st.success(
                            f"{_icon} **Physics module available:** {machine_type} \u2014 {_detail}.  \n"
                            f"Induction motor drive confirmed. All applicable phases will run."
                        )
                    else:
                        st.warning(
                            f"{_icon} **Physics module available for {machine_type}**, but:  \n"
                            f"\u26a0\ufe0f Selected drive type is not yet supported for physics analysis.  \n"
                            f"AI statistical analytics will run. Switch to an induction motor drive to enable physics phases."
                        )
                elif _status == "planned":
                    st.info(
                        f"{_icon} **Physics module in development:** {machine_type} \u2014 {_detail}.  \n"
                        f"AI statistical analytics (Layer 1) will run. Physics module coming in a future update."
                    )
                else:
                    st.warning(f"{_icon} **Physics module planned:** {machine_type}. AI statistical analytics will run.")
            else:
                st.info("\u2139\ufe0f AI statistical analytics will run. No physics module currently planned for this type.")

        # ── Drive type ──────────────────────────────────────────────────
        st.markdown("**Drive type**")
        drive_type = st.selectbox(
            "Drive",
            options=["-- Select drive type --"] + DRIVE_TYPES,
            key="reg_drive_type",
            label_visibility="collapsed",
        )
        drive_type = drive_type if drive_type != "-- Select drive type --" else ""

        # Drive type support badge
        if drive_type and drive_type != "-- Select drive type --":
            if drive_type in DRIVE_TYPES_SUPPORTED:
                st.success(
                    "✅ **Induction motor drive — full physics supported.**  \n"
                    "All applicable physics phases will run for this drive type."
                )
            elif drive_type in ["Not applicable — static equipment"]:
                st.info("ℹ️ Static equipment — no drive physics applicable.")
            elif drive_type in ["Unknown", "Hydraulic coupling"]:
                st.info("ℹ️ AI statistical analytics will run. Drive physics not applicable.")
            else:
                st.warning(
                    f"⚠️ **Physics module not available for this drive type.**  \n"
                    f"**{drive_type}**  \n"
                    f"Only induction motor drives are currently supported for physics-based analytics.  \n"
                    f"AI statistical analytics (Layer 1) will run for all analysis types."
                )

        _drive_extra = ""
        if "VFD" in (drive_type or "") and "Induction motor" in (drive_type or ""):
            _vd1, _vd2 = st.columns(2)
            _vfd_min = _vd1.number_input("Min speed (RPM)", min_value=0, value=300, key="reg_vfd_min")
            _vfd_max = _vd2.number_input("Max speed (RPM)", min_value=0, value=1500, key="reg_vfd_max")
            _vfd_ctrl = st.selectbox("VFD control mode",
                ["Speed control","Pressure control","Flow control","Torque control"], key="reg_vfd_ctrl")
            _drive_extra = f"VFD speed range: {_vfd_min}\u2013{_vfd_max} RPM\nVFD control mode: {_vfd_ctrl}\n"
        elif "belt" in (drive_type or "").lower() and "Induction motor" in (drive_type or ""):
            _bd1, _bd2 = st.columns(2)
            _belt_drive  = _bd1.number_input("Drive pulley ø (mm)", min_value=1, value=200, key="reg_belt_drive")
            _belt_driven = _bd2.number_input("Driven pulley ø (mm)", min_value=1, value=200, key="reg_belt_driven")
            _sr = round(_belt_drive/_belt_driven,3) if _belt_driven else 1.0
            st.caption(f"Speed ratio: {_belt_drive}/{_belt_driven} = **{_sr}**")
            _drive_extra = f"Belt drive pulley (drive): {_belt_drive} mm\nBelt drive pulley (driven): {_belt_driven} mm\nBelt speed ratio: {_sr}\n"
        elif "gearbox" in (drive_type or "").lower() and "Induction motor" in (drive_type or ""):
            _gb1, _gb2 = st.columns(2)
            _gb_ratio = _gb1.number_input("Gear ratio (:1)", min_value=0.1, value=1.0, step=0.1, key="reg_gb_ratio")
            _gb_eff   = _gb2.number_input("Gearbox efficiency (%)", 50, 100, 98, key="reg_gb_eff")
            _drive_extra = f"Gear ratio: {_gb_ratio}:1\nGearbox efficiency: {_gb_eff}%\n"


        machine_desc = ""
        _is_pump_reg = (machine_type == "Centrifugal Pump" or "pump" in (machine_type or "").lower())

        with st.expander("Parameter thresholds (optional)", expanded=False):
            st.caption("Define warning and critical limits per parameter. Leave blank to let Claude decide automatically.")
            thresh_text = st.text_area(
                "Thresholds",
                placeholder="vibration_mm_s: warning=2.8, critical=4.5\ndischarge_temp_C: warning=170, critical=185\nmotor_current_A: warning=46, critical=50",
                height=120,
                help="One parameter per line. Format: param_name: warning=X, critical=Y",
            )

        if st.button("Register", type="primary", use_container_width=True):
            if machine_type and machine_id:
                # Build description from drive type only — specs entered via Data tab
                _structured = ""
                if drive_type:
                    _structured += f"Drive type: {drive_type}\n"
                if _drive_extra:
                    _structured += _drive_extra
                full_desc = _structured.strip()
                if thresh_text and thresh_text.strip():
                    full_desc = (full_desc + "\n\n=== PARAMETER THRESHOLDS ===\n" + thresh_text.strip()).strip()
                db.register_machine(machine_id.strip(), machine_type.strip(), full_desc)
                _phys_msg = ("Physics module: active." if PHYSICS_MODULE_STATUS.get(machine_type,("","",""))[0]=="available" else "AI analytics will run.")
                st.success(f"Machine **{machine_id}** registered as **{machine_type}**.  \nAdd specifications in the **Data tab**.")
                st.rerun()
            else:
                if not machine_id:
                    st.error("Machine ID is required.")
                if not machine_type:
                    st.error("Please select a machine type.")

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
        key="active_machine_select",
    )
    # Clear results when machine changes
    if st.session_state.get("_last_active_machine") != selected_id:
        st.session_state["last_multi_results"]  = None
        st.session_state["last_dq_report"]      = None
        st.session_state["last_data"]           = None
        st.session_state["_pending_analysis"]   = False
        st.session_state["_corrected_csv"]      = None
        st.session_state["_corrected_df"]       = None
        st.session_state["_confirm_pending"]    = False
        st.session_state["_last_active_machine"] = selected_id

    # ── Delete machine ──────────────────────────────────────────────
    with st.expander("\U0001f5d1\ufe0f Delete this machine", expanded=False):
        st.warning(
            f"Permanently delete **{selected_id}** and all its data, "
            "analysis history and maintenance logs. This cannot be undone."
        )
        _confirm_key = f"confirm_delete_{selected_id}"
        _confirmed = st.checkbox(
            f"I confirm I want to delete {selected_id}",
            key=_confirm_key, value=False
        )
        if _confirmed:
            if st.button(
                f"\U0001f5d1\ufe0f Delete {selected_id} permanently",
                type="primary",
                key=f"btn_delete_{selected_id}",
                use_container_width=True,
            ):
                ok = db.delete_machine(selected_id)
                if ok:
                    st.session_state["last_multi_results"]   = None
                    st.session_state["last_data"]            = None
                    st.session_state["_last_active_machine"] = None
                    st.success(f"{selected_id} deleted.")
                    st.rerun()
                else:
                    st.error("Delete failed — check logs.")

    st.divider()

    st.subheader("Upload data")
    uploaded_file = st.file_uploader(
        "CSV or Excel",
        type=["csv", "xlsx", "xls"],
        help="Any column named timestamp/date/time is auto-detected as the time axis.",
    )

    # Duplicate check (only meaningful when a file is selected)
    _existing_files = db.get_file_info(selected_id)
    _upload_stem    = uploaded_file.name.rsplit(".", 1)[0] if uploaded_file else ""
    _dup            = bool(_upload_stem and any(
        f["file"].rsplit(".", 1)[0] == _upload_stem for f in _existing_files
    ))

    # Ingest button — always visible, disabled until a file is selected
    if st.button("Ingest", use_container_width=True, disabled=not uploaded_file):
        if _dup:
            st.warning(
                f"⚠️ **{uploaded_file.name}** has already been ingested. "
                "Delete the existing file first to re-upload."
            )
        else:
            with st.spinner("Reading and storing data…"):
                result = db.ingest_file(uploaded_file, selected_id)
            if result["success"]:
                st.success(f"\u2713 {result['rows']:,} rows ingested")
                st.caption("Columns: " + ", ".join(result["columns"]))
                st.session_state["last_dq_report"]     = None
                st.session_state["last_multi_results"] = None
                st.session_state["last_data"]          = None
                st.session_state["_pending_analysis"]  = False
                st.session_state["_corrected_csv"]     = None
                st.session_state["_corrected_df"]      = None
                st.rerun()
            else:
                st.error(result["error"])

    if uploaded_file:
        # ── Timestamp format pre-check ──────────────────────────────
        if TS_CHECK_AVAILABLE:
            import tempfile, os as _os
            _ext  = uploaded_file.name.split(".")[-1]
            _tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=f".{_ext}")
            _tmp.write(uploaded_file.read())
            _tmp.close()
            uploaded_file.seek(0)  # reset for ingest
            _ts_check = check_timestamp_format(_tmp.name)
            _os.unlink(_tmp.name)

            if not _ts_check["parseable"]:
                st.warning(
                    f"\u26a0\ufe0f **Timestamp format issue detected** in column `{_ts_check['col'] or 'unknown'}`  \n"
                    f"{_ts_check['suggestion']}"
                )
                if _ts_check["sample_raw"]:
                    st.caption(f"Sample raw values: {', '.join(_ts_check['sample_raw'][:3])}")
                if _ts_check.get("corrected_df") is not None:
                    st.success(
                        "\u2705 Timestamps can be auto-corrected to YYYY-MM-DD HH:MM:SS format.  \n"
                        "Click **Fix timestamps and ingest** to correct and load in one step, "
                        "or download the corrected file to keep a copy."
                    )
                    _ts_buf = _ts_check["corrected_df"].to_csv(index=False).encode("utf-8")
                    _fix_col, _dl_col = st.columns(2)
                    with _fix_col:
                        if st.button("\u2705 Fix timestamps and ingest", type="primary",
                                     key="fix_ts_ingest_btn", use_container_width=True):
                            with st.spinner("Correcting timestamps and ingesting…"):
                                import io as _io2
                                _fixed_buf = _io2.BytesIO(_ts_buf)
                                _fixed_buf.name = uploaded_file.name.rsplit(".",1)[0] + "_fixed.csv"
                                _res2 = db.ingest_file(_fixed_buf, selected_id)
                            if _res2["success"]:
                                st.success(
                                    f"\u2713 {_res2['rows']:,} rows ingested with corrected timestamps."
                                )
                                st.caption("Columns: " + ", ".join(_res2["columns"]))
                                st.rerun()
                            else:
                                st.error(f"Ingest failed: {_res2['error']}")
                    with _dl_col:
                        st.download_button(
                            label="\u2b07\ufe0f Download corrected file",
                            data=_ts_buf,
                            file_name=f"{uploaded_file.name.rsplit('.',1)[0]}_ts_fixed.csv",
                            mime="text/csv",
                            key="dl_ts_fixed",
                            use_container_width=True,
                            help="Save a copy of the timestamp-corrected file"
                        )
                else:
                    st.info("Auto-correction was not possible. Please fix the timestamp column manually before uploading.")
            else:
                if _ts_check["sample_raw"]:
                    st.caption(f"\u2705 Timestamps OK · sample: {_ts_check['sample_raw'][0]}")

    file_info = db.get_file_info(selected_id)
    if file_info:
        st.caption(f"{len(file_info)} file(s) stored for this machine")
    if file_info and len(file_info) > 1:
        st.markdown("**📂 Choose which file to use for analysis:**")
        _file_options = {
            _f["file"]: (
                f"\U0001f527 {_f['file']} ({_f['rows']:,} rows \u00b7 corrected)"
                if "_corrected" in _f["file"].lower()
                else f"\U0001f4c2 {_f['file']} ({_f['rows']:,} rows \u00b7 original)"
            )
            for _f in file_info
        }
        _current_active = st.session_state.get(
            f"active_file_{selected_id}",
            file_info[-1]["file"]
        )
        if _current_active not in _file_options:
            _current_active = file_info[-1]["file"]
        _selected_file = st.selectbox(
            "Active file for analysis",
            options=list(_file_options.keys()),
            format_func=lambda x: _file_options[x],
            index=list(_file_options.keys()).index(_current_active),
            key=f"active_file_select_{selected_id}",
        )
        _file_changed = _selected_file != _current_active
        if st.button(
            "\u2713 Use selected file for analysis",
            key="use_selected_file_btn",
            type="primary",
            use_container_width=True,
            disabled=not _file_changed,
        ):
            st.session_state[f"active_file_{selected_id}"] = _selected_file
            st.session_state["last_data"]          = None
            st.session_state["last_multi_results"] = None
            st.session_state["last_dq_report"]     = None
            st.session_state["_pending_analysis"]  = False
            st.session_state["_corrected_csv"]     = None
            st.session_state["_corrected_df"]      = None
            st.rerun()
        if _file_changed:
            st.caption(f"\u26a0\ufe0f Currently using: **{_current_active}**. Click above to switch.")
        else:
            st.caption(f"\u2139\ufe0f Analysis will use: **{_current_active}**.")
        st.markdown("")

        with st.expander("Manage stored files", expanded=len(file_info) > 1):
            # Check for corrected/original pairs
            _has_corrected = any("_corrected" in f["file"].lower() for f in file_info)
            _has_original  = any("_corrected" not in f["file"].lower() for f in file_info)
            if _has_corrected and _has_original:
                st.info(
                    "\u2139\ufe0f Both original and corrected files are stored. "
                    "The corrected file (\U0001f527) contains auto-applied data quality fixes. "
                    "Analysis always uses the most recently uploaded file unless you delete one."
                )
            elif len(file_info) > 1:
                st.warning(f"{len(file_info)} files stored — duplicates may cause incorrect row counts.")

            for _fi in file_info:
                _is_corrected = "_corrected" in _fi["file"].lower()
                _fc1, _fc2, _fc3 = st.columns([0.62, 0.19, 0.19])
                with _fc1:
                    _bg = "#EAF8F4" if _is_corrected else "#F8FAFC"
                    _bc = "#0F6E56" if _is_corrected else "#CCDDEE"
                    _tag_html = (
                        '<span style="color:#0F6E56;font-weight:700">&nbsp;\u00b7&nbsp;\U0001f527 corrected</span>'
                        if _is_corrected else
                        '<span style="color:#888">&nbsp;\u00b7&nbsp;\U0001f4c2 original</span>'
                    )
                    st.markdown(
                        f'<div style="background:{_bg};border-left:4px solid {_bc};' +
                        f'padding:5px 10px;border-radius:3px;margin-bottom:4px;font-size:0.85em;">' +
                        f'<b>{_fi["file"]}</b>' +
                        f' &nbsp;\u00b7&nbsp; {_fi["rows"]:,} rows' +
                        f' &nbsp;\u00b7&nbsp; {str(_fi["ingested_at"])[:10]}' +
                        _tag_html + '</div>',
                        unsafe_allow_html=True
                    )
                with _fc2:
                    try:
                        import pathlib as _pl
                        _fp = _pl.Path(_fi.get("file_path",""))
                        if _fp.exists():
                            _fdata = _fp.read_bytes()
                        else:
                            import io as _io3
                            _fdf = db.get_data_from_file(selected_id, _fi["file"])
                            _buf3 = _io3.StringIO()
                            if _fdf is not None:
                                _fdf.to_csv(_buf3)
                            _fdata = _buf3.getvalue().encode("utf-8")
                        st.download_button(
                            label="\u2b07\ufe0f",
                            data=_fdata,
                            file_name=_fi["file"],
                            mime="text/csv",
                            key=f"dl_file_{_fi['file']}",
                            help=f"Download {'corrected' if _is_corrected else 'original'} file",
                        )
                    except Exception:
                        pass
                with _fc3:
                    if st.button("Delete", key=f"del_file_{_fi['file']}", type="secondary"):
                        db.delete_file(selected_id, _fi["file"])
                        st.success(f"Deleted {_fi['file']}")
                        st.rerun()
            st.markdown("")

            if st.button("Delete ALL files for this machine", type="primary",
                         key="del_all_files", use_container_width=True):
                db.delete_all_files(selected_id)
                st.success("All data files deleted. Please re-upload.")
                st.rerun()




# ================================================================== #
# MAIN AREA
# ================================================================== #

machine_info = db.get_machine_info(selected_id)
st.title(f"{machine_info['machine_type']}  ·  {selected_id}")

# ── Machine type & Drive type section ────────────────────────────────────────
with st.expander("🔧 Machine type & Drive type", expanded=False):
    st.caption("Review or correct the machine type and drive type. Changes take effect immediately.")

    _mt_desc = machine_info.get("description", "")
    _mt_base = _mt_desc.split("=== PARAMETER THRESHOLDS ===")[0]

    # Parse current drive_type from stored description
    _cur_drive = ""
    for _line in _mt_base.splitlines():
        if _line.strip().lower().startswith("drive type:"):
            _cur_drive = _line.split(":", 1)[1].strip()
            break

    _mt_col1, _mt_col2 = st.columns(2)

    with _mt_col1:
        st.markdown("**Machine type**")
        _mt_cat_cur = next(
            (cat for cat, types in MACHINE_TYPES.items()
             if machine_info.get("machine_type","") in types),
            list(MACHINE_TYPES.keys())[0]
        )
        _mt_cat = st.selectbox(
            "Category",
            options=list(MACHINE_TYPES.keys()),
            index=list(MACHINE_TYPES.keys()).index(_mt_cat_cur),
            key="mt_edit_category",
            label_visibility="collapsed",
        )
        _mt_type_opts = ["-- Select machine type --"] + MACHINE_TYPES.get(_mt_cat, [])
        _mt_cur_type  = machine_info.get("machine_type", "")
        _mt_type_idx  = (
            _mt_type_opts.index(_mt_cur_type)
            if _mt_cur_type in _mt_type_opts else 0
        )
        _mt_new_type = st.selectbox(
            "Type",
            options=_mt_type_opts,
            index=_mt_type_idx,
            key="mt_edit_type",
            label_visibility="collapsed",
        )
        _mt_new_type = _mt_new_type if _mt_new_type != "-- Select machine type --" else _mt_cur_type

    with _mt_col2:
        st.markdown("**Drive type**")
        _mt_drive_idx = (
            DRIVE_TYPES.index(_cur_drive) + 1
            if _cur_drive in DRIVE_TYPES else 0
        )
        _mt_new_drive = st.selectbox(
            "Drive",
            options=["-- Unchanged --"] + DRIVE_TYPES,
            index=_mt_drive_idx,
            key="mt_edit_drive",
            label_visibility="collapsed",
        )
        _mt_new_drive = _mt_new_drive if _mt_new_drive != "-- Unchanged --" else _cur_drive

    # Physics badge
    if _mt_new_type:
        _mt_phys = PHYSICS_MODULE_STATUS.get(_mt_new_type)
        if _mt_phys:
            _mt_status, _mt_detail, _mt_icon = _mt_phys
            if _mt_status == "available":
                if _mt_new_drive in DRIVE_TYPES_SUPPORTED:
                    st.success(f"{_mt_icon} Physics module available — {_mt_detail}. Induction motor drive confirmed.")
                elif _mt_new_drive:
                    st.warning(f"{_mt_icon} Physics module available for {_mt_new_type}, but drive type not supported. AI analytics only.")
                else:
                    st.info(f"{_mt_icon} Physics module available — select a drive type to confirm.")
            elif _mt_status == "planned":
                st.info(f"{_mt_icon} Physics module in development. AI analytics will run.")
            else:
                st.info(f"{_mt_icon} AI analytics will run.")

    # VFD / belt / gearbox sub-fields
    _mt_drive_extra = ""
    _mt_drive_extra_lines = []
    # Read existing sub-values from description for pre-population
    def _read_desc_val(desc, key, default=""):
        for ln in desc.splitlines():
            if ln.strip().lower().startswith(key.lower() + ":"):
                return ln.split(":", 1)[1].strip()
        return default

    if "VFD" in (_mt_new_drive or "") and "Induction motor" in (_mt_new_drive or ""):
        _mv1, _mv2 = st.columns(2)
        _vfd_min_cur = int(_read_desc_val(_mt_base, "VFD speed range", "300–").split("–")[0].strip() or 300)
        _vfd_max_cur = int((_read_desc_val(_mt_base, "VFD speed range", "–1500").split("–")[-1].strip() or 1500))
        _mt_vfd_min  = _mv1.number_input("Min speed (RPM)", min_value=0, value=_vfd_min_cur, key="mt_vfd_min")
        _mt_vfd_max  = _mv2.number_input("Max speed (RPM)", min_value=0, value=_vfd_max_cur, key="mt_vfd_max")
        _vfd_modes   = ["Speed control","Pressure control","Flow control","Torque control"]
        _vfd_ctrl_cur = _read_desc_val(_mt_base, "VFD control mode", "Speed control")
        _vfd_ctrl_idx = _vfd_modes.index(_vfd_ctrl_cur) if _vfd_ctrl_cur in _vfd_modes else 0
        _mt_vfd_ctrl  = st.selectbox("VFD control mode", _vfd_modes, index=_vfd_ctrl_idx, key="mt_vfd_ctrl")
        _mt_drive_extra = f"VFD speed range: {_mt_vfd_min}\u2013{_mt_vfd_max} RPM\nVFD control mode: {_mt_vfd_ctrl}"
    elif "belt" in (_mt_new_drive or "").lower() and "Induction motor" in (_mt_new_drive or ""):
        _mb1, _mb2 = st.columns(2)
        _mt_bd = _mb1.number_input("Drive pulley ø (mm)", min_value=1,
                                    value=int(_read_desc_val(_mt_base, "Belt drive pulley (drive)", "200").split()[0] or 200),
                                    key="mt_belt_drive")
        _mt_bn = _mb2.number_input("Driven pulley ø (mm)", min_value=1,
                                    value=int(_read_desc_val(_mt_base, "Belt drive pulley (driven)", "200").split()[0] or 200),
                                    key="mt_belt_driven")
        _mt_sr = round(_mt_bd / _mt_bn, 3) if _mt_bn else 1.0
        st.caption(f"Speed ratio: {_mt_bd}/{_mt_bn} = **{_mt_sr}**")
        _mt_drive_extra = f"Belt drive pulley (drive): {_mt_bd} mm\nBelt drive pulley (driven): {_mt_bn} mm\nBelt speed ratio: {_mt_sr}"
    elif "gearbox" in (_mt_new_drive or "").lower() and "Induction motor" in (_mt_new_drive or ""):
        _mg1, _mg2 = st.columns(2)
        _mt_gr  = _mg1.number_input("Gear ratio (:1)", min_value=0.1,
                                     value=float(_read_desc_val(_mt_base, "Gear ratio", "1.0").rstrip(":1").strip() or 1.0),
                                     step=0.1, key="mt_gb_ratio")
        _mt_ge  = _mg2.number_input("Gearbox efficiency (%)", 50, 100,
                                     value=int(_read_desc_val(_mt_base, "Gearbox efficiency", "98").rstrip("%").strip() or 98),
                                     key="mt_gb_eff")
        _mt_drive_extra = f"Gear ratio: {_mt_gr}:1\nGearbox efficiency: {_mt_ge}%"

    if st.button("Save machine type & drive type", key="save_mt_drive_btn", use_container_width=True):
        # Rebuild description: strip old Drive type + sub-field lines, insert new ones
        _drive_keywords = (
            "drive type:", "vfd speed range:", "vfd control mode:",
            "belt drive pulley", "belt speed ratio:", "gear ratio:", "gearbox efficiency:"
        )
        _clean_lines = [
            ln for ln in _mt_base.splitlines()
            if not any(ln.strip().lower().startswith(kw) for kw in _drive_keywords)
        ]
        if _mt_new_drive:
            _clean_lines.append(f"Drive type: {_mt_new_drive}")
        if _mt_drive_extra:
            _clean_lines.extend(_mt_drive_extra.splitlines())

        _thresh_block = _mt_desc.split("=== PARAMETER THRESHOLDS ===")
        _new_desc = "\n".join(_clean_lines).strip()
        if len(_thresh_block) > 1:
            _new_desc += "\n\n=== PARAMETER THRESHOLDS ===\n" + _thresh_block[1].strip()

        db.register_machine(selected_id, _mt_new_type, _new_desc)
        st.success("✓ Machine type and drive type updated.")
        st.rerun()

# Inline threshold editor
with st.expander("Parameter thresholds", expanded=False):
    st.caption(
        "Set warning and critical limits for each sensor parameter. "
        "Leave both fields blank to let Claude decide automatically."
    )
    _thr_desc = machine_info.get("description", "")
    _thr_base = _thr_desc.split("=== PARAMETER THRESHOLDS ===")[0].strip()

    # Parse existing thresholds into a dict for pre-population
    _thr_existing = {}
    if "=== PARAMETER THRESHOLDS ===" in _thr_desc:
        for _tl in _thr_desc.split("=== PARAMETER THRESHOLDS ===")[-1].strip().splitlines():
            _tl = _tl.strip()
            if not _tl or ":" not in _tl:
                continue
            try:
                _tp, _tr = _tl.split(":", 1)
                _tlims = {}
                for _part in _tr.split(","):
                    _part = _part.strip()
                    if "=" in _part:
                        _k, _v = _part.split("=", 1)
                        _tlims[_k.strip()] = float(_v.strip())
                if _tlims:
                    _thr_existing[_tp.strip()] = _tlims
            except Exception:
                pass

    # Get numeric columns from loaded data (may be None before upload)
    _thr_data = db.get_data(selected_id)
    _thr_cols  = (
        _thr_data.select_dtypes(include="number").columns.tolist()
        if _thr_data is not None and not _thr_data.empty
        else list(_thr_existing.keys())
    )

    if not _thr_cols:
        st.info("Upload data first — parameter columns will appear here.")
    else:
        st.caption(
            "Tick the checkbox to activate thresholds for a parameter. "
            "Only ticked parameters are used in health scoring even if values are entered for others."
        )
        # Header row
        _th1, _th2, _th3, _th4 = st.columns([0.15, 1.85, 1, 1])
        _th1.markdown('<div style="font-size:0.8em;font-weight:600;color:#555;">Use</div>', unsafe_allow_html=True)
        _th2.markdown('<div style="font-size:0.8em;font-weight:600;color:#555;">Parameter</div>', unsafe_allow_html=True)
        _th3.markdown('<div style="font-size:0.8em;font-weight:600;color:#BA7517;">⚠️ Warning</div>', unsafe_allow_html=True)
        _th4.markdown('<div style="font-size:0.8em;font-weight:600;color:#A32D2D;">❌ Critical</div>', unsafe_allow_html=True)

        _thr_new = {}
        for _col in _thr_cols:
            _ex       = _thr_existing.get(_col, {})
            _warn_cur = _ex.get("warning", None)
            _crit_cur = _ex.get("critical", None)
            _is_active = _col in _thr_existing
            _cb, _cl, _cw, _cc = st.columns([0.15, 1.85, 1, 1])
            _use_col = _cb.checkbox(
                " ", value=_is_active,
                key=f"thr_use_{_col}",
                label_visibility="collapsed",
            )
            _cl.markdown(
                f'<div style="font-size:0.85em;padding-top:8px;'
                f'color:{"#222" if _use_col else "#AAA"};">'
                f"<code>{_col}</code></div>",
                unsafe_allow_html=True,
            )
            _warn_val = _cw.text_input(
                f"W_{_col}",
                value=str(_warn_cur) if _warn_cur is not None else "",
                placeholder="e.g. 2.8",
                label_visibility="collapsed",
                key=f"thr_warn_{_col}",
                disabled=not _use_col,
            )
            _crit_val = _cc.text_input(
                f"C_{_col}",
                value=str(_crit_cur) if _crit_cur is not None else "",
                placeholder="e.g. 4.5",
                label_visibility="collapsed",
                key=f"thr_crit_{_col}",
                disabled=not _use_col,
            )
            if _use_col:
                _lims = {}
                try:
                    if _warn_val.strip():
                        _lims["warning"] = float(_warn_val.strip())
                except ValueError:
                    pass
                try:
                    if _crit_val.strip():
                        _lims["critical"] = float(_crit_val.strip())
                except ValueError:
                    pass
                if _lims:
                    _thr_new[_col] = _lims

        # Validate: find ticked params with no values entered
        _thr_incomplete = [
            _col for _col in _thr_cols
            if st.session_state.get(f"thr_use_{_col}", False)
            and not st.session_state.get(f"thr_warn_{_col}", "").strip()
            and not st.session_state.get(f"thr_crit_{_col}", "").strip()
        ]
        if _thr_incomplete:
            st.warning(
                "⚠️ Please enter at least one threshold value (warning or critical) "
                "for the selected parameter(s): "
                + ", ".join(f"`{c}`" for c in _thr_incomplete)
                + "  \nUntick the checkbox to skip a parameter."
            )
        if st.button("Save thresholds", key="save_thresh_btn",
                     use_container_width=True, disabled=bool(_thr_incomplete)):
            _thresh_lines = [
                f"{_p}: " + ", ".join(f"{k}={v}" for k, v in _lims.items())
                for _p, _lims in _thr_new.items()
            ]
            _thresh_text = "\n".join(_thresh_lines)
            if _thresh_text:
                _updated_desc = _thr_base + "\n\n=== PARAMETER THRESHOLDS ===\n" + _thresh_text
            else:
                _updated_desc = _thr_base
            db.register_machine(selected_id, machine_info["machine_type"], _updated_desc.strip())
            st.success("\u2713 Thresholds saved.")
            st.rerun()

# ── Machine specifications ────────────────────────────────────────
with st.expander("✏️ Machine specifications", expanded=False):
    st.caption("Enter or update nameplate data, installation details, and notes. Threshold block is preserved automatically.")
    _tab_mtype  = machine_info.get("machine_type", "")
    _tab_desc   = machine_info.get("description", "")
    _tab_base   = _tab_desc.split("=== PARAMETER THRESHOLDS ===")[0].strip()
    _tab_thresh_block = (
        "\n\n=== PARAMETER THRESHOLDS ===\n" +
        _tab_desc.split("=== PARAMETER THRESHOLDS ===")[1].strip()
        if "=== PARAMETER THRESHOLDS ===" in _tab_desc else ""
    )
    _tab_is_pump = (_tab_mtype == "Centrifugal Pump" or "pump" in _tab_mtype.lower())

    # Pre-parse existing description to pre-populate fields
    _tab_np = parse_nameplate(_tab_base) if PUMP_PHYSICS_AVAILABLE and _tab_is_pump else {}

    def _tab_fv(key, default=0.0):
        """Get float value from parsed nameplate or default."""
        v = _tab_np.get(key, default)
        try: return float(v) if v else default
        except: return default

    _tab_stab1, _tab_stab2 = st.tabs(["✏️ Enter manually", "📎 Upload file"])
    _tab_new_desc = _tab_base

    with _tab_stab1:
        if _tab_is_pump:
            # ── Tier 1: Recommended ───────────────────────────
            st.caption("⭐ Recommended — improves accuracy for all phases")
            _tr1, _tr2 = st.columns(2)
            _tnp_rated_kw  = _tr1.number_input("Rated power (kW)", min_value=0.0,
                value=_tab_fv("rated_power_kw"), step=0.1, format="%.1f", key="tnp_rated_kw",
                help="Motor electrical input rating.")
            _tnp_flow      = _tr2.number_input("Rated flow (m³/h)", min_value=0.0,
                value=_tab_fv("rated_flow"), step=0.1, format="%.1f", key="tnp_flow",
                help="Design flow at rated duty point.")
            _tr3, _tr4 = st.columns(2)
            _tnp_head      = _tr3.number_input("Rated head (m)", min_value=0.0,
                value=_tab_fv("rated_head"), step=0.1, format="%.1f", key="tnp_head",
                help="Design head at rated duty point.")
            _tnp_pump_eff  = _tr4.number_input("Pump efficiency (%)", min_value=0.0, max_value=100.0,
                value=_tab_fv("pump_efficiency"), step=0.1, format="%.1f", key="tnp_pump_eff",
                help="Hydraulic efficiency at rated point.")

            # ── Tier 2: Optional ──────────────────────────────
            with st.expander("Optional — improves accuracy", expanded=False):
                st.caption("Motor nameplate data. Defaults are used if not entered.")
                _to1, _to2, _to3 = st.columns(3)
                _tnp_rated_rpm = _to1.number_input("Rated speed (RPM)", min_value=0,
                    value=int(_tab_fv("rated_speed", 0)), step=10, key="tnp_rated_rpm")
                _tnp_fla       = _to2.number_input("FLA (A)", min_value=0.0,
                    value=_tab_fv("fla"), step=0.1, format="%.1f", key="tnp_fla")
                _tnp_voltage   = _to3.number_input("Voltage (V)", min_value=0.0,
                    value=_tab_fv("voltage", 415.0), step=1.0, format="%.0f", key="tnp_voltage")
                _to4, _to5, _to6 = st.columns(3)
                _tnp_pf        = _to4.number_input("Power factor", min_value=0.0, max_value=1.0,
                    value=_tab_fv("power_factor"), step=0.01, format="%.2f", key="tnp_pf")
                _tnp_motor_eff = _to5.number_input("Motor efficiency (%)", min_value=0.0, max_value=100.0,
                    value=_tab_fv("motor_efficiency"), step=0.1, format="%.1f", key="tnp_motor_eff")
                _tnp_ie        = _to6.selectbox("IE class", ["--","IE1","IE2","IE3","IE4"],
                    index=0, key="tnp_ie")
                _to7, _to8 = st.columns(2)
                _tnp_poles     = _to7.selectbox("Poles", [0,2,4,6,8], index=0, key="tnp_poles")
                _tnp_freq      = _to8.selectbox("Frequency (Hz)", [50,60], index=0, key="tnp_freq")

            # ── Tier 3: Advanced ──────────────────────────────
            with st.expander("Advanced diagnostics", expanded=False):
                st.caption("Enables BEP deviation, cavitation checks, blade pass frequency, and commissioning comparison.")
                _ta1, _ta2, _ta3 = st.columns(3)
                _tnp_bep_flow  = _ta1.number_input("BEP flow (m³/h)", min_value=0.0,
                    value=_tab_fv("bep_flow"), step=0.1, format="%.1f", key="tnp_bep_flow")
                _tnp_npsh_r    = _ta2.number_input("NPSH required (m)", min_value=0.0,
                    value=_tab_fv("npsh_r"), step=0.1, format="%.1f", key="tnp_npsh_r")
                _tnp_imp_dia   = _ta3.number_input("Impeller ø (mm)", min_value=0,
                    value=int(_tab_fv("impeller_diameter", 0)), step=1, key="tnp_imp_dia")
                _ta4, _ta5, _ta6 = st.columns(3)
                _tnp_vanes     = _ta4.number_input("Number of vanes", min_value=0,
                    value=int(_tab_fv("vanes", 0)), step=1, key="tnp_vanes")
                _tnp_pump_rpm  = _ta5.number_input("Pump speed (RPM)", min_value=0,
                    value=int(_tab_fv("pump_speed", 0)), step=10, key="tnp_pump_rpm")
                _tnp_comm_kw   = _ta6.number_input("Commissioning power (kW)", min_value=0.0,
                    value=_tab_fv("commissioning_power"), step=0.1, format="%.1f", key="tnp_comm_kw")
                _ta7, _ = st.columns(2)
                _tnp_density   = _ta7.number_input("Fluid density (kg/m³)", min_value=0.0,
                    value=_tab_fv("fluid_density", 998.0), step=1.0, format="%.0f", key="tnp_density")

            _tnp_notes = st.text_area(
                "Additional notes",
                value=_tab_base if not any([_tnp_rated_kw, _tnp_flow, _tnp_head]) else "",
                placeholder="e.g. Commissioning date: 2024-01-15, seal type: mechanical, bearing: 6308",
                height=60, key="tnp_extra_notes", label_visibility="collapsed",
            )

            # Serialise structured fields to description text
            _tab_desc_parts = []
            if _tnp_rated_kw > 0:   _tab_desc_parts.append(f"Rated power: {_tnp_rated_kw} kW")
            if _tnp_rated_rpm > 0:  _tab_desc_parts.append(f"Rated speed: {_tnp_rated_rpm} RPM")
            if _tnp_poles > 0:      _tab_desc_parts.append(f"{_tnp_poles}-pole")
            if _tnp_freq:           _tab_desc_parts.append(f"Frequency: {_tnp_freq} Hz")
            if _tnp_fla > 0:        _tab_desc_parts.append(f"FLA: {_tnp_fla} A")
            if _tnp_voltage > 0:    _tab_desc_parts.append(f"Voltage: {_tnp_voltage} V")
            if _tnp_pf > 0:         _tab_desc_parts.append(f"Power factor: {_tnp_pf}")
            if _tnp_motor_eff > 0:  _tab_desc_parts.append(f"Motor efficiency: {_tnp_motor_eff}%")
            if _tnp_ie != "--":     _tab_desc_parts.append(f"{_tnp_ie}")
            if _tnp_flow > 0:       _tab_desc_parts.append(f"Rated flow: {_tnp_flow} m3/h")
            if _tnp_head > 0:       _tab_desc_parts.append(f"Rated head: {_tnp_head} m")
            if _tnp_pump_eff > 0:   _tab_desc_parts.append(f"Pump efficiency: {_tnp_pump_eff}%")
            if _tnp_bep_flow > 0:   _tab_desc_parts.append(f"BEP flow: {_tnp_bep_flow} m3/h")
            if _tnp_npsh_r > 0:     _tab_desc_parts.append(f"NPSH_r: {_tnp_npsh_r} m")
            if _tnp_imp_dia > 0:    _tab_desc_parts.append(f"Impeller: {_tnp_imp_dia} mm")
            if _tnp_vanes > 0:      _tab_desc_parts.append(f"Vanes: {_tnp_vanes}")
            if _tnp_pump_rpm > 0:   _tab_desc_parts.append(f"Pump speed: {_tnp_pump_rpm} RPM")
            if _tnp_comm_kw > 0:    _tab_desc_parts.append(f"Commissioning power: {_tnp_comm_kw} kW")
            if _tnp_density != 998: _tab_desc_parts.append(f"Density: {_tnp_density} kg/m3")
            # Preserve drive type line from existing description
            for _ln in _tab_base.splitlines():
                if _ln.strip().lower().startswith("drive type:"):
                    _tab_desc_parts.append(_ln.strip())
                    break
            if _tnp_notes.strip():  _tab_desc_parts.append(_tnp_notes.strip())
            _tab_new_desc = "\n".join(_tab_desc_parts)

        else:
            # Non-pump: free text
            _tab_new_desc = st.text_area(
                "Specifications",
                value=_tab_base,
                height=180,
                label_visibility="collapsed",
                help="Enter nameplate data, installation details, and notes.",
                key="tab_spec_textarea",
            )

    with _tab_stab2:
        st.caption(
            "Upload a PDF, image, or text file containing the machine datasheet, "
            "nameplate photo, or specification document."
        )
        _tab_spec_file = st.file_uploader(
            "Upload specification file",
            type=["pdf", "png", "jpg", "jpeg", "webp", "txt"],
            key="tab_edit_spec_file",
            label_visibility="collapsed",
        )
        if _tab_spec_file:
            _tab_api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if st.button(
                "📤 Extract text from file",
                key="tab_extract_spec_btn",
                type="secondary",
                use_container_width=True,
            ):
                with st.spinner("Extracting text from file…"):
                    try:
                        _tab_extracted = _extract_spec_text(_tab_spec_file, _tab_api_key)
                        st.session_state["_tab_extracted_spec"] = _tab_extracted
                    except Exception as _tex:
                        st.error(f"Extraction failed: {_tex}")
            _tab_extracted_text = st.session_state.get("_tab_extracted_spec", "")
            if _tab_extracted_text:
                st.success(f"✅ Text extracted ({len(_tab_extracted_text)} characters)")
                _tab_new_desc = st.text_area(
                    "Extracted text (review and edit before saving)",
                    value=_tab_extracted_text,
                    height=180,
                    key="tab_extracted_desc",
                    label_visibility="collapsed",
                )
            elif _tab_spec_file:
                st.info("Click 'Extract text from file' to read the content.")

    if st.button("Save specifications", key="tab_save_spec_btn", use_container_width=True):
        _tab_updated = (_tab_new_desc.strip() + _tab_thresh_block).strip()
        db.register_machine(selected_id, _tab_mtype, _tab_updated)
        st.session_state.pop("_tab_extracted_spec", None)
        st.success("✓ Specifications updated.")
        st.rerun()

# ── Maintenance logs ─────────────────────────────────────────────────────────
with st.expander("📋 Maintenance logs", expanded=False):
    st.caption("Upload service records or type notes directly. All stored logs are included automatically in every analysis.")

    _ml_tab_file, _ml_tab_text = st.tabs(["📎 Upload file", "✏️ Enter text"])

    with _ml_tab_file:
        _ml_file = st.file_uploader(
            "Upload log (CSV, PDF, image)",
            type=["csv", "xlsx", "pdf", "png", "jpg", "jpeg", "webp", "txt"],
            key="log_uploader",
            help="Handwritten or printed maintenance logs, service records, inspection reports.",
        )
        if st.button("Read & store log", use_container_width=True, disabled=not _ml_file,
                     key="store_log_file_btn"):
            if _ml_file:
                with st.spinner("Reading log file…"):
                    result = read_log(_ml_file, api_key=os.getenv("ANTHROPIC_API_KEY", ""))
                if result["success"]:
                    db.save_log(selected_id, _ml_file.name, result["method"], result["text"])
                    st.success(f"✓ Log stored ({result['method']}, {len(result['text'])} chars)")
                    st.rerun()
                else:
                    st.error(result["error"])

    with _ml_tab_text:
        _ml_title = st.text_input(
            "Entry title",
            placeholder="e.g. Bearing replacement 2024-03-01",
            key="ml_text_title",
        )
        _ml_text = st.text_area(
            "Log entry",
            placeholder=(
                "e.g. Replaced drive-end bearing (SKF 6308). Vibration was 4.2 mm/s before, "
                "dropped to 1.1 mm/s after replacement. Seal inspected — no leaks found."
            ),
            height=130,
            key="ml_text_body",
            label_visibility="collapsed",
        )
        if st.button("Save log entry", use_container_width=True,
                     disabled=not _ml_text.strip(), key="store_log_text_btn"):
            _ml_fname = (_ml_title.strip() or "Manual entry") + ".txt"
            db.save_log(selected_id, _ml_fname, "text", _ml_text.strip())
            st.success(f"✓ Log entry saved as **{_ml_fname}**")
            st.rerun()

    # Stored logs list
    stored_logs = db.get_logs(selected_id)
    if stored_logs:
        st.markdown("---")
        st.caption(f"{len(stored_logs)} log entry/entries stored — included in every analysis")
        for _sl in stored_logs:
            _slc1, _slc2 = st.columns([0.85, 0.15])
            _icon = "✏️" if _sl["file_type"] == "text" else "📄"
            _slc1.caption(f"{_icon} {_sl['filename']}  ·  {str(_sl['uploaded_at'])[:10]}")
            with _slc2:
                if st.button("Delete", key=f"del_log_{_sl['filename']}", type="secondary"):
                    db.delete_log(selected_id, _sl["filename"])
                    st.rerun()
    else:
        st.caption("No logs stored yet.")

# Load data — always use a single specific file, never union multiple files
_all_file_info  = db.get_file_info(selected_id)
_active_file    = st.session_state.get(f"active_file_{selected_id}")

# If no active file chosen yet, default to the most recently ingested file
if not _active_file and _all_file_info:
    _active_file = _all_file_info[-1]["file"]
    st.session_state[f"active_file_{selected_id}"] = _active_file

if _active_file:
    _file_data = db.get_data_from_file(selected_id, _active_file)
    data = _file_data if _file_data is not None else None
else:
    data = None

# ── Data fingerprint: auto-clear stale DQ/analysis results when data changes ──
# Computed on every render — no need to intercept individual file buttons.
_data_fp = (
    f"{selected_id}|"
    + (
        f"{len(data)}|{str(data.index.min())}|{str(data.index.max())}|"
        + ",".join(sorted(data.columns.tolist()))
        if data is not None and not data.empty
        else "empty"
    )
)
if st.session_state.get("_data_fingerprint") != _data_fp:
    st.session_state["_data_fingerprint"]  = _data_fp
    st.session_state["last_dq_report"]     = None
    st.session_state["last_multi_results"] = None
    st.session_state["last_data"]          = None
    st.session_state["_pending_analysis"]  = False
    st.session_state["_corrected_csv"]     = None
    st.session_state["_corrected_df"]      = None
    st.session_state["_pump_physics_result"] = None


def _run_analysis(meta, dq_ctx, db, selected_id, override_data=None):
    """Run the actual analysis loop and store results in session state."""
    _analyzer    = Analyzer()
    _all_results = {}
    _progress    = st.progress(0)
    _status      = st.empty()
    _sel         = meta["selected_analyses"]
    for _i, _atype in enumerate(_sel):
        _status.text(f"Running {_atype} ({_i+1} of {len(_sel)})...")
        _mid = meta["machine_info"]["machine_id"]
        _af  = st.session_state.get(f"active_file_{_mid}")
        _data_to_use = (
            override_data if override_data is not None
            else (db.get_data_from_file(_mid, _af) if _af else db.get_data(_mid))
        )
        _result = _analyzer.analyze(
            machine_info         = meta["machine_info"],
            data                 = _data_to_use,
            analysis_type        = _atype,
            date_range           = meta["date_range"],
            baseline_period      = meta.get("baseline_period"),
            extra_context        = meta["extra_context"],
            schedule             = meta["schedule"] if _atype == "Operational Schedule Compliance" else None,
            logs_text            = meta["logs_text"],
            data_quality_context = dq_ctx,
        )
        if _result["success"]:
            _all_results[_atype] = _result["insights"]
            db.save_analysis(selected_id, _atype, _result["insights"])
        else:
            _all_results[_atype] = {"error": _result["error"]}
        _progress.progress((_i + 1) / len(_sel))
    _status.empty()
    _progress.empty()
    st.session_state["last_multi_results"] = _all_results
    st.session_state["last_data"]          = override_data if override_data is not None else meta["filtered_data"]
    st.rerun()

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

        # ── Data quality — auto-run and display ─────────────────────────────
        if DQ_AVAILABLE and st.session_state.get("last_dq_report") is None:
            with st.spinner("Running data quality checks…"):
                _dq_auto = run_data_quality_checks(data)
                st.session_state["last_dq_report"] = _dq_auto
                st.session_state["_dq_ctx"] = format_quality_report_for_claude(_dq_auto)

        _tab_dq = st.session_state.get("last_dq_report") or {}
        if _tab_dq:
            _tdq_score = _tab_dq.get("score", 100)
            _tdq_crits = [x for x in _tab_dq.get("issues", []) if x["severity"] == "critical"]
            _tdq_warns = [x for x in _tab_dq.get("issues", []) if x["severity"] == "warning"]
            _tdq_label = (
                (f"  ·  {len(_tdq_crits)} critical" if _tdq_crits else "") +
                (f"  ·  {len(_tdq_warns)} warning(s)" if _tdq_warns else "") +
                ("  ·  All checks passed" if not _tab_dq.get("issues") else "")
            )
            with st.expander(f"Data quality — score {_tdq_score}/100{_tdq_label}",
                             expanded=bool(_tdq_crits)):
                _dq_sev_color = {"critical": "#A32D2D", "warning": "#BA7517", "info": "#185FA5"}
                _dq_sev_bg    = {"critical": "#FFF0F0", "warning": "#FFFBF0", "info": "#EAF4FF"}

                if not _tab_dq.get("issues"):
                    if st.session_state.get("_corrections_applied"):
                        st.session_state["_corrections_applied"] = False
                        st.success("✅ **Auto-corrections applied successfully.** Data quality checks passed.")
                        st.info(
                            "ℹ️ Go to the **Analysis** tab to continue with the analysis."
                        )
                    else:
                        st.success("All sensor data quality checks passed. Data is suitable for analysis.")
                else:
                    for _tdq_iss in _tab_dq.get("issues", []):
                        _tdq_sev  = _tdq_iss["severity"]
                        _tdq_fc   = _dq_sev_color.get(_tdq_sev, "#333")
                        _tdq_bg   = _dq_sev_bg.get(_tdq_sev, "#F8F8F8")
                        _tdq_icon = {
                            "critical": "❌", "warning": "⚠️", "info": "ℹ️"
                        }.get(_tdq_sev, "•")
                        _tdq_sugg = CORRECTION_SUGGESTIONS.get(
                            _tdq_iss["check"], {}).get("suggestion", "")
                        st.markdown(
                            f'<div style="background:{_tdq_bg};border-left:5px solid {_tdq_fc};'
                            f'padding:10px 14px;margin-bottom:4px;border-radius:3px;">'
                            f'<span style="font-weight:700;color:{_tdq_fc}">'
                            f'{_tdq_icon} {_tdq_iss["check"]}</span>'
                            f' &nbsp;·&nbsp; <code>{_tdq_iss["col"]}</code>'
                            f' &nbsp;·&nbsp; <span style="color:#888;font-size:0.82em">'
                            f'{_tdq_iss["affected_pct"]}% affected</span><br>'
                            f'<span style="font-size:0.87em;color:#444">'
                            f'{_tdq_iss["detail"]}</span></div>',
                            unsafe_allow_html=True)
                        if _tdq_sugg:
                            st.markdown(
                                f'<div style="background:#F0F4F8;border-left:3px solid #185FA5;'
                                f'padding:6px 12px;margin-bottom:10px;border-radius:2px;">'
                                f'<span style="font-size:0.85em;color:#185FA5;font-weight:600">'
                                f'💡 Suggestion: </span>'
                                f'<span style="font-size:0.85em;color:#333">{_tdq_sugg}</span></div>',
                                unsafe_allow_html=True)

                    # Only show options panel when there are critical issues
                    if _tdq_crits:
                        st.markdown("---")
                        _toption_cols = st.columns(3)

                        # ── Option 1 — Manual correction ─────────────────────
                        with _toption_cols[0]:
                            st.markdown(
                                '<div style="background:#EAF4FF;border:1.5px solid #185FA5;'
                                'border-radius:6px;padding:12px 14px;min-height:72px;">'
                                '<span style="font-weight:700;color:#185FA5">'
                                '⭐ Option 1 — Recommended</span><br>'
                                '<span style="font-size:0.88em;color:#1A1A2E;">'
                                'Correct the data manually and re-upload.</span>'
                                '</div>',
                                unsafe_allow_html=True)
                            st.markdown("")
                            with st.expander("📋 How to correct manually", expanded=False):
                                st.markdown(
                                    "1. Download your original file below  \n"
                                    "2. Correct or remove the affected rows  \n"
                                    "3. Delete the existing file (Manage stored files)  \n"
                                    "4. Re-upload the corrected file  \n"
                                    "5. Press **Analyze**"
                                )
                            import io as _tdl_io
                            _tdl_buf = _tdl_io.BytesIO(data.to_csv().encode("utf-8"))
                            st.download_button(
                                "⬇️ Download original data",
                                data=_tdl_buf,
                                file_name=f"{selected_id}_original.csv",
                                mime="text/csv",
                                key="tab_dl_original_dq",
                                use_container_width=True,
                            )

                        # ── Option 2 — Ignore and proceed ────────────────────
                        with _toption_cols[1]:
                            st.markdown(
                                '<div style="background:#FFF8F0;border:1.5px solid #BA7517;'
                                'border-radius:6px;padding:12px 14px;min-height:72px;">'
                                '<span style="font-weight:700;color:#BA7517">'
                                '⚠️ Option 2 — Ignore and continue</span><br>'
                                '<span style="font-size:0.88em;color:#1A1A2E;">'
                                'Acknowledge each issue and analyse anyway.</span>'
                                '</div>',
                                unsafe_allow_html=True)
                            st.markdown("")
                            _tdq_ignored = {}
                            for _tdq_ai in _tdq_crits:
                                _tdq_akey = f"tab_ack_{_tdq_ai['col']}_{_tdq_ai['check'].replace(' ','_')}"
                                _ta1, _ta2 = st.columns([0.12, 0.88])
                                with _ta1:
                                    _tdq_ignored[_tdq_akey] = st.checkbox(
                                        " ", key=_tdq_akey, value=False,
                                        label_visibility="collapsed")
                                with _ta2:
                                    _ta_clr  = "#2E7D32" if _tdq_ignored[_tdq_akey] else "#A32D2D"
                                    _ta_icon = "✅" if _tdq_ignored[_tdq_akey] else "❌"
                                    _ta_lbl  = "Acknowledged" if _tdq_ignored[_tdq_akey] else "Pending"
                                    st.markdown(
                                        f'<span style="color:{_ta_clr};font-size:0.85em;font-weight:600">'
                                        f'{_ta_icon} {_tdq_ai["check"]} · <code>{_tdq_ai["col"]}</code>'
                                        f' · <i>{_ta_lbl}</i></span>',
                                        unsafe_allow_html=True)
                            _tdq_all_acked = all(_tdq_ignored.values()) if _tdq_ignored else False
                            if not _tdq_all_acked:
                                st.caption(f"{sum(1 for v in _tdq_ignored.values() if not v)} issue(s) unacknowledged.")
                            else:
                                st.session_state["_pending_analysis"] = False
                                st.session_state["_dq_acknowledged"]  = True
                                st.info(
                                    "ℹ️ **All issues acknowledged.** "
                                    "Go to the **Analysis** tab to continue."
                                )

                        # ── Option 3 — Auto-correct ───────────────────────────
                        with _toption_cols[2]:
                            st.markdown(
                                '<div style="background:#F0FFF4;border:1.5px solid #2E7D32;'
                                'border-radius:6px;padding:12px 14px;min-height:72px;">'
                                '<span style="font-weight:700;color:#2E7D32">'
                                '🔧 Option 3 — Auto-correct</span><br>'
                                '<span style="font-size:0.88em;color:#1A1A2E;">'
                                'Apply suggested fixes and save corrected file.</span>'
                                '</div>',
                                unsafe_allow_html=True)
                            st.markdown("")
                            _tdq_fixable = [
                                i for i in _tdq_crits
                                if CORRECTION_SUGGESTIONS.get(i["check"], {}).get("auto", False)
                                and i["col"] != "[timestamp]"
                            ]
                            _tdq_manual_only = [
                                i for i in _tdq_crits
                                if not CORRECTION_SUGGESTIONS.get(i["check"], {}).get("auto", False)
                                or i["col"] == "[timestamp]"
                            ]
                            if not _tdq_fixable:
                                st.info("No auto-corrections available. Use Option 1.")
                            else:
                                _tdq_fix_keys = {}
                                for _tfi_idx, _tfi in enumerate(_tdq_fixable):
                                    _tfkey = f"tab_apply_{_tfi['col']}_{_tfi['check'].replace(' ','_')}"
                                    _tsugg = CORRECTION_SUGGESTIONS.get(_tfi["check"],{}).get("suggestion","")
                                    _tdq_fix_keys[_tfkey] = st.checkbox(
                                        f"{_tfi['check']} · `{_tfi['col']}`",
                                        key=_tfkey, value=True, help=_tsugg)
                                if _tdq_manual_only:
                                    st.caption(
                                        f"ℹ️ {len(_tdq_manual_only)} issue(s) need manual review: "
                                        + ", ".join(f"`{i['col']}`" for i in _tdq_manual_only))
                                if st.button("🔧 Apply and save to database",
                                             key="tab_apply_fixes_btn", type="primary",
                                             use_container_width=True):
                                    import io as _tio
                                    _tcorr = data.copy()
                                    _tlog  = []
                                    for _tfi in _tdq_fixable:
                                        _tfkey = f"tab_apply_{_tfi['col']}_{_tfi['check'].replace(' ','_')}"
                                        if not _tdq_fix_keys.get(_tfkey, False):
                                            continue
                                        _tfn = CORRECTION_SUGGESTIONS.get(_tfi["check"],{}).get("fn","")
                                        _tc  = _tfi["col"]
                                        try:
                                            if _tfn in ("fix_flatline","fix_frozen_value_run"):
                                                _tr = fix_flatline(_tcorr, _tc)
                                            elif _tfn == "fix_isolated_spikes":
                                                _tr = fix_isolated_spikes(_tcorr, _tc)
                                            elif _tfn == "fix_physical_impossibles":
                                                _tr = fix_physical_impossibles(_tcorr, _tc)
                                            elif _tfn == "fix_missing_gaps":
                                                _tr = fix_missing_gaps(_tcorr)
                                            elif _tfn == "fix_duplicate_timestamps":
                                                _tr = fix_duplicate_timestamps(_tcorr)
                                            else:
                                                continue
                                            _tcorr = _tr["corrected_df"]
                                            _tlog.append(
                                                f"✅ {_tfi['check']} ({_tc}): {_tr['description']}")
                                        except Exception as _tfe:
                                            _tlog.append(
                                                f"❌ {_tfi['check']} ({_tc}): Failed — {_tfe}")
                                    _tcorr_buf  = _tio.BytesIO(_tcorr.to_csv().encode("utf-8"))
                                    _tcorr_name = f"{selected_id}_corrected.csv"
                                    _tcorr_buf.name = _tcorr_name
                                    _tsave = db.ingest_file(_tcorr_buf, selected_id)
                                    if _tsave.get("success"):
                                        for _te in _tlog:
                                            st.caption(_te)
                                        st.session_state["last_dq_report"]     = None
                                        st.session_state["last_multi_results"] = None
                                        st.session_state["_pending_analysis"]  = False
                                        st.session_state["_corrections_applied"] = True
                                        st.rerun()
                                    else:
                                        st.error(f"Save failed: {_tsave.get('error')}")

                    # Warnings-only: still show download button
                    elif _tdq_warns:
                        import io as _tdl_io2
                        _tdl_buf2 = _tdl_io2.BytesIO(data.to_csv().encode("utf-8"))
                        st.download_button(
                            "⬇️ Download data",
                            data=_tdl_buf2,
                            file_name=f"{selected_id}_original.csv",
                            mime="text/csv",
                            key="tab_dl_original_dq",
                            use_container_width=False,
                        )



        st.subheader("Recent readings")
        st.dataframe(data.tail(200), use_container_width=True, height=280)

        st.subheader("Descriptive statistics")
        st.dataframe(data[numeric_cols].describe().round(3), use_container_width=True)

        if numeric_cols:
            st.subheader("Sensor overview")

            # ── Auto-group columns by unit type ──────────────────────
            def _group_by_unit(cols):
                import re as _re
                groups = {
                    "Pressure": [],
                    "Temperature": [],
                    "Current": [],
                    "Power": [],
                    "Vibration / Velocity": [],
                    "Speed / Frequency": [],
                    "Flow / Volume": [],
                    "Voltage": [],
                    "Energy": [],
                    "Other": [],
                }
                # NOTE: Order matters — first match wins.
                # "Voltage" must be before "Current" so voltage_A_V
                # matches "voltage" and not a short current keyword.
                # "Energy" must be before "Power" so kwh columns don't
                # match "kw" from Power.
                # Short keywords (\u22643 chars) use word-boundary regex
                # to prevent partial matches (e.g. "_a" in "voltage_a_v").
                kw_map = [
                    ("Energy",              ["kwh","kw_h","energy","consumption"]),
                    ("Voltage",             ["voltage","volt"]),
                    ("Current",             ["current","amps","ampere"]),
                    ("Power",               ["power","kilowatt","watt","cos_phi","power_factor","pf"]),
                    ("Pressure",            ["pressure","press","_bar","_psi","_kpa","_mbar"]),
                    ("Temperature",         ["temp","temperature","_celsius","_fahrenheit","_degc","deg_c"]),
                    ("Vibration / Velocity",["vibration","vibr","velocity","mm_s","mm/s","acceleration","accel"]),
                    ("Speed / Frequency",   ["speed","rpm","rps","frequency","_hz"]),
                    ("Flow / Volume",       ["flow","volume","_m3","litre","liter","gpm","cfm"]),
                ]
                def _kw_match(keyword, col_lower):
                    """Match keyword in column name; use word-boundary
                    regex for short keywords to avoid false positives."""
                    if len(keyword) <= 3:
                        return bool(_re.search(
                            r"(?<![a-z])" + _re.escape(keyword) + r"(?![a-z])",
                            col_lower))
                    return keyword in col_lower

                for col in cols:
                    cl = col.lower()
                    placed = False
                    for group_name, keywords in kw_map:
                        if any(_kw_match(kw, cl) for kw in keywords):
                            groups[group_name].append(col)
                            placed = True
                            break
                    if not placed:
                        groups["Other"].append(col)
                return {k: v for k, v in groups.items() if v}

            _groups = _group_by_unit(numeric_cols)

            # Build per-column issue lookup for chart overlays
            _dq_now = st.session_state.get("last_dq_report") or {}
            _dq_issues_by_col = {}
            for _iss in (_dq_now.get("issues") or []):
                _c2 = _iss.get("col","")
                if _c2 not in _dq_issues_by_col:
                    _dq_issues_by_col[_c2] = []
                _dq_issues_by_col[_c2].append(_iss)

            # Severity colours for overlays
            _SEV_FILL = {"critical": "rgba(163,45,45,0.12)", "warning": "rgba(186,117,23,0.10)", "info": "rgba(24,95,165,0.08)"}
            _SEV_LINE = {"critical": "rgba(163,45,45,0.7)",  "warning": "rgba(186,117,23,0.6)",  "info": "rgba(24,95,165,0.5)"}
            _SEV_MARKER = {"critical": "red", "warning": "orange", "info": "blue"}

            import plotly.graph_objects as _go2
            for _grp_name, _grp_cols in _groups.items():
                st.markdown(f"**{_grp_name}**")
                _fig = _go2.Figure()

                # Data traces
                for _c in _grp_cols:
                    _fig.add_trace(_go2.Scatter(
                        x=data.index, y=data[_c],
                        name=_c, mode="lines", line=dict(width=1.5),
                        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>" + _c + ": %{y:.3f}<extra></extra>",
                    ))

                # Overlay DQ issue annotations for columns in this group
                _overlay_added = False
                for _c in _grp_cols:
                    for _iss in _dq_issues_by_col.get(_c, []):
                        _sev   = _iss.get("severity","warning")
                        _chk   = _iss.get("check","Issue")
                        _fill  = _SEV_FILL.get(_sev, "rgba(186,117,23,0.10)")
                        _line  = _SEV_LINE.get(_sev, "rgba(186,117,23,0.6)")
                        _mclr  = _SEV_MARKER.get(_sev, "orange")
                        _label = f"{_chk} ({_c})"

                        s_ts = _iss.get("start_ts")
                        e_ts = _iss.get("end_ts")
                        p_ts = _iss.get("point_ts")

                        if s_ts is not None:
                            # Shaded region for range issues
                            _x1 = str(e_ts) if e_ts is not None else str(data.index.max())
                            _fig.add_vrect(
                                x0=str(s_ts), x1=_x1,
                                fillcolor=_fill,
                                line=dict(color=_line, width=1, dash="dot"),
                                annotation_text=_label,
                                annotation_position="top left",
                                annotation=dict(
                                    font=dict(size=9, color=_mclr),
                                    bgcolor="rgba(255,255,255,0.7)",
                                ),
                                layer="below",
                            )
                            _overlay_added = True

                        if p_ts is not None:
                            # Use add_shape + add_annotation separately
                            # (add_vline annotation args are version-dependent)
                            _fig.add_shape(
                                type="line",
                                x0=str(p_ts), x1=str(p_ts),
                                y0=0, y1=1,
                                xref="x", yref="paper",
                                line=dict(color=_mclr, width=1.5, dash="dash"),
                            )
                            _fig.add_annotation(
                                x=str(p_ts),
                                y=0.98,
                                xref="x", yref="paper",
                                text=_label,
                                showarrow=False,
                                font=dict(size=9, color=_mclr),
                                bgcolor="rgba(255,255,255,0.75)",
                                bordercolor=_mclr,
                                borderwidth=1,
                                xanchor="left",
                                yanchor="top",
                            )
                            _overlay_added = True

                # Also overlay [timestamp] issues (gaps) on all charts
                for _iss in _dq_issues_by_col.get("[timestamp]", []):
                    _sev  = _iss.get("severity","warning")
                    _fill = _SEV_FILL.get(_sev, "rgba(186,117,23,0.10)")
                    _line = _SEV_LINE.get(_sev, "rgba(186,117,23,0.6)")
                    _mclr = _SEV_MARKER.get(_sev, "orange")
                    s_ts  = _iss.get("start_ts")
                    e_ts  = _iss.get("end_ts")
                    if s_ts is not None:
                        _fig.add_vrect(
                            x0=str(s_ts), x1=str(e_ts) if e_ts else str(data.index.max()),
                            fillcolor="rgba(100,100,100,0.08)",
                            line=dict(color="rgba(100,100,100,0.5)", width=1, dash="dot"),
                            annotation_text="Missing data",
                            annotation_position="top left",
                            annotation=dict(font=dict(size=9, color="grey")),
                            layer="below",
                        )

                if _overlay_added:
                    st.caption(
                        "\u26a0\ufe0f Shaded / marked regions indicate sensor data quality issues "
                        "(invalid or untrustworthy readings — not performance problems). "
                        "Trends and efficiency changes are separate findings from the AI analysis."
                    )

                _fig.update_layout(
                    title=dict(text=_grp_name, font=dict(size=13)),
                    xaxis_title="Time",
                    yaxis_title=", ".join(_grp_cols[:2]) + (" ..." if len(_grp_cols) > 2 else ""),
                    height=300,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=40, r=20, t=40, b=40),
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="top", y=-0.25,
                                xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
                    font=dict(size=11),
                )
                st.plotly_chart(_fig, use_container_width=True)


# ------------------------------------------------------------------ #
# TAB 2 — Analysis
# ------------------------------------------------------------------ #

with tab_analysis:
    if data is None or data.empty:
        st.info("Upload data first (sidebar), then run an analysis.")
    else:
        # ── Pump physics viability check ─────────────────────────────
        _mtype = machine_info.get("machine_type","")
        _is_pump = _mtype == "Centrifugal Pump" or any(kw in _mtype.lower() for kw in ["pump","centrifugal pump","water pump"])
        if _is_pump and data is not None and not data.empty and PUMP_PHYSICS_AVAILABLE:
            _phase_info = detect_phase(data)
            _phases = _phase_info.get("phases",[])
            if _phases:
                with st.expander(f"\U0001f527 Pump physics — Phase {max(_phases)} available", expanded=False):
                    _pcols = _phase_info["cols"]
                    # Show detected phases
                    _phase_labels = {
                        1: "Power Baseline",
                        2: "Hydraulic Efficiency",
                        3: "Full Duty Point",
                        4: "Thermodynamic Efficiency",
                        5: "Mechanical Signature",
                    }
                    _phase_colours = {1:"\U0001f535",2:"\U0001f7e2",3:"\U0001f7e3",4:"\U0001f7e0",5:"\U0001f7e2"}
                    _col_a, _col_b = st.columns(2)
                    for _ph in [1,2,3,4,5]:
                        _active = _ph in _phases
                        _icon = "\u2705" if _active else "\u274c"
                        _col = _col_a if _ph <= 3 else _col_b
                        _col.caption(f"{_icon} Phase {_ph}: {_phase_labels[_ph]}")

                    # Phase 4 viability check
                    if _phase_info["has_pressure"] and _phase_info["has_temp"]:
                        st.markdown("---")
                        st.markdown("**Phase 4 — Thermodynamic method viability:**")
                        _viab = check_phase4_viability(data, _pcols)
                        _lvl = _viab.get("level","not_viable")
                        if _lvl == "viable":
                            st.success(_viab["message"])
                        elif _lvl == "marginal":
                            st.warning(_viab["message"])
                        else:
                            st.error(_viab["message"])
                        if _viab.get("recommendation"):
                            st.info(f"\U0001f4a1 {_viab['recommendation']}")
                    elif _phase_info["has_pressure"] and not _phase_info["has_temp"]:
                        st.markdown("---")
                        st.caption(
                            "\u2139\ufe0f Phase 4 viability check: add fluid inlet and outlet "
                            "temperature columns (keywords: temp_in, temp_out) to assess "
                            "whether the thermodynamic method is viable for this pump."
                        )

                    # ── Pump curve finder ─────────────────────────────────
                    if CURVE_FINDER_AVAILABLE:
                        st.markdown("---")
                        st.markdown("**\U0001f4c8 Performance curve sourcing**")
                        _api_key = os.getenv("ANTHROPIC_API_KEY","")
                        _cf_mfr  = st.text_input(
                            "Pump manufacturer",
                            placeholder="e.g. Grundfos, KSB, Wilo, Sulzer",
                            key="cf_manufacturer",
                        )
                        _cf_model = st.text_input(
                            "Pump model number (from nameplate)",
                            placeholder="e.g. CR 32-3, Etanorm 040-025-200",
                            key="cf_model",
                        )
                        _cf_flow  = st.number_input("Rated flow (m\u00b3/h)", min_value=0.0, value=0.0, key="cf_flow")
                        _cf_head  = st.number_input("Rated head (m)", min_value=0.0, value=0.0, key="cf_head")

                        if _cf_mfr:
                            _mfr_info = get_manufacturer_info(_cf_mfr)
                            if _mfr_info:
                                st.caption(
                                    f"\u2705 {_mfr_info['name']} is in the manufacturer library. "
                                    f"Will try product selector first.  \n"
                                    f"\U0001f517 Manual lookup: [{_mfr_info['selector_url']}]({_mfr_info['selector_url']})"
                                )
                            else:
                                st.caption(
                                    f"\u2139\ufe0f {_cf_mfr} is not in the known manufacturer library. "
                                    f"Will use web search only."
                                )

                        if st.button(
                            "\U0001f50d Search for pump curves",
                            key="search_curves_btn",
                            type="primary",
                            disabled=not (_cf_mfr and _cf_model and _cf_flow > 0 and _cf_head > 0 and _api_key),
                        ):
                            with st.spinner(
                                f"Searching for {_cf_mfr} {_cf_model} curves... "
                                "Trying manufacturer site first, then web search."
                            ):
                                _result = search_and_extract_curves(
                                    _cf_mfr, _cf_model, _cf_flow, _cf_head, _api_key
                                )
                            st.session_state["_curve_result"] = _result

                        _cr = st.session_state.get("_curve_result")
                        if _cr:
                            _method_labels = {
                                "manufacturer_site": "\u2705 Found on manufacturer site",
                                "web_search":        "\U0001f310 Found via web search",
                                "not_found":         "\u274c Not found automatically",
                            }
                            _mlabel = _method_labels.get(_cr.get("method",""), _cr.get("method",""))
                            if _cr.get("success"):
                                st.success(f"{_mlabel}")
                                st.info(_cr.get("message",""))
                                _hq = _cr.get("hq_points",[])
                                if _hq:
                                    # Validate points
                                    _val = validate_hq_points(_hq, _cf_flow, _cf_head)
                                    if _val["warnings"]:
                                        for _w in _val["warnings"]:
                                            st.warning(f"\u26a0\ufe0f {_w}")
                                    st.markdown(f"**{len(_hq)} H-Q data points extracted — please review:**")
                                    import pandas as _pd2
                                    _hq_df = _pd2.DataFrame(_hq)
                                    st.dataframe(_hq_df, use_container_width=True, height=200)
                                    if _cr.get("eta_points"):
                                        with st.expander("Efficiency curve points", expanded=False):
                                            st.dataframe(_pd2.DataFrame(_cr["eta_points"]), use_container_width=True)
                                # Source reference
                                _ref = _cr.get("source_ref",{})
                                if _ref:
                                    with st.expander("\U0001f4cb Source reference (included in report)", expanded=False):
                                        st.code(format_source_reference_text(_ref))
                                        st.caption(
                                            f"Method: {_ref.get('method','')}  \u00b7  "
                                            f"Confidence: {_ref.get('confidence','')}  \u00b7  "
                                            f"Retrieved: {_ref.get('retrieved_date','')}  \n"
                                            + (f"URL: {_ref.get('url','')}" if _ref.get('url') else "")
                                        )
                                    # Save to session for report
                                    st.session_state["_pump_curve_source_ref"] = _ref
                                    st.warning(
                                        "\u26a0\ufe0f **Review required:** Verify these points match "
                                        "your specific pump variant and impeller diameter before use."
                                    )
                            else:
                                st.error(f"{_mlabel}")
                                st.info(_cr.get("message",""))
                                _mfr_info2 = get_manufacturer_info(_cf_mfr)
                                if _mfr_info2:
                                    st.markdown(
                                        f"\U0001f517 Try manually at "
                                        f"[{_mfr_info2['name']} product selector]"
                                        f"({_mfr_info2['selector_url']}):  \n"
                                        f"{_mfr_info2['datasheet_hint']}"
                                    )
            else:
                with st.expander("\U0001f527 Pump physics — no physics columns detected", expanded=False):
                    st.info(
                        "Machine type is pump but no physics-relevant columns were detected.  \n"
                        "Add columns with these keywords to activate pump physics:  \n"
                        "- **Phase 1:** `current`, `power_kw`  \n"
                        "- **Phase 2:** + `flow`  \n"
                        "- **Phase 3:** + `suction_pressure`, `discharge_pressure`  \n"
                        "- **Phase 4:** + `temp_in`, `temp_out`  \n"
                        "- **Phase 5:** + `vibration`, `rpm`"
                    )

        left, right = st.columns([1, 3])

        with left:
            selected_analyses = []

            # ── Primary analyses ──────────────────────────────────
            with st.expander("Primary analyses", expanded=True):
                primary_options = [
                    ("Overall Health Assessment",      "Comprehensive health score, KPIs, anomalies and narrative"),
                    ("Trend & Drift Analysis",          "How fast is each parameter degrading over time?"),
                    ("Anomaly Detection",               "Full catalogue of outliers and spikes with timestamps"),
                    ("Operational Schedule Compliance", "Running outside permitted hours or days"),
                ]
                for name, desc in primary_options:
                    if st.checkbox(name, value=False, key=f"chk_{name}", help=desc):
                        selected_analyses.append(name)

            # ── Additional analytics ───────────────────────────
            with st.expander("Additional analytics", expanded=False):
                st.caption("Use for focused investigation of a specific relationship or pattern.")
                additional_options = [
                    ("Correlation Analysis",        "Statistical relationships and dependencies between parameters"),
                    ("Parameter Distribution",      "Statistical spread, skewness and operating mode shape"),
                    ("Cross-Parameter Comparison",  "Ratios and balance between physically related parameters"),
                ]
                for name, desc in additional_options:
                    if st.checkbox(name, value=False, key=f"chk_{name}", help=desc):
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

            # ── Normal operation baseline period ──────────────────────
            with st.expander("\U0001f4cf Normal operation baseline period", expanded=False):
                st.markdown(
                    "Define the period when the machine was running **normally** — "
                    "ideally after commissioning and before any faults. "
                    "This period is used to calculate control chart limits (UCL/LCL)."
                )
                _use_baseline = st.checkbox(
                    "Specify normal operation baseline period",
                    key="use_baseline_period", value=False
                )
                if _use_baseline:
                    _total_days = (max_d - min_d).days
                    _rec_min_days = 14
                    _rec_ideal_days = max(21, int(_total_days * 0.20))
                    st.caption(
                        f"\U0001f4cb **Minimum recommended:** {_rec_min_days} days · "
                        f"**Ideal:** {_rec_ideal_days} days (20% of your dataset = "
                        f"{_rec_ideal_days} days) · "
                        f"Must be within {min_d} – {max_d}"
                    )
                    _bl_cols = st.columns(2)
                    _bl_start = _bl_cols[0].date_input(
                        "Baseline start",
                        value=min_d,
                        min_value=min_d,
                        max_value=max_d,
                        key="baseline_start",
                    )
                    _bl_end = _bl_cols[1].date_input(
                        "Baseline end",
                        value=min(min_d + __import__("datetime").timedelta(days=_rec_ideal_days), max_d),
                        min_value=min_d,
                        max_value=max_d,
                        key="baseline_end",
                    )
                    _bl_days = (_bl_end - _bl_start).days
                    if _bl_days < _rec_min_days:
                        st.warning(
                            f"\u26a0\ufe0f Baseline period is only {_bl_days} day(s). "
                            f"A minimum of {_rec_min_days} days is recommended for reliable "
                            f"control limits. Short baselines produce wide UCL/LCL bands that "
                            f"may miss real anomalies."
                        )
                    elif _bl_days < _rec_ideal_days:
                        st.info(
                            f"\u2139\ufe0f {_bl_days} days selected. Acceptable — "
                            f"{_rec_ideal_days} days or more is ideal for stable limits."
                        )
                    else:
                        st.success(
                            f"\u2705 {_bl_days}-day baseline period. "
                            f"This is sufficient for reliable control limits."
                        )
                    baseline_period = (_bl_start, _bl_end)
                else:
                    _auto_days = max(14, int((max_d - min_d).days * 0.20))
                    _auto_end  = min_d + __import__("datetime").timedelta(days=_auto_days)
                    baseline_period = (min_d, _auto_end)
                    st.info(
                        f"\u2139\ufe0f **Default baseline will be used:** first {_auto_days} days "
                        f"of your data ({min_d} – {_auto_end}).  \n"
                        f"For more accurate control limits, tick the checkbox above and specify "
                        f"a period when the machine was operating normally."
                    )
                st.session_state["baseline_period"] = baseline_period

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
                    _data_now = data  # use already-loaded active file data
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
                # Clear old results so they don't show during DQ gate
                st.session_state["last_multi_results"] = None
                st.session_state["_pending_analysis"]  = False
                st.session_state["_corrected_csv"]     = None
                st.session_state["_correction_log"]    = []
                logs_text    = db.get_logs_text(selected_id)
                analyzer_obj = Analyzer()
                all_results  = {}
                progress_bar = st.progress(0)
                status_text  = st.empty()

                # Filter data by date range for display consistency
                from analyzer import Analyzer as _A
                filtered_data = _A._filter_by_date(data, date_range)

                # Store schedule for use in energy display
                st.session_state["_last_schedule"]   = schedule or {}
                st.session_state["_machine_desc"]    = machine_info.get("description", "")

                # ── Pump physics pre-run ───────────────────────────────
                _pump_physics_summary = ""
                if PUMP_PHYSICS_AVAILABLE:
                    _mtype_check = machine_info.get("machine_type","")
                    _is_pump_check = (_mtype_check == "Centrifugal Pump" or
                                      any(k in _mtype_check.lower() for k in ["pump","centrifugal"]))
                    if _is_pump_check and data is not None and not data.empty:
                        _phys_result = run_all_phases(
                            data,
                            machine_info.get("description",""),
                            hq_curve=st.session_state.get("_pump_hq_curve"),
                            bearing_freqs=st.session_state.get("_pump_bearing_freqs"),
                            baseline_period=st.session_state.get("baseline_period"),
                        )
                        _pump_physics_summary = _phys_result.get("summary","")
                        st.session_state["_pump_physics_result"] = _phys_result

                # ── Use DQ already computed in Data tab ───────────────────
                _dq_ctx = st.session_state.get("_dq_ctx", "")
                _dq_stored = st.session_state.get("last_dq_report") or {}
                _dq_crits_now = [x for x in _dq_stored.get("issues", []) if x["severity"] == "critical"]
                _pending_meta_now = {
                    "selected_analyses": selected_analyses,
                    "machine_info":      machine_info,
                    "date_range":        date_range,
                    "baseline_period":   st.session_state.get("baseline_period", (min_d, min_d)),
                    "extra_context":     (extra_context + "\n\n" + _pump_physics_summary).strip() if _pump_physics_summary else extra_context,
                    "schedule":          schedule,
                    "logs_text":         db.get_logs_text(selected_id),
                    "filtered_data":     filtered_data,
                }
                st.session_state["_pending_meta"] = _pending_meta_now
                if _dq_crits_now:
                    # Critical issues — show compact gate in Analysis tab
                    st.session_state["_pending_analysis"] = True
                    st.rerun()
                else:
                    # Clean data — run immediately
                    st.session_state["_pending_analysis"] = False
                    _run_analysis(_pending_meta_now, _dq_ctx, db, selected_id)


            # ── DQ gate — compact banner (full details in Data tab) ──────────
            if not DQ_AVAILABLE:
                st.warning("Data quality module not loaded — install data_checker.py")
            if st.session_state.get("_pending_analysis"):
                _dq_gate = st.session_state.get("last_dq_report") or {}
                _meta    = st.session_state.get("_pending_meta", {})
                _dq_ctx  = st.session_state.get("_dq_ctx", "")
                _crits   = [x for x in _dq_gate.get("issues", []) if x["severity"] == "critical"]

                # Option 2 acknowledged in Data tab — bypass gate and run
                if st.session_state.get("_dq_acknowledged"):
                    st.session_state["_dq_acknowledged"] = False
                    st.session_state["_pending_analysis"] = False
                    _run_analysis(_meta, _dq_ctx, db, selected_id)
                elif _crits:
                    st.error(
                        f"**{len(_crits)} critical data quality issue(s) detected in the sensor data.**  \n"
                        "Review the **Data quality** panel in the **Data** tab for details and "
                        "correction options. You can also acknowledge the issues and proceed."
                    )
                    for _gc_iss in _crits:
                        _gc_sev  = _gc_iss["severity"]
                        _gc_icon = "\u274c" if _gc_sev == "critical" else "\u26a0\ufe0f"
                        st.markdown(
                            f"- {_gc_icon} **{_gc_iss['check']}** "
                            f"\u00b7 `{_gc_iss['col']}` "
                            f"\u00b7 <span style=\"color:#888;font-size:0.9em\">"
                            f"{_gc_iss['affected_pct']}% affected</span>",
                            unsafe_allow_html=True)
                    st.markdown("")
                    _gc_ack = st.checkbox(
                        "I have reviewed the issues and want to proceed with analysis",
                        key="dq_ack_all_chk")
                    if _gc_ack:
                        if st.button("Continue Analysis", type="primary",
                                     key="dq_continue_btn", use_container_width=True):
                            st.session_state["_pending_analysis"] = False
                            _run_analysis(_meta, _dq_ctx, db, selected_id)
                else:
                    st.session_state["_pending_analysis"] = False
                    _run_analysis(_meta, _dq_ctx, db, selected_id)


            # Display results only when no pending DQ gate
            # Show results only when DQ gate is not active
            _show_results = not st.session_state.get("_pending_analysis", False)
            if _show_results:

                # ── Pump physics results panel ────────────────────────
                _phys_res = st.session_state.get("_pump_physics_result")
                if _phys_res and _phys_res.get("phases"):
                    _all_f = _phys_res.get("all_findings", [])
                    _n_crit_ph = sum(1 for f in _all_f if f.get("severity") == "critical")
                    _n_warn_ph = sum(1 for f in _all_f if f.get("severity") == "warning")
                    _ph_label  = (f"  ·  {_n_crit_ph} critical" if _n_crit_ph else "") +                                  (f"  ·  {_n_warn_ph} warning(s)" if _n_warn_ph else "") +                                  ("  ·  No issues" if not _all_f else "")
                    _ph_nums   = sorted(k for k in _phys_res["phases"].keys() if isinstance(k, int))
                    _ph_names  = {1:"Power Baseline",2:"Hydraulic Efficiency",
                                  3:"Full Duty Point",4:"Thermodynamic",5:"Mechanical",
                                  "TS":"Time-Segmented Events"}
                    _has_ts    = "TS" in _phys_res["phases"]
                    _ts_label  = " + Event Detection" if _has_ts else ""
                    with st.expander(
                        f"\U0001f527 Pump physics \u2014 Phases {_ph_nums}{_ts_label}{_ph_label}",
                        expanded=(_n_crit_ph > 0)
                    ):
                        for _ph_n, _ph_r in sorted(_phys_res["phases"].items(),
                                                    key=lambda x: (isinstance(x[0], str), x[0])):
                            _ph_display = f"Phase {_ph_n} \u2014 {_ph_r['name']}" if isinstance(_ph_n, int) else _ph_r.get("name", str(_ph_n))
                            st.markdown(f"**{_ph_display}**")
                            # Warnings
                            for _w in _ph_r.get("warnings", []):
                                st.warning(_w)
                            # Metrics table
                            _mets = _ph_r.get("metrics", {})
                            if _mets:
                                _met_cols = st.columns(min(len(_mets), 3))
                                for _mi, (_mk, _mv) in enumerate(_mets.items()):
                                    _met_cols[_mi % 3].metric(_mk, str(_mv))
                            # Findings
                            for _f in _ph_r.get("findings", []):
                                _sev = _f.get("severity","info")
                                _conf = _f.get("confidence","")
                                _icon = {
                                    "critical": "❌",
                                    "warning":  "⚠️",
                                    "info":     "ℹ️",
                                }.get(_sev, "•")
                                _bg  = {"critical":"#FFF0F0","warning":"#FFFBF0","info":"#EAF4FF"}.get(_sev,"#F8F8F8")
                                _bc  = {"critical":"#A32D2D","warning":"#BA7517","info":"#185FA5"}.get(_sev,"#888")
                                st.markdown(
                                    f'<div style="background:{_bg};border-left:4px solid {_bc};' +
                                    f'padding:8px 12px;margin-bottom:6px;border-radius:3px;">' +
                                    f'<span style="font-weight:700;color:{_bc}">{_icon} {_f["finding"]}</span>' +
                                    f' &nbsp;·&nbsp; <span style="color:#888;font-size:0.82em">Phase {_ph_n} · Confidence: {_conf}</span><br>' +
                                    f'<span style="font-size:0.87em">{_f["detail"]}</span></div>',
                                    unsafe_allow_html=True)
                            st.markdown("")

                dq = st.session_state.get("last_dq_report")
                if dq:
                    _sev_color = {"critical":"#A32D2D","warning":"#BA7517","info":"#185FA5"}
                    _sev_bg    = {"critical":"#FFF0F0","warning":"#FFFBF0","info":"#EAF4FF"}
                    _n_crit = dq["summary"]["critical"]
                    _n_warn = dq["summary"]["warning"]
                    _score  = dq["score"]
                    _label  = (f"  \u00b7  {_n_crit} critical" if _n_crit else "") + \
                              (f"  \u00b7  {_n_warn} warning(s)" if _n_warn else "") + \
                              ("  \u00b7  All checks passed" if not dq["issues"] else "")
                    with st.expander(f"Data quality check \u2014 score {_score}/100{_label}", expanded=(_n_crit>0)):
                        if not dq["issues"]:
                            st.success("All sensor data quality checks passed. Data is suitable for analysis.")
                        else:
                            for _iss2 in dq["issues"]:
                                _sev2  = _iss2["severity"]
                                _icon2 = {"critical":"\u274c","warning":"\u26a0\ufe0f","info":"\u2139\ufe0f"}.get(_sev2,"\u2022")
                                _bg2   = _sev_bg.get(_sev2,"#F8F8F8")
                                _fc2   = _sev_color.get(_sev2,"#333")
                                st.markdown(
                                    f'<div style="background:{_bg2};border-left:4px solid {_fc2};padding:8px 12px;margin-bottom:6px;border-radius:2px;">' +
                                    f'<span style="font-weight:600;color:{_fc2}">{_icon2} {_iss2["check"]}</span>' +
                                    f' &nbsp;\u00b7&nbsp; <code>{_iss2["col"]}</code>' +
                                    f' &nbsp;\u00b7&nbsp; <span style="color:#888;font-size:0.85em">{_iss2["affected_pct"]}% affected</span><br>' +
                                    f'<span style="font-size:0.88em">{_iss2["detail"]}</span></div>',
                                    unsafe_allow_html=True)
                    st.markdown("")

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
                    if not st.session_state.get("_pending_analysis"):
                        st.markdown(
                            "_Select one or more analysis types and press **Analyze** to generate insights._"
                        )


            # PDF download button — shown when results exist
            if st.session_state.get("last_multi_results") and REPORT_AVAILABLE:
                st.divider()
                if st.button("Download PDF report", type="secondary", use_container_width=True):
                    with st.spinner("Generating PDF report..."):
                        try:
                            _data = st.session_state.get("last_data")
                            _data_info = None
                            if _data is not None:
                                _data_info = {
                                    "rows": len(_data),
                                    "columns": len(_data.select_dtypes(include="number").columns),
                                }
                            # ── Build data_sources for report traceability ──
                            _file_info_list = db.get_file_info(selected_id)
                            _active_fname   = st.session_state.get(f"active_file_{selected_id}", "")
                            _active_fi      = next(
                                (f for f in _file_info_list if f["file"] == _active_fname),
                                _file_info_list[-1] if _file_info_list else {}
                            )
                            _logs_list      = db.get_logs(selected_id) if hasattr(db, "get_logs") else []
                            _dq_report      = st.session_state.get("last_dq_report") or {}
                            _corr_log       = st.session_state.get("_correction_log", [])
                            _ml_tier        = st.session_state.get("last_multi_results", {})
                            _tier_label     = ""
                            for _at, _ins in (_ml_tier or {}).items():
                                if isinstance(_ins, dict) and "_ml_tier" in _ins:
                                    _tier_label = _ins.get("tier_label","")
                                    break
                            _bl_period      = st.session_state.get("baseline_period")

                            # ── Parse machine description into structured nameplate data ──
                            def _parse_desc_to_nameplate(desc: str, mtype: str) -> dict:
                                """Extract nameplate parameters from machine description text."""
                                import re as _re
                                _is_pump = any(k in mtype.lower() for k in ["pump","centrifugal"])
                                _motor, _pump_np, _sys = {}, {}, {}
                                lines = desc.replace("\r","").split("\n")
                                for line in lines:
                                    line = line.strip()
                                    if not line or line.startswith("==="):
                                        continue
                                    # Common key: value patterns
                                    m = _re.match(r"^([^:=]+)[:\s=]+(.+)$", line)
                                    if not m:
                                        continue
                                    key   = m.group(1).strip().lower()
                                    value = m.group(2).strip()
                                    # Motor fields
                                    if any(k in key for k in ["rated power","motor power","kw","hp"]):
                                        _motor["Rated power"] = value
                                    elif any(k in key for k in ["full load amp","fla","rated current","amps"]):
                                        _motor["Full load amps (FLA)"] = value
                                    elif any(k in key for k in ["voltage","supply volt"]):
                                        _motor["Supply voltage"] = value
                                    elif any(k in key for k in ["power factor","cos"]):
                                        _motor["Power factor"] = value
                                    elif any(k in key for k in ["motor speed","motor rpm","rated speed","synchronous"]):
                                        _motor["Rated speed"] = value
                                    elif any(k in key for k in ["pole","poles"]):
                                        _motor["Number of poles"] = value
                                    elif any(k in key for k in ["ie class","efficiency class"]):
                                        _motor["IE efficiency class"] = value
                                    elif any(k in key for k in ["insulation","insul class"]):
                                        _motor["Insulation class"] = value
                                    elif any(k in key for k in ["motor efficiency","motor eta","\u03b7_motor"]):
                                        _motor["Motor efficiency"] = value
                                    elif any(k in key for k in ["commissioning power","baseline power"]):
                                        _motor["Commissioning power"] = value
                                    # Pump fields
                                    elif any(k in key for k in ["rated flow","design flow","q_rated"]):
                                        _pump_np["Rated flow"] = value
                                    elif any(k in key for k in ["rated head","design head","h_rated","total head"]):
                                        _pump_np["Rated head"] = value
                                    elif any(k in key for k in ["pump efficiency","\u03b7_pump","pump eta"]):
                                        _pump_np["Pump efficiency at BEP"] = value
                                    elif any(k in key for k in ["bep flow","q_bep"]):
                                        _pump_np["BEP flow"] = value
                                    elif any(k in key for k in ["pump speed","pump rpm"]):
                                        _pump_np["Pump rated speed"] = value
                                    elif any(k in key for k in ["impeller","vane","blade"]):
                                        _pump_np["Impeller details"] = value
                                    elif any(k in key for k in ["npsh","suction head"]):
                                        _pump_np["NPSH required"] = value
                                    elif any(k in key for k in ["seal type","seal"]):
                                        _pump_np["Seal type"] = value
                                    elif any(k in key for k in ["stage","stages"]):
                                        _pump_np["Stage count"] = value
                                    # System fields
                                    elif any(k in key for k in ["fluid","medium","liquid"]):
                                        _sys["Fluid type"] = value
                                    elif any(k in key for k in ["coupling","drive"]):
                                        _sys["Coupling / drive type"] = value
                                    elif any(k in key for k in ["elevation","\u0394z","delta z"]):
                                        _sys["Elevation difference"] = value
                                    elif any(k in key for k in ["commission date","install date","year"]):
                                        _sys["Commissioning date"] = value
                                    elif any(k in key for k in ["bearing","de bearing","nde bearing"]):
                                        _sys["Bearing details"] = value
                                    elif any(k in key for k in ["operating temp","fluid temp"]):
                                        _sys["Fluid operating temperature"] = value
                                return {"motor": _motor, "pump": _pump_np, "system": _sys}

                            _desc_text = machine_info.get("description","")
                            _np_data   = _parse_desc_to_nameplate(_desc_text, machine_info.get("machine_type",""))
                            _eng_notes = st.session_state.get("_pending_meta",{}).get("extra_context","") or ""

                            # All files stored for this machine (not just active)
                            _all_files = [
                                {
                                    "filename":    f.get("file","—"),
                                    "file_type":   "CSV" if str(f.get("file","")).endswith(".csv") else "Excel",
                                    "file_status": "Corrected" if "_corrected" in str(f.get("file","")).lower() else "Original",
                                    "rows":        f.get("rows",0),
                                    "ingested_at": str(f.get("ingested_at","—"))[:19],
                                    "active":      f.get("file","") == _active_fi.get("file",""),
                                }
                                for f in _file_info_list
                            ]

                            _data_sources = {
                                "file_info": {
                                    "filename":    _active_fi.get("file","—"),
                                    "file_type":   "CSV" if str(_active_fi.get("file","")).endswith(".csv") else "Excel",
                                    "file_status": "Corrected (auto-correction applied)" if "_corrected" in str(_active_fi.get("file","")).lower() else "Original upload",
                                    "ingested_at": str(_active_fi.get("ingested_at","—"))[:19],
                                    "uploaded_by": "Engineer",
                                },
                                "all_files":       _all_files,
                                "baseline_period": _bl_period,
                                "nameplate":       _np_data,
                                "engineer_notes":  _eng_notes,
                                "pump_curve_source_ref": st.session_state.get("_pump_curve_source_ref"),
                                "maintenance_logs": [
                                    {"filename": _l.get("filename","—"),
                                     "method":   _l.get("method","—"),
                                     "uploaded_at": _l.get("uploaded_at","—")}
                                    for _l in (_logs_list or [])
                                ],
                                "data_quality": {
                                    "score":               _dq_report.get("score"),
                                    "issues":              _dq_report.get("issues",[]),
                                    "corrections_applied": _corr_log,
                                },
                                "engine": {
                                    "model":            "Claude Sonnet 4 (Anthropic)",
                                    "tier_label":       _tier_label,
                                    "analysis_types":   list((st.session_state.get("last_multi_results") or {}).keys()),
                                    "analysis_date":    datetime.now().strftime("%d %b %Y %H:%M"),
                                    "platform_version": "AI Based Machine Analytics v1.0",
                                    "baseline_used":    f"{_bl_period[0]} to {_bl_period[1]}" if _bl_period else "Default (first 20% of dataset)",
                                },
                            }

                            pdf_bytes = generate_report(
                                machine_info=machine_info,
                                multi_results=st.session_state["last_multi_results"],
                                date_range=date_range,
                                data_info=_data_info,
                                data_sources=_data_sources,
                            )
                            st.download_button(
                                label="Click here to save PDF",
                                data=pdf_bytes,
                                file_name=f"{selected_id}_analytics_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                        except Exception as e:
                            st.error(f"PDF generation failed: {e}")


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
