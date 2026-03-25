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
def get_services(_version=5):
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

        # Physics module status badge
        if machine_type:
            _phys = PHYSICS_MODULE_STATUS.get(machine_type)
            if _phys:
                _status, _detail, _icon = _phys
                if _status == "available":
                    st.success(f"{_icon} **Physics module available:** {machine_type} \u2014 {_detail}. Physics-based analytics will activate automatically when sensor columns are detected.")
                elif _status == "planned":
                    st.info(f"{_icon} **Physics module in development:** {machine_type} \u2014 {_detail}. AI statistical analytics (Layer 1) will run. Physics module coming in a future update.")
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

        # Show badge for unsupported drive types
        if drive_type and drive_type not in DRIVE_TYPES_SUPPORTED and drive_type not in ["-- Select drive type --", "Unknown", "Not applicable — static equipment", "Hydraulic coupling"]:
            st.warning(
                f"⚠️ **{drive_type}:** AI statistical analytics will run.  \n"
                f"Physics-based analytics for this drive type are not yet available. "
                f"Only induction motor drives are currently supported for physics modules."
            )
        elif drive_type in DRIVE_TYPES_SUPPORTED:
            st.success(
                f"✅ **Induction motor drive selected.** "
                f"Full physics-based analytics are supported for this drive type."
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

        # ── Specifications and notes ───────────────────────────
        st.markdown("**Specifications and notes**")
        _spec_tab1, _spec_tab2 = st.tabs(["✏️ Enter manually", "📎 Upload file"])

        machine_desc = ""
        with _spec_tab1:
            machine_desc = st.text_area(
                "Specifications",
                placeholder=(
                    "Rated power: 75 kW\n"
                    "Rated flow: 185 m\u00b3/h\n"
                    "Rated head: 45 m\n"
                    "Rated speed: 1480 RPM\n"
                    "Full load amps: 138 A\n"
                    "Motor efficiency: 94.5%\n"
                    "IE class: IE3\n"
                    "Commissioning date: 2024-01-15"
                ),
                height=130,
                help="Enter nameplate data, installation details, and any relevant notes.",
                label_visibility="collapsed",
            )

        with _spec_tab2:
            st.caption(
                "Upload a PDF, image, or text file containing the machine datasheet, "
                "nameplate photo, or specification document. "
                "The text will be extracted automatically and added to the machine profile."
            )
            _spec_file = st.file_uploader(
                "Upload specification file",
                type=["pdf", "png", "jpg", "jpeg", "webp", "txt", "csv"],
                key="reg_spec_file",
                label_visibility="collapsed",
                help="PDF datasheet, photo of nameplate, or text specification file.",
            )
            if _spec_file:
                _api_key_spec = os.getenv("ANTHROPIC_API_KEY","")
                if st.button(
                    "📤 Extract text from file",
                    key="extract_spec_btn",
                    type="secondary",
                    use_container_width=True,
                ):
                    with st.spinner("Extracting text from file…"):
                        try:
                            _extracted = _extract_spec_text(_spec_file, _api_key_spec)
                            st.session_state["_extracted_spec"] = _extracted
                        except Exception as _ex:
                            st.error(f"Extraction failed: {_ex}")

                _extracted_text = st.session_state.get("_extracted_spec","")
                if _extracted_text:
                    st.success(f"✅ Text extracted ({len(_extracted_text)} characters)")
                    machine_desc = st.text_area(
                        "Extracted text (review and edit before registering)",
                        value=_extracted_text,
                        height=160,
                        key="reg_extracted_desc",
                        label_visibility="collapsed",
                    )
                elif _spec_file:
                    st.info("Click ‘Extract text from file’ to read the content.")
                    machine_desc = ""


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
                _structured = ""
                if drive_type:
                    _structured += f"Drive type: {drive_type}\n"
                if _drive_extra:
                    _structured += _drive_extra
                full_desc = (_structured + machine_desc).strip() if _structured else machine_desc
                if thresh_text and thresh_text.strip():
                    full_desc = full_desc + "\n\n=== PARAMETER THRESHOLDS ===\n" + thresh_text.strip()
                db.register_machine(machine_id.strip(), machine_type.strip(), full_desc)
                _phys_msg = ("Physics module: active." if PHYSICS_MODULE_STATUS.get(machine_type,("","",""))[0]=="available" else "AI analytics will run.")
                st.success(f"Machine **{machine_id}** registered as **{machine_type}**.  \n{_phys_msg}")
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

        if st.button("Ingest", use_container_width=True):
            with st.spinner("Reading and storing data…"):
                result = db.ingest_file(uploaded_file, selected_id)
            if result["success"]:
                st.success(f"\u2713 {result['rows']:,} rows ingested")
                st.caption("Columns: " + ", ".join(result["columns"]))
                st.rerun()
            else:
                st.error(result["error"])

    file_info = db.get_file_info(selected_id)
    if file_info:
        st.caption(f"{len(file_info)} file(s) stored for this machine")
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
                            _fdf = db.get_data(selected_id)
                            _buf3 = _io3.StringIO()
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

            # ── Active file selector ─────────────────────────────────
            if len(file_info) > 1:
                st.markdown("---")
                st.markdown("**📂 Choose which file to use for analysis:**")
                _file_options = {
                    _f["file"]: (
                        f"🔧 {_f['file']} ({_f['rows']:,} rows · corrected)"
                        if "_corrected" in _f["file"].lower()
                        else f"📂 {_f['file']} ({_f['rows']:,} rows · original)"
                    )
                    for _f in file_info
                }
                _current_active = st.session_state.get(
                    f"active_file_{selected_id}",
                    file_info[-1]["file"]  # default: most recently ingested
                )
                # Ensure current active is still valid
                if _current_active not in _file_options:
                    _current_active = file_info[-1]["file"]
                _selected_file = st.selectbox(
                    "Active file for analysis",
                    options=list(_file_options.keys()),
                    format_func=lambda x: _file_options[x],
                    index=list(_file_options.keys()).index(_current_active),
                    key=f"active_file_select_{selected_id}",
                )
                if _selected_file != st.session_state.get(f"active_file_{selected_id}"):
                    st.session_state[f"active_file_{selected_id}"] = _selected_file
                    st.session_state["last_data"] = None  # force reload
                    st.session_state["last_multi_results"] = None
                    st.rerun()
                st.caption(
                    f"ℹ️ Analysis will use: **{_current_active}**. "
                    f"Change selection above to switch files."
                )
            if st.button("Delete ALL files for this machine", type="primary",
                         key="del_all_files", use_container_width=True):
                db.delete_all_files(selected_id)
                st.success("All data files deleted. Please re-upload.")
                st.rerun()

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

# Load data from active file if one is selected, otherwise load all files
_active_file = st.session_state.get(f"active_file_{selected_id}")
if _active_file and db.get_file_info(selected_id) and len(db.get_file_info(selected_id)) > 1:
    _file_data = db.get_data_from_file(selected_id, _active_file)
    data = _file_data if _file_data is not None else db.get_data(selected_id)
else:
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

            # ── Auto-group columns by unit type ──────────────────────
            def _group_by_unit(cols):
                groups = {
                    "Pressure": [],
                    "Temperature": [],
                    "Current / Power": [],
                    "Vibration / Velocity": [],
                    "Speed / Frequency": [],
                    "Flow / Volume": [],
                    "Energy": [],
                    "Other": [],
                }
                kw_map = [
                    ("Energy",              ["kwh","kw_h","energy","consumption"]),
                    ("Current / Power",     ["current","_a","amps","power_kw","kilowatt","watt","cos_phi","power_factor","pf"]),
                    ("Pressure",            ["pressure","press","_bar","_psi","_kpa","_mbar"]),
                    ("Temperature",         ["temp","temperature","_celsius","_fahrenheit","_c","_degc","deg_c"]),
                    ("Vibration / Velocity",["vibration","vibr","velocity","mm_s","mm/s","acceleration","accel"]),
                    ("Speed / Frequency",   ["speed","rpm","rps","frequency","_hz"]),
                    ("Flow / Volume",       ["flow","volume","_m3","litre","liter","gpm","cfm"]),
                    ("Voltage",             ["voltage","_v","volt"]),
                ]
                for col in cols:
                    cl = col.lower()
                    placed = False
                    for group_name, keywords in kw_map:
                        if any(kw in cl for kw in keywords):
                            groups[group_name].append(col)
                            placed = True
                            break
                    if not placed:
                        groups["Other"].append(col)
                return {k: v for k, v in groups.items() if v}

            _groups = _group_by_unit(numeric_cols)

            for _grp_name, _grp_cols in _groups.items():
                st.markdown(f"**{_grp_name}**")
                import plotly.graph_objects as _go2
                _fig = _go2.Figure()
                for _c in _grp_cols:
                    _fig.add_trace(_go2.Scatter(
                        x=data.index, y=data[_c],
                        name=_c, mode="lines", line=dict(width=1.5),
                        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>" + _c + ": %{y:.3f}<extra></extra>",
                    ))
                _fig.update_layout(
                    title=dict(text=_grp_name, font=dict(size=13)),
                    xaxis_title="Time",
                    yaxis_title=", ".join(_grp_cols[:2]) + (" ..." if len(_grp_cols) > 2 else ""),
                    height=280,
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

        def _run_analysis(meta, dq_ctx, db, selected_id, override_data=None):
            """Run the actual analysis loop and store results in session state."""
            _analyzer    = Analyzer()
            _all_results = {}
            _progress    = st.progress(0)
            _status      = st.empty()
            _sel         = meta["selected_analyses"]
            for _i, _atype in enumerate(_sel):
                _status.text(f"Running {_atype} ({_i+1} of {len(_sel)})...")
                # Use corrected data if provided, otherwise fetch from DB
                _data_to_use = override_data if override_data is not None                                else db.get_data(meta["machine_info"]["machine_id"])
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
                        )
                        _pump_physics_summary = _phys_result.get("summary","")
                        st.session_state["_pump_physics_result"] = _phys_result

                # ── Data quality check ─────────────────────────────────
                _dq_ctx = ""
                if data is not None and not data.empty:
                    _dq = run_data_quality_checks(data)
                    st.session_state["last_dq_report"]     = _dq
                    st.session_state["_dq_ctx"]            = format_quality_report_for_claude(_dq)
                    st.session_state["_pending_analysis"]  = True
                    st.session_state["_pending_meta"]      = {
                        "selected_analyses": selected_analyses,
                        "machine_info":      machine_info,
                        "date_range":        date_range,
                        "baseline_period":   st.session_state.get("baseline_period", (min_d, min_d)),
                        "extra_context":     (extra_context + "\n\n" + _pump_physics_summary).strip() if _pump_physics_summary else extra_context,
                        "schedule":          schedule,
                        "logs_text":         db.get_logs_text(selected_id),
                        "filtered_data":     filtered_data,
                    }
                    st.rerun()


            # ── DQ gate UI + analysis runner ──────────────────────────
            if not DQ_AVAILABLE:
                st.warning("Data quality module not loaded — install data_checker.py")
            if st.session_state.get("_pending_analysis"):
                _dq      = st.session_state.get("last_dq_report", {})
                _meta    = st.session_state.get("_pending_meta", {})
                _dq_ctx  = st.session_state.get("_dq_ctx", "")
                _crits   = [x for x in _dq.get("issues",[]) if x["severity"]=="critical"]
                _warns   = [x for x in _dq.get("issues",[]) if x["severity"]=="warning"]

                if _crits:
                    # ── Header ───────────────────────────────────────────────
                    st.error(
                        f"**Data quality check — {len(_crits)} critical issue(s) detected.**  \n"
                        "Please choose one of the three options below before continuing."
                    )

                    # ── Issue cards with suggestions ─────────────────────────
                    for _iss in _crits:
                        _check   = _iss["check"]
                        _col     = _iss["col"]
                        _sugg    = CORRECTION_SUGGESTIONS.get(_check, {}).get("suggestion", "Review and correct manually.")
                        st.markdown(
                            f'<div style="background:#FFF0F0;border-left:5px solid #A32D2D;padding:10px 14px;margin-bottom:4px;border-radius:3px;">' +
                            f'<span style="font-weight:700;color:#A32D2D">\u274c {_check}</span>' +
                            f' &nbsp;\u00b7&nbsp; <code>{_col}</code>' +
                            f' &nbsp;\u00b7&nbsp; <span style="color:#888;font-size:0.82em">{_iss["affected_pct"]}% affected</span><br>' +
                            f'<span style="font-size:0.87em;color:#444">{_iss["detail"]}</span></div>',
                            unsafe_allow_html=True)
                        if _sugg:
                            st.markdown(
                                f'<div style="background:#F0F4F8;border-left:3px solid #185FA5;padding:6px 12px;margin-bottom:10px;border-radius:2px;">' +
                                f'<span style="font-size:0.85em;color:#185FA5;font-weight:600">\U0001f4a1 Suggestion: </span>' +
                                f'<span style="font-size:0.85em;color:#333">{_sugg}</span></div>',
                                unsafe_allow_html=True)


                    # ── Three options ─────────────────────────────────────────
                    _opt1, _opt2 = st.columns(2)

                    with _opt1:
                        st.markdown(
                            '<div style="background:#EAF4FF;border:1.5px solid #185FA5;border-radius:6px;padding:14px 16px;min-height:80px;">' +
                            '<span style="font-weight:700;color:#185FA5;font-size:1em">\u2b50 Option 1 — Recommended</span><br>' +
                            '<span style="font-size:0.9em;color:#1A1A2E;">Correct the data manually and re-upload for accurate analysis.</span>' +
                            '</div>',
                            unsafe_allow_html=True)
                        st.markdown("")
                        with st.expander("\U0001f4cb How to correct the data", expanded=True):
                            st.markdown(
                                "1. Download your original data file  \n"
                                "2. Correct or remove the affected rows/columns  \n"
                                "3. In the **Data** tab, delete the existing uploaded file  \n"
                                "4. Re-upload the corrected file  \n"
                                "5. Press **Analyze** again"
                            )

                    with _opt2:
                        st.markdown(
                            '<div style="background:#FFF8F0;border:1.5px solid #BA7517;border-radius:6px;padding:14px 16px;min-height:80px;">' +
                            '<span style="font-weight:700;color:#BA7517;font-size:1em">\u26a0\ufe0f Option 2 — Ignore and continue</span><br>' +
                            '<span style="font-size:0.9em;color:#1A1A2E;">Acknowledge each issue and proceed. Analysis may be affected.</span>' +
                            '</div>',
                            unsafe_allow_html=True)
                        st.markdown("")
                        _ignored = {}
                        for _iss in _crits:
                            _key = f"dq_ack_{_iss['col']}_{_iss['check'].replace(' ','_')}"
                            _c1, _c2 = st.columns([0.12, 0.88])
                            with _c1:
                                _ignored[_key] = st.checkbox(" ", key=_key, value=False, label_visibility="collapsed")
                            with _c2:
                                _bc   = "#2E7D32" if _ignored[_key] else "#A32D2D"
                                _icon = "\u2705" if _ignored[_key] else "\u274c"
                                _lbl  = "Acknowledged" if _ignored[_key] else "Unacknowledged"
                                st.markdown(
                                    f'<span style="color:{_bc};font-size:0.88em;font-weight:600">{_icon} {_iss["check"]} &nbsp;·&nbsp; <code>{_iss["col"]}</code> &nbsp;·&nbsp; <i>{_lbl}</i></span>',
                                    unsafe_allow_html=True)
                        st.markdown("")
                        _all_acked = all(_ignored.values()) if _ignored else False
                        _n_remain  = sum(1 for v in _ignored.values() if not v)
                        if not _all_acked:
                            st.caption(f"{_n_remain} issue(s) still unacknowledged. Tick all to enable Continue Analysis.")
                        else:
                            st.success("All issues acknowledged.")
                            if st.button("Continue Analysis", type="primary", key="dq_continue_btn", use_container_width=True):
                                st.session_state["_pending_analysis"] = False
                                _run_analysis(_meta, _dq_ctx, db, selected_id)

                    # ── Option 3 — full width below ────────────────────────────
                    _fixable = [
                        i for i in _crits
                        if CORRECTION_SUGGESTIONS.get(i["check"], {}).get("auto", False)
                        and i["col"] != "[timestamp]"
                    ]
                    _manual_only = [
                        i for i in _crits
                        if not CORRECTION_SUGGESTIONS.get(i["check"], {}).get("auto", False)
                        or i["col"] == "[timestamp]"
                    ]
                    _raw_data = st.session_state.get("last_data") or                                 db.get_data(_meta["machine_info"]["machine_id"])

                    st.markdown("")
                    st.markdown(
                        '<div style="background:#F0FFF4;border:1.5px solid #2E7D32;border-radius:6px;padding:14px 16px;">' +
                        '<span style="font-weight:700;color:#2E7D32;font-size:1em">\U0001f527 Option 3 — Auto-correct and download</span><br>' +
                        '<span style="font-size:0.9em;color:#1A1A2E;">Apply suggested corrections automatically. Download the corrected file, re-upload and analyze.</span>' +
                        '</div>',
                        unsafe_allow_html=True)
                    st.markdown("")

                    if not _fixable:
                        st.info("No auto-corrections available for the detected issues. Use Option 1 to correct manually.")
                    elif _raw_data is None:
                        st.warning("Data not available for auto-correction.")
                    else:
                        st.caption(f"{len(_fixable)} auto-correction(s) available. Select which to apply:")
                        _fix_keys = {}
                        _fc1, _fc2 = st.columns(2)
                        for _fi_idx, _fi in enumerate(_fixable):
                            _fkey = f"apply_{_fi['col']}_{_fi['check'].replace(' ','_')}"
                            _sugg = CORRECTION_SUGGESTIONS.get(_fi["check"],{}).get("suggestion","")
                            _col_sel = _fc1 if _fi_idx % 2 == 0 else _fc2
                            with _col_sel:
                                _fix_keys[_fkey] = st.checkbox(
                                    f"{_fi['check']} · `{_fi['col']}`",
                                    key=_fkey, value=True, help=_sugg)

                        if _manual_only:
                            st.caption(f"\u2139\ufe0f {len(_manual_only)} issue(s) cannot be auto-corrected (manual review needed): " +
                                       ", ".join(f"`{i['col']}`" for i in _manual_only))

                        st.markdown("")
                        if st.button("\U0001f527 Apply selected corrections and prepare download",
                                     key="apply_fixes_btn", type="secondary", use_container_width=True):
                            import io as _io
                            _corrected = _raw_data.copy()
                            _log = []
                            for _fi in _fixable:
                                _fkey = f"apply_{_fi['col']}_{_fi['check'].replace(' ','_')}"
                                if not _fix_keys.get(_fkey, False):
                                    continue
                                _fn_name = CORRECTION_SUGGESTIONS.get(_fi["check"],{}).get("fn","")
                                _col     = _fi["col"]
                                try:
                                    if _fn_name in ("fix_flatline", "fix_frozen_value_run"):
                                        _res = fix_flatline(_corrected, _col)
                                    elif _fn_name == "fix_isolated_spikes":
                                        _res = fix_isolated_spikes(_corrected, _col)
                                    elif _fn_name == "fix_physical_impossibles":
                                        _res = fix_physical_impossibles(_corrected, _col)
                                    elif _fn_name == "fix_missing_gaps":
                                        _res = fix_missing_gaps(_corrected)
                                    elif _fn_name == "fix_duplicate_timestamps":
                                        _res = fix_duplicate_timestamps(_corrected)
                                    else:
                                        continue
                                    _corrected = _res["corrected_df"]
                                    _log.append(f"\u2705 {_fi['check']} ({_col}): {_res['description']}")
                                except Exception as _fe:
                                    _log.append(f"\u274c {_fi['check']} ({_col}): Failed — {_fe}")
                            _buf = _io.StringIO()
                            _corrected.to_csv(_buf)
                            st.session_state["_corrected_csv"]  = _buf.getvalue().encode("utf-8")
                            st.session_state["_correction_log"] = _log
                            st.session_state["_corrected_df"]   = _corrected
                            st.session_state["_confirm_pending"] = False
                            st.rerun()

                        if st.session_state.get("_corrected_csv"):
                            st.success("Corrections applied successfully.")
                            for _entry in st.session_state.get("_correction_log", []):
                                st.caption(_entry)
                            st.markdown("---")
                            st.markdown("**What would you like to do with the corrected data?**")
                            _dl_col, _use_col = st.columns(2)
                            with _dl_col:
                                _fname = f"{_meta['machine_info'].get('machine_id','machine')}_corrected.csv"
                                st.download_button(
                                    label="\u2b07\ufe0f Download corrected file",
                                    data=st.session_state["_corrected_csv"],
                                    file_name=_fname,
                                    mime="text/csv",
                                    key="dl_corrected_csv",
                                    use_container_width=True,
                                    help="Save corrected CSV to your computer"
                                )
                            with _use_col:
                                if st.button(
                                    "\u25b6\ufe0f Use corrected data for analysis now",
                                    key="use_corrected_btn",
                                    type="primary",
                                    use_container_width=True,
                                    help="Run analysis immediately using the corrected data"
                                ):
                                    st.session_state["_confirm_pending"] = True
                                    st.rerun()
                            # Confirmation dialog
                            # Also show download original button alongside download corrected
                            _raw_data_for_dl = st.session_state.get("last_data") or                                                db.get_data(_meta["machine_info"]["machine_id"])
                            if _raw_data_for_dl is not None:
                                import io as _io_orig
                                _orig_buf = _io_orig.StringIO()
                                _raw_data_for_dl.to_csv(_orig_buf)
                                _orig_fname = f"{_meta['machine_info'].get('machine_id','machine')}_original.csv"
                                st.download_button(
                                    label="\u2b07\ufe0f Download original file",
                                    data=_orig_buf.getvalue().encode("utf-8"),
                                    file_name=_orig_fname,
                                    mime="text/csv",
                                    key="dl_original_csv",
                                    use_container_width=True,
                                    help="Download the original unmodified data file"
                                )

                            # Confirmation dialog
                            if st.session_state.get("_confirm_pending"):
                                st.warning(
                                    "\u26a0\ufe0f **Confirm:** The corrected file will be saved to the "
                                    "database with a **_corrected** tag and used for this analysis.  \n"
                                    "The original file remains unchanged and available in File Management."
                                )
                                _conf1, _conf2 = st.columns(2)
                                with _conf1:
                                    if st.button("\u2705 Yes, save corrected and analyze",
                                                 key="confirm_yes_btn", type="primary",
                                                 use_container_width=True):
                                        _corrected_df = st.session_state.get("_corrected_df")
                                        if _corrected_df is not None:
                                            # Save corrected file to DB with _corrected tag
                                            import io as _io_save
                                            _machine_id  = _meta["machine_info"].get("machine_id","machine")
                                            _corr_fname  = f"{_machine_id}_corrected.csv"
                                            _save_buf    = _io_save.BytesIO(
                                                st.session_state["_corrected_csv"])
                                            _save_buf.name = _corr_fname
                                            _save_res    = db.ingest_file(_save_buf, selected_id)
                                            if _save_res.get("success"):
                                                st.session_state["_corrected_saved"] = _corr_fname
                                            # Run analysis with corrected data
                                            _meta["filtered_data"] = _corrected_df
                                            st.session_state["_pending_meta"]     = _meta
                                            st.session_state["_pending_analysis"] = False
                                            st.session_state["_confirm_pending"]  = False
                                            st.session_state["_corrected_csv"]    = None
                                            _run_analysis(_meta, _dq_ctx, db, selected_id,
                                                          override_data=_corrected_df)
                                with _conf2:
                                    if st.button("\u274c Cancel", key="confirm_no_btn",
                                                 use_container_width=True):
                                        st.session_state["_confirm_pending"] = False
                                        st.rerun()
                    # ── Warnings ───────────────────────────────────────────────
                    if _warns:
                        st.markdown("")
                        with st.expander(f"\u26a0\ufe0f {len(_warns)} additional warning(s) — informational only", expanded=False):
                            for _iss in _warns:
                                st.markdown(
                                    f'<div style="background:#FFFBF0;border-left:4px solid #BA7517;padding:8px 12px;margin-bottom:6px;border-radius:2px;">' +
                                    f'<span style="font-weight:600;color:#BA7517">\u26a0\ufe0f {_iss["check"]}</span>' +
                                    f' &nbsp;\u00b7&nbsp; <code>{_iss["col"]}</code><br>' +
                                    f'<span style="font-size:0.88em">{_iss["detail"]}</span></div>',
                                    unsafe_allow_html=True)


                else:
                    # No critical issues — run immediately
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
                    _ph_nums   = sorted(_phys_res["phases"].keys())
                    _ph_names  = {1:"Power Baseline",2:"Hydraulic Efficiency",
                                  3:"Full Duty Point",4:"Thermodynamic",5:"Mechanical"}
                    with st.expander(
                        f"🔧 Pump physics — Phases {_ph_nums}{_ph_label}",
                        expanded=(_n_crit_ph > 0)
                    ):
                        for _ph_n, _ph_r in sorted(_phys_res["phases"].items()):
                            st.markdown(f"**Phase {_ph_n} — {_ph_r['name']}**")
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
                            st.success("All data quality checks passed.")
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
                            _dq_report      = st.session_state.get("last_dq_report", {})
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
                                    "analysis_types":   list(st.session_state.get("last_multi_results",{}).keys()),
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
