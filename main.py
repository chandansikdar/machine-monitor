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
    from data_checker import run_data_quality_checks, format_quality_report_for_claude, check_timestamp_format
    DQ_AVAILABLE = True
except ImportError:
    DQ_AVAILABLE = False
    def run_data_quality_checks(df, **kw): return {"issues":[],"summary":{"total":0,"critical":0,"warning":0,"info":0},"passed":True,"score":100}
    def format_quality_report_for_claude(r): return ""
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

@st.cache_resource
def get_services(_version=2):
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

    st.divider()

    st.subheader("Upload data")
    uploaded_file = st.file_uploader(
        "CSV or Excel",
        type=["csv", "xlsx", "xls"],
        help="Any column named timestamp/date/time is auto-detected as the time axis.",
    )
    if uploaded_file:
        # ── Timestamp format pre-check ──────────────────────────────
        if DQ_AVAILABLE:
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
                    f"⚠️ **Timestamp format issue detected** in column `{_ts_check['col'] or 'unknown'}`  \n"
                    f"{_ts_check['suggestion']}"
                )
                if _ts_check["sample_raw"]:
                    st.caption(f"Sample raw values: {', '.join(_ts_check['sample_raw'][:3])}")
                if _ts_check.get("corrected_df") is not None:
                    st.success(
                        "✅ Timestamps can be auto-corrected to YYYY-MM-DD HH:MM:SS format.  \n"
                        "Click **Fix timestamps and ingest** to correct and load in one step, "
                        "or download the corrected file to keep a copy."
                    )
                    _ts_buf = _ts_check["corrected_df"].to_csv(index=False).encode("utf-8")
                    _fix_col, _dl_col = st.columns(2)
                    with _fix_col:
                        if st.button("✅ Fix timestamps and ingest", type="primary",
                                     key="fix_ts_ingest_btn", use_container_width=True):
                            with st.spinner("Correcting timestamps and ingesting…"):
                                import io as _io2
                                _fixed_buf = _io2.BytesIO(_ts_buf)
                                _fixed_buf.name = uploaded_file.name.rsplit(".",1)[0] + "_fixed.csv"
                                _res2 = db.ingest_file(_fixed_buf, selected_id)
                            if _res2["success"]:
                                st.success(
                                    f"✓ {_res2['rows']:,} rows ingested with corrected timestamps."
                                )
                                st.caption("Columns: " + ", ".join(_res2["columns"]))
                                st.rerun()
                            else:
                                st.error(f"Ingest failed: {_res2['error']}")
                    with _dl_col:
                        st.download_button(
                            label="⬇️ Download corrected file",
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
                    st.caption(f"✅ Timestamps OK · sample: {_ts_check['sample_raw'][0]}")

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
                        "extra_context":     extra_context,
                        "schedule":          schedule,
                        "logs_text":         db.get_logs_text(selected_id),
                        "filtered_data":     filtered_data,
                    }
                    st.rerun()


            # ── DQ gate UI + analysis runner ──────────────────────────
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
                            f'<span style="font-weight:700;color:#A32D2D">❌ {_check}</span>' +
                            f' &nbsp;\u00b7&nbsp; <code>{_col}</code>' +
                            f' &nbsp;\u00b7&nbsp; <span style="color:#888;font-size:0.82em">{_iss["affected_pct"]}% affected</span><br>' +
                            f'<span style="font-size:0.87em;color:#444">{_iss["detail"]}</span></div>',
                            unsafe_allow_html=True)
                        if _sugg:
                            st.markdown(
                                f'<div style="background:#F0F4F8;border-left:3px solid #185FA5;padding:6px 12px;margin-bottom:10px;border-radius:2px;">' +
                                f'<span style="font-size:0.85em;color:#185FA5;font-weight:600">💡 Suggestion: </span>' +
                                f'<span style="font-size:0.85em;color:#333">{_sugg}</span></div>',
                                unsafe_allow_html=True)


                    # ── Three options ─────────────────────────────────────────
                    _opt1, _opt2 = st.columns(2)

                    with _opt1:
                        st.markdown(
                            '<div style="background:#EAF4FF;border:1.5px solid #185FA5;border-radius:6px;padding:14px 16px;min-height:80px;">' +
                            '<span style="font-weight:700;color:#185FA5;font-size:1em">⭐ Option 1 — Recommended</span><br>' +
                            '<span style="font-size:0.9em;color:#1A1A2E;">Correct the data manually and re-upload for accurate analysis.</span>' +
                            '</div>',
                            unsafe_allow_html=True)
                        st.markdown("")
                        with st.expander("📋 How to correct the data", expanded=True):
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
                            '<span style="font-weight:700;color:#BA7517;font-size:1em">⚠️ Option 2 — Ignore and continue</span><br>' +
                            '<span style="font-size:0.9em;color:#1A1A2E;">Acknowledge each issue and proceed. Analysis may be affected.</span>' +
                            '</div>',
                            unsafe_allow_html=True)
                        st.markdown("")
                        _ignored = {}
                        for _iss in _crits:
                            _key = f"dq_ack_{_iss['col']}_{_iss['check'].replace(' ','_')}"
                            _c1, _c2 = st.columns([0.12, 0.88])
                            with _c1:
                                _ignored[_key] = st.checkbox("", key=_key, value=False)
                            with _c2:
                                _bc   = "#2E7D32" if _ignored[_key] else "#A32D2D"
                                _icon = "✅" if _ignored[_key] else "❌"
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
                        '<span style="font-weight:700;color:#2E7D32;font-size:1em">🔧 Option 3 — Auto-correct and download</span><br>' +
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
                            st.caption(f"ℹ️ {len(_manual_only)} issue(s) cannot be auto-corrected (manual review needed): " +
                                       ", ".join(f"`{i['col']}`" for i in _manual_only))

                        st.markdown("")
                        if st.button("🔧 Apply selected corrections and prepare download",
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
                                    _log.append(f"✅ {_fi['check']} ({_col}): {_res['description']}")
                                except Exception as _fe:
                                    _log.append(f"❌ {_fi['check']} ({_col}): Failed — {_fe}")
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
                                    label="⬇️ Download corrected file",
                                    data=st.session_state["_corrected_csv"],
                                    file_name=_fname,
                                    mime="text/csv",
                                    key="dl_corrected_csv",
                                    use_container_width=True,
                                    help="Save corrected CSV to your computer"
                                )
                            with _use_col:
                                if st.button(
                                    "▶️ Use corrected data for analysis now",
                                    key="use_corrected_btn",
                                    type="primary",
                                    use_container_width=True,
                                    help="Run analysis immediately using the corrected data"
                                ):
                                    st.session_state["_confirm_pending"] = True
                                    st.rerun()
                            # Confirmation dialog
                            if st.session_state.get("_confirm_pending"):
                                st.warning(
                                    "⚠️ **Confirm:** The corrected data will be used for analysis.  \n"
                                    "The original uploaded file is unchanged — this only affects the current session.  \n"
                                    "Download the corrected file too if you want to keep the corrections permanently."
                                )
                                _conf1, _conf2 = st.columns(2)
                                with _conf1:
                                    if st.button("✅ Yes, analyze with corrected data", key="confirm_yes_btn", type="primary", use_container_width=True):
                                        _corrected_df = st.session_state.get("_corrected_df")
                                        if _corrected_df is not None:
                                            # Override last_data and pending_meta with corrected df
                                            _meta["filtered_data"] = _corrected_df
                                            st.session_state["_pending_meta"]      = _meta
                                            st.session_state["_pending_analysis"]  = False
                                            st.session_state["_confirm_pending"]   = False
                                            st.session_state["_corrected_csv"]     = None
                                            _run_analysis(_meta, _dq_ctx, db, selected_id, override_data=_corrected_df)
                                with _conf2:
                                    if st.button("❌ Cancel", key="confirm_no_btn", use_container_width=True):
                                        st.session_state["_confirm_pending"] = False
                                        st.rerun()

                    # ── Warnings ───────────────────────────────────────────────
                    if _warns:
                        st.markdown("")
                        with st.expander(f"⚠️ {len(_warns)} additional warning(s) — informational only", expanded=False):
                            for _iss in _warns:
                                st.markdown(
                                    f'<div style="background:#FFFBF0;border-left:4px solid #BA7517;padding:8px 12px;margin-bottom:6px;border-radius:2px;">' +
                                    f'<span style="font-weight:600;color:#BA7517">⚠️ {_iss["check"]}</span>' +
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
                            pdf_bytes = generate_report(
                                machine_info=machine_info,
                                multi_results=st.session_state["last_multi_results"],
                                date_range=date_range,
                                data_info=_data_info,
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
