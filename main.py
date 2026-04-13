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
    integrity_gate,
    clean_samples,
    ingest_baseline,
    run_assessment,
    assessment_summary,
)

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


def _migrate_cleaning_report(cr_d: dict) -> CleaningReport:
    """Reconstruct CleaningReport from a dict, tolerating old field names.

    Old schema had:  n_after_running_mask, n_after_integrity, n_after_iqr (wrong position)
    New schema has:  n_after_load_precondition, n_after_start_transient,
                     n_after_user_filter, n_after_iqr  (n_cleaned = n_after_iqr)
    """
    valid_fields = {f.name for f in CleaningReport.__dataclass_fields__.values()}
    # Keep only fields that exist in the current dataclass
    filtered = {k: v for k, v in cr_d.items() if k in valid_fields}
    # Back-fill missing fields with best available value
    if "n_after_start_transient" not in filtered:
        # Old records had no start transient step — use load precondition count as proxy
        filtered["n_after_start_transient"] = filtered.get(
            "n_after_load_precondition", filtered.get("n_raw", 0)
        )
    if "n_after_iqr" not in filtered:
        # Old records stored final cleaned count as n_after_user_filter
        filtered["n_after_iqr"] = filtered.get("n_after_user_filter", 0)
    return CleaningReport(**filtered)


def baseline_from_dict(d: dict) -> BaselineMetadata:
    """Reconstruct BaselineMetadata from a plain dict (loaded from JSON)."""
    cr_d = d.get("cleaning_report")
    cleaning = _migrate_cleaning_report(cr_d) if cr_d else None

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
    Returns None if the minimum set of fields is unavailable.

    Hard requirements: pf_rated, eta_rated (both dimensionless and stable defaults exist).
    Soft defaults: v_nominal_phase=230, p_rated_shaft_kw=0, i_rated=0.
    When p_rated=0 or i_rated=0, integrity checks fall back to data-derived estimates
    and skip checks that require nameplate current bounds respectively.
    """
    desc = machine_info.get("description", "")
    em   = parse_electrical_meta(desc)
    # Minimum hard requirements — platform cannot compute anything meaningful without these
    hard_required = ["pf_rated", "eta_rated"]
    missing = [r for r in hard_required if r not in em]
    if missing:
        return None
    # Soft defaults for optional fields
    em.setdefault("v_nominal_phase",  230.0)   # standard Swiss/EU L-N voltage
    em.setdefault("p_rated_shaft_kw",   0.0)   # 0 triggers data-derived estimation
    em.setdefault("i_rated",            0.0)   # 0 skips current plausibility check
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
    power_unit: str = "W",
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
        f"power_unit: {power_unit}\n"
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


def scale_power_to_watts(df: pd.DataFrame, power_unit: str) -> pd.DataFrame:
    """If power columns are in kW, multiply by 1000 to convert to Watts.
    The electrical_diagnostics module always works internally in Watts.
    """
    if power_unit.lower() == "kw":
        df = df.copy()
        for col in ["phase_1_active_power", "phase_2_active_power", "phase_3_active_power"]:
            if col in df.columns:
                df[col] = df[col] * 1000.0
    return df


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


@st.cache_data(show_spinner="Running integrity checks…")
def run_integrity_checks(df_json: str, meta_json: str):
    """Vectorised §3.1 integrity checks — cached so they only run once per dataset.

    Parameters are JSON strings so st.cache_data can hash them.
    Returns a dict with failed DataFrame (JSON), fail_checks list, fail_reasons list.
    """
    import json as _json, io as _io
    df    = pd.read_json(_io.StringIO(df_json), orient="split")
    meta  = _json.loads(meta_json)

    # Note: Check 0 (non-numeric) runs in the caller before JSON conversion
    # because JSON serialisation silently drops non-numeric strings.
    # By the time data arrives here all measurement columns are numeric.
    fail_check  = pd.Series("", index=df.index)
    fail_reason = pd.Series("", index=df.index)

    v_nom       = float(meta.get("v_nominal_phase", 230))
    p_shaft_kw  = float(meta.get("p_rated_shaft_kw", 0))
    eta         = float(meta.get("eta_rated", 0.90))

    v1 = df["phase_1_voltage"]; v2 = df["phase_2_voltage"]; v3 = df["phase_3_voltage"]
    i1 = df["phase_1_current"]; i2 = df["phase_2_current"]; i3 = df["phase_3_current"]
    p1 = df["phase_1_active_power"]; p2 = df["phase_2_active_power"]; p3 = df["phase_3_active_power"]
    p_total = p1 + p2 + p3

    # Determine rated electrical input — fall back to data estimate if not saved
    p_rated_estimated = False
    if p_shaft_kw > 0 and eta > 0:
        p_rated = p_shaft_kw / eta            # kW electrical
    else:
        # Estimate: 95th percentile of p_total ÷ 0.95 (assume machine reaches ~95% rated)
        p95 = float(p_total[p_total > 0].quantile(0.95)) / 1000.0   # kW
        p_rated = p95 / 0.95 if p95 > 0 else 0.0
        p_rated_estimated = True

    run_thr = 0.05 * p_rated * 1000.0   # 5% of rated electrical input (W)
    running = p_total > run_thr

    # Check 1 — Voltage plausibility (skip V=0/NaN & I=0 simultaneously = powered down)
    v_lo = 0.85 * v_nom;  v_hi = 1.15 * v_nom;  v_max = 1.5 * v_nom
    for ph, v, i_x in [("1", v1, i1), ("2", v2, i2), ("3", v3, i3)]:
        v_missing = v.isna() | (v == 0)       # NaN or zero voltage
        powered   = ~(v_missing & (i_x == 0)) # powered-down = missing V AND zero I → skip

        # Missing voltage while current is flowing (sensor dropout / logging gap)
        c1_missing = powered & v_missing & (fail_check == "")
        fail_check  = fail_check.where(~c1_missing, "check_1_voltage_plausibility")
        fail_reason = fail_reason.where(~c1_missing,
            f"Phase {ph} voltage missing (null/zero) while current is flowing "
            f"- possible voltage sensor dropout or data logging gap")

        # Voltage present but below 50 V floor (channel failure or short)
        c1_floor = powered & ~v_missing & (v < 50.0) & (fail_check == "")
        fail_check  = fail_check.where(~c1_floor, "check_1_voltage_plausibility")
        fail_reason = fail_reason.where(~c1_floor,
            f"Phase {ph} voltage below 50 V floor (channel failure or short suspected)")

        # Voltage present but above 1.5x nominal (reference lead on phase conductor)
        c1_high = powered & ~v_missing & (v > v_max) & (fail_check == "")
        fail_check  = fail_check.where(~c1_high, "check_1_voltage_plausibility")
        fail_reason = fail_reason.where(~c1_high,
            f"Phase {ph} voltage above 1.5x V_nominal ({v_max:.0f} V) "
            f"- reference lead may be on phase conductor")

        # Voltage outside +-15% nominal band
        c1_range = (powered & ~v_missing
                    & ~v.between(v_lo, v_hi)
                    & (fail_check == "")
                    & ~c1_floor & ~c1_high)
        fail_check  = fail_check.where(~c1_range, "check_1_voltage_plausibility")
        fail_reason = fail_reason.where(~c1_range,
            f"Phase {ph} voltage outside +-15% nominal band "
            f"[{v_lo:.0f}, {v_hi:.0f}] V")

    # Check 2 — Current plausibility (running samples only)
    # Skip entirely if i_rated is 0 or missing — cannot compute meaningful bounds
    i_rated = float(meta.get("i_rated", 0))
    if i_rated > 0:
        i_lo = 0.005 * i_rated;  i_hi = 1.5 * i_rated
        for ph, i_x in [("1", i1), ("2", i2), ("3", i3)]:
            c2 = running & ~i_x.between(i_lo, i_hi) & (fail_check == "")
            fail_check  = fail_check.where(~c2, "check_2_current_plausibility")
            fail_reason = fail_reason.where(~c2,
                f"Phase {ph} current outside [{i_lo:.1f}, {i_hi:.1f}] A "
                f"(i_rated={i_rated:.0f} A)")
        # 5% cross-phase check
        for ph, i_x, ia, ib in [("1",i1,i2,i3),("2",i2,i1,i3),("3",i3,i1,i2)]:
            avg_others = (ia + ib) / 2.0
            c2x = running & (avg_others > 0) & (i_x < 0.05 * avg_others) & (fail_check == "")
            fail_check  = fail_check.where(~c2x, "check_2_current_plausibility")
            fail_reason = fail_reason.where(~c2x,
                f"Phase {ph} current < 5% of other phases (CT fault suspected)")

    # Check 3 — Negative per-phase active power
    # For a passive motor load, negative active power on any phase is physically
    # impossible under correct wiring. Any value below -100 W (well above measurement
    # noise) indicates a wiring fault: V-I channel pairing error, CT polarity reversal,
    # or voltage reference error. The -100 W noise floor is machine-size-independent
    # and catches faults at any load level including light load.
    NOISE_FLOOR_W = -100.0
    for _ph, _p in [("1", p1), ("2", p2), ("3", p3)]:
        _c3 = (_p < NOISE_FLOOR_W) & (fail_check == "")
        fail_check  = fail_check.where(~_c3, "check_3_negative_active_power")
        fail_reason = fail_reason.where(~_c3,
            f"Phase {_ph} active power is negative (below -100 W) - "
            f"impossible for a passive motor load - check CT polarity and "
            f"V-I channel assignment")

    # Checks 4 & 5 — PF plausibility and spread (only when meaningfully loaded)
    # Use 10% of rated as minimum for PF to be interpretable.
    # If p_rated is 0 (electrical params not saved), skip both PF checks.
    if p_rated > 0:
        pf_run_thr = 0.10 * p_rated * 1000.0   # 10% of rated electrical input (W)
        pf_running = p_total > pf_run_thr

        # Check 4 — Per-phase PF plausibility
        for ph, p_x, v_x, i_x in [("1",p1,v1,i1),("2",p2,v2,i2),("3",p3,v3,i3)]:
            s_x  = v_x * i_x
            pf_x = p_x / s_x.replace(0, np.nan)
            c4   = pf_running & ((pf_x < 0.30) | (pf_x > 1.00)) & (fail_check == "")
            fail_check  = fail_check.where(~c4, "check_4_pf_plausibility")
            fail_reason = fail_reason.where(~c4, f"Phase {ph} PF outside [0.30, 1.00]")

        # Check 5 — Per-phase PF spread
        pf_list = []
        for p_x, v_x, i_x in [(p1,v1,i1),(p2,v2,i2),(p3,v3,i3)]:
            s_x = v_x * i_x
            pf_list.append(p_x / s_x.replace(0, np.nan))
        pf_spread = pd.concat(pf_list, axis=1).max(axis=1) - pd.concat(pf_list, axis=1).min(axis=1)
        c5 = pf_running & (pf_spread > 0.15) & (fail_check == "")
        fail_check  = fail_check.where(~c5, "check_5_pf_consistency")
        fail_reason = fail_reason.where(~c5, "Per-phase PF spread > 0.15 (channel pairing error suspected)")

    failed_mask = fail_check != ""
    failed_df   = df[failed_mask].copy()
    failed_df["failure_check"]  = fail_check[failed_mask].values
    failed_df["failure_reason"] = fail_reason[failed_mask].values

    return {
        "failed_json":        failed_df.to_json(orient="split", date_format="iso"),
        "fail_checks":        fail_check[failed_mask].tolist(),
        "fail_reasons":       fail_reason[failed_mask].tolist(),
        "n_total":            len(df),
        "n_failed":           int(failed_mask.sum()),
        "p_rated_kw":         round(p_rated, 1),
        "p_rated_estimated":  p_rated_estimated,
    }


def render_cleaning_report(report: CleaningReport, title: str = "Data cleaning"):
    try:
        st.markdown(f"**{title}**")
        steps = [
            ("Raw samples",                    report.n_raw),
            ("Step 1 \u2014 Load \u226540% rated",   report.n_after_load_precondition),
            ("Step 2 \u2014 Start transient",        getattr(report, "n_after_start_transient",
                                                     report.n_after_load_precondition)),
            ("Step 3 \u2014 User filter",            report.n_after_user_filter),
            ("Step 4 \u2014 IQR rejection",          report.n_cleaned),
        ]
        rows_html = ""
        prev = None
        for label, count in steps:
            dropped = f"\u2212{prev - count:,}" if prev is not None and prev > count else ""
            dropped_colour = "#A32D2D" if (prev and prev - count > 0) else "#888"
            rows_html += (
                f'<tr>'
                f'<td style="padding:4px 12px;font-size:0.88em;color:#444">{label}</td>'
                f'<td style="padding:4px 12px;font-size:0.95em;font-weight:600;text-align:right">{count:,}</td>'
                f'<td style="padding:4px 12px;font-size:0.82em;color:{dropped_colour};text-align:right">{dropped}</td>'
                f'</tr>'
            )
            prev = count

        retained_pct = report.fraction_retained * 100
        colour = "green" if retained_pct >= 70 else "orange" if retained_pct >= 40 else "red"
        st.markdown(
            f'<table style="border-collapse:collapse;width:100%">'
            f'<thead><tr style="background:#f0f4f8">'
            f'<th style="padding:4px 12px;font-size:0.82em;text-align:left">Step</th>'
            f'<th style="padding:4px 12px;font-size:0.82em;text-align:right">Samples</th>'
            f'<th style="padding:4px 12px;font-size:0.82em;text-align:right">Removed</th>'
            f'</tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table>',
            unsafe_allow_html=True,
        )
        st.caption(f":{colour}[{retained_pct:.0f}% of raw samples retained for analysis]")
    except Exception as _e:
        st.warning(f"Could not render cleaning report: {_e}. Re-run the assessment to refresh.")


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
        with st.expander(f"PF Drift \u2014 {len(active_bands)} active band(s)", expanded=False):
            st.caption(
                "Each band is a narrow operating-point window (2% of rated power). "
                "**Baseline PF** = mean PF during the ingested baseline period. "
                "**Recent PF** = mean PF during the selected assessment date range. "
                "**Drift** = Recent \u2212 Baseline (negative = degradation)."
            )
            # Table — convert band centres from W to kW for display
            rows = []
            for b in active_bands:
                drift = b.pf_drift if b.pf_drift is not None else 0.0
                if drift <= PF_DRIFT_ACTION:
                    status = "\U0001f534 Action"
                elif drift <= PF_DRIFT_ALERT:
                    status = "\U0001f7e0 Alert"
                elif drift <= PF_DRIFT_WATCH:
                    status = "\U0001f7e1 Watch"
                else:
                    status = "\U0001f7e2 Normal"
                rows.append({
                    "Band centre (kW)":  f"{b.centre_kw / 1000:.1f}",
                    "Baseline PF":       f"{b.mean_pf_baseline:.4f}",
                    "Recent PF":         f"{b.mean_pf_recent:.4f}" if b.mean_pf_recent else "\u2014",
                    "Drift":             f"{b.pf_drift:+.4f}" if b.pf_drift is not None else "\u2014",
                    "Status":            status,
                    "n baseline":        b.n_baseline,
                    "n recent":          b.n_recent,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Visual — use kW for x-axis
            centres     = [b.centre_kw / 1000 for b in active_bands]
            bl_pf_vals  = [b.mean_pf_baseline for b in active_bands]
            rec_pf_vals = [b.mean_pf_recent if b.mean_pf_recent else None for b in active_bands]
            drift_vals  = [b.pf_drift if b.pf_drift is not None else 0.0 for b in active_bands]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=centres, y=bl_pf_vals,
                mode="lines+markers",
                name="Baseline PF",
                line=dict(color="#054D5F", width=2),
                marker=dict(size=6),
            ))
            if any(v is not None for v in rec_pf_vals):
                fig.add_trace(go.Scatter(
                    x=centres, y=rec_pf_vals,
                    mode="lines+markers",
                    name="Recent PF",
                    line=dict(color="#C8A84B", width=2, dash="dash"),
                    marker=dict(size=6),
                ))
            # Drift bar (secondary y)
            bar_colours = [
                "#A32D2D" if d <= PF_DRIFT_ACTION
                else "#E67E22" if d <= PF_DRIFT_ALERT
                else "#F1C40F" if d <= PF_DRIFT_WATCH
                else "#177E40"
                for d in drift_vals
            ]
            fig.add_trace(go.Bar(
                x=centres, y=drift_vals,
                name="PF drift",
                marker_color=bar_colours,
                opacity=0.5,
                yaxis="y2",
                width=[centres[1] - centres[0]] * len(centres) if len(centres) > 1 else [1],
            ))
            # Threshold lines on drift axis
            for val, colour, label in [
                (PF_DRIFT_WATCH,  "#F1C40F", f"Watch {PF_DRIFT_WATCH}"),
                (PF_DRIFT_ALERT,  "#E67E22", f"Alert {PF_DRIFT_ALERT}"),
                (PF_DRIFT_ACTION, "#A32D2D", f"Action {PF_DRIFT_ACTION}"),
            ]:
                fig.add_hline(
                    y=val, line_color=colour, line_dash="dot", line_width=1,
                    annotation_text=label, annotation_position="bottom right",
                    annotation_font_size=9, yref="y2",
                )
            fig.update_layout(
                title=dict(text="PF by operating-point band", font=dict(size=13)),
                xaxis_title="Band centre (kW)",
                yaxis=dict(title="Power Factor", range=[0, 1.05]),
                yaxis2=dict(
                    title="PF drift",
                    overlaying="y", side="right",
                    range=[min(min(drift_vals) * 1.2, PF_DRIFT_ACTION * 1.2), 0.02],
                    showgrid=False,
                    tickformat=".3f",
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=40, r=60, t=45, b=50),
                legend=dict(orientation="h", yanchor="top", y=-0.18,
                            xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
                hovermode="x unified", font=dict(size=11), height=320,
            )
            st.plotly_chart(fig, use_container_width=True)

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

    # Cleaning report — detect stale records from before the 4-step pipeline update
    cr = record.cleaning_report
    if cr:
        _is_stale = not hasattr(cr, "n_after_start_transient")
        if _is_stale:
            st.info(
                "\u2139\ufe0f This assessment was saved before the cleaning pipeline was updated. "
                "**Re-run the assessment** to see the current four-step cleaning report."
            )
        else:
            render_cleaning_report(cr, title="Assessment data cleaning")
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

def build_assessment_charts(
    data: pd.DataFrame,
    record: AssessmentRecord,
    cleaned_data: pd.DataFrame | None = None,
) -> list:
    """Build control charts for VUF, P_total and PF_machine.

    Shows cleaned samples (used for analysis) in solid blue.
    Non-cleaned samples are shown as faded grey in the background so
    the user can see the full time window without confusing stopped
    periods with analysed data.
    """
    figs = []
    if data is None or data.empty:
        return figs
    missing = check_required_columns(data)
    if missing:
        return figs

    def _derive(df):
        v1 = df["phase_1_voltage"]
        v2 = df["phase_2_voltage"]
        v3 = df["phase_3_voltage"]
        v_avg = (v1 + v2 + v3) / 3.0
        vuf = (pd.concat([(v1-v_avg).abs(),(v2-v_avg).abs(),(v3-v_avg).abs()],
                         axis=1).max(axis=1) / v_avg.replace(0, np.nan) * 100.0)
        i1=df["phase_1_current"]; i2=df["phase_2_current"]; i3=df["phase_3_current"]
        i_avg = (i1+i2+i3) / 3.0
        iuf = (pd.concat([(i1-i_avg).abs(),(i2-i_avg).abs(),(i3-i_avg).abs()],
                         axis=1).max(axis=1) / i_avg.replace(0, np.nan) * 100.0)
        p_w = df["phase_1_active_power"]+df["phase_2_active_power"]+df["phase_3_active_power"]
        p_kw = p_w / 1000.0
        s_sum = v1*i1 + v2*i2 + v3*i3
        pf = (p_w / s_sum.replace(0, np.nan)).clip(0, 1)
        return vuf, iuf, p_kw, pf

    # Raw (full window) — used as faded background
    raw_vuf, raw_iuf, raw_p_kw, raw_pf = _derive(data)

    # Cleaned (analysis samples) — primary series
    if cleaned_data is not None and not cleaned_data.empty:
        cln_idx = cleaned_data.index if hasattr(cleaned_data.index, "name") else cleaned_data.index
        cl_vuf, cl_iuf, cl_p_kw, cl_pf = _derive(cleaned_data)
        has_cleaned = True
    else:
        has_cleaned = False

    baseline = record.motor_side

    def _chart(raw_x, raw_y, cl_x, cl_y, title, y_label,
               h_lines=None, y_range=None, show_cleaned=True):
        fig = go.Figure()

        # Background: all raw data (faded grey)
        fig.add_trace(go.Scatter(
            x=raw_x, y=raw_y, mode="lines",
            line=dict(color="rgba(180,180,180,0.45)", width=0.8),
            name="All data (not analysed)",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>" + y_label + ": %{y:.3f} (raw)<extra></extra>",
            showlegend=True,
        ))

        # Foreground: cleaned / analysed data (solid blue)
        if show_cleaned and cl_x is not None and cl_y is not None:
            fig.add_trace(go.Scatter(
                x=cl_x, y=cl_y, mode="markers",
                marker=dict(color="#185FA5", size=2.5, opacity=0.85),
                name="Cleaned (used for analysis)",
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>" + y_label + ": %{y:.3f}<extra></extra>",
                showlegend=True,
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
            hovermode="x unified", font=dict(size=11), height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        xanchor="left", x=0, font=dict(size=10)),
        )
        return fig

    cl_idx = cleaned_data.index if has_cleaned else None

    # VUF chart
    figs.append(_chart(
        data.index, raw_vuf,
        cl_idx, cl_vuf if has_cleaned else None,
        "Zone 1 \u2014 Voltage Unbalance Factor (VUF)", "VUF (%)",
        h_lines=[
            (VUF_CRITICAL, "#C0392B", "solid",  f"Critical {VUF_CRITICAL:.1f}%"),
            (VUF_WATCH,    "#E67E22", "dash",   f"Watch {VUF_WATCH:.1f}%"),
        ],
        y_range=[0, max(float(raw_vuf.max()) * 1.3, VUF_CRITICAL * 1.5)],
    ))

    # IUF chart — only on cleaned data (IUF is meaningless at low / zero load)
    iuf_max = float(cl_iuf.max()) if has_cleaned else float(raw_iuf.max())
    figs.append(_chart(
        data.index, raw_iuf,
        cl_idx, cl_iuf if has_cleaned else None,
        "Zone 2 \u2014 Current Imbalance Factor (IUF)", "IUF (%)",
        h_lines=[
            (IUF_CRITICAL, "#C0392B", "solid",  f"Critical {IUF_CRITICAL:.0f}%"),
            (IUF_WATCH,    "#E67E22", "dash",   f"Watch {IUF_WATCH:.0f}%"),
        ],
        y_range=[0, max(iuf_max * 1.3, IUF_CRITICAL * 1.5)],
    ))

    # P_total chart
    p_hlines = []
    if record.zone4 and record.zone4.p_baseline_avg_kw:
        p_hlines.append((
            record.zone4.p_baseline_avg_kw, "#054D5F", "dashdot",
            f"Baseline avg {record.zone4.p_baseline_avg_kw:.1f} kW",
        ))
    figs.append(_chart(
        data.index, raw_p_kw,
        cl_idx, cl_p_kw if has_cleaned else None,
        "Total Active Power (P_total)", "P_total (kW)",
        h_lines=p_hlines or None,
    ))

    # PF_machine chart
    pf_hlines = []
    if baseline and baseline.bands:
        for b in baseline.bands[:6]:
            pf_hlines.append((
                b.mean_pf_baseline, "rgba(180,180,180,0.6)", "dot",
                f"Band {b.centre_kw:.0f} kW baseline PF",
            ))
    figs.append(_chart(
        data.index, raw_pf,
        cl_idx, cl_pf if has_cleaned else None,
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
    ("last_assessment",   None),
    ("last_data",         None),
    ("last_cleaned_data", None),
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

        reg_p_rated = st.number_input(
            "Rated shaft power (kW)  \u2014 optional",
            min_value=0.0, value=0.0, step=1.0, format="%.1f",
            key="reg_p_rated",
            help=(
                "Motor nameplate rated shaft power in kW. "
                "If left as 0, the platform estimates from your data with a warning."
            ),
        )

        _reg_cols = st.columns(2)
        reg_v_nom = _reg_cols[0].number_input(
            "Rated voltage L-N (V)  \u2014 optional",
            min_value=0.0, value=0.0, step=1.0, format="%.0f",
            key="reg_v_nom",
            help=(
                "Phase-to-neutral voltage (V). "
                "For 400 V three-phase systems enter 230 V. "
                "Used for voltage plausibility checks."
            ),
        )
        reg_i_rated = _reg_cols[1].number_input(
            "Full-load current (A)  \u2014 optional",
            min_value=0.0, value=0.0, step=0.5, format="%.1f",
            key="reg_i_rated",
            help=(
                "Motor nameplate full-load current (FLA) in Amperes. "
                "Used for current plausibility checks."
            ),
        )

        _any_nameplate = reg_p_rated > 0 or reg_v_nom > 0 or reg_i_rated > 0

        if st.button("Register", type="primary", use_container_width=True,
                     disabled=not (machine_id and machine_type)):
            _reg_desc = ""
            if _any_nameplate:
                _reg_app = APP_TYPE_MAP.get(machine_type.strip(), "compressed_air")
                _reg_desc = serialise_meta_block(
                    v_nominal = reg_v_nom  if reg_v_nom   > 0 else 230.0,
                    p_rated   = reg_p_rated,
                    pf_rated  = 0.88,
                    eta_rated = 0.90,
                    i_rated   = reg_i_rated,
                    four_wire = True,
                    at_panel  = True,
                    app_type  = _reg_app,
                    power_unit = "W",
                )
            db.register_machine(machine_id.strip(), machine_type.strip(), _reg_desc)

            _saved = []
            if reg_p_rated > 0: _saved.append(f"{reg_p_rated:.0f} kW")
            if reg_v_nom   > 0: _saved.append(f"{reg_v_nom:.0f} V L-N")
            if reg_i_rated > 0: _saved.append(f"{reg_i_rated:.1f} A FLA")

            if _saved:
                st.success(
                    f"**{machine_id}** registered \u2014 saved: {', '.join(_saved)}. "
                    "Complete remaining nameplate values in the "
                    "\u26a1 **Electrical parameters** expander."
                )
            else:
                st.success(
                    f"**{machine_id}** registered. "
                    "Enter nameplate values in the "
                    "\u26a1 **Electrical parameters** expander."
                )
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
        st.session_state["last_cleaned_data"] = None
        st.session_state["last_data"]       = None
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

    # ── Power unit selector — required before ingest ──────────────────────
    # Auto-detect from file, let user confirm. Unit is a property of the file,
    # not the machine nameplate — different exports may use W or kW.
    _detected_unit = "W"
    if uploaded_file is not None:
        try:
            import io
            _peek = uploaded_file.read(262144)  # read first 256 KB
            uploaded_file.seek(0)               # reset for actual ingest
            if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
                _df_peek = pd.read_excel(io.BytesIO(_peek), nrows=200)
            else:
                _df_peek = pd.read_csv(io.BytesIO(_peek), nrows=200, on_bad_lines="skip")
            _power_cols = [c for c in _df_peek.columns if "active_power" in c.lower()]
            if _power_cols:
                # Pool all power columns so stopped-period zeros don't mask the signal
                _all_vals = pd.concat(
                    [_df_peek[c] for c in _power_cols]
                ).dropna()
                _nonzero = _all_vals[_all_vals > 0]
                if len(_nonzero) > 0:
                    _median_val = float(_nonzero.median())
                    # Values > 500 are almost certainly W; <= 500 are almost certainly kW
                    _detected_unit = "W" if _median_val > 500 else "kW"
        except Exception:
            _detected_unit = "W"

    _current_saved_unit = parse_electrical_meta(
        db.get_machine_info(selected_id).get("description", "") if selected_id else ""
    ).get("power_unit", _detected_unit).upper()

    _upload_unit = st.radio(
        "Active power unit in this file",
        options=["W", "kW"],
        index=0 if _current_saved_unit == "W" else 1,
        horizontal=True,
        key="upload_power_unit",
        help=(
            "\u26a1 Auto-detected from first rows of file. "
            "Confirm before ingesting. "
            "W = raw Watts (e.g. 31,000); kW = kilowatts (e.g. 31.0)."
        ),
    )
    if uploaded_file is not None and _upload_unit != _detected_unit:
        st.caption(
            f"\u26a0\ufe0f Auto-detected **{_detected_unit}** but you selected **{_upload_unit}**. "
            "Make sure this matches your file."
        )

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
                # Save the confirmed power unit into machine metadata
                _mi_now   = db.get_machine_info(selected_id) or {}
                _meta_now = parse_electrical_meta(_mi_now.get("description", ""))
                if _meta_now.get("power_unit", "").upper() != _upload_unit:
                    _app_type = APP_TYPE_MAP.get(_mi_now.get("machine_type", ""), "compressed_air")
                    _new_block = serialise_meta_block(
                        float(_meta_now.get("v_nominal_phase", 230)),
                        float(_meta_now.get("p_rated_shaft_kw", 0)),
                        float(_meta_now.get("pf_rated", 0.88)),
                        float(_meta_now.get("eta_rated", 0.90)),
                        float(_meta_now.get("i_rated", 0)),
                        bool(_meta_now.get("four_wire", True)),
                        bool(_meta_now.get("measurement_at_panel", True)),
                        _app_type,
                        _upload_unit,
                    )
                    _updated_desc = replace_meta_block(
                        _mi_now.get("description", ""), _new_block
                    )
                    db.register_machine(
                        selected_id,
                        _mi_now.get("machine_type", "screw_compressor"),
                        _updated_desc,
                    )
                st.success(
                    f"\u2713 {result['rows']:,} rows ingested  \u00b7  "
                    f"Power unit: **{_upload_unit}**"
                )
                st.session_state["last_assessment"] = None
                st.session_state["last_cleaned_data"] = None
                st.session_state["last_data"]       = None
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

    # Show which fields were already saved (e.g. from registration)
    _pre_filled = []
    if _em.get("v_nominal_phase", 0) > 0:    _pre_filled.append(f"V_nom={_em['v_nominal_phase']:.0f} V")
    if _em.get("p_rated_shaft_kw", 0) > 0:   _pre_filled.append(f"P_rated={_em['p_rated_shaft_kw']:.0f} kW")
    if _em.get("i_rated", 0) > 0:            _pre_filled.append(f"FLA={_em['i_rated']:.0f} A")
    if _pre_filled:
        st.info(f"\u2139\ufe0f Pre-filled from registration: {', '.join(_pre_filled)}. "
                "Confirm or update below, then save.")

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
    # Power unit is set at upload time (sidebar), not here.
    # Read current saved value for round-trip when user clicks Save.
    _power_unit = _em.get("power_unit", "W").upper()

    if st.button("Save electrical parameters", key="save_ep_btn", use_container_width=True):
        _app_type = APP_TYPE_MAP.get(machine_info["machine_type"], "compressed_air")
        _new_block = serialise_meta_block(
            _v_nom, _p_rated, _pf_rated, _eta_rated, _i_rated,
            _fw, _at_panel, _app_type, _power_unit,
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
    st.session_state["last_cleaned_data"] = None
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

        # ── Integrity checks §3.1 ─────────────────────────────────────────
        meta_for_check = build_meta(machine_info)
        if not missing_cols and meta_for_check:
            with st.expander("\U0001f50d Integrity checks (§3.1) \u2014 Data tab", expanded=False):
                st.caption(
                    "Six integrity checks per §3.1 of the methodology. "
                    "Check 0 validates data type (non-numeric values excluded). "
                    "Checks 1\u20135 validate physical plausibility. "
                    "Failed samples are flagged below \u2014 they will not enter the analysis pipeline."
                )
                _raw_reset = data.reset_index()
                # Drop rows with null/NaT timestamp — empty trailing rows from CSV
                _raw_reset = _raw_reset[_raw_reset["timestamp"].notna()].copy()
                _scaled    = scale_power_to_watts(_raw_reset, meta_for_check.get("power_unit", "W"))

                # ── Check 0: Non-numeric — scan BEFORE any coercion ──────────
                # Must use the raw source data (data.reset_index()) because
                # _coerce_numeric already replaced strings with NaN in `data`.
                # Re-read the stored CSV directly to get original string values.
                _pre_fail_check  = pd.Series("", index=_scaled.index)
                _pre_fail_reason = pd.Series("", index=_scaled.index)

                _MEAS = [
                    "phase_1_voltage","phase_2_voltage","phase_3_voltage",
                    "phase_1_current","phase_2_current","phase_3_current",
                    "phase_1_active_power","phase_2_active_power","phase_3_active_power",
                ]

                # Primary: read from _orig_* sidecar columns set by _coerce_numeric
                _flags_col = "_non_numeric_flags"
                _has_sidecar = _flags_col in _scaled.columns
                if _has_sidecar:
                    _flag_series = _scaled[_flags_col].fillna("")
                    _has_flag    = _flag_series != ""
                    if _has_flag.any():
                        for _idx in _flag_series[_has_flag].index:
                            _entries = _flag_series[_idx].split("|")
                            for _entry in _entries:
                                if ":" in _entry:
                                    _col_name, _bad_val = _entry.split(":", 1)
                                    _c0 = (_scaled.index == _idx) & (_pre_fail_check == "")
                                    _pre_fail_check  = _pre_fail_check.where(~_c0, "check_0_non_numeric")
                                    _pre_fail_reason = _pre_fail_reason.where(~_c0,
                                        f"{_col_name} contains non-numeric value "
                                        f"'{_bad_val}' - cannot be used for diagnostics")

                # Fallback: re-read stored CSV to scan for strings that survived to disk
                if not _has_sidecar or (_pre_fail_check == "").all():
                    try:
                        _file_info_now = db.get_file_info(selected_id)
                        if _file_info_now:
                            import pathlib as _pl
                            _csv_path = _pl.Path(_file_info_now[-1]["file"])
                            if _csv_path.exists():
                                _raw_str = pd.read_csv(_csv_path, dtype=str)
                                # Align index with _scaled by timestamp
                                _ts_col = next(
                                    (c for c in _raw_str.columns
                                     if any(k in c.lower() for k in ["time","date","timestamp"])),
                                    _raw_str.columns[0]
                                )
                                _raw_str[_ts_col] = pd.to_datetime(
                                    _raw_str[_ts_col], dayfirst=True, errors="coerce"
                                )
                                _raw_str = _raw_str.set_index(_ts_col).sort_index()
                                for _mc in _MEAS:
                                    if _mc not in _raw_str.columns:
                                        continue
                                    _col_orig = _raw_str[_mc]
                                    _col_num  = pd.to_numeric(_col_orig, errors="coerce")
                                    _bad      = _col_num.isna() & ~_col_orig.isna()
                                    if _bad.any():
                                        for _ts, _bv in _col_orig[_bad].items():
                                            if _ts in _pre_fail_check.index:
                                                _c0 = (_scaled.index == _ts) & (_pre_fail_check == "")
                                                _pre_fail_check  = _pre_fail_check.where(~_c0, "check_0_non_numeric")
                                                _pre_fail_reason = _pre_fail_reason.where(~_c0,
                                                    f"{_mc} contains non-numeric value "
                                                    f"'{_bv}' - cannot be used for diagnostics")
                    except Exception:
                        pass  # Fallback failed silently — Check 0 proceeds without it

                # Build pre-failed DataFrame and restore original bad values
                _pre_failed_mask = _pre_fail_check != ""
                _pre_failed_df   = _scaled[_pre_failed_mask].copy()
                _pre_failed_df["failure_check"]  = _pre_fail_check[_pre_failed_mask].values
                _pre_failed_df["failure_reason"] = _pre_fail_reason[_pre_failed_mask].values

                # Restore original string values from _orig_* sidecar columns if present
                for _mc in _MEAS:
                    _orig_col = f"_orig_{_mc}"
                    if _orig_col in _pre_failed_df.columns:
                        _orig_vals = _pre_failed_df[_orig_col].astype(str)
                        _has_orig  = _orig_vals.str.strip() != ""
                        if _has_orig.any():
                            # Convert to object dtype first to allow string assignment
                            _pre_failed_df[_mc] = _pre_failed_df[_mc].astype(object)
                            _pre_failed_df.loc[_has_orig, _mc] = _orig_vals[_has_orig]
                        _pre_failed_df = _pre_failed_df.drop(columns=[_orig_col])

                import json as _json
                # Drop all sidecar columns before passing to run_integrity_checks
                _sidecar_cols = [c for c in _scaled.columns
                                 if c.startswith("_orig_") or c == _flags_col]
                _scaled_clean = _scaled.drop(columns=_sidecar_cols, errors="ignore")
                _ig = run_integrity_checks(
                    _scaled_clean.to_json(orient="split", date_format="iso"),
                    _json.dumps(meta_for_check),
                )
                _n_total  = _ig["n_total"]

                # Merge Check 0 failures with Checks 1-5 failures
                import io as _io
                _phys_failed = pd.read_json(_io.StringIO(_ig["failed_json"]), orient="split")
                _phys_failed["failure_check"]  = _ig["fail_checks"]
                _phys_failed["failure_reason"] = _ig["fail_reasons"]

                # Combine: Check 0 rows + physical check rows
                if len(_pre_failed_df) > 0 and "timestamp" in _pre_failed_df.columns:
                    _all_failed = pd.concat(
                        [_pre_failed_df, _phys_failed], ignore_index=True
                    ).drop_duplicates(subset=["timestamp", "failure_check"]).reset_index(drop=True)
                else:
                    _all_failed = _phys_failed

                _n_failed     = len(_all_failed)
                _n_passed     = _n_total - _n_failed
                _fail_pct     = _n_failed / _n_total * 100 if _n_total > 0 else 0
                _fail_checks  = _all_failed["failure_check"].tolist()  if len(_all_failed) else []
                _fail_reasons = _all_failed["failure_reason"].tolist() if len(_all_failed) else []
                _failed       = _all_failed

                # Show estimated power warning if nameplate not saved
                if _ig.get("p_rated_estimated"):
                    st.warning(
                        f"\u26a0\ufe0f **Rated power not saved.** "
                        f"Load thresholds estimated from data "
                        f"(95th percentile \u2192 ~**{_ig['p_rated_kw']:.0f} kW** electrical). "
                        f"Enter the motor nameplate rated shaft power in the "
                        f"\u26a1 **Electrical parameters** expander for accurate checks."
                    )

                ic1, ic2, ic3 = st.columns(3)
                ic1.metric("Total samples",  f"{_n_total:,}")
                ic2.metric("Passed ✅",       f"{_n_passed:,}")
                ic3.metric("Failed ❌",        f"{_n_failed:,}",
                           delta=f"{_fail_pct:.1f}% of total" if _n_failed else None,
                           delta_color="inverse")

                if _n_failed == 0:
                    st.success("\u2705 All samples passed all five integrity checks.")
                    # Download passed data
                    _passed_df = _scaled_clean.copy()
                    _passed_df = _passed_df[[c for c in _passed_df.columns
                                             if not c.startswith("_")]]
                    if "timestamp" not in _passed_df.columns and _passed_df.index.name == "timestamp":
                        _passed_df = _passed_df.reset_index()
                    st.download_button(
                        label=f"\u2b07\ufe0f Download all {_n_total:,} passed rows (CSV)",
                        data=_passed_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"integrity_passed_{selected_id}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                else:
                    # Summary by check type
                    from collections import Counter
                    _check_counts = Counter(_fail_checks)
                    _check_labels = {
                        "check_0_non_numeric":           "Check 0 \u2014 Non-numeric value in measurement column",
                        "check_1_voltage_plausibility":  "Check 1 \u2014 Voltage plausibility",
                        "check_2_current_plausibility":  "Check 2 \u2014 Current plausibility",
                        "check_3_negative_active_power": "Check 3 \u2014 Negative active power (wiring fault)",
                        "check_4_pf_plausibility":       "Check 4 \u2014 Per-phase PF plausibility",
                        "check_5_pf_consistency":        "Check 5 \u2014 Per-phase PF spread consistency",
                    }
                    st.warning(
                        f"\u26a0\ufe0f **{_n_failed:,} samples ({_fail_pct:.1f}%) failed one or more integrity checks.** "
                        "These samples will not be used in the analysis pipeline."
                    )
                    for _chk, _cnt in _check_counts.items():
                        _lbl = _check_labels.get(_chk, _chk)
                        st.markdown(
                            f'<div style="background:#FFF0F0;border-left:4px solid #A32D2D;'
                            f'padding:8px 12px;margin-bottom:4px;border-radius:3px;font-size:0.88em">'
                            f'\u274c <b>{_lbl}</b>: {_cnt:,} sample(s) failed</div>',
                            unsafe_allow_html=True,
                        )

                    # Build full failed rows export
                    _failed_full = _failed.copy()
                    _failed_full["failure_check"]  = _fail_checks
                    _failed_full["failure_reason"] = _fail_reasons
                    _export_cols = (
                        ["timestamp"] +
                        [c for c in REQUIRED_COLS if c in _failed_full.columns] +
                        ["failure_check", "failure_reason"]
                    )
                    _failed_full = _failed_full[_export_cols].reset_index(drop=True)

                    # Two-column layout: failed | passed
                    _col_fail, _col_pass = st.columns(2)

                    with _col_fail:
                        st.markdown(f"**\u274c Failed rows ({_n_failed:,})**")
                        st.download_button(
                            label=f"\u2b07\ufe0f Download {_n_failed:,} failed rows (CSV)",
                            data=_failed_full.to_csv(index=False).encode("utf-8"),
                            file_name=f"integrity_failures_{selected_id}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                        with st.expander(
                            f"Preview first {min(10, _n_failed)} failed rows",
                            expanded=False
                        ):
                            st.dataframe(
                                _failed_full.head(10),
                                use_container_width=True,
                                hide_index=True,
                            )

                    with _col_pass:
                        st.markdown(f"**\u2705 Passed rows ({_n_passed:,})**")
                        # Build passed rows: all rows NOT in failed set
                        _failed_ts  = set(_failed_full["timestamp"].astype(str).tolist()) \
                                      if "timestamp" in _failed_full.columns else set()
                        _passed_df  = _scaled_clean.copy()
                        _passed_df  = _passed_df[[c for c in _passed_df.columns
                                                   if not c.startswith("_")]]
                        if "timestamp" not in _passed_df.columns:
                            _passed_df = _passed_df.reset_index()
                        _passed_df  = _passed_df[
                            ~_passed_df["timestamp"].astype(str).isin(_failed_ts)
                        ].reset_index(drop=True)
                        st.download_button(
                            label=f"\u2b07\ufe0f Download {_n_passed:,} passed rows (CSV)",
                            data=_passed_df.to_csv(index=False).encode("utf-8"),
                            file_name=f"integrity_passed_{selected_id}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                        with st.expander(
                            f"Preview first {min(10, _n_passed)} passed rows",
                            expanded=False
                        ):
                            st.dataframe(
                                _passed_df.head(10),
                                use_container_width=True,
                                hide_index=True,
                            )
        elif not missing_cols and meta_for_check is None:
            st.info(
                "\u2139\ufe0f Integrity checks require electrical parameters (nameplate values). "
                "Fill in the \u26a1 Electrical parameters expander above to enable them."
            )

        # ── Download raw data as seen by platform ─────────────────────────
        _dl_raw = data.reset_index()
        # Drop sidecar columns before download
        _dl_raw = _dl_raw[[c for c in _dl_raw.columns if not c.startswith("_")]]
        st.download_button(
            label=f"\u2b07\ufe0f Download full dataset as seen by platform ({len(_dl_raw):,} rows)",
            data=_dl_raw.to_csv(index=False).encode("utf-8"),
            file_name=f"platform_data_{selected_id}.csv",
            mime="text/csv",
            help=(
                "Downloads the data exactly as the platform has loaded and processed it "
                "(timestamps parsed, power scaled, non-numeric values coerced to blank). "
                "Use this to verify what the platform sees before running analysis."
            ),
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
                st.markdown("**Assessment period**")
                st.caption(
                    "The window of recent data to evaluate. "
                    "Should be **after** the baseline period. "
                    "PF drift, IUF, VUF and Zone 4 are all computed over this window."
                )
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
                        f"Baseline avg power: {_bl_p_avg:.1f} kW"
                        if _bl_p_avg else
                        f"\u2705 Baseline ingested {_stored_at} \u2014 {_n_bands} PF band(s)"
                    )
                    for _w in _bl_warns:
                        st.caption(f"\u26a0\ufe0f {_w}")

                    # Baseline cleaning report
                    _bl_cr = _stored_bl_dict.get("cleaning_report")
                    if _bl_cr:
                        with st.expander("Baseline data cleaning", expanded=False):
                            _cr = _migrate_cleaning_report(_bl_cr)
                            render_cleaning_report(_cr, title="Baseline data cleaning")

                    # Baseline raw data viewer — shows exactly what was used
                    with st.expander(
                        f"\U0001f4cb View baseline data  ({_bl_start} to {_bl_end})",
                        expanded=False
                    ):
                        try:
                            _bl_start_ts2 = pd.Timestamp(_bl_start)
                            _bl_end_ts2   = pd.Timestamp(_bl_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                            _bl_view_data = data.loc[
                                (_bl_start_ts2 <= data.index) & (data.index <= _bl_end_ts2)
                            ]
                            if _bl_view_data.empty:
                                st.warning(
                                    "No data found for the stored baseline period. "
                                    "This may indicate a date parsing issue in the ingested file. "
                                    "Try deleting and re-ingesting the data file."
                                )
                                st.caption(
                                    f"Baseline period: {_bl_start} to {_bl_end}  |  "
                                    f"Data available: {data.index.min().date()} to {data.index.max().date()}"
                                )
                            else:
                                _n_bl_rows = len(_bl_view_data)
                                st.caption(
                                    f"{_n_bl_rows:,} rows  |  "
                                    f"{_bl_view_data.index.min().strftime('%Y-%m-%d %H:%M')} "
                                    f"to {_bl_view_data.index.max().strftime('%Y-%m-%d %H:%M')}"
                                )
                                st.dataframe(
                                    _bl_view_data.head(100),
                                    use_container_width=True,
                                    height=250,
                                )
                                _bl_dl = _bl_view_data.reset_index()
                                _bl_dl = _bl_dl[[c for c in _bl_dl.columns if not c.startswith("_")]]
                                st.download_button(
                                    label=f"\u2b07\ufe0f Download baseline data ({_n_bl_rows:,} rows, CSV)",
                                    data=_bl_dl.to_csv(index=False).encode("utf-8"),
                                    file_name=f"baseline_{selected_id}_{_bl_start}_to_{_bl_end}.csv",
                                    mime="text/csv",
                                    use_container_width=True,
                                )
                        except Exception as _bl_e:
                            st.error(f"Could not load baseline data: {_bl_e}")

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
                                _raw_bl_reset = scale_power_to_watts(_raw_bl_reset, meta.get("power_unit", "W"))
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
                            st.session_state["last_cleaned_data"] = None
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
                        # Strip timezone from index if present to avoid comparison errors
                        _data_idx = data.index
                        if hasattr(_data_idx, "tz") and _data_idx.tz is not None:
                            _data_idx = _data_idx.tz_localize(None)
                            _recent = data.copy()
                            _recent.index = _data_idx
                        else:
                            _recent = data
                        _recent = _recent.loc[(_start_ts <= _recent.index) & (_recent.index <= _end_ts)]
                        if _recent.empty:
                            st.error(
                                f"No data in selected date range "
                                f"({date_range[0]} to {date_range[1]}). "
                                f"Data available: {data.index.min().date()} "
                                f"to {data.index.max().date()}."
                            )
                        else:
                            _bm_loaded = baseline_from_dict(db.get_baseline(selected_id))
                            with st.spinner("Running electrical diagnostics\u2026"):
                                _raw_reset = _recent.reset_index()
                                _raw_reset = scale_power_to_watts(_raw_reset, meta.get("power_unit", "W"))
                                _record = run_assessment(_raw_reset, _bm_loaded, meta)
                                # Also capture cleaned data for download
                                _user_filter = _bm_loaded.user_filter_expr if _bm_loaded else None
                                _cleaned_df, _ = clean_samples(_raw_reset, meta, _user_filter)

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
                            st.session_state["last_assessment"]   = _record
                            st.session_state["last_data"]         = _recent
                            st.session_state["last_cleaned_data"] = _cleaned_df
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

                    # ── Cleaned data download ─────────────────────────────
                    _cleaned = st.session_state.get("last_cleaned_data")
                    if _cleaned is not None and not _cleaned.empty:
                        _cr = record.cleaning_report
                        _n_cleaned = _cr.n_cleaned if _cr else len(_cleaned)
                        with st.expander(
                            f"\u2b07\ufe0f Download cleaned assessment data "
                            f"({_n_cleaned:,} samples)",
                            expanded=False,
                        ):
                            st.caption(
                                "The cleaned dataset used for this assessment — "
                                "after load precondition (≥40%), start transient exclusion, "
                                "user filter, and IQR rejection. "
                                "Active power is in Watts."
                            )
                            # Convert power back to original unit if needed
                            _dl_unit = meta.get("power_unit", "W").upper()
                            _dl_df   = _cleaned.copy()
                            if _dl_unit == "KW":
                                for _pc in ["phase_1_active_power",
                                            "phase_2_active_power",
                                            "phase_3_active_power"]:
                                    if _pc in _dl_df.columns:
                                        _dl_df[_pc] = (_dl_df[_pc] / 1000.0).round(6)
                                if "p_total_kw" not in _dl_df.columns:
                                    _dl_df["p_total_kw"] = (
                                        _dl_df["phase_1_active_power"] +
                                        _dl_df["phase_2_active_power"] +
                                        _dl_df["phase_3_active_power"]
                                    ).round(6)

                            # Preview
                            st.dataframe(
                                _dl_df.head(10),
                                use_container_width=True,
                                hide_index=True,
                            )
                            st.download_button(
                                label=f"\u2b07\ufe0f Download all {_n_cleaned:,} cleaned rows (CSV)",
                                data=_dl_df.to_csv(index=False).encode("utf-8"),
                                file_name=(
                                    f"cleaned_{selected_id}_"
                                    f"{str(date_range[0])}_to_"
                                    f"{str(date_range[1])}.csv"
                                ),
                                mime="text/csv",
                                use_container_width=True,
                            )

                    # Control charts
                    _chart_data = st.session_state.get("last_data")
                    if _chart_data is None:
                        _chart_data = data
                    if _chart_data is not None:
                        _chart_data_w  = scale_power_to_watts(
                            _chart_data.reset_index(), meta.get("power_unit", "W")
                        ).set_index("timestamp")
                        _cleaned_chart = st.session_state.get("last_cleaned_data")
                        # Ensure cleaned data has a proper datetime index
                        if _cleaned_chart is not None and not _cleaned_chart.empty:
                            if "timestamp" in _cleaned_chart.columns:
                                _cleaned_chart = _cleaned_chart.set_index("timestamp")
                            _cleaned_chart.index = pd.to_datetime(_cleaned_chart.index)
                        st.markdown("---")
                        st.subheader("Charts")
                        st.caption(
                            "\U0001f7e2 **Blue dots** = cleaned samples used for analysis  "
                            "\u2502  \U0001f6ab **Grey line** = all raw data (not analysed)"
                        )
                        for fig in build_assessment_charts(
                            _chart_data_w, record, cleaned_data=_cleaned_chart
                        ):
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
