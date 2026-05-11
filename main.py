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
import plotly.io as pio
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
    PF_DRIFT_WATCH,
    VUF_CRITICAL,
    VUF_WATCH,
    ZONE4_SIGNIFICANCE_PCT,
    integrity_gate,
    clean_samples,
    ingest_baseline,
    run_assessment,
    compute_pf_drift_phase,
    select_pf_bands,
    select_pf_bands_phase,
    assessment_summary,
)

load_dotenv()
import electrical_diagnostics as _ed_mod
_DIAG_VERSION = getattr(_ed_mod, "_DIAG_VERSION", "unknown")

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
    Returns None only when no metadata block exists yet (new machine, nothing saved).
    All five numeric parameters have soft defaults — resolve_effective_meta
    fills in data-derived estimates for any that are zero.

    Soft defaults (only applied when key absent from DB):
      v_nominal_phase  = 0.0   → triggers data-derived estimation in §2.5
      p_rated_shaft_kw = 0.0   → triggers data-derived estimation in §2.5
      i_rated          = 0.0   → triggers data-derived estimation in §2.5
      pf_rated         = 0.87  → assumed default if not entered
      eta_rated        = 0.90  → assumed default if not entered
    """
    desc = machine_info.get("description", "")
    em   = parse_electrical_meta(desc)
    # Return None only if no metadata has ever been saved for this machine
    if not em:
        return None
    # Soft defaults — only for fields not saved in the DB.
    # v_nominal_phase intentionally has NO default here so resolve_effective_meta
    # can detect "not saved" (missing/0) and estimate from data per §2.5.
    em.setdefault("p_rated_shaft_kw",   0.0)
    em.setdefault("i_rated",            0.0)
    em.setdefault("pf_rated",           0.87)
    em.setdefault("eta_rated",          0.90)
    em.setdefault("v_nominal_phase",    0.0)   # 0 = not saved → estimated in §2.5
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
    user_entered: dict | None = None,
) -> str:
    """Serialise electrical metadata to a text block.

    user_entered: optional dict of {field: bool} indicating which values
    were explicitly typed by the user vs filled in as platform defaults.
    E.g. {"v_nominal_phase": True, "pf_rated": False}
    """
    ue = user_entered or {}
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
        f"user_entered_v_nominal: {str(ue.get('v_nominal_phase', False)).lower()}\n"
        f"user_entered_p_rated: {str(ue.get('p_rated_shaft_kw', False)).lower()}\n"
        f"user_entered_pf: {str(ue.get('pf_rated', False)).lower()}\n"
        f"user_entered_eta: {str(ue.get('eta_rated', False)).lower()}\n"
        f"user_entered_i_rated: {str(ue.get('i_rated', False)).lower()}\n"
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


def resolve_effective_meta(saved_meta: dict, data_w: pd.DataFrame,
                           raw_em: dict | None = None) -> dict:
    """Resolve effective nameplate values per §2.5.

    raw_em  : output of parse_electrical_meta() BEFORE build_meta setdefaults.
              If a key is non-zero here, the user explicitly saved it.
    saved_meta : output of build_meta() — has setdefaults applied.
    data_w  : full dataset in Watts.

    Rule for every parameter:
      - non-zero in raw_em  → from nameplate  (user saved it)
      - zero / missing      → estimate from data (or assume default for PF/eta)

    Extra keys added:
      p_rated_elec_kw, p_rated_source, v_nominal_source,
      i_rated_source, pf_rated_source, eta_rated_source
    """
    meta = dict(saved_meta)
    raw  = raw_em or {}

    power_cols   = ["phase_1_active_power", "phase_2_active_power", "phase_3_active_power"]
    voltage_cols = ["phase_1_voltage",      "phase_2_voltage",      "phase_3_voltage"]
    current_cols = ["phase_1_current",      "phase_2_current",      "phase_3_current"]
    has_power   = all(c in data_w.columns for c in power_cols)
    has_voltage = all(c in data_w.columns for c in voltage_cols)
    has_current = all(c in data_w.columns for c in current_cols)

    _pt = (data_w[power_cols[0]] + data_w[power_cols[1]] + data_w[power_cols[2]]
           if has_power else pd.Series(dtype=float))
    _running = (_pt > 0) if len(_pt) > 0 else pd.Series(dtype=bool)

    # ── Efficiency (needed first for p_rated_elec) ───────────────────────────
    _eta_raw = float(raw.get("eta_rated", 0))
    if _eta_raw > 0:
        meta["eta_rated"]        = _eta_raw
        meta["eta_rated_source"] = "nameplate"
    else:
        meta["eta_rated"]        = 0.90
        meta["eta_rated_source"] = "assumed_default"
    eta = meta["eta_rated"]

    # ── Shaft power ──────────────────────────────────────────────────────────
    _p_raw = float(raw.get("p_rated_shaft_kw", 0))
    if _p_raw > 0:
        meta["p_rated_shaft_kw"] = _p_raw
        meta["p_rated_elec_kw"]  = round(_p_raw / eta, 3)
        meta["p_rated_source"]   = "nameplate"
    else:
        _p95_kw = float(_pt[_pt > 0].quantile(0.95)) / 1000.0 if has_power and (_pt > 0).any() else 0.0
        if _p95_kw > 0:
            p_elec = _p95_kw / 0.95
            meta["p_rated_source"] = "estimated_from_data"
        else:
            p_elec = 1.0
            meta["p_rated_source"] = "assumed_default"
        meta["p_rated_elec_kw"]  = round(p_elec, 3)
        meta["p_rated_shaft_kw"] = round(p_elec * eta, 3)

    # ── Voltage ──────────────────────────────────────────────────────────────
    _v_raw = float(raw.get("v_nominal_phase", 0))
    if _v_raw > 0:
        meta["v_nominal_phase"]  = _v_raw
        meta["v_nominal_source"] = "nameplate"
    elif has_voltage and _running.any():
        _v_all    = pd.concat([data_w.loc[_running, c] for c in voltage_cols])
        _v_valid  = _v_all[_v_all > 10]
        if len(_v_valid) > 0:
            meta["v_nominal_phase"]  = round(float(_v_valid.median()), 1)
            meta["v_nominal_source"] = "estimated_from_data"
        else:
            meta["v_nominal_phase"]  = 230.0
            meta["v_nominal_source"] = "assumed_default"
    else:
        meta["v_nominal_phase"]  = 230.0
        meta["v_nominal_source"] = "assumed_default"

    # ── Current ──────────────────────────────────────────────────────────────
    _i_raw = float(raw.get("i_rated", 0))
    if _i_raw > 0:
        meta["i_rated"]        = _i_raw
        meta["i_rated_source"] = "nameplate"
    elif has_current and _running.any():
        _i_avg = (data_w.loc[_running, current_cols[0]] +
                  data_w.loc[_running, current_cols[1]] +
                  data_w.loc[_running, current_cols[2]]) / 3.0
        _i95 = float(_i_avg[_i_avg > 0].quantile(0.95)) if (_i_avg > 0).any() else 0.0
        if _i95 > 0:
            meta["i_rated"]        = round(_i95 / 0.95, 1)
            meta["i_rated_source"] = "estimated_from_data"
        else:
            meta["i_rated"]        = 0.0
            meta["i_rated_source"] = "assumed_default"
    else:
        meta["i_rated"]        = 0.0
        meta["i_rated_source"] = "assumed_default"

    # ── PF ───────────────────────────────────────────────────────────────────
    _pf_raw = float(raw.get("pf_rated", 0))
    if _pf_raw > 0:
        meta["pf_rated"]        = _pf_raw
        meta["pf_rated_source"] = "nameplate"
    else:
        meta["pf_rated"]        = 0.87
        meta["pf_rated_source"] = "assumed_default"

    return meta


def apply_integrity_filter(
    df: pd.DataFrame,
    passed_ts: set | None,
) -> tuple[pd.DataFrame, int]:
    """Filter a DataFrame to only rows whose timestamp passed integrity checks.

    Parameters
    ----------
    df          : DataFrame with DatetimeIndex (timestamp as index)
    passed_ts   : set of ISO timestamp strings that passed all checks,
                  or None if the integrity check has not been run yet

    Returns
    -------
    (filtered_df, n_excluded)
    If passed_ts is None (check not run), returns df unchanged with n_excluded=0.
    """
    if passed_ts is None or len(passed_ts) == 0:
        return df, 0
    ts_strs  = df.index.astype(str)
    mask     = ts_strs.isin(passed_ts)
    excluded = int((~mask).sum())
    return df[mask], excluded


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
            ("Raw samples",                         report.n_raw),
            ("Step 1 \u2014 Load \u226520% rated",  report.n_after_load_precondition),
            ("Step 2 \u2014 Start transient",        getattr(report, "n_after_start_transient",
                                                     report.n_after_load_precondition)),
            ("Step 3 \u2014 User filter",            report.n_after_user_filter),
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
                   f"critical \u2264 {PF_DRIFT_ACTION:.2f})")
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
    total_bands  = len(m.bands)
    if active_bands:
        with st.expander(
            f"PF Drift \u2014 {len(active_bands)} active band(s) of {total_bands} total",
            expanded=False,
        ):
            st.caption(
                "Each band is an equal-width power bin (bin width = 1% of actual operating range, giving 100 bins). "
                "Only bins with \u22655 samples in both baseline and recent period are used for PF drift calculation. "
                "**Baseline PF** = mean PF during the ingested baseline period. "
                "**Recent PF** = mean PF during the selected assessment date range. "
                "**Drift** = Recent \u2212 Baseline (negative = degradation). "
                "**Drift %** = Drift as percentage of Baseline PF.  \n"
                "**p-value** = Welch\u2019s t-test (two-tailed). "
                "**Significant** = Yes if p < 0.05 (drift exceeds normal statistical variation).  \n"
                f"\U0001f7e1 Watch: drift \u2264 {PF_DRIFT_WATCH*100:.0f}%  \u2002"
                                f"\U0001f534 Action: drift \u2264 {PF_DRIFT_ACTION*100:.0f}%"
            )
            # Table — convert band centres from W to kW for display
            rows = []
            for b in active_bands:
                drift = b.pf_drift if b.pf_drift is not None else 0.0
                drift_pct = (drift / b.mean_pf_baseline * 100) if b.mean_pf_baseline else 0.0
                if drift <= PF_DRIFT_ACTION:
                    status = "\U0001f534 Action"

                    status = "\U0001f7e0 Alert"
                elif drift <= PF_DRIFT_WATCH:
                    status = "\U0001f7e1 Watch"
                else:
                    status = "\U0001f7e2 Normal"
                rows.append({
                    "Low (kW)":          f"{b.low_kw / 1000:.3f}" if b.low_kw else "\u2014",
                    "Centre (kW)":       f"{b.centre_kw / 1000:.3f}",
                    "High (kW)":         f"{b.high_kw / 1000:.3f}" if b.high_kw else "\u2014",
                    "Baseline PF":       f"{b.mean_pf_baseline:.4f}",
                    "Baseline \u03c3":   f"{b.std_pf_baseline:.5f}",
                    "Recent PF":         f"{b.mean_pf_recent:.4f}" if b.mean_pf_recent is not None else "\u2014",
                    "Recent \u03c3":     f"{b.std_pf_recent:.5f}" if b.std_pf_recent is not None else "\u2014",
                    "Drift":             f"{b.pf_drift:+.4f}" if b.pf_drift is not None else "\u2014",
                    "Drift %":           f"{drift_pct:+.2f}%" if b.pf_drift is not None else "\u2014",
                    "p-value":           f"{b.p_value:.4f}" if b.p_value is not None else "\u2014",
                    "Significant":       ("\u2705 Yes" if b.drift_significant
                                          else ("\u274c No" if b.drift_significant is False
                                                else "\u2753 n/a")),
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

    # Per-phase PF drift tables
    _ph_bands_all = st.session_state.get("last_phase_bands") or {}
    for _ph in (1, 2, 3):
        _ph_bands = _ph_bands_all.get(_ph, [])
        _ph_active = [b for b in _ph_bands if not b.suppressed and b.pf_drift is not None]
        if not _ph_active:
            continue
        _ph_total = len(_ph_bands)
        with st.expander(
            f"Phase {_ph} PF Drift \u2014 {len(_ph_active)} active band(s) of {_ph_total} total",
            expanded=False,
        ):
            st.caption(
                f"Bins defined by Phase {_ph} power P_{_ph} (1% of phase operating range → 100 bins). "
                f"PF = P_{_ph} / (V_{_ph} \u00d7 I_{_ph}). Fully independent of other phases. "
                "**p-value** = Welch\u2019s t-test. "
                "**Significant** = Yes if p < 0.05.  \n"
                f"\U0001f7e1 Watch: \u2264 {PF_DRIFT_WATCH*100:.0f}%  \u2002"
                                f"\U0001f534 Action: \u2264 {PF_DRIFT_ACTION*100:.0f}%"
            )
            _ph_rows = []
            for b in _ph_active:
                drift = b.pf_drift
                drift_pct = (drift / b.mean_pf_baseline * 100) if b.mean_pf_baseline else 0.0
                if drift <= PF_DRIFT_ACTION:   status = "\U0001f534 Critical"
                elif drift <= PF_DRIFT_WATCH:  status = "\U0001f7e1 Watch"
                else:                          status = "\U0001f7e2 Normal"
                _ph_rows.append({
                    "Low (kW)":        f"{b.low_kw / 1000:.3f}",
                    "Centre (kW)":     f"{b.centre_kw / 1000:.3f}",
                    "High (kW)":       f"{b.high_kw / 1000:.3f}",
                    "Baseline PF":     f"{b.mean_pf_baseline:.4f}",
                    "Baseline \u03c3": f"{b.std_pf_baseline:.5f}",
                    "Recent PF":       f"{b.mean_pf_recent:.4f}" if b.mean_pf_recent is not None else "\u2014",
                    "Recent \u03c3":   f"{b.std_pf_recent:.5f}"  if b.std_pf_recent  is not None else "\u2014",
                    "Drift":           f"{drift:+.4f}",
                    "Drift %":         f"{drift_pct:+.2f}%",
                    "p-value":         f"{b.p_value:.4f}" if b.p_value is not None else "\u2014",
                    "Significant":     ("\u2705 Yes" if b.drift_significant
                                        else ("\u274c No" if b.drift_significant is False
                                              else "\u2753 n/a")),
                    "Status":          status,
                    "n baseline":      b.n_baseline,
                    "n recent":        b.n_recent,
                })
            st.dataframe(pd.DataFrame(_ph_rows), use_container_width=True, hide_index=True)


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




from io import BytesIO as _BytesIO

# ---------------------------------------------------------------------------
# Assessment report generator
# ---------------------------------------------------------------------------

def _fig_to_png_bytes(go_fig, figsize=(8.5, 2.8), dpi=130) -> bytes:
    """Convert a go.Figure (Scatter traces) to PNG bytes via matplotlib.

    Handles:
    * go.Scatter — lines / markers / lines+markers
    * go.Indicator (gauge) — rendered as a coloured value box
    * layout.shapes (add_hline results) — drawn as axhline
    * layout.annotations — drawn as ax.annotate for threshold labels
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    is_gauge = any(hasattr(t, "value") and not hasattr(t, "x") for t in go_fig.data)
    if is_gauge:
        # Render gauge as a simple coloured box
        trace = go_fig.data[0]
        val   = trace.value if trace.value is not None else 0.0
        title = (go_fig.layout.title.text or "").replace("<br>", "\n").replace("<sup>", "").replace("</sup>", "")
        # Derive fill colour from bar colour (set by _gauge/_gauge_inverted)
        bar_col = "#177E40"
        if hasattr(trace, "gauge") and trace.gauge and trace.gauge.bar:
            bar_col = trace.gauge.bar.color or bar_col
        fig_m, ax = plt.subplots(figsize=(4, 1.6))
        ax.set_facecolor(bar_col)
        fig_m.patch.set_facecolor(bar_col)
        ax.text(0.5, 0.6, f"{val:.2f}", ha="center", va="center",
                fontsize=32, fontweight="bold", color="white",
                transform=ax.transAxes)
        ax.text(0.5, 0.15, title.split("\n")[0], ha="center", va="center",
                fontsize=9, color="white", alpha=0.85,
                transform=ax.transAxes)
        ax.axis("off")
        buf = _BytesIO()
        fig_m.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                      facecolor=bar_col)
        plt.close(fig_m)
        buf.seek(0)
        return buf.read()

    fig_m, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#FAFBFC")
    fig_m.patch.set_facecolor("white")

    _is_ts = False   # will detect if x axis is datetime
    for trace in go_fig.data:
        if not hasattr(trace, "x") or trace.x is None:
            continue
        xs = list(trace.x)
        ys = list(trace.y) if trace.y is not None else []
        if not xs or not ys:
            continue

        # Detect datetime x-axis
        if xs and hasattr(xs[0], "year"):
            _is_ts = True
        elif xs and isinstance(xs[0], str):
            try:
                import pandas as _pd
                xs = list(_pd.to_datetime(xs))
                _is_ts = True
            except Exception:
                pass

        col   = "#888888"
        alpha = 1.0
        lw    = 1.5
        mode  = trace.mode or "lines"
        label = trace.name or ""

        if hasattr(trace, "line") and trace.line:
            col = trace.line.color or col
            if trace.line.width:
                lw = float(trace.line.width)
        if hasattr(trace, "opacity") and trace.opacity:
            alpha = float(trace.opacity)
        # Grey background traces get lower alpha
        if "All data" in label or "rgba(180" in col:
            col, alpha, lw = "#BBBBBB", 0.4, 0.8

        if "lines" in mode:
            ax.plot(xs, ys, color=col, linewidth=lw, alpha=alpha,
                    label=label, solid_capstyle="round")
        elif "markers" in mode:
            ax.scatter(xs, ys, color=col, s=8, alpha=alpha, label=label)

    # Threshold hlines from layout.shapes
    for shape in (go_fig.layout.shapes or []):
        if getattr(shape, "type", None) == "line":
            y0 = getattr(shape, "y0", None)
            y1 = getattr(shape, "y1", None)
            if y0 is not None and y0 == y1:
                sc = getattr(shape.line, "color", "#888") if shape.line else "#888"
                sd = "dashed" if getattr(shape.line, "dash", "") in ("dash", "dashdot") else "solid"
                ax.axhline(y=y0, color=sc, linestyle=sd, linewidth=1.0, alpha=0.75)

    # Labels
    layout = go_fig.layout
    title  = (layout.title.text if layout.title and layout.title.text else "")
    title  = title.replace("<br>", " ").replace("<sup>", "").replace("</sup>", "").replace("&#x2014;", "—")
    ax.set_title(title, fontsize=10, pad=6, loc="left", color="#333")

    if layout.xaxis and layout.xaxis.title and layout.xaxis.title.text:
        ax.set_xlabel(layout.xaxis.title.text, fontsize=8)
    if layout.yaxis and layout.yaxis.title and layout.yaxis.title.text:
        ax.set_ylabel(layout.yaxis.title.text, fontsize=8)
    if layout.yaxis and layout.yaxis.range:
        try:
            ax.set_ylim(layout.yaxis.range[0], layout.yaxis.range[1])
        except Exception:
            pass

    if _is_ts:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig_m.autofmt_xdate(rotation=30, ha="right")
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6, color="#CCCCCC")
    ax.spines[["top", "right"]].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(handles, labels, fontsize=7, loc="upper right",
                  framealpha=0.7, edgecolor="none")

    fig_m.tight_layout(pad=0.4)
    buf = _BytesIO()
    fig_m.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig_m)
    buf.seek(0)
    return buf.read()


def generate_assessment_report_pdf(
    record: "AssessmentRecord",
    meta: dict,
    data: "pd.DataFrame",
    cleaned_data: "pd.DataFrame | None",
    phase_bands: dict,
    gauge_thresholds: dict,
    figs: list,
) -> bytes:
    """Build a ReportLab PDF assessment report and return as bytes.

    Charts are rendered via matplotlib from the Plotly figure trace data.
    No kaleido / Chrome required.
    """
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, HRFlowable, PageBreak, KeepTogether,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

    PAGE_W, PAGE_H = A4
    MARGIN = 1.8 * cm
    COL_W  = PAGE_W - 2 * MARGIN

    # ── Brand colours ────────────────────────────────────────────────────────
    TEAL   = rl_colors.HexColor("#054D5F")
    GOLD   = rl_colors.HexColor("#C8A84B")
    GREEN  = rl_colors.HexColor("#177E40")
    AMBER  = rl_colors.HexColor("#E67E22")
    RED    = rl_colors.HexColor("#C0392B")
    LTGREY = rl_colors.HexColor("#F5F7FA")
    MGREY  = rl_colors.HexColor("#D0D8E4")
    DGREY  = rl_colors.HexColor("#4A5568")

    _TIER_COL = {"critical": RED, "watch": AMBER, None: GREEN, "none": GREEN}
    _TIER_LBL = {"critical": "CRITICAL", "watch": "WATCH", None: "NORMAL", "none": "NORMAL"}

    def _tier_col(tier): return _TIER_COL.get(tier, DGREY)
    def _tier_lbl(tier): return _TIER_LBL.get(tier, "—")

    # ── Styles ────────────────────────────────────────────────────────────────
    base = getSampleStyleSheet()

    def _style(name, parent="Normal", **kw):
        s = ParagraphStyle(name, parent=base[parent], **kw)
        return s

    S = {
        "title":    _style("title",   "Title",   fontSize=18, textColor=TEAL,  spaceAfter=2),
        "sub":      _style("sub",     "Normal",  fontSize=9,  textColor=DGREY, spaceAfter=6),
        "h2":       _style("h2",      "Heading2",fontSize=13, textColor=TEAL,  spaceBefore=14, spaceAfter=4),
        "h3":       _style("h3",      "Heading3",fontSize=11, textColor=TEAL,  spaceBefore=8,  spaceAfter=3),
        "body":     _style("body",    "Normal",  fontSize=9,  leading=13),
        "small":    _style("small",   "Normal",  fontSize=8,  textColor=DGREY),
        "bold":     _style("bold",    "Normal",  fontSize=9,  fontName="Helvetica-Bold"),
        "badge_ok": _style("badge_ok","Normal",  fontSize=8,  textColor=GREEN, fontName="Helvetica-Bold"),
        "badge_wa": _style("badge_wa","Normal",  fontSize=8,  textColor=AMBER, fontName="Helvetica-Bold"),
        "badge_cr": _style("badge_cr","Normal",  fontSize=8,  textColor=RED,   fontName="Helvetica-Bold"),
    }
    def _badge_style(tier):
        return {"critical": S["badge_cr"], "watch": S["badge_wa"]}.get(tier, S["badge_ok"])

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _tbl(data_rows, col_widths, style_cmds=None):
        ts = TableStyle([
            ("BACKGROUND", (0,0), (-1,0), LTGREY),
            ("TEXTCOLOR",  (0,0), (-1,0), TEAL),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [rl_colors.white, LTGREY]),
            ("GRID",       (0,0), (-1,-1), 0.3, MGREY),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ])
        if style_cmds:
            for cmd in style_cmds:
                ts.add(*cmd)
        return Table(data_rows, colWidths=col_widths, style=ts, repeatRows=1)

    def _fig_img(go_fig, w_cm=16, h_cm=4.5, gauge=False):
        """Convert a go.Figure to a ReportLab Image flowable."""
        png = _fig_to_png_bytes(go_fig, figsize=(w_cm*0.5, h_cm*0.5) if not gauge else (4, 1.6))
        buf = _BytesIO(png)
        return Image(buf, width=w_cm*cm, height=h_cm*cm, kind="proportional")

    def _hr(): return HRFlowable(width="100%", thickness=0.5,
                                 color=MGREY, spaceAfter=4, spaceBefore=4)

    # ── Data ──────────────────────────────────────────────────────────────────
    now_str   = datetime.now().strftime("%Y-%m-%d %H:%M")
    machine   = meta.get("machine_name") or meta.get("machine_id") or "—"
    mtype     = meta.get("machine_type", "—")
    mapp      = meta.get("application_type", "—")
    p_shaft   = meta.get("p_rated_shaft_kw")
    p_shaft_s = f"{p_shaft:.1f} kW" if p_shaft else "—"
    eta       = meta.get("eta_rated")
    eta_s     = f"{eta:.2f}" if eta else "—"

    if data is not None and len(data) > 0:
        idx = pd.to_datetime(data.index)
        period_str = f"{idx.min().strftime('%Y-%m-%d')}  to  {idx.max().strftime('%Y-%m-%d')}"
        n_raw = len(data)
    else:
        period_str = "—"
        n_raw = 0
    n_clean = len(cleaned_data) if cleaned_data is not None else 0
    ret_pct = f"{100 * n_clean / n_raw:.0f}%" if n_raw > 0 else "—"

    # Zone tiers
    z1_tier = record.supply_alarm.tier    if record.supply_alarm  else None
    z1_vuf  = record.supply_alarm.vuf_pct if record.supply_alarm  else None
    z2_tier = record.motor_side.iuf_tier     if record.motor_side else None
    z2_iuf  = record.motor_side.iuf_mean_pct if record.motor_side else None

    mside_pf_tier = None
    if record.motor_side and record.motor_side.bands:
        sig = [b for b in record.motor_side.bands if b.drift_significant]
        if sig:
            worst = min(b.pf_drift for b in sig)
            mside_pf_tier = ("critical" if worst <= -0.03 else
                             "watch"    if worst <= -0.02 else None)

    tiers   = [z1_tier, z2_tier, mside_pf_tier]
    overall = ("critical" if "critical" in tiers else
               "watch"    if "watch"    in tiers else None)

    # ── Story ─────────────────────────────────────────────────────────────────
    story = []

    # ── Cover / Header ────────────────────────────────────────────────────────
    header_data = [[
        Paragraph("<b><font color='#054D5F' size=16>Symbion</font>"
                  "<font color='#C8A84B' size=16> Machine Analytics</font></b><br/>"
                  "<font size=9 color='#6B7280'>Machine Health Assessment Report</font>", S["body"]),
        Paragraph(f"<b>{machine}</b><br/>"
                  f"<font size=8 color='#6B7280'>Generated: {now_str}<br/>"
                  f"Period: {period_str}</font>", ParagraphStyle("rt", parent=S["body"],
                  alignment=TA_RIGHT)),
    ]]
    header_tbl = Table(header_data, colWidths=[COL_W*0.6, COL_W*0.4])
    header_tbl.setStyle(TableStyle([
        ("LINEBELOW", (0,0), (-1,0), 2, TEAL),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 10))

    # ── Overall health banner ─────────────────────────────────────────────────
    ov_col  = _tier_col(overall)
    ov_lbl  = _tier_lbl(overall)
    ov_msg  = ("No issues detected across all diagnostic zones." if overall is None else
               "One or more diagnostic zones require attention — see zone findings below.")
    banner_data = [[
        Paragraph(f"<b><font size=13 color='#{ov_col.hexval()[2:]}'>Overall Health: {ov_lbl}</font></b><br/>"
                  f"<font size=9>{ov_msg}</font>", S["body"]),
    ]]
    banner_tbl = Table(banner_data, colWidths=[COL_W])
    banner_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), rl_colors.HexColor(
            "#EAF7EE" if overall is None else "#FDECEA" if overall == "critical" else "#FEF3E7")),
        ("LEFTPADDING",  (0,0), (-1,-1), 12),
        ("TOPPADDING",   (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0), (-1,-1), 8),
        ("LINEBEFORECOLOR", (0,0), (0,-1), ov_col),
        ("LINEBEFORE",   (0,0), (0,-1), 4, ov_col),
        ("ROUNDEDCORNERS", (0,0), (-1,-1), 4),
    ]))
    story.append(banner_tbl)
    story.append(Spacer(1, 10))

    # ── Machine details ────────────────────────────────────────────────────────
    story.append(Paragraph("Machine Details", S["h2"]))
    meta_rows = [
        ["Machine", machine, "Type", mtype],
        ["Application", mapp, "Rated shaft power", p_shaft_s],
        ["Rated efficiency", eta_s, "Assessment period", period_str],
    ]
    meta_tbl = _tbl(
        [[Paragraph(c, S["bold"] if i%2==0 else S["body"]) for i,c in enumerate(r)]
         for r in meta_rows],
        [COL_W*0.2, COL_W*0.3, COL_W*0.2, COL_W*0.3],
    )
    story.append(meta_tbl)
    story.append(Spacer(1, 8))

    # ── Data quality ──────────────────────────────────────────────────────────
    story.append(Paragraph("Data Quality", S["h2"]))
    cr = record.cleaning_report
    if cr:
        n_st  = getattr(cr, "n_after_start_transient", cr.n_after_load_precondition)
        cl_rows = [
            [Paragraph(h, S["bold"]) for h in ["Step", "Samples", "Removed"]],
            ["Raw samples",              f"{cr.n_raw:,}",                      "—"],
            ["Step 1 — Load \u226520%",  f"{cr.n_after_load_precondition:,}",
             Paragraph(f'<font color="#C0392B">-{cr.n_raw - cr.n_after_load_precondition:,}</font>',S["body"])
             if cr.n_raw > cr.n_after_load_precondition else "0"],
            ["Step 2 — Start transient", f"{n_st:,}",
             Paragraph(f'<font color="#C0392B">-{cr.n_after_load_precondition - n_st:,}</font>',S["body"])
             if cr.n_after_load_precondition > n_st else "0"],
            ["Step 3 — User filter",     f"{cr.n_after_user_filter:,}",
             Paragraph(f'<font color="#C0392B">-{n_st - cr.n_after_user_filter:,}</font>',S["body"])
             if n_st > cr.n_after_user_filter else "0"],
            [Paragraph("<b>Cleaned (analysis)</b>", S["body"]), f"{cr.n_cleaned:,}", ""],
        ]
        story.append(_tbl(cl_rows, [COL_W*0.55, COL_W*0.22, COL_W*0.23]))
        story.append(Paragraph(f"{ret_pct} of raw samples retained for analysis.",
                                ParagraphStyle("gr", parent=S["small"], textColor=GREEN)))
    story.append(Spacer(1, 8))

    # ── Zone findings ─────────────────────────────────────────────────────────
    story.append(Paragraph("Zone Diagnostic Findings", S["h2"]))

    def _zone_row(z_title, tier, msg, thresh_str):
        col = _tier_col(tier)
        lbl = _tier_lbl(tier)
        return KeepTogether([
            Table([[
                Paragraph(f"<b>{z_title}</b>", S["bold"]),
                Paragraph(f"<b><font color='#{col.hexval()[2:]}'>{lbl}</font></b>",
                          S["body"]),
            ]], colWidths=[COL_W*0.75, COL_W*0.25],
                style=TableStyle([
                    ("BACKGROUND", (0,0), (-1,-1), LTGREY),
                    ("TOPPADDING", (0,0), (-1,-1), 5),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                    ("LEFTPADDING", (0,0), (-1,-1), 8),
                    ("LINEBEFORE", (0,0), (0,-1), 3, col),
                ])),
            Table([[
                Paragraph(msg, S["body"]),
            ], [
                Paragraph(thresh_str, S["small"]),
            ]], colWidths=[COL_W],
                style=TableStyle([
                    ("TOPPADDING", (0,0), (-1,-1), 3),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
                    ("LEFTPADDING", (0,0), (-1,-1), 14),
                ])),
            Spacer(1, 6),
        ])

    z1_msg = (f"VUF = {z1_vuf:.2f}% — " +
              ("above critical threshold." if z1_tier == "critical" else
               "above watch threshold." if z1_tier == "watch" else
               "within normal limits.")) if z1_vuf is not None else "No supply data."
    story.append(_zone_row("Zone 1 — Supply Quality (VUF)", z1_tier, z1_msg,
        f"Watch \u2265{gauge_thresholds.get('vuf_watch',1.0):.1f}%  |  "
        f"Critical \u2265{gauge_thresholds.get('vuf_critical',2.0):.1f}%"))

    z2_msg = (f"IUF = {z2_iuf:.1f}% — " +
              ("above critical threshold." if z2_tier == "critical" else
               "above watch threshold." if z2_tier == "watch" else
               "within normal limits.")) if z2_iuf is not None else "No current data."
    story.append(_zone_row("Zone 2 — Current Imbalance (IUF)", z2_tier, z2_msg,
        f"Watch \u2265{gauge_thresholds.get('iuf_watch',5.0):.0f}%  |  "
        f"Critical \u2265{gauge_thresholds.get('iuf_critical',10.0):.0f}%"))

    n_sig = sum(1 for b in record.motor_side.bands if b.drift_significant) \
            if record.motor_side else 0
    z3_msg = (f"Statistically significant PF drift in {n_sig} load band(s)."
              if n_sig else "No statistically significant PF drift detected.")
    story.append(_zone_row("Zone 3 — Motor Health (PF Drift)", mside_pf_tier, z3_msg,
        "Watch \u2264-0.01  |  Alert \u2264-0.02  |  Action \u2264-0.03 (absolute PF)"))

    z4_msg = "—"
    if record.zone4:
        z4r = record.zone4
        if hasattr(z4r, "finding") and z4r.finding:
            z4_msg = z4r.finding
        elif hasattr(z4r, "delta_pct") and z4r.delta_pct is not None:
            z4_msg = f"Power change: {z4r.delta_pct:+.1f}% vs baseline average."
    story.append(_zone_row("Zone 4 — Driven Equipment", None, z4_msg, ""))
    story.append(Spacer(1, 6))

    # ── Recommendations ───────────────────────────────────────────────────────
    story.append(Paragraph("Recommendations", S["h2"]))
    recs = []
    if z1_tier == "critical": recs.append(("Critical", "Zone 1", "Investigate supply voltage quality immediately. Check upstream transformer and busbars."))
    elif z1_tier == "watch":  recs.append(("Watch",    "Zone 1", "Monitor supply voltage balance. Check for single-phase loads on the feeder."))
    if z2_tier == "critical": recs.append(("Critical", "Zone 2", "Current imbalance critical — inspect cabling, contactor, and fuse on all three phases."))
    elif z2_tier == "watch":  recs.append(("Watch",    "Zone 2", "Elevated current imbalance — check panel connections and phase fuse ratings."))
    if mside_pf_tier == "critical": recs.append(("Critical", "Zone 3", "Significant PF degradation. Schedule motor inspection: winding insulation, bearing condition."))
    elif mside_pf_tier == "watch":  recs.append(("Watch",    "Zone 3", "Developing PF drift. Monitor closely; plan inspection at next maintenance window."))
    if not recs: recs.append(("Normal", "All zones", "No corrective action required. Continue scheduled monitoring."))

    rec_data = [[Paragraph(h, S["bold"]) for h in ["Priority", "Zone", "Recommended Action"]]]
    for tier, zone, text in recs:
        col = _tier_col(tier.lower() if tier != "Normal" else None)
        rec_data.append([
            Paragraph(f"<b><font color='#{col.hexval()[2:]}'>{tier}</font></b>", S["body"]),
            Paragraph(zone, S["bold"]),
            Paragraph(text, S["body"]),
        ])
    story.append(_tbl(rec_data, [COL_W*0.14, COL_W*0.14, COL_W*0.72]))
    story.append(Spacer(1, 6))

    # ── PF Drift tables ───────────────────────────────────────────────────────
    if record.motor_side and record.motor_side.bands:
        sig_bands = [b for b in record.motor_side.bands
                     if not b.suppressed and b.pf_drift is not None and b.drift_significant]
        if sig_bands:
            story.append(Paragraph("PF Drift — Significant Bands (Machine Level)", S["h2"]))
            d_data = [[Paragraph(h, S["bold"]) for h in
                       ["Band centre (kW)", "Baseline PF", "Recent PF", "Drift"]]]
            for b in sig_bands[:20]:
                dc = ("#C0392B" if b.pf_drift <= -0.03 else
                      "#E67E22" if b.pf_drift <= -0.02 else "#333333")
                d_data.append([
                    f"{b.centre_kw/1000:.2f}",
                    f"{b.mean_pf_baseline:.4f}",
                    f"{b.mean_pf_recent:.4f}",
                    Paragraph(f"<b><font color='{dc}'>{b.pf_drift:+.4f}</font></b>", S["body"]),
                ])
            story.append(_tbl(d_data, [COL_W*0.25]*4))
            story.append(Spacer(1, 6))

    # ── Charts ────────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Diagnostic Charts", S["h2"]))
    story.append(Paragraph(
        "Blue = cleaned data used for analysis  |  Grey = all raw data",
        S["small"]))
    story.append(Spacer(1, 6))

    chart_labels = [
        "Daily Imbalance Run Chart — VUF & IUF",
        "Zone 1 — VUF Gauge",
        "Zone 2 — IUF Gauge",
        "P_total Time Series", "PF Machine Time Series", "PF Gauge",
    ]
    drift_labels = ["PF Drift — Machine", "PF Drift — Phase 1",
                    "PF Drift — Phase 2", "PF Drift — Phase 3"]
    all_labels = chart_labels + drift_labels

    for i, go_fig in enumerate(figs):
        label = all_labels[i] if i < len(all_labels) else f"Chart {i+1}"
        is_gauge = any(hasattr(t, "value") and not hasattr(t, "x")
                       for t in go_fig.data)
        try:
            if is_gauge:
                img = _fig_img(go_fig, w_cm=6, h_cm=3.2, gauge=True)
                story.append(KeepTogether([
                    Paragraph(label, S["h3"]),
                    img,
                    Spacer(1, 8),
                ]))
            else:
                img = _fig_img(go_fig, w_cm=16, h_cm=4.5)
                story.append(KeepTogether([
                    Paragraph(label, S["h3"]),
                    img,
                    Spacer(1, 8),
                ]))
        except Exception as _ce:
            story.append(Paragraph(f"[Chart could not be rendered: {_ce}]", S["small"]))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(_hr())
    story.append(Table([[
        Paragraph("Symbion Machine Analytics Platform  |  Methodology v0.8", S["small"]),
        Paragraph(f"Report generated {now_str}  |  Confidential",
                  ParagraphStyle("fr", parent=S["small"], alignment=TA_RIGHT)),
    ]], colWidths=[COL_W*0.6, COL_W*0.4],
        style=TableStyle([("TOPPADDING",(0,0),(-1,-1),3)])))

    # ── Build ─────────────────────────────────────────────────────────────────
    buf = _BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title=f"Machine Health Assessment — {machine}",
        author="Symbion Machine Analytics",
    )
    doc.build(story)
    return buf.getvalue()



def generate_assessment_report_html(
    record: "AssessmentRecord",
    meta: dict,
    data: "pd.DataFrame",
    cleaned_data: "pd.DataFrame | None",
    phase_bands: dict,
    gauge_thresholds: dict,
    figs: list,
) -> str:
    """Generate a self-contained HTML assessment report.

    Parameters
    ----------
    record           : completed AssessmentRecord
    meta             : effective machine metadata dict
    data             : raw measurement DataFrame (full window)
    cleaned_data     : cleaned DataFrame used for analysis
    phase_bands      : {1: [BandRecord], ...} per-phase drift bands
    gauge_thresholds : {vuf_watch, vuf_critical, iuf_watch, iuf_critical,
                        pf_watch, pf_critical}
    figs             : list of go.Figure from build_assessment_charts
    """
    import io as _io

    now_str   = datetime.now().strftime("%Y-%m-%d %H:%M")
    machine   = meta.get("machine_name") or meta.get("machine_id") or "—"
    mtype     = meta.get("machine_type", "—")
    mapp      = meta.get("application_type", "—")
    p_shaft   = meta.get("p_rated_shaft_kw")
    p_shaft_s = f"{p_shaft:.1f} kW" if p_shaft else "—"
    eta       = meta.get("eta_rated")
    eta_s     = f"{eta:.2f}" if eta else "—"

    # ── Date range ───────────────────────────────────────────────────────────
    if data is not None and len(data) > 0:
        idx = pd.to_datetime(data.index)
        period_str = f"{idx.min().strftime('%Y-%m-%d')} → {idx.max().strftime('%Y-%m-%d')}"
        n_raw = len(data)
    else:
        period_str = "—"
        n_raw = 0
    n_clean = len(cleaned_data) if cleaned_data is not None else 0
    ret_pct = f"{100 * n_clean / n_raw:.0f}%" if n_raw > 0 else "—"

    # ── Zone colours ─────────────────────────────────────────────────────────
    _TIER_COL = {"critical": "#C0392B", "watch": "#E67E22", None: "#177E40", "none": "#177E40"}
    _TIER_BG  = {"critical": "#FDECEA", "watch":  "#FEF3E7", None: "#EAF7EE", "none": "#EAF7EE"}
    _TIER_LBL = {"critical": "CRITICAL", "watch": "WATCH",  None: "NORMAL",   "none": "NORMAL"}

    def _zone_badge(tier):
        col = _TIER_COL.get(tier, "#888")
        lbl = _TIER_LBL.get(tier, str(tier).upper() if tier else "—")
        return (f'<span style="background:{col};color:#fff;padding:2px 10px;'
                f'border-radius:10px;font-size:11px;font-weight:700;'
                f'letter-spacing:.06em">{lbl}</span>')

    # ── Overall health ────────────────────────────────────────────────────────
    tiers = []
    if record.supply_alarm:       tiers.append(record.supply_alarm.tier)
    if record.motor_side:         tiers.append(record.motor_side.iuf_tier)
    mside_pf_tier = None
    if record.motor_side and record.motor_side.bands:
        sig = [b for b in record.motor_side.bands if b.drift_significant]
        if sig:
            worst = min(b.pf_drift for b in sig)
            if worst <= -0.03:   mside_pf_tier = "critical"
            elif worst <= -0.02: mside_pf_tier = "watch"
            else:                mside_pf_tier = None
    tiers.append(mside_pf_tier)
    overall = "critical" if "critical" in tiers else ("watch" if "watch" in tiers else None)
    overall_col = _TIER_COL.get(overall)
    overall_lbl = _TIER_LBL.get(overall)

    # ── Chart HTML fragments ──────────────────────────────────────────────────
    chart_htmls = []
    first = True
    for fig in figs:
        fig2 = go.Figure(fig)
        fig2.update_layout(height=300, margin=dict(l=40, r=120, t=55, b=40))
        html_frag = pio.to_html(
            fig2,
            include_plotlyjs="cdn" if first else False,
            full_html=False,
            config={"displayModeBar": False, "responsive": True},
        )
        chart_htmls.append(html_frag)
        first = False

    # ── Zone 3 PF drift table ─────────────────────────────────────────────────
    def _drift_table(bands, label):
        rows = [b for b in bands if not b.suppressed and b.pf_drift is not None]
        if not rows:
            return f"<p style='color:#888;font-size:12px'>No qualifying bands for {label}.</p>"
        sig_rows = [b for b in rows if b.drift_significant]
        html = (
            '<table style="width:100%;border-collapse:collapse;font-size:12px">'
            '<thead><tr style="background:#F5F7FA">'
            '<th style="padding:6px 8px;text-align:left;border-bottom:2px solid #E0E4EA">Band centre (kW)</th>'
            '<th style="padding:6px 8px;text-align:right;border-bottom:2px solid #E0E4EA">Baseline PF</th>'
            '<th style="padding:6px 8px;text-align:right;border-bottom:2px solid #E0E4EA">Recent PF</th>'
            '<th style="padding:6px 8px;text-align:right;border-bottom:2px solid #E0E4EA">Drift</th>'
            '<th style="padding:6px 8px;text-align:center;border-bottom:2px solid #E0E4EA">Significant</th>'
            '</tr></thead><tbody>'
        )
        for b in sig_rows[:15]:
            drift_col = "#C0392B" if b.pf_drift <= -0.03 else ("#E67E22" if b.pf_drift <= -0.02 else "#333")
            html += (
                f'<tr style="border-bottom:1px solid #F0F0F0">'
                f'<td style="padding:5px 8px">{b.centre_kw/1000:.2f}</td>'
                f'<td style="padding:5px 8px;text-align:right">{b.mean_pf_baseline:.4f}</td>'
                f'<td style="padding:5px 8px;text-align:right">{b.mean_pf_recent:.4f}</td>'
                f'<td style="padding:5px 8px;text-align:right;color:{drift_col};font-weight:600">'
                f'{b.pf_drift:+.4f}</td>'
                f'<td style="padding:5px 8px;text-align:center">&#10003;</td>'
                f'</tr>'
            )
        html += '</tbody></table>'
        if len(rows) > 15:
            html += f'<p style="font-size:11px;color:#888">Showing top 15 of {len(sig_rows)} significant bands.</p>'
        return html

    machine_drift_table = ""
    if record.motor_side and record.motor_side.bands:
        machine_drift_table = _drift_table(record.motor_side.bands, "Machine")

    phase_drift_tables = ""
    for ph in (1, 2, 3):
        ph_bands = phase_bands.get(ph, [])
        if ph_bands:
            phase_drift_tables += (
                f'<h4 style="margin:16px 0 6px;font-size:13px;color:#054D5F">Phase {ph}</h4>'
                + _drift_table(ph_bands, f"Phase {ph}")
            )

    # ── Zone messages ─────────────────────────────────────────────────────────
    z1_tier = record.supply_alarm.tier    if record.supply_alarm  else None
    z1_vuf  = record.supply_alarm.vuf_pct if record.supply_alarm  else None
    z1_msg  = (f"VUF = {z1_vuf:.2f}% — "
               + ("above critical threshold." if z1_tier == "critical"
                  else "above watch threshold." if z1_tier == "watch"
                  else "within normal limits.")) if z1_vuf is not None else "No supply data."

    z2_tier = record.motor_side.iuf_tier     if record.motor_side else None
    z2_iuf  = record.motor_side.iuf_mean_pct if record.motor_side else None
    z2_msg  = (f"IUF = {z2_iuf:.1f}% — "
               + ("above critical threshold." if z2_tier == "critical"
                  else "above watch threshold." if z2_tier == "watch"
                  else "within normal limits.")) if z2_iuf is not None else "No current data."

    z4_msg  = "—"
    if record.zone4:
        z4r = record.zone4
        if hasattr(z4r, "finding") and z4r.finding:
            z4_msg = z4r.finding
        elif hasattr(z4r, "delta_pct") and z4r.delta_pct is not None:
            z4_msg = f"Power change: {z4r.delta_pct:+.1f}% vs baseline average."

    # ── Recommendations ───────────────────────────────────────────────────────
    recs = []
    if z1_tier == "critical": recs.append(("Critical", "Zone 1", "Investigate supply voltage quality immediately. Check upstream transformer and busbars."))
    elif z1_tier == "watch":  recs.append(("Watch",    "Zone 1", "Monitor supply voltage balance. Check for single-phase loads on the feeder."))
    if z2_tier == "critical": recs.append(("Critical", "Zone 2", "Current imbalance critical — inspect cabling, contactor, and fuse condition on all three phases."))
    elif z2_tier == "watch":  recs.append(("Watch",    "Zone 2", "Elevated current imbalance — check panel connections and phase fuse ratings."))
    if mside_pf_tier == "critical": recs.append(("Critical", "Zone 3", "Significant PF degradation detected. Schedule motor inspection: winding insulation, bearing condition."))
    elif mside_pf_tier == "watch":  recs.append(("Watch",    "Zone 3", "Developing PF drift. Monitor closely; plan inspection at next maintenance window."))
    if not recs: recs.append(("Normal", "All zones", "No corrective action required. Continue scheduled monitoring."))

    rec_rows = ""
    for tier, zone, text in recs:
        rc = _TIER_COL.get(tier.lower(), "#177E40")
        rec_rows += (
            f'<tr><td style="padding:7px 10px">{_zone_badge(tier.lower() if tier != "Normal" else None)}</td>'
            f'<td style="padding:7px 10px;font-weight:600;color:#054D5F">{zone}</td>'
            f'<td style="padding:7px 10px">{text}</td></tr>'
        )

    # ── Assemble charts into labelled sections ────────────────────────────────
    chart_labels = [
        "Daily Imbalance Run Chart \u2014 VUF & IUF",
        "Zone 1 \u2014 VUF Gauge",
        "Zone 2 \u2014 IUF Gauge",
        "P_total \u00b7 Time Series",
        "Machine Power Factor \u00b7 Time Series",
        "PF Gauge",
    ]
    # Drift charts follow (Machine + Phase 1/2/3)
    drift_labels = ["PF Drift · Machine", "PF Drift · Phase 1", "PF Drift · Phase 2", "PF Drift · Phase 3"]
    all_labels = chart_labels + drift_labels

    charts_html = ""
    for i, (frag, lbl) in enumerate(zip(chart_htmls, all_labels + [""] * max(0, len(chart_htmls) - len(all_labels)))):
        label = lbl or f"Chart {i+1}"
        charts_html += (
            f'<div style="margin:20px 0">'
            f'<div style="font-size:12px;font-weight:600;color:#4A5568;'
            f'margin-bottom:4px;letter-spacing:.04em;text-transform:uppercase">{label}</div>'
            f'{frag}</div>'
        )

    # ── HTML ──────────────────────────────────────────────────────────────────
    cr = record.cleaning_report
    cleaning_rows = ""
    if cr:
        steps = [
            ("Raw samples",                   cr.n_raw,                    None),
            ("Step 1 — Load \u226520% rated", cr.n_after_load_precondition, cr.n_raw - cr.n_after_load_precondition),
            ("Step 2 — Start transient",      getattr(cr, "n_after_start_transient", cr.n_after_load_precondition),
             cr.n_after_load_precondition - getattr(cr, "n_after_start_transient", cr.n_after_load_precondition)),
            ("Step 3 — User filter",          cr.n_after_user_filter,
             getattr(cr, "n_after_start_transient", cr.n_after_load_precondition) - cr.n_after_user_filter),
            ("Cleaned (used for analysis)",   cr.n_cleaned,                None),
        ]
        for label, count, removed in steps:
            rm_s = (f'<span style="color:#C0392B;font-weight:600">-{removed}</span>' if removed and removed > 0 else "")
            cleaning_rows += (
                f'<tr><td style="padding:5px 10px">{label}</td>'
                f'<td style="padding:5px 10px;text-align:right;font-weight:600">{count:,}</td>'
                f'<td style="padding:5px 10px;text-align:right">{rm_s}</td></tr>'
            )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Machine Health Assessment — {machine}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',Arial,sans-serif;font-size:14px;color:#333;background:#fff;padding:32px}}
  h1{{font-size:22px;color:#054D5F;margin-bottom:4px}}
  h2{{font-size:16px;color:#054D5F;margin:28px 0 10px;padding-bottom:6px;border-bottom:2px solid #054D5F}}
  h3{{font-size:14px;color:#054D5F;margin:16px 0 8px}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{background:#EEF2F7;padding:7px 10px;text-align:left;border-bottom:2px solid #D0D8E4;font-weight:600;color:#333}}
  td{{padding:6px 10px;border-bottom:1px solid #F0F2F5}}
  tr:last-child td{{border-bottom:none}}
  .header{{display:flex;justify-content:space-between;align-items:flex-start;
           border-bottom:3px solid #054D5F;padding-bottom:16px;margin-bottom:24px}}
  .logo{{font-size:24px;font-weight:700;color:#054D5F;letter-spacing:-.5px}}
  .logo span{{color:#C8A84B}}
  .meta-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin:12px 0}}
  .meta-box{{background:#F5F8FA;border-radius:6px;padding:10px 14px}}
  .meta-label{{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:#6B7280;margin-bottom:2px}}
  .meta-value{{font-size:15px;font-weight:600;color:#054D5F}}
  .health-banner{{border-radius:8px;padding:14px 20px;margin:16px 0;
                  background:{_TIER_BG.get(overall,'#EAF7EE')};
                  border-left:5px solid {overall_col}}}
  .health-title{{font-size:18px;font-weight:700;color:{overall_col}}}
  .zone-card{{border:1px solid #E0E4EA;border-radius:8px;padding:14px 18px;margin:10px 0;
              background:#FAFBFC}}
  .zone-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}}
  .zone-title{{font-size:14px;font-weight:700;color:#054D5F}}
  .zone-body{{font-size:13px;color:#444;line-height:1.5}}
  .footer{{margin-top:40px;padding-top:16px;border-top:1px solid #E0E4EA;
           font-size:11px;color:#9CA3AF;display:flex;justify-content:space-between}}
  @media print{{
    body{{padding:16px}}
    h2{{page-break-before:auto}}
    .no-print{{display:none}}
  }}
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div>
    <div class="logo">Sym<span>bion</span> Machine Analytics</div>
    <div style="font-size:12px;color:#6B7280;margin-top:2px">Machine Health Assessment Report</div>
  </div>
  <div style="text-align:right;font-size:12px;color:#6B7280">
    <div><strong>{machine}</strong></div>
    <div>Generated: {now_str}</div>
    <div>Assessment period: {period_str}</div>
  </div>
</div>

<!-- Executive Summary -->
<h2>Executive Summary</h2>
<div class="health-banner">
  <div class="health-title">Overall Health: {overall_lbl}</div>
  <div style="font-size:13px;margin-top:6px;color:#444">
    {'No issues detected across all diagnostic zones.' if overall is None else
     'One or more diagnostic zones require attention. See zone findings below.'}
  </div>
</div>

<!-- Machine Details -->
<h2>Machine Details</h2>
<div class="meta-grid">
  <div class="meta-box"><div class="meta-label">Machine</div><div class="meta-value">{machine}</div></div>
  <div class="meta-box"><div class="meta-label">Type</div><div class="meta-value">{mtype}</div></div>
  <div class="meta-box"><div class="meta-label">Application</div><div class="meta-value">{mapp}</div></div>
  <div class="meta-box"><div class="meta-label">Rated shaft power</div><div class="meta-value">{p_shaft_s}</div></div>
  <div class="meta-box"><div class="meta-label">Rated efficiency</div><div class="meta-value">{eta_s}</div></div>
  <div class="meta-box"><div class="meta-label">Assessment period</div><div class="meta-value" style="font-size:12px">{period_str}</div></div>
</div>

<!-- Data Quality -->
<h2>Data Quality</h2>
<table>
  <thead><tr><th>Step</th><th style="text-align:right">Samples</th><th style="text-align:right">Removed</th></tr></thead>
  <tbody>{cleaning_rows}</tbody>
</table>
<div style="margin-top:8px;font-size:12px;color:#177E40;font-weight:600">
  {ret_pct} of raw samples retained for analysis
</div>

<!-- Zone Findings -->
<h2>Zone Diagnostic Findings</h2>

<div class="zone-card">
  <div class="zone-header">
    <span class="zone-title">Zone 1 — Supply Quality (VUF)</span>
    {_zone_badge(z1_tier)}
  </div>
  <div class="zone-body">{z1_msg}
    <br><span style="font-size:11px;color:#888">Watch ≥{gauge_thresholds.get('vuf_watch',1.0):.1f}%
    &nbsp;|&nbsp; Critical ≥{gauge_thresholds.get('vuf_critical',2.0):.1f}%</span>
  </div>
</div>

<div class="zone-card">
  <div class="zone-header">
    <span class="zone-title">Zone 2 — Current Imbalance (IUF)</span>
    {_zone_badge(z2_tier)}
  </div>
  <div class="zone-body">{z2_msg}
    <br><span style="font-size:11px;color:#888">Watch ≥{gauge_thresholds.get('iuf_watch',5.0):.0f}%
    &nbsp;|&nbsp; Critical ≥{gauge_thresholds.get('iuf_critical',10.0):.0f}%</span>
  </div>
</div>

<div class="zone-card">
  <div class="zone-header">
    <span class="zone-title">Zone 3 — Motor Health (PF Drift)</span>
    {_zone_badge(mside_pf_tier)}
  </div>
  <div class="zone-body">
    {'Statistically significant PF drift detected in ' + str(sum(1 for b in record.motor_side.bands if b.drift_significant)) + ' load band(s).'
      if record.motor_side and any(b.drift_significant for b in record.motor_side.bands)
      else 'No statistically significant PF drift detected.'}
    <br><span style="font-size:11px;color:#888">Watch ≤-0.01 &nbsp;|&nbsp; Alert ≤-0.02 &nbsp;|&nbsp; Action ≤-0.03 (absolute)</span>
  </div>
</div>

<div class="zone-card">
  <div class="zone-header">
    <span class="zone-title">Zone 4 — Driven Equipment</span>
    {_zone_badge(None)}
  </div>
  <div class="zone-body">{z4_msg}</div>
</div>

<!-- PF Drift Tables -->
<h2>PF Drift Detail — Significant Bands</h2>
<h3>Machine Level</h3>
{machine_drift_table if machine_drift_table else '<p style="color:#888;font-size:12px">No machine-level bands available.</p>'}
{('<h3>Per-Phase</h3>' + phase_drift_tables) if phase_drift_tables else ''}

<!-- Recommendations -->
<h2>Recommendations</h2>
<table>
  <thead><tr><th>Priority</th><th>Zone</th><th>Action</th></tr></thead>
  <tbody>{rec_rows}</tbody>
</table>

<!-- Charts -->
<h2>Diagnostic Charts</h2>
<div style="font-size:12px;color:#6B7280;margin-bottom:12px">
  Blue = cleaned data used for analysis &nbsp;|&nbsp; Grey = all raw data
</div>
{charts_html}

<!-- Footer -->
<div class="footer">
  <div>Symbion Machine Analytics Platform &nbsp;|&nbsp; Methodology v0.8</div>
  <div>Report generated {now_str} &nbsp;|&nbsp; Confidential</div>
</div>

</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Assessment charts
# ---------------------------------------------------------------------------

def build_assessment_charts(
    data: pd.DataFrame,
    record: AssessmentRecord,
    cleaned_data: pd.DataFrame | None = None,
    meta: dict | None = None,
    phase_bands: dict | None = None,
    iuf_gauge_watch: float | None = None,
    iuf_gauge_critical: float | None = None,
    vuf_gauge_watch: float | None = None,
    vuf_gauge_critical: float | None = None,
    pf_gauge_watch: float | None = None,
    pf_gauge_critical: float | None = None,
    vuf_gauge_value: float | None = None,   # override: latest daily mean (default: assessment mean)
    iuf_gauge_value: float | None = None,   # override: latest daily mean (default: assessment mean)
    pf_gauge_value: float | None = None,    # override: latest daily mean PF
    pf_drift_watch: float | None = None,    # override PF_DRIFT_WATCH  (default -0.01)
    pf_drift_critical: float | None = None, # override PF_DRIFT_ACTION (default -0.03)
    baseline_p_daily: dict | None = None,   # {date_str: kw} baseline daily P_total
) -> list:
    """Build control charts for VUF, IUF, P_total, PF_machine, and PF drift.

    Time-series charts show cleaned samples (solid blue) against all raw
    data (faded grey background).  The PF drift chart (last) plots per-band
    drift vs. load for the machine level and each phase; marker size
    indicates statistical significance (larger = p < 0.05).

    Parameters
    ----------
    phase_bands : {1: [BandRecord], 2: [...], 3: [...]} from session state.
                  Pass an empty dict if phase analysis was not run.
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
               h_lines=None, y_range=None, show_cleaned=True, show_raw=True):
        fig = go.Figure()

        # Background: all raw data (faded grey).
        # show_raw=False suppresses this trace — used for the IUF chart where the
        # raw series contains values near 100% at near-shutdown (tiny I_avg
        # denominator), which poisons Plotly's autoscale when the user clicks
        # the home / autorange button.
        if show_raw:
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

    def _pf_drift_charts(machine_bands: list, phase_bands_dict: dict,
                         drift_watch: float = PF_DRIFT_WATCH,
                         drift_action: float = PF_DRIFT_ACTION) -> list:
        """One PF-drift band-profile chart per signal (Machine + Phase 1/2/3).

        Each chart: X = band centre (kW), Y = PF drift.
        Larger solid markers = statistically significant (p < 0.05).
        """
        _COLOURS = {
            "Machine":  "#185FA5",
            "Phase 1":  "#E74C3C",
            "Phase 2":  "#27AE60",
            "Phase 3":  "#8E44AD",
        }

        def _one(bands, name) -> go.Figure | None:
            active = [
                (b.centre_kw / 1000, b.pf_drift, bool(b.drift_significant))
                for b in bands
                if not b.suppressed and b.pf_drift is not None
            ]
            if not active:
                return None
            xs, ys, sigs = zip(*active)
            colour = _COLOURS[name]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(xs), y=list(ys),
                mode="lines+markers",
                name=name,
                line=dict(color=colour, width=1.8),
                marker=dict(
                    color=colour,
                    size=[8 if s else 4 for s in sigs],
                    opacity=[1.0 if s else 0.45 for s in sigs],
                ),
                customdata=[[("Yes" if s else "No")] for s in sigs],
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "Band centre: %{x:.3f} kW<br>"
                    "Drift: %{y:+.4f}<br>"
                    "Significant (p\u202f<\u202f0.05): %{customdata[0]}"
                    "<extra></extra>"
                ),
            ))

            y_lo = min(min(ys) * 1.35, drift_action * 1.5)
            y_hi = max(max(ys) * 1.35 if max(ys) > 0 else 0.005, 0.02)

            fig.add_hline(y=0, line_color="#AAAAAA", line_width=1, line_dash="dot")
            for val, col, label in [
                (drift_watch,  "#F1C40F", f"Watch {drift_watch:+.2f}"),
                (drift_action, "#A32D2D", f"Critical {drift_action:+.2f}"),
            ]:
                fig.add_hline(
                    y=val, line_color=col, line_dash="dash", line_width=1.2,
                    annotation_text=label, annotation_position="top right",
                    annotation_font_size=9,
                )
            fig.update_layout(
                title=dict(
                    text=(
                        f"PF Drift by Load Band \u2014 {name}"
                        "<br><sup>Larger solid markers = statistically significant"
                        " (p\u202f<\u202f0.05)"
                        " \u2502 Only bands with \u22655 recent samples shown</sup>"
                    ),
                    font=dict(size=13),
                ),
                xaxis_title="Band centre (kW)",
                yaxis=dict(title="PF drift", range=[y_lo, y_hi], tickformat="+.3f"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=50, r=130, t=65, b=50),
                hovermode="x unified", font=dict(size=11), height=320,
                showlegend=False,
            )
            return fig

        out = []
        f = _one(machine_bands, "Machine")
        if f:
            out.append(f)
        for ph in (1, 2, 3):
            f = _one(phase_bands_dict.get(ph, []), f"Phase {ph}")
            if f:
                out.append(f)
        return out

    def _gauge(value: float, watch: float, critical: float,
               title: str, unit: str,
               axis_max: float | None = None) -> go.Figure:
        """Indicator gauge — high-is-bad (VUF, IUF).

        Colour bands:
          0 -> watch    : green  (#177E40)
          watch -> crit : amber  (#E67E22)
          crit -> max   : red    (#C0392B)

        axis_max overrides the default 1.4x critical scale.
        """
        g_watch    = watch
        g_critical = critical
        g_max      = axis_max if axis_max is not None else max(critical * 1.4, 15.0)

        if value < g_watch:
            bar_colour = "#177E40"
        elif value < g_critical:
            bar_colour = "#E67E22"
        else:
            bar_colour = "#C0392B"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            number=dict(
                suffix=f" {unit}",
                font=dict(size=28, color=bar_colour),
                valueformat=".1f",
            ),
            title=dict(text=title, font=dict(size=12)),
            gauge=dict(
                axis=dict(
                    range=[0, g_max],
                    tickwidth=1,
                    tickcolor="#555",
                    tickfont=dict(size=9),
                    nticks=6,
                ),
                bar=dict(color=bar_colour, thickness=0.22),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0,          g_watch],    color="rgba(23,126,64,0.15)"),
                    dict(range=[g_watch,    g_critical],  color="rgba(230,126,34,0.15)"),
                    dict(range=[g_critical, g_max],       color="rgba(192,57,43,0.15)"),
                ],
                threshold=dict(
                    line=dict(color=bar_colour, width=3),
                    thickness=0.80,
                    value=value,
                ),
            ),
        ))
        fig.update_layout(
            height=220,
            margin=dict(l=30, r=30, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        )
        return fig

    def _gauge_inverted(value: float, watch: float, critical: float,
                        title: str, unit: str) -> go.Figure:
        """Indicator gauge — low-is-bad (PF).

        Colour bands (axis runs 0 → 1):
          0 -> critical  : red    (#C0392B)
          critical->watch: amber  (#E67E22)
          watch -> 1.0   : green  (#177E40)

        watch > critical (e.g. watch=0.85, critical=0.75).
        """
        if value >= watch:
            bar_colour = "#177E40"
        elif value >= critical:
            bar_colour = "#E67E22"
        else:
            bar_colour = "#C0392B"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            number=dict(
                suffix=f" {unit}",
                font=dict(size=28, color=bar_colour),
                valueformat=".3f",
            ),
            title=dict(text=title, font=dict(size=12)),
            gauge=dict(
                axis=dict(
                    range=[0, 1.0],
                    tickwidth=1,
                    tickcolor="#555",
                    tickfont=dict(size=9),
                    nticks=6,
                ),
                bar=dict(color=bar_colour, thickness=0.22),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0,        critical], color="rgba(192,57,43,0.15)"),
                    dict(range=[critical, watch],    color="rgba(230,126,34,0.15)"),
                    dict(range=[watch,    1.0],      color="rgba(23,126,64,0.15)"),
                ],
                threshold=dict(
                    line=dict(color=bar_colour, width=3),
                    thickness=0.80,
                    value=value,
                ),
            ),
        ))
        fig.update_layout(
            height=220,
            margin=dict(l=30, r=30, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        )
        return fig

    # Resolve gauge thresholds (caller overrides take priority over constants)
    _iuf_watch    = iuf_gauge_watch    if iuf_gauge_watch    is not None else float(IUF_WATCH)
    _iuf_critical = iuf_gauge_critical if iuf_gauge_critical is not None else float(IUF_CRITICAL)
    _vuf_watch    = vuf_gauge_watch    if vuf_gauge_watch    is not None else float(VUF_WATCH)
    _vuf_critical = vuf_gauge_critical if vuf_gauge_critical is not None else float(VUF_CRITICAL)
    _pf_watch     = pf_gauge_watch     if pf_gauge_watch     is not None else 0.85
    _pf_critical  = pf_gauge_critical  if pf_gauge_critical  is not None else 0.75

    cl_idx = cleaned_data.index if has_cleaned else None

    # VUF gauge
    if record.supply_alarm and record.supply_alarm.vuf_pct is not None:
        _vuf_display = vuf_gauge_value if vuf_gauge_value is not None \
                       else record.supply_alarm.vuf_pct
        figs.append(_gauge(
            value=_vuf_display,
            watch=_vuf_watch,
            critical=_vuf_critical,
            title=(
                f"VUF Gauge \u2014 Latest Daily Mean<br>"
                f"<sup>Watch \u2265{_vuf_watch:.1f}%  \u2502  Critical \u2265{_vuf_critical:.1f}%"
                f"  \u2502  Axis: 0 \u2192 5%</sup>"
            ),
            unit="%",
            axis_max=5.0,
        ))

    # IUF gauge — per-phase contribution for subtitle
    _ph_iuf_subtitle = ""
    if has_cleaned:
        try:
            _i1 = cleaned_data["phase_1_current"]
            _i2 = cleaned_data["phase_2_current"]
            _i3 = cleaned_data["phase_3_current"]
            _i_avg_cl  = (_i1 + _i2 + _i3) / 3.0
            _i_avg_safe = _i_avg_cl.replace(0, np.nan)
            _ph_dev = {
                1: float(((_i1 - _i_avg_cl).abs() / _i_avg_safe * 100).mean()),
                2: float(((_i2 - _i_avg_cl).abs() / _i_avg_safe * 100).mean()),
                3: float(((_i3 - _i_avg_cl).abs() / _i_avg_safe * 100).mean()),
            }
            _worst_ph  = max(_ph_dev, key=_ph_dev.get)
            _worst_val = _ph_dev[_worst_ph]
            _ph_iuf_subtitle = (
                f"  \u2502  Highest imbalance: Phase\u00a0{_worst_ph}"
                f" ({_worst_val:.1f}%)"
            )
        except Exception:
            pass

    if record.motor_side and record.motor_side.iuf_mean_pct is not None:
        _iuf_display = iuf_gauge_value if iuf_gauge_value is not None \
                       else record.motor_side.iuf_mean_pct
        figs.append(_gauge(
            value=_iuf_display,
            watch=_iuf_watch,
            critical=_iuf_critical,
            title=(
                f"IUF Gauge \u2014 Latest Daily Mean<br>"
                f"<sup>Watch \u2265{_iuf_watch:.0f}%  \u2502  Critical"
                f" \u2265{_iuf_critical:.0f}%{_ph_iuf_subtitle}</sup>"
            ),
            unit="%",
        ))

    # P_total daily average run chart
    p_hlines = []
    if record.zone4 and record.zone4.p_baseline_avg_kw:
        p_hlines.append((
            record.zone4.p_baseline_avg_kw, "#054D5F", "dashdot",
            f"Baseline avg {record.zone4.p_baseline_avg_kw:.1f} kW",
        ))
    if cleaned_data is not None and not cleaned_data.empty:
        _p_cols = ["phase_1_active_power", "phase_2_active_power", "phase_3_active_power"]
        if all(c in cleaned_data.columns for c in _p_cols):
            _assess_avg_kw = float(
                (cleaned_data[_p_cols[0]] + cleaned_data[_p_cols[1]] + cleaned_data[_p_cols[2]]).mean()
            ) / 1000.0
            if _assess_avg_kw > 0:
                p_hlines.append((
                    _assess_avg_kw, "#C8A84B", "dash",
                    f"Assessment avg {_assess_avg_kw:.1f} kW",
                ))
    _p40_kw = None
    if meta:
        _p_shaft = float(meta.get("p_rated_shaft_kw", 0))
        _eta     = float(meta.get("eta_rated", 0.90))
        if _p_shaft > 0 and _eta > 0:
            _p40_kw = 0.20 * (_p_shaft / _eta)
        else:
            _p95_raw = float(raw_p_kw[raw_p_kw > 0].quantile(0.95)) if (raw_p_kw > 0).any() else 0.0
            _p_rated_est = _p95_raw / 0.95 if _p95_raw > 0 else 0.0
            _p40_kw = 0.20 * _p_rated_est if _p_rated_est > 0 else None
    if _p40_kw and _p40_kw > 0:
        p_hlines.append((
            _p40_kw, "#177E40", "dot",
            f"20% load threshold ({_p40_kw:.1f} kW)",
        ))

    try:
        _p_src = cleaned_data if has_cleaned else data
        _p_idx = pd.to_datetime(_p_src.index)
        _p_daily = (cl_p_kw if has_cleaned else raw_p_kw).copy()
        _p_daily.index = _p_idx
        _p_daily_avg = _p_daily.resample("D").mean().dropna()
        if len(_p_daily_avg) >= 1:
            _prun = go.Figure()
            _pd_x = _p_daily_avg.index.tolist()
            _pd_y = _p_daily_avg.values.tolist()
            # Assessment trace (blue)
            _prun.add_trace(go.Scatter(
                x=_pd_x, y=_pd_y,
                mode="lines+markers",
                name="Assessment",
                line=dict(color="#185FA5", width=2),
                marker=dict(size=5),
                hovertemplate="<b>Assessment</b> %{y:.1f} kW<extra></extra>",
            ))
            if len(_pd_y) >= 2:
                _pxt = np.arange(len(_pd_x))
                _pyt = np.polyval(np.polyfit(_pxt, _pd_y, 1), _pxt)
                _prun.add_trace(go.Scatter(
                    x=_pd_x, y=_pyt.tolist(),
                    mode="lines", showlegend=False, hoverinfo="skip",
                    line=dict(color="#185FA5", width=1.2, dash="dot"),
                ))
            # Baseline trace (green) — plotted on same axes, different date range
            _bl_pd = baseline_p_daily or {}
            _bl_y_vals = []
            if _bl_pd:
                _bl_sorted = sorted(_bl_pd.items())
                _bl_x = [pd.Timestamp(d) for d, _ in _bl_sorted]
                _bl_y = [v for _, v in _bl_sorted]
                _bl_y_vals = _bl_y
                _prun.add_trace(go.Scatter(
                    x=_bl_x, y=_bl_y,
                    mode="lines+markers",
                    name="Baseline",
                    line=dict(color="#177E40", width=2),
                    marker=dict(size=5, color="#177E40"),
                    hovertemplate="<b>Baseline</b> %{y:.1f} kW<extra></extra>",
                ))
                if len(_bl_y) >= 2:
                    _blxt = np.arange(len(_bl_x))
                    _blyt = np.polyval(np.polyfit(_blxt, _bl_y, 1), _blxt)
                    _prun.add_trace(go.Scatter(
                        x=_bl_x, y=_blyt.tolist(),
                        mode="lines", showlegend=False, hoverinfo="skip",
                        line=dict(color="#177E40", width=1.2, dash="dot"),
                    ))
            _p_y_max = max(_pd_y + _bl_y_vals) if (_pd_y or _bl_y_vals) else 0
            for _hv, _hc, _hd, _hl in (p_hlines or []):
                _prun.add_shape(
                    type="line", xref="paper", x0=0, x1=1,
                    yref="y", y0=_hv, y1=_hv,
                    line=dict(color=_hc, width=1.2,
                              dash={"dashdot": "dashdot", "dash": "dash",
                                    "dot": "dot", "solid": "solid"}.get(_hd, "dash")),
                )
                _prun.add_annotation(
                    xref="paper", x=1.01, yref="y", y=_hv,
                    text=_hl, showarrow=False, xanchor="left",
                    font=dict(size=8, color=_hc),
                )
                _p_y_max = max(_p_y_max, _hv)
            _prun.update_layout(
                title=dict(
                    text="Total Active Power — Daily Average"
                         "<br><sup>Blue = assessment  \u2502  Green = baseline"
                         "  \u2502  Dotted = linear trend</sup>",
                    font=dict(size=13),
                ),
                xaxis=dict(title="Date", tickformat="%d %b"),
                yaxis=dict(title="P_total (kW)",
                           range=[0, _p_y_max * 1.25 if _p_y_max > 0 else 10]),
                showlegend=True,
                legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10)),
                hovermode="x unified",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=55, r=160, t=65, b=50),
                height=300, font=dict(size=11),
            )
            figs.append(_prun)
    except Exception:
        figs.append(_chart(
            data.index, raw_p_kw,
            cl_idx, cl_p_kw if has_cleaned else None,
            "Total Active Power (P_total)", "P_total (kW)",
            h_lines=p_hlines or None,
        ))

    # PF_machine daily average run chart
    pf_hlines = []
    if baseline and baseline.bands:
        for b in baseline.bands[:6]:
            pf_hlines.append((
                b.mean_pf_baseline, "rgba(100,100,100,0.55)", "dot",
                f"Baseline PF @ {b.centre_kw/1000:.0f} kW",
            ))

    try:
        _pf_src = cl_pf if has_cleaned else raw_pf
        _pf_idx = pd.to_datetime(
            (cleaned_data if has_cleaned else data).index
        )
        _pf_s = _pf_src.copy()
        _pf_s.index = _pf_idx
        _pf_daily_avg = _pf_s.resample("D").mean().dropna()
        if len(_pf_daily_avg) >= 1:
            _pfrun = go.Figure()
            _pfd_x = _pf_daily_avg.index.tolist()
            _pfd_y = _pf_daily_avg.values.tolist()
            _pfrun.add_trace(go.Scatter(
                x=_pfd_x, y=_pfd_y,
                mode="lines+markers",
                name="Daily avg PF",
                line=dict(color="#185FA5", width=2),
                marker=dict(size=5),
                hovertemplate="<b>PF</b> %{y:.4f}<extra></extra>",
            ))
            if len(_pfd_y) >= 2:
                _pfxt = np.arange(len(_pfd_x))
                _pfyt = np.polyval(np.polyfit(_pfxt, _pfd_y, 1), _pfxt)
                _pfrun.add_trace(go.Scatter(
                    x=_pfd_x, y=_pfyt.tolist(),
                    mode="lines", showlegend=False, hoverinfo="skip",
                    line=dict(color="#185FA5", width=1.2, dash="dot"),
                ))
            for _hv, _hc, _hd, _hl in (pf_hlines or []):
                _pfrun.add_shape(
                    type="line", xref="paper", x0=0, x1=1,
                    yref="y", y0=_hv, y1=_hv,
                    line=dict(color=_hc, width=1,
                              dash={"dot": "dot", "dash": "dash",
                                    "dashdot": "dashdot"}.get(_hd, "dot")),
                )
                _pfrun.add_annotation(
                    xref="paper", x=1.01, yref="y", y=_hv,
                    text=_hl, showarrow=False, xanchor="left",
                    font=dict(size=8, color="#888"),
                )
            _pf_y_min = min(_pfd_y) if _pfd_y else 0.7
            _pfrun.update_layout(
                title=dict(
                    text="Machine Power Factor — Daily Average"
                         "<br><sup>Each point = daily mean (cleaned samples)"
                         "  \u2502  Dotted = linear trend"
                         "  \u2502  Grey lines = baseline band PF values</sup>",
                    font=dict(size=13),
                ),
                xaxis=dict(title="Date", tickformat="%d %b"),
                yaxis=dict(
                    title="PF",
                    range=[max(0, _pf_y_min * 0.97), 1.02],
                    tickformat=".3f",
                ),
                showlegend=False,
                hovermode="x unified",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=55, r=160, t=65, b=50),
                height=300, font=dict(size=11),
            )
            figs.append(_pfrun)
    except Exception:
        figs.append(_chart(
            data.index, raw_pf,
            cl_idx, cl_pf if has_cleaned else None,
            "Machine Power Factor (PF_machine)", "PF_machine",
            h_lines=pf_hlines or None,
            y_range=[0, 1.05],
        ))

    # PF gauge — inverted scale (high PF = healthy)
    if has_cleaned and len(cl_pf) > 0:
        _pf_display = pf_gauge_value if pf_gauge_value is not None else float(cl_pf.mean())
        figs.append(_gauge_inverted(
            value=_pf_display,
            watch=_pf_watch,
            critical=_pf_critical,
            title=(
                f"PF Gauge \u2014 Latest Daily Mean<br>"
                f"<sup>Watch \u2264{_pf_watch:.2f}  \u2502  Critical \u2264{_pf_critical:.2f}"
                f"  \u2502  Axis: 0 \u2192 1</sup>"
            ),
            unit="",
        ))

    # PF drift by load band — one chart per signal (Machine + Phase 1/2/3)
    _machine_bands  = record.motor_side.bands if record.motor_side else []
    _phase_bands    = phase_bands or {}
    _eff_pf_watch    = pf_drift_watch    if pf_drift_watch    is not None else float(PF_DRIFT_WATCH)
    _eff_pf_critical = pf_drift_critical if pf_drift_critical is not None else float(PF_DRIFT_ACTION)
    figs.extend(_pf_drift_charts(_machine_bands, _phase_bands,
                                 _eff_pf_watch, _eff_pf_critical))

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
    ("last_assessment",          None),
    ("last_data",                None),
    ("last_cleaned_data",        None),
    ("last_integrity_passed_ts", None),
    ("baseline_ic_excluded",     0),
    ("last_integrity_failure_summary", {}),
    ("effective_meta",           None),   # resolved meta per §2.5 — single source of truth
    ("_ep_just_saved",           False),  # flag to show save confirmation after rerun
    ("last_phase_bands",         {}),     # {1: [BandRecord], 2: [...], 3: [...]}
    ("last_phase_bl_all_bins",   {}),     # {1: all 100 bins, 2: ..., 3: ...}
    ("baseline_vuf_mean",        None),   # baseline period mean VUF %
    ("baseline_iuf_mean",        None),   # baseline period mean IUF %
    ("baseline_ph_iuf",          {}),     # {1: mean%, 2: mean%, 3: mean%}
    ("baseline_p_daily",         {}),     # {date_str: kw} daily P_total baseline
    # Gauge threshold defaults — version-stamped so a code change forces a clean reset.
    # _GAUGE_SS_VER should be bumped whenever the defaults or valid ranges change.
    ("vuf_gauge_watch",    float(VUF_WATCH)),
    ("vuf_gauge_critical", float(VUF_CRITICAL)),
    ("iuf_gauge_watch",    float(IUF_WATCH)),
    ("iuf_gauge_critical", float(IUF_CRITICAL)),
    ("pf_gauge_watch",     0.85),
    ("pf_gauge_critical",  0.75),
    ("pf_drift_watch",    float(PF_DRIFT_WATCH)),
    ("pf_drift_critical", float(PF_DRIFT_ACTION)),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Force-correct gauge thresholds that got corrupted to min_value in earlier versions.
# Key: if any value is out of its expected range, the entire set is reset to defaults.
_GAUGE_SS_VER = "v5"
if st.session_state.get("_gauge_ss_ver") != _GAUGE_SS_VER:
    st.session_state["vuf_gauge_watch"]    = float(VUF_WATCH)
    st.session_state["vuf_gauge_critical"] = float(VUF_CRITICAL)
    st.session_state["iuf_gauge_watch"]    = float(IUF_WATCH)
    st.session_state["iuf_gauge_critical"] = float(IUF_CRITICAL)
    st.session_state["pf_gauge_watch"]     = 0.85
    st.session_state["pf_gauge_critical"]  = 0.75
    st.session_state["_gauge_ss_ver"]      = _GAUGE_SS_VER


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
    st.caption(f"Electrical Diagnostics \u2014 Symbion  |  {_DIAG_VERSION}")

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
                    v_nominal  = reg_v_nom   if reg_v_nom   > 0 else 230.0,
                    p_rated    = reg_p_rated,
                    pf_rated   = 0.88,
                    eta_rated  = 0.90,
                    i_rated    = reg_i_rated,
                    four_wire  = True,
                    at_panel   = True,
                    app_type   = _reg_app,
                    power_unit = "W",
                    # Only mark fields the user actually typed as entered
                    user_entered = {
                        "v_nominal_phase":  reg_v_nom   > 0,
                        "p_rated_shaft_kw": reg_p_rated > 0,
                        "pf_rated":         False,   # not asked at registration
                        "eta_rated":        False,   # not asked at registration
                        "i_rated":          reg_i_rated > 0,
                    },
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
        st.session_state["last_integrity_passed_ts"] = None
        st.session_state["last_data"]       = None
        st.session_state["_last_machine"]   = selected_id
        # Clear electrical parameter widget keys so they re-render from DB values
        for _ep_key in ["ep_v_nom", "ep_p_rated", "ep_i_rated",
                         "ep_pf_rated", "ep_eta_rated"]:
            st.session_state.pop(_ep_key, None)

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
                        # Preserve existing user_entered flags
                        user_entered={
                            "v_nominal_phase":  str(_meta_now.get("user_entered_v_nominal", "false")).lower() == "true",
                            "p_rated_shaft_kw": str(_meta_now.get("user_entered_p_rated",   "false")).lower() == "true",
                            "pf_rated":         str(_meta_now.get("user_entered_pf",        "false")).lower() == "true",
                            "eta_rated":        str(_meta_now.get("user_entered_eta",       "false")).lower() == "true",
                            "i_rated":          str(_meta_now.get("user_entered_i_rated",   "false")).lower() == "true",
                        },
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
                st.session_state["last_integrity_passed_ts"] = None
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
        "Enter values from the motor nameplate. Fields left blank use the estimated or assumed "
        "value shown below them \u2014 derived per \u00a72.5 and flagged throughout the platform."
    )
    _desc = machine_info.get("description", "")
    _em   = parse_electrical_meta(_desc)

    # Effective (resolved) meta — has estimated values where nameplate is missing
    _eff = st.session_state.get("effective_meta") or {}

    # Helper: is a field saved from the nameplate (not just a default)?
    def _is_saved(key):
        v = _em.get(key, 0)
        return v is not None and v != 0

    # Helper: caption to show under each field
    def _est_note(col, key, label, fmt, unit, note=""):
        """Show 'Estimated/Assumed: X' note when value was not entered from nameplate."""
        src = _eff.get(key + "_source", "")
        if src == "nameplate":
            return  # entered from nameplate — no annotation needed
        eff_val = _eff.get(key, 0)
        if eff_val:
            tag = "Assumed default" if src == "assumed_default" else "Estimated from data"
            col.caption(f"\u26a0\ufe0f {tag}: {eff_val:{fmt}} {unit}{' \u2014 ' + note if note else ''}")
        else:
            col.caption(f"\u26a0\ufe0f {label} not available")

    # Helper: field display value — saved value or estimated value or blank
    def _field_display(key, fmt):
        """Return the saved nameplate value for the form field.
        Never pre-fill with estimated values — estimates are shown as captions only,
        so the user always sees a blank field when no nameplate value is saved."""
        v = _em.get(key, 0)
        if v and v != 0:
            return f"{v:{fmt}}"
        return ""

    # Row 1 — Voltage, Power, Current
    _c1, _c2, _c3 = st.columns(3)

    _v_nom_txt = _c1.text_input(
        "Phase-to-neutral voltage (V)",
        value=_field_display("v_nominal_phase", ".0f"),
        key="ep_v_nom",
        placeholder="e.g. 230",
        help="e.g. 230 V for a 400/230 V system.",
    )
    if not _v_nom_txt.strip():
        _c1.caption("\u26a0\ufe0f Rated value not entered \u2014 will be estimated from data")

    _p_rated_txt = _c2.text_input(
        "Rated shaft power (kW)",
        value=_field_display("p_rated_shaft_kw", ".1f"),
        key="ep_p_rated",
        placeholder="e.g. 75",
        help="Motor nameplate shaft power.",
    )
    if not _p_rated_txt.strip():
        _c2.caption("\u26a0\ufe0f Rated value not entered \u2014 will be estimated from data")

    _i_rated_txt = _c3.text_input(
        "Full-load current / FLA (A)",
        value=_field_display("i_rated", ".1f"),
        key="ep_i_rated",
        placeholder="e.g. 140",
        help="Nameplate full-load current (FLA).",
    )
    if not _i_rated_txt.strip():
        _c3.caption("\u26a0\ufe0f Rated value not entered \u2014 will be estimated from data")

    # Row 2 — PF, Efficiency, Panel checkbox
    _c4, _c5, _c6 = st.columns(3)

    _pf_rated_txt = _c4.text_input(
        "Rated full-load PF",
        value=_field_display("pf_rated", ".2f"),
        key="ep_pf_rated",
        placeholder="e.g. 0.87",
    )
    if not _pf_rated_txt.strip():
        _c4.caption("\u26a0\ufe0f Rated value not entered \u2014 0.87 will be assumed")

    _eta_rated_txt = _c5.text_input(
        "Rated efficiency (0\u20131)",
        value=_field_display("eta_rated", ".2f"),
        key="ep_eta_rated",
        placeholder="e.g. 0.9",
        help="e.g. 0.9 for 90% efficiency",
    )
    if not _eta_rated_txt.strip():
        _c5.caption("\u26a0\ufe0f Rated value not entered \u2014 0.9 will be assumed")

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

    def _parse_field(txt, default=0.0, lo=None, hi=None):
        """Parse a text field to float. Return default if blank or invalid."""
        txt = txt.strip()
        if not txt:
            return default
        try:
            v = float(txt)
            if lo is not None and v < lo:
                return default
            if hi is not None and v > hi:
                return default
            return v
        except ValueError:
            return default

    _v_nom     = _parse_field(_v_nom_txt,    default=0.0, lo=0.0)
    _p_rated   = _parse_field(_p_rated_txt,  default=0.0, lo=0.0)
    _i_rated   = _parse_field(_i_rated_txt,  default=0.0, lo=0.0)
    _pf_rated  = _parse_field(_pf_rated_txt, default=0.0, lo=0.0, hi=1.0)
    _eta_rated = _parse_field(_eta_rated_txt,default=0.0, lo=0.0, hi=1.0)

    # Validate and show field-level feedback
    _ep_errors = []
    if _v_nom_txt.strip()     and _v_nom     == 0.0: _ep_errors.append("Voltage: enter a positive number (e.g. 230)")
    if _pf_rated_txt.strip()  and _pf_rated  == 0.0: _ep_errors.append("PF: must be between 0 and 1 (e.g. 0.87)")
    if _eta_rated_txt.strip() and _eta_rated == 0.0: _ep_errors.append("Efficiency: must be between 0 and 1 (e.g. 0.9)")
    for _ep_err in _ep_errors:
        st.error(f"\u274c {_ep_err}")

    # Power unit is set at upload time (sidebar), not here.
    # Read current saved value for round-trip when user clicks Save.
    _power_unit = _em.get("power_unit", "W").upper()

    if st.button("Save electrical parameters", key="save_ep_btn",
                 use_container_width=True, disabled=bool(_ep_errors)):
        _app_type = APP_TYPE_MAP.get(machine_info["machine_type"], "compressed_air")
        _user_entered_flags = {
            "v_nominal_phase":  bool(_v_nom_txt.strip()),
            "p_rated_shaft_kw": bool(_p_rated_txt.strip()),
            "pf_rated":         bool(_pf_rated_txt.strip()),
            "eta_rated":        bool(_eta_rated_txt.strip()),
            "i_rated":          bool(_i_rated_txt.strip()),
        }
        _new_block = serialise_meta_block(
            _v_nom, _p_rated, _pf_rated, _eta_rated, _i_rated,
            _fw, _at_panel, _app_type, _power_unit,
            user_entered=_user_entered_flags,
        )
        _new_desc = replace_meta_block(_desc, _new_block)
        db.register_machine(selected_id, machine_info["machine_type"], _new_desc)
        # Clear widget keys so they re-render with the newly saved values
        for _ep_key in ["ep_v_nom", "ep_p_rated", "ep_i_rated",
                         "ep_pf_rated", "ep_eta_rated"]:
            st.session_state.pop(_ep_key, None)
        st.session_state["effective_meta"]  = None
        st.session_state["_ep_just_saved"]  = True
        st.rerun()

    # Persistent save confirmation — shown on the render after save rerun
    if st.session_state.get("_ep_just_saved"):
        st.session_state["_ep_just_saved"] = False
        st.success("\u2713 Electrical parameters saved. Banner above now reflects the updated values.")

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
    st.session_state["_data_fp"]           = _data_fp
    st.session_state["last_assessment"]    = None
    st.session_state["last_cleaned_data"]  = None
    st.session_state["last_integrity_passed_ts"] = None
    st.session_state["last_data"]          = None
    st.session_state["effective_meta"]     = None   # recompute below

# ── Resolve effective meta (§2.5) ─────────────────────────────────────────
# Always recomputes fresh every render from current machine_info + data.
# Never reads from session state cache — the cache was causing stale values.
_saved_meta = build_meta(machine_info)
_raw_em     = parse_electrical_meta(machine_info.get("description", ""))
if data is not None and not data.empty and _saved_meta is not None:
    _data_w_full = scale_power_to_watts(
        data.reset_index(), _saved_meta.get("power_unit", "W")
    )
    meta = resolve_effective_meta(_saved_meta, _data_w_full, _raw_em)
elif _saved_meta is not None:
    meta = resolve_effective_meta(_saved_meta, pd.DataFrame(), _raw_em)
else:
    meta = None
st.session_state["effective_meta"] = meta


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

        # ── Effective parameters banner ───────────────────────────────────
        if meta:
            _src_p   = meta.get("p_rated_source",   "estimated_from_data")
            _src_v   = meta.get("v_nominal_source",  "estimated_from_data")
            _src_i   = meta.get("i_rated_source",    "estimated_from_data")
            _src_pf  = meta.get("pf_rated_source",   "assumed_default")
            _src_eta = meta.get("eta_rated_source",  "assumed_default")
            _p_shaft = meta.get("p_rated_shaft_kw", 0)
            _p_elec  = meta.get("p_rated_elec_kw", 0)
            _v_nom   = meta.get("v_nominal_phase", 230)
            _i_fla   = meta.get("i_rated", 0)
            _pf      = meta.get("pf_rated", 0.87)
            _eta     = meta.get("eta_rated", 0.90)

            def _src_label(src):
                if src == "nameplate":           return "\u2705 from nameplate"
                if src == "estimated_from_data": return "\u26a0\ufe0f estimated from data"
                if src == "assumed_default":     return "\u26a0\ufe0f assumed default"
                return src

            _p_line = f"P_shaft = **{_p_shaft:.1f} kW** ({_src_label(_src_p)})"
            _any_estimated = any(s != "nameplate" for s in [_src_p, _src_v, _src_i, _src_pf, _src_eta])
            _param_lines = (
                f"{_p_line}  |  "
                f"V_nominal = **{_v_nom:.1f} V** ({_src_label(_src_v)})  |  "
                f"FLA = **{_i_fla:.0f} A** ({_src_label(_src_i)})  |  "
                f"PF = **{_pf:.2f}** ({_src_label(_src_pf)})  |  "
                f"Efficiency = **{_eta:.2f}** ({_src_label(_src_eta)})"
            )
            if _any_estimated:
                st.warning(
                    f"\u26a0\ufe0f **Some parameters not entered \u2014 estimated/assumed values used per \u00a72.5.**  \n"
                    f"{_param_lines}  \n"
                    f"Enter correct values in the \u26a1 **Electrical parameters** expander above."
                )
            else:
                # All from nameplate — show as persistent green banner (not expander)
                st.success(
                    f"\u2705 **All electrical parameters entered from nameplate.**  \n"
                    f"{_param_lines}"
                )

        # ── Integrity checks §3.1 ─────────────────────────────────────────
        meta_for_check = meta   # already resolved via §2.5 — single source of truth
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
                                    _raw_str[_ts_col], format="mixed",
                                    dayfirst=False, errors="coerce"
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
                    # Store all timestamps as passed
                    _passed_df = _scaled_clean.copy()
                    _passed_df = _passed_df[[c for c in _passed_df.columns
                                             if not c.startswith("_")]]
                    if "timestamp" not in _passed_df.columns and _passed_df.index.name == "timestamp":
                        _passed_df = _passed_df.reset_index()
                    # Save passed timestamps and empty failure summary to session state
                    st.session_state["last_integrity_passed_ts"] = set(
                        _passed_df["timestamp"].astype(str).tolist()
                    )
                    st.session_state["last_integrity_failure_summary"] = {}
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
                        # Save passed timestamps and failure reason summary to session state
                        st.session_state["last_integrity_passed_ts"] = set(
                            _passed_df["timestamp"].astype(str).tolist()
                        )
                        # Build reason summary: {reason_string: count}
                        if "failure_reason" in _all_failed.columns:
                            _reason_summary = (
                                _all_failed["failure_reason"]
                                .value_counts()
                                .to_dict()
                            )
                        else:
                            _reason_summary = {}
                        st.session_state["last_integrity_failure_summary"] = _reason_summary
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
        meta = st.session_state.get("effective_meta") or build_meta(machine_info)
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
                    _bl_ic_excl = st.session_state.get("baseline_ic_excluded", 0)
                    if _bl_ic_excl and _bl_ic_excl > 0:
                        _fail_summary = st.session_state.get("last_integrity_failure_summary", {})
                        if _fail_summary:
                            _reason_lines = "\n".join(
                                f"- {reason}: {cnt:,} row(s)"
                                for reason, cnt in sorted(
                                    _fail_summary.items(), key=lambda x: -x[1]
                                )
                            )
                            st.warning(
                                f"\u26a0\ufe0f **{_bl_ic_excl:,} rows excluded from this baseline** "
                                f"by integrity checks. Only clean rows were used to build PF bands.\n\n"
                                f"{_reason_lines}"
                            )
                        else:
                            st.warning(
                                f"\u26a0\ufe0f **{_bl_ic_excl:,} rows excluded from this baseline** "
                                f"by integrity checks. Only clean rows were used to build PF bands."
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
                            # Apply integrity filter so viewer shows only rows
                            # that actually entered the baseline
                            _ic_ts_view = st.session_state.get("last_integrity_passed_ts")
                            _bl_view_data, _bl_view_excluded = apply_integrity_filter(
                                _bl_view_data, _ic_ts_view
                            )
                            if _bl_view_excluded > 0:
                                pass  # Warning already shown in the baseline info card above
                            elif _ic_ts_view is None:
                                st.info(
                                    "\u2139\ufe0f Integrity checks have not been run for this session. "
                                    "Row count below may include wiring-fault rows. "
                                    "Open the \U0001f50d Integrity checks expander in the Data tab "
                                    "and re-ingest baseline to apply filtering."
                                )
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
                                _dl_unit = meta.get("power_unit", "W").upper() if meta else "W"

                                def _bl_to_unit(df):
                                    df = df.copy()
                                    if _dl_unit == "KW":
                                        for _pc in ["phase_1_active_power",
                                                    "phase_2_active_power",
                                                    "phase_3_active_power"]:
                                            if _pc in df.columns:
                                                df[_pc] = (df[_pc] / 1000).round(6)
                                    return df[[c for c in df.columns if not c.startswith("_")]]

                                # Stage 1: Raw baseline
                                _bl_raw_all = data.loc[
                                    (_bl_start_ts2 <= data.index) & (data.index <= _bl_end_ts2)
                                ].reset_index()
                                _bl_raw_all_w = scale_power_to_watts(
                                    _bl_raw_all, meta.get("power_unit", "W") if meta else "W"
                                )
                                _dl_bl_raw = _bl_to_unit(_bl_raw_all_w)

                                # Stage 3: Post-integrity (already have _bl_view_data)
                                _bl_raw_w_seq = scale_power_to_watts(
                                    _bl_view_data.reset_index(),
                                    meta.get("power_unit", "W") if meta else "W"
                                )
                                _dl_bl_post_ic = _bl_to_unit(_bl_raw_w_seq)

                                # Stage 2: Integrity-removed
                                _bl_ic_pass_ts = set(_dl_bl_post_ic["timestamp"].astype(str)) if "timestamp" in _dl_bl_post_ic.columns else set()
                                _dl_bl_ic_removed = _dl_bl_raw[
                                    ~_dl_bl_raw["timestamp"].astype(str).isin(_bl_ic_pass_ts)
                                ].copy() if "timestamp" in _dl_bl_raw.columns else pd.DataFrame()
                                _bl_n_ic_excl = len(_dl_bl_ic_removed)

                                # Stage 5: Cleaned baseline via clean_samples (same as ingest)
                                _bl_user_filter = _stored_bl_dict.get("user_filter_expr")
                                _bl_cleaned_seq = pd.DataFrame()
                                _bl_removed_seq = pd.DataFrame()
                                if meta:
                                    try:
                                        _bl_cleaned_seq, _ = clean_samples(_bl_raw_w_seq, meta, _bl_user_filter,
                                                        load_precondition_fraction=st.session_state.get("load_precondition_pct", 20) / 100.0)
                                        _bl_cl_ts = set(_bl_cleaned_seq["timestamp"].astype(str)) if "timestamp" in _bl_cleaned_seq.columns else set()
                                        _bl_removed_seq = _bl_raw_w_seq[
                                            ~_bl_raw_w_seq["timestamp"].astype(str).isin(_bl_cl_ts)
                                        ].copy() if "timestamp" in _bl_raw_w_seq.columns else pd.DataFrame()
                                    except Exception as _e:
                                        st.caption(f"Could not compute cleaned baseline: {_e}")

                                _dl_bl_cleaned = _bl_to_unit(_bl_cleaned_seq) if not _bl_cleaned_seq.empty else pd.DataFrame()
                                _dl_bl_removed = _bl_to_unit(_bl_removed_seq) if not _bl_removed_seq.empty else pd.DataFrame()

                                # ── Sequential display ────────────────────────
                                st.markdown("---")
                                st.markdown(f"**\U0001f4e5 1. Raw baseline data — {len(_dl_bl_raw):,} rows**")
                                st.download_button(
                                    label=f"\u2b07\ufe0f Download raw baseline ({len(_dl_bl_raw):,} rows, CSV)",
                                    data=_dl_bl_raw.to_csv(index=False).encode("utf-8"),
                                    file_name=f"baseline_raw_{selected_id}_{_bl_start}_to_{_bl_end}.csv",
                                    mime="text/csv", use_container_width=True, key="dl_bl_raw",
                                )
                                st.dataframe(_dl_bl_raw.head(5), use_container_width=True, hide_index=True)

                                st.markdown("---")
                                _bl_ic_c1, _bl_ic_c2 = st.columns(2)
                                with _bl_ic_c1:
                                    st.markdown(f"**\U0001f6e1\ufe0f 2. Removed by integrity — {_bl_n_ic_excl:,} rows**")
                                    if _bl_n_ic_excl > 0:
                                        st.download_button(
                                            label=f"\u2b07\ufe0f Download integrity-removed ({_bl_n_ic_excl:,} rows, CSV)",
                                            data=_dl_bl_ic_removed.to_csv(index=False).encode("utf-8"),
                                            file_name=f"baseline_integrity_removed_{selected_id}_{_bl_start}_to_{_bl_end}.csv",
                                            mime="text/csv", use_container_width=True, key="dl_bl_ic_rm",
                                        )
                                        st.dataframe(_dl_bl_ic_removed.head(5), use_container_width=True, hide_index=True)
                                    else:
                                        st.caption("\u2705 No rows removed by integrity check")
                                with _bl_ic_c2:
                                    st.markdown(f"**\u2705 3. After integrity — {len(_dl_bl_post_ic):,} rows**")
                                    st.download_button(
                                        label=f"\u2b07\ufe0f Download post-integrity ({len(_dl_bl_post_ic):,} rows, CSV)",
                                        data=_dl_bl_post_ic.to_csv(index=False).encode("utf-8"),
                                        file_name=f"baseline_post_integrity_{selected_id}_{_bl_start}_to_{_bl_end}.csv",
                                        mime="text/csv", use_container_width=True, key="dl_bl_post_ic",
                                    )
                                    st.dataframe(_dl_bl_post_ic.head(5), use_container_width=True, hide_index=True)

                                st.markdown("---")
                                _bl_cl_c1, _bl_cl_c2 = st.columns(2)
                                with _bl_cl_c1:
                                    st.markdown(f"**\u274c 4. Removed by cleaning — {len(_dl_bl_removed):,} rows**")
                                    if not _dl_bl_removed.empty:
                                        st.download_button(
                                            label=f"\u2b07\ufe0f Download cleaning-removed ({len(_dl_bl_removed):,} rows, CSV)",
                                            data=_dl_bl_removed.to_csv(index=False).encode("utf-8"),
                                            file_name=f"baseline_removed_{selected_id}_{_bl_start}_to_{_bl_end}.csv",
                                            mime="text/csv", use_container_width=True, key="dl_bl_cl_rm",
                                        )
                                        st.dataframe(_dl_bl_removed.head(5), use_container_width=True, hide_index=True)
                                    else:
                                        st.caption("\u2705 No rows removed by cleaning")
                                with _bl_cl_c2:
                                    st.markdown(f"**\U0001f4ca 5. Final baseline for analysis — {len(_dl_bl_cleaned):,} rows**")
                                    if not _dl_bl_cleaned.empty:
                                        st.download_button(
                                            label=f"\u2b07\ufe0f Download cleaned baseline ({len(_dl_bl_cleaned):,} rows, CSV)",
                                            data=_dl_bl_cleaned.to_csv(index=False).encode("utf-8"),
                                            file_name=f"baseline_cleaned_{selected_id}_{_bl_start}_to_{_bl_end}.csv",
                                            mime="text/csv", use_container_width=True, key="dl_bl_cleaned",
                                        )
                                        st.dataframe(_dl_bl_cleaned.head(5), use_container_width=True, hide_index=True)
                        except Exception as _bl_e:
                            st.error(f"Could not load baseline data: {_bl_e}")

                    if st.button("Delete baseline", key="del_baseline_btn",
                                 type="secondary", use_container_width=True):
                        db.delete_baseline(selected_id)
                        st.rerun()

                    # ── Total power baseline PF bins (computed from cleaned baseline) ──
                    if _stored_bl_dict and meta:
                        try:
                            _bl_ts_h1 = pd.Timestamp(_stored_bl_dict.get("timestamp_start", ""))
                            _bl_ts_h2 = pd.Timestamp(_stored_bl_dict.get("timestamp_end", ""))
                            _bl_raw_h  = data.loc[
                                (_bl_ts_h1 <= data.index) & (data.index <= _bl_ts_h2)
                            ].reset_index()
                            _bl_w_h   = scale_power_to_watts(_bl_raw_h, meta.get("power_unit", "W"))
                            _bl_cl_h, _ = clean_samples(
                                _bl_w_h, meta, _stored_bl_dict.get("user_filter_expr"),
                                load_precondition_fraction=st.session_state.get("load_precondition_pct", 20) / 100.0,
                            )
                            if not _bl_cl_h.empty:
                                _tot_all_bins = select_pf_bands(_bl_cl_h, min_samples=0)
                                _tot_qual = [b for b in _tot_all_bins if b.n_baseline >= 5]
                                if _tot_all_bins:
                                    import plotly.graph_objects as _go2
                                    with st.expander(
                                        f"\U0001f4ca Total power baseline PF bins "
                                        f"({len(_tot_qual)} qualifying of {len(_tot_all_bins)} total)",
                                        expanded=False,
                                    ):
                                        st.caption(
                                            "Bins by P_total, 1% of operating range (100 bins). "
                                            "Computed from cleaned baseline data. "
                                            "Colour = mean baseline PF. "
                                            "Only bins with \u22655 samples qualify for PF drift detection."
                                        )
                                        _bands_df = pd.DataFrame([
                                            {
                                                "centre_kw":        round(b.centre_kw / 1000, 3),
                                                "low_kw":           round(b.low_kw     / 1000, 3),
                                                "high_kw":          round(b.high_kw    / 1000, 3),
                                                "n_baseline":       b.n_baseline,
                                                "mean_pf_baseline": round(b.mean_pf_baseline, 4),
                                                "qualifies":        b.n_baseline >= 5,
                                            }
                                            for b in _tot_all_bins
                                        ]).sort_values("centre_kw")

                                        _pf_min = _bands_df["mean_pf_baseline"].min()
                                        _pf_max = _bands_df["mean_pf_baseline"].max()
                                        _pf_rng = max(_pf_max - _pf_min, 0.01)
                                        _colors = [
                                            f"rgba({int(5+200*(1-(pf-_pf_min)/_pf_rng))},"
                                            f"{int(77+150*((pf-_pf_min)/_pf_rng))},"
                                            f"{int(95+100*((pf-_pf_min)/_pf_rng))},0.85)"
                                            for pf in _bands_df["mean_pf_baseline"]
                                        ]
                                        _fig_h = _go2.Figure()
                                        _fig_h.add_trace(_go2.Bar(
                                            x=_bands_df["centre_kw"],
                                            y=_bands_df["n_baseline"],
                                            width=(_bands_df["high_kw"] - _bands_df["low_kw"]) * 0.9,
                                            marker_color=_colors,
                                            customdata=list(zip(
                                                _bands_df["mean_pf_baseline"],
                                                _bands_df["low_kw"],
                                                _bands_df["high_kw"],
                                                _bands_df["n_baseline"],
                                            )),
                                            hovertemplate=(
                                                "Band: %{customdata[1]:.3f} \u2013 %{customdata[2]:.3f} kW<br>"
                                                "Samples: %{customdata[3]}<br>"
                                                "Baseline PF: %{customdata[0]:.4f}<extra></extra>"
                                            ),
                                        ))
                                        _fig_h.add_hline(
                                            y=5, line_dash="dash", line_color="#C8A84B",
                                            line_width=1.5,
                                            annotation_text="Min 5 samples",
                                            annotation_position="top right",
                                            annotation_font_size=10,
                                        )
                                        _fig_h.update_layout(
                                            xaxis_title="Band centre (kW)",
                                            yaxis_title="Baseline samples",
                                            height=300,
                                            plot_bgcolor="rgba(0,0,0,0)",
                                            paper_bgcolor="rgba(0,0,0,0)",
                                            margin=dict(l=40, r=20, t=30, b=40),
                                            font=dict(size=11),
                                            showlegend=False, bargap=0.05,
                                        )
                                        st.plotly_chart(_fig_h, use_container_width=True)
                                        st.dataframe(
                                            _bands_df.rename(columns={
                                                "centre_kw":        "Centre (kW)",
                                                "low_kw":           "Low (kW)",
                                                "high_kw":          "High (kW)",
                                                "n_baseline":       "Baseline samples",
                                                "mean_pf_baseline": "Baseline PF",
                                                "qualifies":        "Qualifies",
                                            }),
                                            use_container_width=True,
                                            hide_index=True,
                                        )
                        except Exception as _tot_e:
                            st.caption(f"\u26a0\ufe0f Could not compute total power bins: {_tot_e}")
                else:
                    st.info("No baseline ingested yet.")

                    # ── Per-phase baseline PF bins ────────────────────────────
                # Compute and show phase baseline bins directly from stored baseline data
                if _stored_bl_dict and meta:
                    try:
                        _bl_ts1_ph = pd.Timestamp(_stored_bl_dict.get("timestamp_start", ""))
                        _bl_ts2_ph = pd.Timestamp(_stored_bl_dict.get("timestamp_end", ""))
                        _bl_raw_ph = data.loc[
                            (_bl_ts1_ph <= data.index) & (data.index <= _bl_ts2_ph)
                        ].reset_index()
                        _bl_w_ph = scale_power_to_watts(_bl_raw_ph, meta.get("power_unit", "W"))
                        _bl_cl_ph, _ = clean_samples(
                            _bl_w_ph, meta,
                            _stored_bl_dict.get("user_filter_expr"),
                            load_precondition_fraction=st.session_state.get("load_precondition_pct", 20) / 100.0,
                        )
                        if not _bl_cl_ph.empty:
                            for _ph in (1, 2, 3):
                                _ph_all_bl = select_pf_bands_phase(
                                    _bl_cl_ph, _ph, min_samples=0
                                )
                                if not _ph_all_bl:
                                    continue
                                _nq_ph = sum(1 for b in _ph_all_bl if b.n_baseline >= 5)
                                with st.expander(
                                    f"\U0001f4ca Phase {_ph} baseline PF bins "
                                    f"({_nq_ph} qualifying of {len(_ph_all_bl)} total)",
                                    expanded=False,
                                ):
                                    st.caption(
                                        f"Bins by P_{_ph} power, 1% of phase operating range. "
                                        f"Bins with \u22655 samples qualify for PF drift detection."
                                    )
                                    _ph_bl_rows = []
                                    for _b in _ph_all_bl:
                                        _q = _b.n_baseline >= 5
                                        _ph_bl_rows.append({
                                            "Low (kW)":        f"{_b.low_kw  / 1000:.3f}",
                                            "Centre (kW)":     f"{_b.centre_kw / 1000:.3f}",
                                            "High (kW)":       f"{_b.high_kw  / 1000:.3f}",
                                            "n baseline":      _b.n_baseline,
                                            "Baseline PF":     f"{_b.mean_pf_baseline:.4f}" if _b.n_baseline > 0 else "\u2014",
                                            "Baseline \u03c3": f"{_b.std_pf_baseline:.5f}" if _b.n_baseline > 0 else "\u2014",
                                            "Qualifies":       "\u2705 Yes" if _q else "\u274c No (<5)",
                                        })
                                    st.dataframe(
                                        pd.DataFrame(_ph_bl_rows),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                                    st.caption(
                                        f"{_nq_ph} of {len(_ph_all_bl)} bins qualify. "
                                        f"{len(_ph_all_bl) - _nq_ph} excluded."
                                    )
                    except Exception:
                        pass  # silently skip if baseline data unavailable

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
                        # Apply integrity filter — exclude rows that failed checks
                        _ic_passed_ts = st.session_state.get("last_integrity_passed_ts")
                        _raw_bl, _bl_excluded = apply_integrity_filter(_raw_bl, _ic_passed_ts)
                        st.session_state["baseline_ic_excluded"] = _bl_excluded
                        if _ic_passed_ts is None:
                            st.info(
                                "\u2139\ufe0f Integrity checks have not been run yet. "
                                "Open the \U0001f50d Integrity checks expander in the Data tab "
                                "to screen for wiring faults before ingesting baseline."
                            )
                        elif _bl_excluded > 0:
                            st.warning(
                                f"\u26a0\ufe0f {_bl_excluded:,} rows excluded from baseline "
                                f"— failed integrity checks (wiring/CT faults). "
                                f"{len(_raw_bl):,} rows used."
                            )
                        if _raw_bl.empty:
                            st.error("No data in selected baseline period after integrity filtering.")
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
                            st.session_state["last_integrity_passed_ts"] = None
                            st.rerun()

                st.markdown("---")
                # Load precondition threshold
                _load_pct = st.number_input(
                    "Minimum load threshold (% of rated)",
                    min_value=1, max_value=50,
                    value=int(st.session_state.get("load_precondition_pct", 20)),
                    step=1,
                    key="load_precondition_pct",
                    help=(
                        "Step 1 of data cleaning removes samples below this fraction "
                        "of rated electrical power. Default 20 % (methodology §4.1). "
                        "Lower values retain more low-load samples; raise if CT accuracy "
                        "is poor at light load."
                    ),
                )
                _load_frac_ui = _load_pct / 100.0
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
                        # Apply integrity filter — exclude rows that failed checks
                        _ic_passed_ts_assess = st.session_state.get("last_integrity_passed_ts")
                        _recent, _ic_excluded_assess = apply_integrity_filter(_recent, _ic_passed_ts_assess)
                        if _ic_passed_ts_assess is None:
                            st.info(
                                "\u2139\ufe0f Integrity checks have not been run. "
                                "Open the \U0001f50d Integrity checks expander in the Data tab "
                                "to screen for wiring faults before running assessment."
                            )
                        elif _ic_excluded_assess > 0:
                            st.warning(
                                f"\u26a0\ufe0f {_ic_excluded_assess:,} rows excluded from assessment "
                                f"— failed integrity checks. "
                                f"{len(_recent):,} rows used."
                            )
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
                                # Load raw baseline data for band computation at analysis time
                                _bl_start_ts3 = pd.Timestamp(_bm_loaded.timestamp_start) if _bm_loaded and _bm_loaded.timestamp_start else None
                                _bl_end_ts3   = pd.Timestamp(_bm_loaded.timestamp_end)   if _bm_loaded and _bm_loaded.timestamp_end   else None
                                if _bl_start_ts3 is not None and _bl_end_ts3 is not None:
                                    _raw_bl_for_assess = data.loc[
                                        (_bl_start_ts3 <= data.index) & (data.index <= _bl_end_ts3)
                                    ].reset_index()
                                    _raw_bl_for_assess = scale_power_to_watts(_raw_bl_for_assess, meta.get("power_unit", "W"))
                                else:
                                    _raw_bl_for_assess = None
                                _record = run_assessment(_raw_reset, _bm_loaded, meta,
                                                         raw_baseline=_raw_bl_for_assess,
                                                         load_precondition_fraction=_load_frac_ui)
                                # Also capture cleaned data for download
                                _user_filter = _bm_loaded.user_filter_expr if _bm_loaded else None
                                _cleaned_df, _ = clean_samples(_raw_reset, meta, _user_filter,
                                                               load_precondition_fraction=_load_frac_ui)

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
                            # Compute per-phase PF drift bands (phase-specific bins)
                            # Use cleaned data for both baseline and assessment
                            _user_filter = _bm_loaded.user_filter_expr if _bm_loaded else None
                            _cleaned_bl_for_phase = None
                            if _raw_bl_for_assess is not None:
                                _cleaned_bl_for_phase, _ = clean_samples(
                                    _raw_bl_for_assess, meta, _user_filter,
                                    load_precondition_fraction=_load_frac_ui,
                                )
                            if _cleaned_bl_for_phase is not None and len(_cleaned_bl_for_phase) > 0:
                                _ph_bands = {}
                                _ph_bl_all_bins = {}  # all bins including <5 samples
                                for _ph in (1, 2, 3):
                                    # Build phase-specific baseline bands, extended to cover assessment range
                                    _ph_bl_all = select_pf_bands_phase(
                                        _cleaned_bl_for_phase, _ph,
                                        min_samples=0,
                                        cleaned_recent=_cleaned_df
                                    )
                                    _ph_bl_bands = [b for b in _ph_bl_all
                                                    if b.n_baseline >= 5]
                                    _ph_bl_all_bins[_ph] = _ph_bl_all
                                    # Compute drift using cleaned assessment data
                                    _ph_bands[_ph] = compute_pf_drift_phase(
                                        _cleaned_df, _ph_bl_bands,
                                        _ph, _cleaned_bl_for_phase
                                    )
                                st.session_state["last_phase_bands"]       = _ph_bands
                                st.session_state["last_phase_bl_all_bins"] = _ph_bl_all_bins
                            else:
                                st.session_state["last_phase_bands"]       = {}
                                st.session_state["last_phase_bl_all_bins"] = {}

                            # Baseline imbalance averages — stored for gauge comparison display
                            _bl_vuf_mean = None
                            _bl_iuf_mean = None
                            _bl_ph_iuf   = {}
                            if _cleaned_bl_for_phase is not None and len(_cleaned_bl_for_phase) > 0:
                                try:
                                    _bl_df = _cleaned_bl_for_phase.copy()
                                    if "timestamp" in _bl_df.columns:
                                        _bl_df = _bl_df.set_index("timestamp")
                                    _bv1=_bl_df["phase_1_voltage"]
                                    _bv2=_bl_df["phase_2_voltage"]
                                    _bv3=_bl_df["phase_3_voltage"]
                                    _bva=(_bv1+_bv2+_bv3)/3.0
                                    _bl_vuf_mean = round(float(
                                        (pd.concat([(_bv1-_bva).abs(),(_bv2-_bva).abs(),(_bv3-_bva).abs()],axis=1)
                                         .max(axis=1)/_bva.replace(0,np.nan)*100.0).mean()
                                    ), 3)
                                    _bi1=_bl_df["phase_1_current"]
                                    _bi2=_bl_df["phase_2_current"]
                                    _bi3=_bl_df["phase_3_current"]
                                    _bia=(_bi1+_bi2+_bi3)/3.0
                                    _bis=_bia.replace(0,np.nan)
                                    _bl_iuf_mean = round(float(
                                        (pd.concat([(_bi1-_bia).abs(),(_bi2-_bia).abs(),(_bi3-_bia).abs()],axis=1)
                                         .max(axis=1)/_bis*100.0).mean()
                                    ), 2)
                                    _bl_ph_iuf = {
                                        1: round(float(((_bi1-_bia).abs()/_bis*100).mean()), 2),
                                        2: round(float(((_bi2-_bia).abs()/_bis*100).mean()), 2),
                                        3: round(float(((_bi3-_bia).abs()/_bis*100).mean()), 2),
                                    }
                                except Exception:
                                    pass
                            st.session_state["baseline_vuf_mean"] = _bl_vuf_mean
                            st.session_state["baseline_iuf_mean"] = _bl_iuf_mean
                            st.session_state["baseline_ph_iuf"]   = _bl_ph_iuf

                            # Baseline daily P_total for run chart
                            _bl_p_daily = {}
                            if _cleaned_bl_for_phase is not None and len(_cleaned_bl_for_phase) > 0:
                                try:
                                    _bp_cols = ["phase_1_active_power",
                                                "phase_2_active_power",
                                                "phase_3_active_power"]
                                    if all(c in _cleaned_bl_for_phase.columns for c in _bp_cols):
                                        _bp_df = _cleaned_bl_for_phase.copy()
                                        # Handle both DatetimeIndex and timestamp-as-column
                                        if "timestamp" in _bp_df.columns:
                                            _bp_df = _bp_df.set_index("timestamp")
                                        _bp_df.index = pd.to_datetime(_bp_df.index)
                                        _bp_s = (_bp_df[_bp_cols[0]] +
                                                 _bp_df[_bp_cols[1]] +
                                                 _bp_df[_bp_cols[2]]) / 1000.0
                                        _bp_daily = _bp_s.resample("D").mean().dropna()
                                        _bl_p_daily = {
                                            str(d.date()): round(float(v), 3)
                                            for d, v in zip(_bp_daily.index, _bp_daily.values)
                                        }
                                except Exception:
                                    pass
                            st.session_state["baseline_p_daily"] = _bl_p_daily
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

                    # ── Cleaned data download + removed rows per step ────────
                    _cleaned = st.session_state.get("last_cleaned_data")
                    _raw_for_dl = st.session_state.get("last_data")
                    if _cleaned is not None and not _cleaned.empty and record.cleaning_report:
                        _cr        = record.cleaning_report
                        _n_cleaned = _cr.n_cleaned
                        _dl_unit   = meta.get("power_unit", "W").upper()

                        def _to_dl_unit(df):
                            """Convert power columns to original unit for download."""
                            df = df.copy()
                            if _dl_unit == "KW":
                                for _pc in ["phase_1_active_power",
                                            "phase_2_active_power",
                                            "phase_3_active_power"]:
                                    if _pc in df.columns:
                                        df[_pc] = (df[_pc] / 1000.0).round(6)
                            return df[[c for c in df.columns if not c.startswith("_")]]

                        with st.expander(
                            f"\U0001f4ca Assessment data — cleaned & removed rows",
                            expanded=False,
                        ):
                            st.caption(
                                "Download the cleaned dataset used for analysis, or the rows "
                                "removed at each cleaning step."
                            )

                            # Use clean_samples directly — guarantees same result as run_assessment
                            _raw_w = scale_power_to_watts(
                                (_raw_for_dl if _raw_for_dl is not None else
                                 data.loc[
                                     (pd.Timestamp(date_range[0]) <= data.index) &
                                     (data.index <= pd.Timestamp(date_range[1]) +
                                      pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
                                 ]).reset_index(), meta.get("power_unit", "W")
                            )
                            _uf_for_dl = (baseline_from_dict(db.get_baseline(selected_id)).user_filter_expr
                                          if db.get_baseline(selected_id) else None)
                            _cleaned_redone, _cr_redone = clean_samples(
                                _raw_w, meta, _uf_for_dl,
                                load_precondition_fraction=st.session_state.get("load_precondition_pct", 20) / 100.0,
                            )

                            # Derive removed rows by timestamp comparison
                            _raw_ts_set     = set(_raw_w["timestamp"].astype(str)) if "timestamp" in _raw_w.columns else set()
                            _cleaned_ts_set = set(_cleaned_redone["timestamp"].astype(str)) if "timestamp" in _cleaned_redone.columns else set()
                            _removed_ts_set = _raw_ts_set - _cleaned_ts_set

                            # Tag each removed row with step label using cleaning report counts
                            _removed_all_labelled = _raw_w[
                                _raw_w["timestamp"].astype(str).isin(_removed_ts_set)
                            ].copy() if "timestamp" in _raw_w.columns else pd.DataFrame()

                            # Assign step labels based on cleaning report thresholds
                            if not _removed_all_labelled.empty:
                                _pt_rm = (_removed_all_labelled["phase_1_active_power"] +
                                          _removed_all_labelled["phase_2_active_power"] +
                                          _removed_all_labelled["phase_3_active_power"])
                                # Estimate load_min same way clean_samples does
                                _p_shaft_rm = float(meta.get("p_rated_shaft_kw", 0) or 0)
                                _eta_rm     = float(meta.get("eta_rated", 0.9) or 0.9)
                                if _p_shaft_rm > 0 and _eta_rm > 0:
                                    _lm_rm = 0.20 * (_p_shaft_rm / _eta_rm) * 1000.0
                                else:
                                    _p95_rm = float((_raw_w["phase_1_active_power"] +
                                                     _raw_w["phase_2_active_power"] +
                                                     _raw_w["phase_3_active_power"]
                                                    ).quantile(0.95))
                                    _lm_rm = 0.20 * (_p95_rm / 0.95)
                                _removed_all_labelled["removed_at_step"] = _removed_all_labelled.apply(
                                    lambda r: (
                                        "Step 1 - Load precondition (<20% rated)"
                                        if (r["phase_1_active_power"] + r["phase_2_active_power"] +
                                            r["phase_3_active_power"]) < _lm_rm
                                        else "Step 2/3 - Transient or user filter"
                                    ), axis=1
                                )

                            _n_removed = len(_removed_all_labelled)

                            # Two download columns
                            # ── Derive each stage sequentially ───────────────
                            # _raw_w = assessment period, post-integrity, scaled to Watts
                            # Reconstruct true assessment-period raw by adding back
                            # integrity-excluded rows using the same date range
                            _assess_start = _raw_w["timestamp"].min() if "timestamp" in _raw_w.columns else None
                            _assess_end   = _raw_w["timestamp"].max() if "timestamp" in _raw_w.columns else None

                            if _assess_start is not None:
                                _raw_assess = data.loc[
                                    (pd.Timestamp(_assess_start) <= data.index) &
                                    (data.index <= pd.Timestamp(_assess_end))
                                ].reset_index()
                                _raw_assess = scale_power_to_watts(
                                    _raw_assess, meta.get("power_unit", "W")
                                )
                            else:
                                _raw_assess = _raw_w.copy()

                            _dl_raw_assess = _to_dl_unit(_raw_assess)

                            # Integrity-removed rows = raw - post-integrity
                            _dl_post_ic = _to_dl_unit(_raw_w)
                            if "timestamp" in _dl_raw_assess.columns and "timestamp" in _dl_post_ic.columns:
                                _ic_pass_ts_set = set(_dl_post_ic["timestamp"].astype(str))
                                _dl_ic_removed  = _dl_raw_assess[
                                    ~_dl_raw_assess["timestamp"].astype(str).isin(_ic_pass_ts_set)
                                ].copy()
                            else:
                                _dl_ic_removed = pd.DataFrame(columns=_dl_raw_assess.columns)

                            _n_ic_excl   = len(_dl_ic_removed)
                            _dl_removed  = _to_dl_unit(_removed_all_labelled)
                            _dl_cleaned  = _to_dl_unit(
                                _cleaned if "timestamp" in _cleaned.columns
                                else _cleaned.reset_index()
                            )

                            # ── Sequential display ────────────────────────────
                            st.markdown("---")
                            # Stage 1: Raw assessment data
                            st.markdown(f"**\U0001f4e5 1. Raw assessment data — {len(_dl_raw_assess):,} rows**")
                            st.download_button(
                                label=f"\u2b07\ufe0f Download raw assessment data ({len(_dl_raw_assess):,} rows, CSV)",
                                data=_dl_raw_assess.to_csv(index=False).encode("utf-8"),
                                file_name=f"raw_{selected_id}_{date_range[0]}_to_{date_range[1]}.csv",
                                mime="text/csv", use_container_width=True, key="dl_raw_assess",
                            )
                            st.dataframe(_dl_raw_assess.head(5), use_container_width=True, hide_index=True)

                            st.markdown("---")
                            # Stage 2: Removed by integrity check
                            _ic_col1, _ic_col2 = st.columns(2)
                            with _ic_col1:
                                st.markdown(f"**\U0001f6e1\ufe0f 2. Removed by integrity check — {_n_ic_excl:,} rows**")
                                if _n_ic_excl > 0:
                                    st.download_button(
                                        label=f"\u2b07\ufe0f Download integrity-removed rows ({_n_ic_excl:,} rows, CSV)",
                                        data=_dl_ic_removed.to_csv(index=False).encode("utf-8"),
                                        file_name=f"integrity_removed_{selected_id}_{date_range[0]}_to_{date_range[1]}.csv",
                                        mime="text/csv", use_container_width=True, key="dl_ic_removed",
                                    )
                                    st.dataframe(_dl_ic_removed.head(5), use_container_width=True, hide_index=True)
                                else:
                                    st.caption("\u2705 No rows removed by integrity check")
                            with _ic_col2:
                                # Stage 3: After integrity check
                                st.markdown(f"**\u2705 3. After integrity check — {len(_dl_post_ic):,} rows**")
                                st.download_button(
                                    label=f"\u2b07\ufe0f Download post-integrity data ({len(_dl_post_ic):,} rows, CSV)",
                                    data=_dl_post_ic.to_csv(index=False).encode("utf-8"),
                                    file_name=f"post_integrity_{selected_id}_{date_range[0]}_to_{date_range[1]}.csv",
                                    mime="text/csv", use_container_width=True, key="dl_post_ic",
                                )
                                st.dataframe(_dl_post_ic.head(5), use_container_width=True, hide_index=True)

                            st.markdown("---")
                            # Stage 4 & 5: Cleaning removed and final cleaned
                            _cl_col1, _cl_col2 = st.columns(2)
                            with _cl_col1:
                                st.markdown(f"**\u274c 4. Removed by cleaning — {_n_removed:,} rows**")
                                st.download_button(
                                    label=f"\u2b07\ufe0f Download cleaning-removed rows ({_n_removed:,} rows, CSV)",
                                    data=_dl_removed.to_csv(index=False).encode("utf-8"),
                                    file_name=f"removed_{selected_id}_{date_range[0]}_to_{date_range[1]}.csv",
                                    mime="text/csv", use_container_width=True, key="dl_cleaning_removed",
                                )
                                st.dataframe(_dl_removed.head(5), use_container_width=True, hide_index=True)
                            with _cl_col2:
                                st.markdown(f"**\U0001f4ca 5. Final data for analysis — {_n_cleaned:,} rows**")
                                st.download_button(
                                    label=f"\u2b07\ufe0f Download cleaned data ({_n_cleaned:,} rows, CSV)",
                                    data=_dl_cleaned.to_csv(index=False).encode("utf-8"),
                                    file_name=f"cleaned_{selected_id}_{date_range[0]}_to_{date_range[1]}.csv",
                                    mime="text/csv", use_container_width=True, key="dl_cleaned_final",
                                )
                                st.dataframe(_dl_cleaned.head(5), use_container_width=True, hide_index=True)

                    # ── Operating zone report ───────────────────────────────
                    _oz_raw = st.session_state.get("last_data")
                    if _oz_raw is not None and meta:
                        _oz_raw_w = scale_power_to_watts(
                            _oz_raw.reset_index(), meta.get("power_unit", "W")
                        )
                        _oz_pt = (_oz_raw_w["phase_1_active_power"] +
                                  _oz_raw_w["phase_2_active_power"] +
                                  _oz_raw_w["phase_3_active_power"])
                        _oz_n_total = len(_oz_pt)

                        # Derive rated electrical power (same as clean_samples)
                        _oz_p_shaft = float(meta.get("p_rated_shaft_kw", 0) or 0)
                        _oz_eta     = float(meta.get("eta_rated", 0.9) or 0.9)
                        if _oz_p_shaft > 0 and _oz_eta > 0:
                            _oz_p_rated_w = (_oz_p_shaft / _oz_eta) * 1000.0
                        else:
                            _oz_p95 = float(_oz_pt[_oz_pt > 0].quantile(0.95)) if (_oz_pt > 0).any() else 1.0
                            _oz_p_rated_w = _oz_p95 / 0.95

                        # Zone boundaries as % of rated electrical input
                        _oz_zones = [
                            ("< 20%",    0,    0.20),
                            ("20–40%",   0.20, 0.40),
                            ("40–60%",   0.40, 0.60),
                            ("60–80%",   0.60, 0.80),
                            ("80–100%",  0.80, 1.00),
                            ("> 100%",   1.00, float("inf")),
                        ]

                        _oz_rows = []
                        # Infer sampling interval from median timestamp difference
                        if len(_oz_pt) > 1 and "timestamp" in _oz_raw_w.columns:
                            _ts_sorted = pd.to_datetime(_oz_raw_w["timestamp"]).sort_values()
                            _oz_interval_min = float(
                                _ts_sorted.diff().dropna().dt.total_seconds().median() / 60.0
                            )
                        else:
                            _oz_interval_min = 1.0  # fallback: 1 minute per sample

                        for _zname, _zlo, _zhi in _oz_zones:
                            _lo_w = _zlo * _oz_p_rated_w
                            _hi_w = _zhi * _oz_p_rated_w
                            _mask = (_oz_pt >= _lo_w) & (_oz_pt < _hi_w)
                            _cnt  = int(_mask.sum())
                            _pct  = _cnt / _oz_n_total * 100 if _oz_n_total > 0 else 0.0
                            _mins = _cnt * _oz_interval_min
                            _hrs  = _mins / 60.0
                            _time_str = (
                                f"{_hrs:.1f} h" if _hrs >= 1.0
                                else f"{_mins:.0f} min"
                            )
                            _oz_rows.append({
                                "Zone":              _zname,
                                "P range (kW)":      (
                                    f"{_lo_w/1000:.1f} – {_hi_w/1000:.1f}"
                                    if _zhi != float("inf")
                                    else f"> {_lo_w/1000:.1f}"
                                ),
                                "Time":              _time_str,
                                "% Operating time":  f"{_pct:.1f}%",
                                "_hrs":              _hrs,   # for chart
                            })

                        import plotly.graph_objects as _go_oz
                        with st.expander(
                            "\U0001f4ca Operating zone distribution", expanded=False
                        ):
                            st.caption(
                                f"Distribution of all assessment samples (post-integrity, {_oz_n_total:,} rows) "
                                f"across load zones relative to rated electrical input "
                                f"({_oz_p_rated_w/1000:.1f} kW). "
                                f"Best operating zone: **80–100%**."
                            )
                            _oz_df = pd.DataFrame(_oz_rows)

                            # Bar chart
                            _oz_colors = [
                                "#C0392B",  # <20% — stopped/very light
                                "#E67E22",  # 20-40% — light load
                                "#F1C40F",  # 40-60% — medium-light
                                "#2ECC71",  # 60-80% — good
                                "#27AE60",  # 80-100% — optimal
                                "#8E44AD",  # >100% — overload
                            ]
                            _fig_oz = _go_oz.Figure()
                            _fig_oz.add_trace(_go_oz.Bar(
                                x=[r["Zone"] for r in _oz_rows],
                                y=[round(r["_hrs"], 2) for r in _oz_rows],
                                marker_color=_oz_colors,
                                text=[f"{r['Time']} ({r['% Operating time']})" for r in _oz_rows],
                                textposition="outside",
                                hovertemplate=(
                                    "<b>%{x}</b><br>"
                                    "Time: %{text}<extra></extra>"
                                ),
                            ))
                            _fig_oz.update_layout(
                                xaxis_title="Load zone",
                                yaxis_title="Time (hours)",
                                height=320,
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                margin=dict(l=40, r=20, t=40, b=40),
                                font=dict(size=11),
                                showlegend=False,
                                bargap=0.15,
                            )
                            st.plotly_chart(_fig_oz, use_container_width=True)
                            st.caption(f"Sampling interval: {_oz_interval_min:.1f} min/sample")
                            st.dataframe(
                                pd.DataFrame(_oz_rows).drop(columns=["_hrs"]),
                                use_container_width=True, hide_index=True
                            )

                    # ── PF Drift Excel export ────────────────────────────────
                    _cleaned_for_xl = st.session_state.get("last_cleaned_data")
                    if _cleaned_for_xl is not None and not _cleaned_for_xl.empty and record:

                        with st.expander(
                            "\U0001f4c8 PF drift Excel export", expanded=False
                        ):
                            st.caption(
                                "Each cleaned sample tagged with its bin number and the "
                                "band statistics. Sheet 1 = sample-level data. "
                                "Sheet 2 = band summary."
                            )
                            if st.button(
                                "\U0001f4be Generate PF drift Excel",
                                key="gen_pf_xl_btn",
                                use_container_width=True,
                            ):
                                import io as _io
                                from openpyxl import Workbook as _WB
                                from openpyxl.styles import (
                                    Font as _Font, PatternFill as _Fill,
                                    Alignment as _Align, Border as _Border,
                                    Side as _Side,
                                )

                                # ── Build band lookup from all active bands ──
                                _bands_src = []
                                if record.motor_side and record.motor_side.bands:
                                    for _b in record.motor_side.bands:
                                        if not _b.suppressed and _b.pf_drift is not None:
                                            _bands_src.append(_b)

                                # ── Tag each sample with bin number ──────────
                                _xl_df = (_cleaned_for_xl.reset_index()
                                          if "timestamp" not in _cleaned_for_xl.columns
                                          else _cleaned_for_xl.copy())
                                _xl_df = scale_power_to_watts(
                                    _xl_df, meta.get("power_unit", "W")
                                )
                                _pt_xl = (_xl_df["phase_1_active_power"] +
                                          _xl_df["phase_2_active_power"] +
                                          _xl_df["phase_3_active_power"])
                                _pf_xl = _pt_xl / (
                                    _xl_df["phase_1_voltage"] * _xl_df["phase_1_current"] +
                                    _xl_df["phase_2_voltage"] * _xl_df["phase_2_current"] +
                                    _xl_df["phase_3_voltage"] * _xl_df["phase_3_current"]
                                ).replace(0, float("nan"))

                                # Assign bin number using numpy arrays for speed
                                _pt_arr = _pt_xl.values
                                _n_rows = len(_xl_df)
                                _bin_no     = [None] * _n_rows
                                _bin_low    = [None] * _n_rows
                                _bin_high   = [None] * _n_rows
                                _bin_bl_pf  = [None] * _n_rows
                                _bin_bl_std = [None] * _n_rows
                                _bin_rc_pf  = [None] * _n_rows
                                _bin_drift  = [None] * _n_rows
                                _bin_driftp = [None] * _n_rows
                                _bin_pval   = [None] * _n_rows
                                _bin_sig    = [None] * _n_rows
                                _bin_status = [None] * _n_rows

                                for _bi, _b in enumerate(_bands_src, start=1):
                                    _mask_arr = ((_pt_arr >= _b.low_kw) &
                                                 (_pt_arr < _b.high_kw))
                                    _dp = ((_b.pf_drift / _b.mean_pf_baseline * 100)
                                           if _b.mean_pf_baseline else None)
                                    _st = ("Action" if _b.pf_drift <= -0.03 else
                                           "Critical" if _b.pf_drift <= float(PF_DRIFT_ACTION) else
                                           "Watch"  if _b.pf_drift <= -0.01 else
                                           "Normal")
                                    _sig = ("Yes" if _b.drift_significant else
                                            "No"  if _b.drift_significant is False
                                            else "n/a")
                                    for _pos in range(_n_rows):
                                        if _mask_arr[_pos]:
                                            _bin_no[_pos]     = _bi
                                            _bin_low[_pos]    = round(_b.low_kw  / 1000, 2)
                                            _bin_high[_pos]   = round(_b.high_kw / 1000, 2)
                                            _bin_bl_pf[_pos]  = _b.mean_pf_baseline
                                            _bin_bl_std[_pos] = _b.std_pf_baseline
                                            _bin_rc_pf[_pos]  = _b.mean_pf_recent
                                            _bin_drift[_pos]  = _b.pf_drift
                                            _bin_driftp[_pos] = round(_dp, 4) if _dp else None
                                            _bin_pval[_pos]   = _b.p_value
                                            _bin_sig[_pos]    = _sig
                                            _bin_status[_pos] = _st

                                _xl_df["p_total_kw"]       = (_pt_xl / 1000).round(3)
                                _xl_df["pf_sample"]        = _pf_xl.round(5)
                                _xl_df["bin_number"]       = _bin_no
                                _xl_df["bin_low_kw"]       = _bin_low
                                _xl_df["bin_high_kw"]      = _bin_high
                                _xl_df["baseline_pf_mean"] = _bin_bl_pf
                                _xl_df["baseline_pf_std"]  = _bin_bl_std
                                _xl_df["recent_pf_mean"]   = _bin_rc_pf
                                _xl_df["pf_drift"]         = _bin_drift
                                _xl_df["pf_drift_pct"]     = _bin_driftp
                                _xl_df["p_value"]          = _bin_pval
                                _xl_df["significant"]      = _bin_sig
                                _xl_df["status"]           = _bin_status

                                # ── Build Excel workbook ─────────────────────
                                _hdr_fill = _Fill("solid", start_color="054D5F")
                                _hdr_font = _Font(bold=True, color="FFFFFF",
                                                  name="Arial", size=10)
                                _body_font = _Font(name="Arial", size=10)
                                _thin = _Border(
                                    left=_Side(style="thin"),
                                    right=_Side(style="thin"),
                                    top=_Side(style="thin"),
                                    bottom=_Side(style="thin"),
                                )

                                def _style_header(ws, headers):
                                    for col, h in enumerate(headers, 1):
                                        c = ws.cell(row=1, column=col, value=h)
                                        c.font = _hdr_font
                                        c.fill = _hdr_fill
                                        c.alignment = _Align(
                                            horizontal="center", wrap_text=True
                                        )
                                        c.border = _thin

                                _wb = _WB()

                                # ── Sheet 1: Sample-level ────────────────────
                                _ws1 = _wb.active
                                _ws1.title = "Sample Data"
                                _keep_cols = [
                                    "timestamp", "p_total_kw", "pf_sample",
                                    "bin_number", "bin_low_kw", "bin_high_kw",
                                    "baseline_pf_mean", "baseline_pf_std",
                                    "recent_pf_mean", "pf_drift", "pf_drift_pct",
                                    "p_value", "significant", "status",
                                ]
                                _out_cols = [c for c in _keep_cols
                                             if c in _xl_df.columns]
                                _hdr1 = [
                                    "Timestamp", "P_total (kW)", "PF (sample)",
                                    "Bin #", "Bin Low (kW)", "Bin High (kW)",
                                    "Baseline PF mean", "Baseline PF std",
                                    "Recent PF mean", "PF Drift",
                                    "Drift %", "p-value", "Significant", "Status",
                                ]
                                _style_header(_ws1, _hdr1[:len(_out_cols)])
                                for _ri, _row in enumerate(
                                    _xl_df[_out_cols].itertuples(index=False), start=2
                                ):
                                    for _ci, _val in enumerate(_row, start=1):
                                        _c = _ws1.cell(row=_ri, column=_ci,
                                                        value=_val)
                                        _c.font = _body_font
                                        _c.border = _thin
                                # Column widths
                                for _ci, _w in enumerate(
                                    [18,12,12,8,12,12,16,14,14,10,10,10,12,10],
                                    start=1
                                ):
                                    _ws1.column_dimensions[
                                        _ws1.cell(1, _ci).column_letter
                                    ].width = _w
                                _ws1.freeze_panes = "A2"

                                # ── Sheet 2: Band summary ────────────────────
                                _ws2 = _wb.create_sheet("Band Summary")
                                _hdr2 = [
                                    "Bin #", "Low (kW)", "Centre (kW)", "High (kW)",
                                    "Baseline PF", "Baseline σ",
                                    "Recent PF", "Recent σ",
                                    "Drift", "Drift %", "p-value",
                                    "Significant", "Status",
                                    "n baseline", "n recent",
                                ]
                                _style_header(_ws2, _hdr2)
                                for _bi, _b in enumerate(_bands_src, start=1):
                                    _dp2 = ((_b.pf_drift / _b.mean_pf_baseline * 100)
                                            if _b.mean_pf_baseline else None)
                                    _st2 = ("Action" if _b.pf_drift <= -0.03 else
                                            "Critical" if _b.pf_drift <= float(PF_DRIFT_ACTION) else
                                            "Watch"  if _b.pf_drift <= -0.01 else
                                            "Normal")
                                    _row2 = [
                                        _bi,
                                        round(_b.low_kw    / 1000, 2),
                                        round(_b.centre_kw / 1000, 2),
                                        round(_b.high_kw   / 1000, 2),
                                        _b.mean_pf_baseline,
                                        _b.std_pf_baseline,
                                        _b.mean_pf_recent,
                                        _b.std_pf_recent,
                                        _b.pf_drift,
                                        round(_dp2, 4) if _dp2 else None,
                                        _b.p_value,
                                        ("Yes" if _b.drift_significant else
                                         "No"  if _b.drift_significant is False
                                         else "n/a"),
                                        _st2,
                                        _b.n_baseline,
                                        _b.n_recent,
                                    ]
                                    for _ci, _val in enumerate(_row2, start=1):
                                        _c = _ws2.cell(
                                            row=_bi + 1, column=_ci, value=_val
                                        )
                                        _c.font = _body_font
                                        _c.border = _thin
                                for _ci, _w in enumerate(
                                    [8,10,12,10,12,12,12,10,10,10,10,12,10,12,10],
                                    start=1
                                ):
                                    _ws2.column_dimensions[
                                        _ws2.cell(1, _ci).column_letter
                                    ].width = _w
                                _ws2.freeze_panes = "A2"

                                # ── Sheet 3: Baseline sample data ────────────
                                _ws3 = _wb.create_sheet("Baseline Data")
                                _hdr3 = [
                                    "Timestamp", "P_total (kW)", "PF (sample)",
                                    "Bin #", "Bin Low (kW)", "Bin High (kW)",
                                    "Baseline PF mean", "Baseline PF std",
                                ]
                                _style_header(_ws3, _hdr3)

                                # Load raw baseline data using stored timestamps
                                try:
                                    _bm_xl = baseline_from_dict(
                                        db.get_baseline(selected_id)
                                    )
                                    if (_bm_xl and _bm_xl.timestamp_start
                                            and _bm_xl.timestamp_end):
                                        _bl_ts1 = pd.Timestamp(_bm_xl.timestamp_start)
                                        _bl_ts2 = pd.Timestamp(_bm_xl.timestamp_end)
                                        _bl_raw = data.loc[
                                            (_bl_ts1 <= data.index) &
                                            (data.index <= _bl_ts2)
                                        ].reset_index()
                                        _bl_w = scale_power_to_watts(
                                            _bl_raw, meta.get("power_unit", "W")
                                        )
                                        # Clean baseline same way as assessment
                                        _bl_cleaned, _ = clean_samples(
                                            _bl_w, meta,
                                            _bm_xl.user_filter_expr,
                                            load_precondition_fraction=st.session_state.get("load_precondition_pct", 20) / 100.0,
                                        )
                                        _pt_bl = (
                                            _bl_cleaned["phase_1_active_power"] +
                                            _bl_cleaned["phase_2_active_power"] +
                                            _bl_cleaned["phase_3_active_power"]
                                        )
                                        _s_bl = (
                                            _bl_cleaned["phase_1_voltage"] *
                                            _bl_cleaned["phase_1_current"] +
                                            _bl_cleaned["phase_2_voltage"] *
                                            _bl_cleaned["phase_2_current"] +
                                            _bl_cleaned["phase_3_voltage"] *
                                            _bl_cleaned["phase_3_current"]
                                        ).replace(0, float("nan"))
                                        _pf_bl = (_pt_bl / _s_bl).round(5)
                                        _pt_bl_arr = _pt_bl.values
                                        _n_bl = len(_bl_cleaned)

                                        # Bin arrays for baseline
                                        _bl_bin_no  = [None] * _n_bl
                                        _bl_bin_low = [None] * _n_bl
                                        _bl_bin_hi  = [None] * _n_bl
                                        _bl_bl_pf   = [None] * _n_bl
                                        _bl_bl_std  = [None] * _n_bl

                                        for _bi3, _b3 in enumerate(_bands_src, start=1):
                                            _m3 = ((_pt_bl_arr >= _b3.low_kw) &
                                                   (_pt_bl_arr <  _b3.high_kw))
                                            for _p3 in range(_n_bl):
                                                if _m3[_p3]:
                                                    _bl_bin_no[_p3]  = _bi3
                                                    _bl_bin_low[_p3] = round(_b3.low_kw  / 1000, 2)
                                                    _bl_bin_hi[_p3]  = round(_b3.high_kw / 1000, 2)
                                                    _bl_bl_pf[_p3]   = _b3.mean_pf_baseline
                                                    _bl_bl_std[_p3]  = _b3.std_pf_baseline

                                        _ts_bl = (
                                            _bl_cleaned["timestamp"].tolist()
                                            if "timestamp" in _bl_cleaned.columns
                                            else [None] * _n_bl
                                        )

                                        for _ri3 in range(_n_bl):
                                            _row3 = [
                                                _ts_bl[_ri3],
                                                round(float(_pt_bl.iloc[_ri3]) / 1000, 3),
                                                float(_pf_bl.iloc[_ri3])
                                                if not pd.isna(_pf_bl.iloc[_ri3]) else None,
                                                _bl_bin_no[_ri3],
                                                _bl_bin_low[_ri3],
                                                _bl_bin_hi[_ri3],
                                                _bl_bl_pf[_ri3],
                                                _bl_bl_std[_ri3],
                                            ]
                                            for _ci3, _val3 in enumerate(_row3, start=1):
                                                _c3 = _ws3.cell(
                                                    row=_ri3 + 2,
                                                    column=_ci3,
                                                    value=_val3,
                                                )
                                                _c3.font = _body_font
                                                _c3.border = _thin

                                        for _ci3, _w3 in enumerate(
                                            [18, 12, 12, 8, 12, 12, 16, 14],
                                            start=1
                                        ):
                                            _ws3.column_dimensions[
                                                _ws3.cell(1, _ci3).column_letter
                                            ].width = _w3
                                        _ws3.freeze_panes = "A2"
                                except Exception as _e3:
                                    _ws3.cell(row=2, column=1,
                                              value=f"Could not load baseline data: {_e3}")

                                # ── Save to buffer and offer download ─────────
                                _buf = _io.BytesIO()
                                _wb.save(_buf)
                                _buf.seek(0)
                                st.download_button(
                                    label="\u2b07\ufe0f Download PF drift Excel",
                                    data=_buf.getvalue(),
                                    file_name=(
                                        f"pf_drift_{selected_id}_"
                                        f"{date_range[0]}_to_{date_range[1]}.xlsx"
                                    ),
                                    mime="application/vnd.openxmlformats-officedocument"
                                         ".spreadsheetml.sheet",
                                    use_container_width=True,
                                )
                                st.success(
                                    f"\u2713 Excel ready — "
                                    f"{len(_xl_df):,} samples, "
                                    f"{len(_bands_src)} bands."
                                )
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
                        # Gauge threshold controls — VUF, IUF and PF
                        with st.expander(
                            "\u2699\ufe0f Gauge Thresholds (VUF, IUF & PF)", expanded=False
                        ):
                            st.caption("Override methodology defaults for the gauge displays only. Does not affect alarm logic.")
                            _gt_c1, _gt_c2, _gt_c3 = st.columns(3)
                            _gt_c1.markdown("**Voltage Imbalance (VUF)**")
                            _gt_c2.markdown("**Current Imbalance (IUF)**")
                            _gt_c3.markdown("**Power Factor (PF)**")
                            _gt_c1.number_input(
                                "Watch threshold (%)",
                                min_value=0.1, max_value=10.0,
                                step=0.1,
                                key="vuf_gauge_watch",
                                help=f"Default: {VUF_WATCH:.1f}% (VUF_WATCH)",
                            )
                            _gt_c2.number_input(
                                "Watch threshold (%)",
                                min_value=0.1, max_value=50.0,
                                step=0.5,
                                key="iuf_gauge_watch",
                                help=f"Default: {IUF_WATCH:.0f}% (IUF_WATCH)",
                            )
                            _gt_c3.number_input(
                                "Watch threshold (PF)",
                                min_value=0.50, max_value=0.99,
                                step=0.01,
                                format="%.2f",
                                key="pf_gauge_watch",
                                help="PF below this value = amber. Default: 0.85",
                            )
                            _gt_c1.number_input(
                                "Critical threshold (%)",
                                min_value=0.1, max_value=20.0,
                                step=0.1,
                                key="vuf_gauge_critical",
                                help=f"Default: {VUF_CRITICAL:.1f}% (VUF_CRITICAL)",
                            )
                            _gt_c2.number_input(
                                "Critical threshold (%)",
                                min_value=0.1, max_value=100.0,
                                step=0.5,
                                key="iuf_gauge_critical",
                                help=f"Default: {IUF_CRITICAL:.0f}% (IUF_CRITICAL)",
                            )
                            _gt_c3.number_input(
                                "Critical threshold (PF)",
                                min_value=0.30, max_value=0.98,
                                step=0.01,
                                format="%.2f",
                                key="pf_gauge_critical",
                                help="PF below this value = red. Default: 0.75",
                            )
                            st.markdown("**PF Drift thresholds (absolute)**")
                            _gd1, _gd2 = st.columns(2)
                            _gd1.number_input(
                                "Watch (drift)",
                                min_value=-0.20, max_value=-0.001,
                                step=0.005, format="%.3f",
                                key="pf_drift_watch",
                                help=f"Default: {PF_DRIFT_WATCH:+.2f}",
                            )
                            _gd2.number_input(
                                "Critical (drift)",
                                min_value=-0.20, max_value=-0.001,
                                step=0.005, format="%.3f",
                                key="pf_drift_critical",
                                help=f"Default: {PF_DRIFT_ACTION:+.2f}",
                            )

                        # Read current widget values from session state
                        _vuf_g_watch = float(st.session_state.get("vuf_gauge_watch",  VUF_WATCH))
                        _vuf_g_crit  = float(st.session_state.get("vuf_gauge_critical", VUF_CRITICAL))
                        _iuf_g_watch = float(st.session_state.get("iuf_gauge_watch",  IUF_WATCH))
                        _iuf_g_crit  = float(st.session_state.get("iuf_gauge_critical", IUF_CRITICAL))
                        _pf_g_watch  = float(st.session_state.get("pf_gauge_watch",  0.85))
                        _pf_g_crit   = float(st.session_state.get("pf_gauge_critical", 0.75))
                        _pf_d_watch    = float(st.session_state.get("pf_drift_watch",    PF_DRIFT_WATCH))
                        _pf_d_critical = float(st.session_state.get("pf_drift_critical", PF_DRIFT_ACTION))
                        _warn_vuf      = _vuf_g_watch >= _vuf_g_crit
                        _warn_iuf      = _iuf_g_watch >= _iuf_g_crit
                        _warn_pf       = _pf_g_watch  <= _pf_g_crit
                        # Drift: Watch must be less negative than Critical
                        # (Watch -0.01 > Critical -0.03 numerically)
                        _warn_pf_drift = _pf_d_watch  <= _pf_d_critical
                        if _warn_vuf:
                            st.warning("VUF: Watch threshold must be below Critical. Reset to defaults.")
                            _vuf_g_watch = float(VUF_WATCH)
                            _vuf_g_crit  = float(VUF_CRITICAL)
                        if _warn_iuf:
                            st.warning("IUF: Watch threshold must be below Critical. Reset to defaults.")
                            _iuf_g_watch = float(IUF_WATCH)
                            _iuf_g_crit  = float(IUF_CRITICAL)
                        if _warn_pf:
                            st.warning("PF level: Watch threshold must be above Critical. Reset to defaults.")
                            _pf_g_watch = 0.85
                            _pf_g_crit  = 0.75
                        if _warn_pf_drift:
                            st.warning("PF drift: Watch must be less negative than Critical (e.g. Watch \u2212 0.01, Critical \u2212 0.03). Reset to defaults.")
                            _pf_d_watch    = float(PF_DRIFT_WATCH)
                            _pf_d_critical = float(PF_DRIFT_ACTION)

                        # Latest daily VUF and IUF — computed before build so
                        # gauges show the most recent day's value
                        _latest_vuf = None
                        _latest_iuf = None
                        _latest_pf  = None
                        _latest_ph_iuf: dict = {}
                        if _cleaned_chart is not None and not _cleaned_chart.empty:
                            try:
                                _rc_pre = _cleaned_chart.copy()
                                _rc_pre.index = pd.to_datetime(_rc_pre.index)
                                _rpv1=_rc_pre["phase_1_voltage"]; _rpv2=_rc_pre["phase_2_voltage"]; _rpv3=_rc_pre["phase_3_voltage"]
                                _rpva=(_rpv1+_rpv2+_rpv3)/3.0
                                _vuf_pre=(pd.concat([(_rpv1-_rpva).abs(),(_rpv2-_rpva).abs(),(_rpv3-_rpva).abs()],axis=1)
                                          .max(axis=1)/_rpva.replace(0,np.nan)*100.0)
                                _rpi1=_rc_pre["phase_1_current"]; _rpi2=_rc_pre["phase_2_current"]; _rpi3=_rc_pre["phase_3_current"]
                                _rpia=(_rpi1+_rpi2+_rpi3)/3.0
                                _iuf_pre=(pd.concat([(_rpi1-_rpia).abs(),(_rpi2-_rpia).abs(),(_rpi3-_rpia).abs()],axis=1)
                                          .max(axis=1)/_rpia.replace(0,np.nan)*100.0)
                                _dv_pre = _vuf_pre.resample("D").mean().dropna()
                                _di_pre = _iuf_pre.resample("D").mean().dropna()
                                if len(_dv_pre) > 0:
                                    _latest_vuf = round(float(_dv_pre.iloc[-1]), 3)
                                if len(_di_pre) > 0:
                                    _latest_iuf = round(float(_di_pre.iloc[-1]), 2)
                                # Latest daily PF
                                _pf_pre = ((_rc_pre["phase_1_active_power"] +
                                            _rc_pre["phase_2_active_power"] +
                                            _rc_pre["phase_3_active_power"]) /
                                           (_rc_pre["phase_1_voltage"] * _rc_pre["phase_1_current"] +
                                            _rc_pre["phase_2_voltage"] * _rc_pre["phase_2_current"] +
                                            _rc_pre["phase_3_voltage"] * _rc_pre["phase_3_current"])
                                           .replace(0, np.nan))
                                _pf_daily_pre = _pf_pre.resample("D").mean().dropna()
                                if len(_pf_daily_pre) > 0:
                                    _latest_pf = round(float(_pf_daily_pre.iloc[-1]), 4)
                                # Per-phase latest daily values — walk back through
                                # available days until a day with enough samples is found
                                _latest_ph_iuf: dict = {}
                                _ph_date_used: str = ""
                                for _candidate_ts in reversed(_di_pre.index.tolist()):
                                    _candidate_day = _candidate_ts.date()
                                    _day_mask = _rc_pre.index.date == _candidate_day
                                    _ld = _rc_pre[_day_mask]
                                    if _ld.empty:
                                        continue
                                    try:
                                        _ldi1=_ld["phase_1_current"]; _ldi2=_ld["phase_2_current"]; _ldi3=_ld["phase_3_current"]
                                        _lda=(_ldi1+_ldi2+_ldi3)/3.0
                                        _lds=_lda.replace(0,np.nan)
                                        _ph_vals = {
                                            1: round(float(((_ldi1-_lda).abs()/_lds*100).mean()), 2),
                                            2: round(float(((_ldi2-_lda).abs()/_lds*100).mean()), 2),
                                            3: round(float(((_ldi3-_lda).abs()/_lds*100).mean()), 2),
                                        }
                                        # Accept if all three phases have valid values
                                        if all(not np.isnan(v) for v in _ph_vals.values()):
                                            _latest_ph_iuf = _ph_vals
                                            _ph_date_used = _candidate_day.strftime("%d %b")
                                            break
                                    except Exception:
                                        continue
                            except Exception:
                                pass   # fall back to assessment mean in gauges

                        _report_figs = build_assessment_charts(
                            _chart_data_w, record,
                            cleaned_data=_cleaned_chart,
                            meta=meta,
                            phase_bands=st.session_state.get("last_phase_bands") or {},
                            iuf_gauge_watch=_iuf_g_watch,
                            iuf_gauge_critical=_iuf_g_crit,
                            vuf_gauge_watch=_vuf_g_watch,
                            vuf_gauge_critical=_vuf_g_crit,
                            pf_gauge_watch=_pf_g_watch,
                            pf_gauge_critical=_pf_g_crit,
                            vuf_gauge_value=_latest_vuf,
                            iuf_gauge_value=_latest_iuf,
                            pf_gauge_value=_latest_pf,
                            pf_drift_watch=_pf_d_watch,
                            pf_drift_critical=_pf_d_critical,
                            baseline_p_daily=st.session_state.get("baseline_p_daily") or {},
                        )
                        # ── Helper: single-signal daily run chart ──────
                        def _daily_run_chart(df, signal, colour, thresholds,
                                             y_fmt=".2f", title_suffix=""):
                            """Render a daily-average run chart for one signal.
                            signal : pd.Series with DatetimeIndex
                            thresholds : [(value, colour, label), ...]
                            """
                            _daily = signal.resample("D").mean().dropna()
                            if len(_daily) < 2:
                                st.caption(
                                    f"Daily run chart requires \u22652 days of data.")
                                return
                            _xd = _daily.index.tolist()
                            _yv = _daily.values.tolist()
                            _xt = np.arange(len(_xd))
                            _yt = np.polyval(np.polyfit(_xt, _yv, 1), _xt)
                            _fig = go.Figure()
                            _fig.add_trace(go.Scatter(
                                x=_xd, y=_yv, mode="lines+markers",
                                name=title_suffix,
                                line=dict(color=colour, width=2),
                                marker=dict(size=5, color=colour),
                                hovertemplate=f"<b>{title_suffix}</b>"
                                              " %{y:" + y_fmt + "}"
                                              "%<extra></extra>",
                            ))
                            _fig.add_trace(go.Scatter(
                                x=_xd, y=_yt.tolist(), mode="lines",
                                showlegend=False, hoverinfo="skip",
                                line=dict(color=colour, width=1.2, dash="dot"),
                            ))
                            _y_max = max(_yv) if _yv else thresholds[-1][0]
                            for _tv, _tc, _tl in thresholds:
                                _fig.add_shape(
                                    type="line", xref="paper", x0=0, x1=1,
                                    yref="y", y0=_tv, y1=_tv,
                                    line=dict(color=_tc, width=1.2, dash="dash"),
                                )
                                _fig.add_annotation(
                                    xref="paper", x=1.01, yref="y", y=_tv,
                                    text=_tl, showarrow=False, xanchor="left",
                                    font=dict(size=8, color=_tc),
                                )
                            _fig.update_layout(
                                title=dict(
                                    text=(
                                        f"Daily Run Chart \u2014 {title_suffix}"
                                        "<br><sup>Each point = daily mean"
                                        " (cleaned samples) \u2502"
                                        " Dotted = linear trend</sup>"
                                    ),
                                    font=dict(size=13),
                                ),
                                xaxis=dict(title="Date", tickformat="%d %b"),
                                yaxis=dict(
                                    title=title_suffix,
                                    range=[0, max(_y_max * 1.4,
                                                  thresholds[-1][0] * 1.6)],
                                    tickformat=y_fmt,
                                ),
                                showlegend=False,
                                hovermode="x unified",
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                margin=dict(l=55, r=130, t=65, b=50),
                                height=300, font=dict(size=11),
                            )
                            st.plotly_chart(_fig, use_container_width=True)

                        # Pre-compute series if cleaned data available
                        _rc_vuf_s = None
                        _rc_iuf_s = None
                        if _cleaned_chart is not None and not _cleaned_chart.empty:
                            try:
                                _rc = _cleaned_chart.copy()
                                _rc.index = pd.to_datetime(_rc.index)
                                _rv1=_rc["phase_1_voltage"]; _rv2=_rc["phase_2_voltage"]; _rv3=_rc["phase_3_voltage"]
                                _rva=(_rv1+_rv2+_rv3)/3.0
                                _rc_vuf_s=(pd.concat([(_rv1-_rva).abs(),(_rv2-_rva).abs(),(_rv3-_rva).abs()],axis=1)
                                           .max(axis=1)/_rva.replace(0,np.nan)*100.0)
                                _ri1=_rc["phase_1_current"]; _ri2=_rc["phase_2_current"]; _ri3=_rc["phase_3_current"]
                                _ria=(_ri1+_ri2+_ri3)/3.0
                                _rc_iuf_s=(pd.concat([(_ri1-_ria).abs(),(_ri2-_ria).abs(),(_ri3-_ria).abs()],axis=1)
                                           .max(axis=1)/_ria.replace(0,np.nan)*100.0)
                            except Exception as _rce:
                                st.warning(f"Imbalance run chart data error: {_rce}")

                        for _fig_idx, fig in enumerate(_report_figs):
                            # Title can be on layout (Scatter charts) or on the
                            # Indicator trace (gauges) — check both
                            _layout_title = getattr(
                                getattr(fig.layout, "title", None), "text", ""
                            ) or ""
                            _trace_title = ""
                            if fig.data and hasattr(fig.data[0], "title"):
                                _trace_title = getattr(
                                    getattr(fig.data[0], "title", None), "text", ""
                                ) or ""
                            _fig_title = _layout_title or _trace_title

                            # Insert VUF run chart before VUF gauge
                            if "VUF Gauge" in _fig_title and _rc_vuf_s is not None:
                                try:
                                    _daily_run_chart(
                                        _cleaned_chart, _rc_vuf_s, "#054D5F",
                                        [(VUF_WATCH,    "#E67E22", f"Watch {VUF_WATCH:.1f}%"),
                                         (VUF_CRITICAL, "#C0392B", f"Critical {VUF_CRITICAL:.1f}%")],
                                        y_fmt=".2f", title_suffix="VUF (%)",
                                    )
                                except Exception as _e:
                                    st.warning(f"VUF run chart: {_e}")

                            # Insert IUF run chart before IUF gauge
                            if "IUF Gauge" in _fig_title and _rc_iuf_s is not None:
                                try:
                                    _daily_run_chart(
                                        _cleaned_chart, _rc_iuf_s, "#C8A84B",
                                        [(IUF_WATCH,    "#E67E22", f"Watch {IUF_WATCH:.0f}%"),
                                         (IUF_CRITICAL, "#C0392B", f"Critical {IUF_CRITICAL:.0f}%")],
                                        y_fmt=".1f", title_suffix="IUF (%)",
                                    )
                                except Exception as _e:
                                    st.warning(f"IUF run chart: {_e}")

                            st.plotly_chart(fig, use_container_width=True)

                            # Below VUF gauge — baseline comparison
                            if "VUF Gauge" in _fig_title:
                                _bl_vuf = st.session_state.get("baseline_vuf_mean")
                                if _bl_vuf is not None:
                                    _delta_vuf = round(_latest_vuf - _bl_vuf, 3) \
                                                 if _latest_vuf is not None else None
                                    _bvc1, _bvc2 = st.columns([1, 2])
                                    _bvc1.metric(
                                        label="Baseline mean VUF",
                                        value=f"{_bl_vuf:.2f}\u00a0%",
                                        help="Mean VUF over the full baseline period (cleaned samples)",
                                    )
                                    if _delta_vuf is not None:
                                        _bvc2.metric(
                                            label="Change vs baseline",
                                            value=f"{_delta_vuf:+.2f}\u00a0%",
                                            delta=f"{_delta_vuf:+.2f}%",
                                            delta_color="inverse",
                                            help="Latest daily mean minus baseline mean (negative = improvement)",
                                        )
                            if "IUF Gauge" in _fig_title and _latest_ph_iuf:
                                # Per-phase: latest vs baseline
                                _bl_ph = st.session_state.get("baseline_ph_iuf") or {}
                                try:
                                    _worst_ph = max(_latest_ph_iuf, key=_latest_ph_iuf.get)
                                    _mc1, _mc2, _mc3 = st.columns(3)
                                    for _ph, _mc in [(1, _mc1), (2, _mc2), (3, _mc3)]:
                                        _pval = _latest_ph_iuf[_ph]
                                        _tier_s = (
                                            "\U0001f534 Critical" if _pval >= _iuf_g_crit else
                                            "\U0001f7e0 Watch"    if _pval >= _iuf_g_watch else
                                            "\U0001f7e2 Normal"
                                        )
                                        _bl_pval = _bl_ph.get(_ph)
                                        _ph_delta = f"{_pval - _bl_pval:+.1f}% vs baseline" \
                                                    if _bl_pval is not None else _tier_s
                                        _mc.metric(
                                            label=f"Phase\u00a0{_ph} imbalance"
                                                  + (" \u2605 Worst" if _ph == _worst_ph else ""),
                                            value=f"{_pval:.1f}\u00a0%",
                                            delta=_ph_delta,
                                            delta_color="inverse" if _bl_pval is not None else "off",
                                            help=f"Per-phase current deviation \u2014 {_ph_date_used or 'latest available day'}"
                                                 + (f" | Baseline: {_bl_pval:.1f}%" if _bl_pval is not None else ""),
                                        )
                                except Exception as _pe:
                                    st.caption(f"Per-phase breakdown unavailable: {_pe}")

                        # ── Download Report ──────────────────────────────
                        st.markdown("---")
                        _gauge_thresholds = {
                            "vuf_watch":    _vuf_g_watch,
                            "vuf_critical": _vuf_g_crit,
                            "iuf_watch":    _iuf_g_watch,
                            "iuf_critical": _iuf_g_crit,
                            "pf_watch":     _pf_g_watch,
                            "pf_critical":  _pf_g_crit,
                        }
                        _report_args = dict(
                            record=record,
                            meta=meta or {},
                            data=_chart_data_w,
                            cleaned_data=_cleaned_chart,
                            phase_bands=st.session_state.get("last_phase_bands") or {},
                            gauge_thresholds=_gauge_thresholds,
                            figs=_report_figs,
                        )
                        _fname_stem = (
                            f"assessment_report_"
                            f"{(meta or {}).get('machine_id', 'machine')}_"
                            f"{datetime.now().strftime('%Y%m%d_%H%M')}"
                        )
                        _dl_c1, _dl_c2 = st.columns(2)
                        # HTML report
                        try:
                            _html_bytes = generate_assessment_report_html(
                                **_report_args
                            ).encode("utf-8")
                            _dl_c1.download_button(
                                label="\U0001f4e5 Download Report (HTML)",
                                data=_html_bytes,
                                file_name=f"{_fname_stem}.html",
                                mime="text/html",
                                help="Interactive report with Plotly charts. Open in browser, then Print \u2192 Save as PDF.",
                            )
                        except Exception as _e:
                            _dl_c1.warning(f"HTML report failed: {_e}")
                        # PDF report
                        try:
                            with st.spinner("Building PDF\u2026"):
                                _pdf_bytes = generate_assessment_report_pdf(
                                    **_report_args
                                )
                            _dl_c2.download_button(
                                label="\U0001f4f4 Download Report (PDF)",
                                data=_pdf_bytes,
                                file_name=f"{_fname_stem}.pdf",
                                mime="application/pdf",
                                help="Static PDF with embedded charts. Ready to share or print.",
                            )
                        except Exception as _e:
                            _dl_c2.warning(f"PDF report failed: {_e}")


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
