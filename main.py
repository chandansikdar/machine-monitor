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
    compute_pf_drift_phase,
    select_pf_bands_phase,
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
                f"\U0001f7e0 Alert: drift \u2264 {PF_DRIFT_ALERT*100:.0f}%  \u2002"
                f"\U0001f534 Action: drift \u2264 {PF_DRIFT_ACTION*100:.0f}%"
            )
            # Table — convert band centres from W to kW for display
            rows = []
            for b in active_bands:
                drift = b.pf_drift if b.pf_drift is not None else 0.0
                drift_pct = (drift / b.mean_pf_baseline * 100) if b.mean_pf_baseline else 0.0
                if drift <= PF_DRIFT_ACTION:
                    status = "\U0001f534 Action"
                elif drift <= PF_DRIFT_ALERT:
                    status = "\U0001f7e0 Alert"
                elif drift <= PF_DRIFT_WATCH:
                    status = "\U0001f7e1 Watch"
                else:
                    status = "\U0001f7e2 Normal"
                rows.append({
                    "Low (kW)":          f"{b.low_kw / 1000:.1f}" if b.low_kw else "\u2014",
                    "Centre (kW)":       f"{b.centre_kw / 1000:.1f}",
                    "High (kW)":         f"{b.high_kw / 1000:.1f}" if b.high_kw else "\u2014",
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
                f"\U0001f7e0 Alert: \u2264 {PF_DRIFT_ALERT*100:.0f}%  \u2002"
                f"\U0001f534 Action: \u2264 {PF_DRIFT_ACTION*100:.0f}%"
            )
            _ph_rows = []
            for b in _ph_active:
                drift = b.pf_drift
                drift_pct = (drift / b.mean_pf_baseline * 100) if b.mean_pf_baseline else 0.0
                if drift <= PF_DRIFT_ACTION:   status = "\U0001f534 Action"
                elif drift <= PF_DRIFT_ALERT:  status = "\U0001f7e0 Alert"
                elif drift <= PF_DRIFT_WATCH:  status = "\U0001f7e1 Watch"
                else:                          status = "\U0001f7e2 Normal"
                _ph_rows.append({
                    "Low (kW)":        f"{b.low_kw / 1000:.1f}",
                    "Centre (kW)":     f"{b.centre_kw / 1000:.1f}",
                    "High (kW)":       f"{b.high_kw / 1000:.1f}",
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

    # Baseline bin report — all 100 bins with qualify/disqualify status
    _bl_bands = getattr(record, "baseline_bands", None) or []
    if _bl_bands:
        import pandas as _pd2
        with st.expander(
            f"\U0001f4cb Baseline PF bins \u2014 all {len(_bl_bands)} bins",
            expanded=False,
        ):
            st.caption(
                "All bins computed from the baseline data (bin width = 1% of operating range). "
                "Bins with \u22655 baseline samples qualify for PF drift detection. "
                "Bins below this threshold are shown for reference only."
            )
            _bl_rows = []
            for _b in _bl_bands:
                _qualifies = _b.n_baseline >= 5
                _bl_rows.append({
                    "Low (kW)":       f"{_b.low_kw / 1000:.1f}",
                    "Centre (kW)":    f"{_b.centre_kw / 1000:.1f}",
                    "High (kW)":      f"{_b.high_kw / 1000:.1f}",
                    "n baseline":     _b.n_baseline,
                    "Baseline PF":    f"{_b.mean_pf_baseline:.4f}" if _b.n_baseline > 0 else "\u2014",
                    "Qualifies":      "\u2705 Yes" if _qualifies else "\u274c No (<5 samples)",
                })
            _bl_df = _pd2.DataFrame(_bl_rows)
            st.dataframe(_bl_df, use_container_width=True, hide_index=True)
            _n_qualify = sum(1 for b in _bl_bands if b.n_baseline >= 5)
            _n_total   = len(_bl_bands)
            st.caption(
                f"{_n_qualify} of {_n_total} bins qualify (\u22655 samples). "
                f"{_n_total - _n_qualify} bins excluded from PF drift calculation."
            )


# ---------------------------------------------------------------------------
# Assessment charts
# ---------------------------------------------------------------------------

def build_assessment_charts(
    data: pd.DataFrame,
    record: AssessmentRecord,
    cleaned_data: pd.DataFrame | None = None,
    meta: dict | None = None,
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

    # P_total chart — baseline avg + 40% load precondition threshold
    p_hlines = []
    if record.zone4 and record.zone4.p_baseline_avg_kw:
        p_hlines.append((
            record.zone4.p_baseline_avg_kw, "#054D5F", "dashdot",
            f"Baseline avg {record.zone4.p_baseline_avg_kw:.1f} kW",
        ))
    # 40% load precondition threshold — derive from meta or estimate from data
    _p40_kw = None
    if meta:
        _p_shaft = float(meta.get("p_rated_shaft_kw", 0))
        _eta     = float(meta.get("eta_rated", 0.90))
        if _p_shaft > 0 and _eta > 0:
            _p40_kw = 0.40 * (_p_shaft / _eta)
        else:
            # Estimate from data 95th percentile (same logic as clean_samples)
            _p95_raw = float(raw_p_kw[raw_p_kw > 0].quantile(0.95)) if (raw_p_kw > 0).any() else 0.0
            _p_rated_est = _p95_raw / 0.95 if _p95_raw > 0 else 0.0
            _p40_kw = 0.40 * _p_rated_est if _p_rated_est > 0 else None
    if _p40_kw and _p40_kw > 0:
        p_hlines.append((
            _p40_kw, "#177E40", "dot",
            f"40% load threshold ({_p40_kw:.1f} kW)",
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
    ("last_assessment",          None),
    ("last_data",                None),
    ("last_cleaned_data",        None),
    ("last_integrity_passed_ts", None),
    ("baseline_ic_excluded",     0),
    ("last_integrity_failure_summary", {}),
    ("effective_meta",           None),   # resolved meta per §2.5 — single source of truth
    ("_ep_just_saved",           False),  # flag to show save confirmation after rerun
    ("last_phase_bands",         {}),     # {1: [BandRecord], 2: [...], 3: [...]}
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
                                _n_bl_rows = len(_bl_view_data)
                                st.caption(
                                    f"{_n_bl_rows:,} rows (integrity-passed)  |  "
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

                    # PF band histogram — bands computed at last assessment time
                    _last_rec = st.session_state.get("last_assessment")
                    _hist_bands_raw = []
                    if _last_rec and isinstance(_last_rec, dict):
                        _hist_bands_raw = _last_rec.get("bands", []) or []
                    # Fall back to stored baseline bands for backwards compat
                    if not _hist_bands_raw:
                        _hist_bands_raw = _stored_bl_dict.get("bands") or []
                    if _hist_bands_raw:
                        with st.expander(
                            f"\U0001f4ca PF band histogram ({len(_hist_bands_raw)} bands)",
                            expanded=False,
                        ):
                            if _last_rec:
                                st.caption("\u2139\ufe0f Bands computed from last assessment using current rated power.")
                            else:
                                st.info("\u2139\ufe0f Run an assessment to see bands computed with current rated power.")
                            st.caption(
                                "Each bar is one of 100 equal-width bins — bin width = 1% of "
                                "the actual operating range (P_max \u2212 P_min) in the cleaned baseline. "
                                "Outliers already removed by the cleaning pipeline. "
                                "Height = baseline samples in bin. Colour = mean baseline PF. "
                                "Only bins with \u22655 samples qualify for PF drift detection."
                            )
                            _bands_df = pd.DataFrame([
                                {
                                    "centre_kw":        round(b["centre_kw"] / 1000, 2),
                                    "low_kw":           round(b.get("low_kw",  b["centre_kw"]) / 1000, 2),
                                    "high_kw":          round(b.get("high_kw", b["centre_kw"]) / 1000, 2),
                                    "n_baseline":       b["n_baseline"],
                                    "mean_pf_baseline": round(b["mean_pf_baseline"], 4),
                                }
                                for b in _hist_bands_raw
                            ]).sort_values("centre_kw")

                            import plotly.graph_objects as _go2
                            _fig_hist = _go2.Figure()

                            # Bar chart — height = sample count, colour = PF
                            _pf_min = _bands_df["mean_pf_baseline"].min()
                            _pf_max = _bands_df["mean_pf_baseline"].max()
                            _pf_range = max(_pf_max - _pf_min, 0.01)

                            _colors = [
                                f"rgba({int(5 + 200*(1 - (pf - _pf_min)/_pf_range))}, "
                                f"{int(77 + 150*((pf - _pf_min)/_pf_range))}, "
                                f"{int(95 + 100*((pf - _pf_min)/_pf_range))}, 0.85)"
                                for pf in _bands_df["mean_pf_baseline"]
                            ]

                            _fig_hist.add_trace(_go2.Bar(
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
                                    "Band: %{customdata[1]:.1f} \u2013 %{customdata[2]:.1f} kW<br>"
                                    "Samples: %{customdata[3]}<br>"
                                    "Baseline PF: %{customdata[0]:.4f}<extra></extra>"
                                ),
                            ))

                            # Minimum sample threshold line
                            _fig_hist.add_hline(
                                y=5, line_dash="dash", line_color="#C8A84B",
                                line_width=1.5,
                                annotation_text="Min 5 samples",
                                annotation_position="top right",
                                annotation_font_size=10,
                            )

                            _fig_hist.update_layout(
                                xaxis_title="Band centre (kW)",
                                yaxis_title="Baseline samples",
                                height=300,
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                margin=dict(l=40, r=20, t=30, b=40),
                                font=dict(size=11),
                                showlegend=False,
                                bargap=0.05,
                            )
                            st.plotly_chart(_fig_hist, use_container_width=True)

                            # Table below the chart
                            st.dataframe(
                                _bands_df.rename(columns={
                                    "centre_kw":        "Centre (kW)",
                                    "low_kw":           "Low (kW)",
                                    "high_kw":          "High (kW)",
                                    "n_baseline":       "Baseline samples",
                                    "mean_pf_baseline": "Baseline PF",
                                }),
                                use_container_width=True,
                                hide_index=True,
                            )
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
                                                         raw_baseline=_raw_bl_for_assess)
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
                            # Compute per-phase PF drift bands (phase-specific bins)
                            if _raw_bl_for_assess is not None:
                                _ph_bands = {}
                                for _ph in (1, 2, 3):
                                    # Build phase-specific baseline bands
                                    _ph_bl_bands = select_pf_bands_phase(
                                        _raw_bl_for_assess, _ph
                                    )
                                    # Compute drift using phase bands
                                    _ph_bands[_ph] = compute_pf_drift_phase(
                                        _raw_reset, _ph_bl_bands,
                                        _ph, _raw_bl_for_assess
                                    )
                                st.session_state["last_phase_bands"] = _ph_bands
                            else:
                                st.session_state["last_phase_bands"] = {}
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

                            # Re-derive removed rows per step using same logic as clean_samples
                            _raw_w = scale_power_to_watts(
                                (_raw_for_dl if _raw_for_dl is not None else
                                 data.loc[
                                     (pd.Timestamp(date_range[0]) <= data.index) &
                                     (data.index <= pd.Timestamp(date_range[1]) +
                                      pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
                                 ]).reset_index(), meta.get("power_unit", "W")
                            )
                            _p_rated_e = (float(meta.get("p_rated_shaft_kw", 0)) /
                                          float(meta.get("eta_rated", 0.9)))
                            _load_min  = 0.40 * _p_rated_e * 1000.0
                            _cold_min  = 0.01 * _p_rated_e * 1000.0
                            _pt_raw    = (_raw_w["phase_1_active_power"] +
                                          _raw_w["phase_2_active_power"] +
                                          _raw_w["phase_3_active_power"])

                            # Step 1 — load precondition
                            _s1_pass  = _raw_w[_pt_raw >= _load_min].copy()
                            _s1_fail  = _raw_w[_pt_raw <  _load_min].copy()
                            _s1_fail["removed_at_step"] = "Step 1 - Load precondition (<40% rated)"

                            # Step 2 — start transient
                            _prev_below = _pt_raw.shift(1, fill_value=0.0) < _cold_min
                            _curr_above = _pt_raw >= _cold_min
                            _crossing   = _raw_w.index[_prev_below & _curr_above]
                            _transient_ts: set = set()
                            for _ci in _crossing:
                                _loc = _raw_w.index.get_loc(_ci)
                                for _off in range(2):   # COLD_START_TRANSIENT_SAMPLES = 2
                                    if _loc + _off < len(_raw_w):
                                        _transient_ts.add(
                                            str(_raw_w.iloc[_loc + _off]["timestamp"])
                                        )
                            _s1_pass_ts = _s1_pass["timestamp"].astype(str)
                            _s2_fail    = _s1_pass[_s1_pass_ts.isin(_transient_ts)].copy()
                            _s2_pass    = _s1_pass[~_s1_pass_ts.isin(_transient_ts)].copy()
                            _s2_fail["removed_at_step"] = "Step 2 - Start transient exclusion"

                            # Step 3 — user filter
                            _bm_for_dl2 = baseline_from_dict(db.get_baseline(selected_id))
                            _uf = _bm_for_dl2.user_filter_expr if _bm_for_dl2 else None
                            if _uf and len(_s2_pass) > 0:
                                try:
                                    _s3_pass = _s2_pass.query(_uf).copy()
                                    _s3_fail = _s2_pass[
                                        ~_s2_pass.index.isin(_s3_pass.index)
                                    ].copy()
                                    _s3_fail["removed_at_step"] = "Step 3 - User filter"
                                except Exception:
                                    _s3_pass = _s2_pass.copy()
                                    _s3_fail = pd.DataFrame(columns=_s2_pass.columns)
                            else:
                                _s3_pass = _s2_pass.copy()
                                _s3_fail = pd.DataFrame(columns=_s2_pass.columns)

                            # Step 4 — IQR rejection
                            _s4_fail = pd.DataFrame(columns=_s3_pass.columns)
                            if len(_s3_pass) >= 4:
                                _pt4  = (_s3_pass["phase_1_active_power"] +
                                         _s3_pass["phase_2_active_power"] +
                                         _s3_pass["phase_3_active_power"])
                                _ia4  = (_s3_pass["phase_1_current"] +
                                         _s3_pass["phase_2_current"] +
                                         _s3_pass["phase_3_current"]) / 3.0
                                _ss4  = (_s3_pass["phase_1_voltage"] * _s3_pass["phase_1_current"] +
                                         _s3_pass["phase_2_voltage"] * _s3_pass["phase_2_current"] +
                                         _s3_pass["phase_3_voltage"] * _s3_pass["phase_3_current"])
                                _pf4  = _pt4 / _ss4.replace(0, float("nan"))
                                _keep4 = pd.Series(True, index=_s3_pass.index)
                                for _sig in (_pt4, _ia4, _pf4):
                                    _q25 = _sig.quantile(0.25); _q75 = _sig.quantile(0.75)
                                    _iqr = _q75 - _q25
                                    _keep4 &= _sig.between(
                                        _q25 - 1.5 * _iqr, _q75 + 1.5 * _iqr, inclusive="both"
                                    )
                                _s4_fail = _s3_pass[~_keep4].copy()
                                _s4_fail["removed_at_step"] = "Step 4 - IQR outlier rejection"

                            # Combine all removed rows
                            _removed_all_labelled = pd.concat(
                                [_s1_fail, _s2_fail, _s3_fail, _s4_fail],
                                ignore_index=True
                            )
                            _n_removed = len(_removed_all_labelled)

                            # Two download columns
                            _dc1, _dc2 = st.columns(2)

                            with _dc1:
                                st.markdown(f"**\u2705 Cleaned ({_n_cleaned:,} rows)**")
                                _dl_cleaned = _to_dl_unit(
                                    _cleaned if "timestamp" in _cleaned.columns
                                    else _cleaned.reset_index()
                                )
                                st.download_button(
                                    label=f"\u2b07\ufe0f Download {_n_cleaned:,} cleaned rows (CSV)",
                                    data=_dl_cleaned.to_csv(index=False).encode("utf-8"),
                                    file_name=f"cleaned_{selected_id}_{date_range[0]}_to_{date_range[1]}.csv",
                                    mime="text/csv", use_container_width=True,
                                )
                                st.dataframe(_dl_cleaned.head(10), use_container_width=True,
                                             hide_index=True)

                            with _dc2:
                                st.markdown(f"**\u274c Removed ({_n_removed:,} rows)**")
                                _dl_removed = _to_dl_unit(_removed_all_labelled)
                                st.download_button(
                                    label=f"\u2b07\ufe0f Download {_n_removed:,} removed rows (CSV)",
                                    data=_dl_removed.to_csv(index=False).encode("utf-8"),
                                    file_name=f"removed_{selected_id}_{date_range[0]}_to_{date_range[1]}.csv",
                                    mime="text/csv", use_container_width=True,
                                )
                                st.dataframe(_dl_removed.head(10), use_container_width=True,
                                             hide_index=True)

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
                                           "Alert"  if _b.pf_drift <= -0.02 else
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
                                            "Alert"  if _b.pf_drift <= -0.02 else
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
                                            _bm_xl.user_filter_expr
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
                        for fig in build_assessment_charts(
                            _chart_data_w, record,
                            cleaned_data=_cleaned_chart,
                            meta=meta,
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
