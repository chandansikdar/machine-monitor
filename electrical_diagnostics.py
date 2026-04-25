"""
electrical_diagnostics.py
=========================
Electrical diagnostics module for the Symbion Machine Analytics Platform.

Implements methodology v0.6 for accumulating-degradation detection on
three-phase induction motors using per-phase RMS measurements.

Zone structure
--------------
Integrity gate  : \u00a73   measurement chain data quality precondition
Zone 1          : \u00a75   supply channel \u2014 voltage unbalance factor (VUF)
Zones 2 & 3     : \u00a76   motor-side combined detection \u2014 IUF + multi-band PF drift
Zone 4          : \u00a77   driven equipment indication \u2014 average power change

All diagnostic rules operate on cleaned running samples produced by the
shared data-cleaning procedure (\u00a74). The same procedure is applied at
baseline ingestion and at every assessment to guarantee like-for-like
comparisons.

Input contract
--------------
DataFrame columns: timestamp,
    phase_1_voltage, phase_2_voltage, phase_3_voltage,
    phase_1_current, phase_2_current, phase_3_current,
    phase_1_active_power, phase_2_active_power, phase_3_active_power
Optional column  : i_n

Power unit convention
---------------------
Per-phase active power columns (phase_1/2/3_active_power) store values in **Watts**.
This is consistent with the V \u00d7 I = VA (Watts) derivation. The metadata
carries p_rated_shaft_kw in kW; wherever code compares p_total (W) against a
kW-based limit, multiply the kW limit by 1000.
Machine metadata dict keys (required):
    v_nominal_phase     float   phase-to-neutral nominal voltage (V)
    p_rated_shaft_kw    float   nameplate shaft power (kW)
    pf_rated            float   nameplate full-load power factor
    eta_rated           float   nameplate full-load efficiency
    i_rated             float   nameplate full-load current (A)
    application_type    str     one of: compressed_air, process_pump,
                                process_chiller, fan, hvac_chiller, other
    four_wire           bool    True = four-wire with neutral
    measurement_at_panel bool   True = voltage measured at panel (not motor terminals)

Derived by this module:
    p_rated_elec_kw  =  p_rated_shaft_kw / eta_rated

Project-wide conventions
------------------------
- Unicode escapes only (no raw emoji in string literals)
- Word-boundary regex for short keyword matching
- All shutdown samples excluded via _running_mask() helper
- All thresholds at module level as named constants
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _welch_p(drift: float, std1: float, n1: int, std2: float, n2: int) -> float | None:
    """Two-tailed p-value for Welch's t-test using regularised incomplete beta.

    Returns None if the test cannot be computed (zero variance or n < 2).
    """
    if std1 <= 0 or std2 <= 0 or n1 < 2 or n2 < 2:
        return None
    try:
        var1 = std1 ** 2 / n1
        var2 = std2 ** 2 / n2
        se   = (var1 + var2) ** 0.5
        if se <= 0:
            return 1.0
        t  = abs(drift / se)
        df = (var1 + var2) ** 2 / (var1 ** 2 / (n1 - 1) + var2 ** 2 / (n2 - 1))
        x  = df / (df + t ** 2)
        # Regularised incomplete beta I(x; df/2, 0.5) via Lentz continued fraction
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        a, b_val = df / 2, 0.5
        lb   = math.lgamma(a) + math.lgamma(b_val) - math.lgamma(a + b_val)
        front = math.exp(math.log(x) * a + math.log(1 - x) * b_val - lb) / a
        f = 1.0; c = 1.0
        d = 1.0 - (a + b_val) * x / (a + 1)
        if abs(d) < 1e-30: d = 1e-30
        d = 1.0 / d; f = d
        for m in range(1, 200):
            m2  = 2 * m
            nu  = m * (b_val - m) * x / ((a + m2 - 1) * (a + m2))
            d   = 1.0 + nu * d; c = 1.0 + nu / c
            if abs(d) < 1e-30: d = 1e-30
            if abs(c) < 1e-30: c = 1e-30
            d = 1.0 / d; f *= c * d
            nu  = -(a + m) * (a + b_val + m) * x / ((a + m2) * (a + m2 + 1))
            d   = 1.0 + nu * d; c = 1.0 + nu / c
            if abs(d) < 1e-30: d = 1e-30
            if abs(c) < 1e-30: c = 1e-30
            d = 1.0 / d; delta = c * d; f *= delta
            if abs(delta - 1.0) < 1e-10:
                break
        return max(0.0, min(1.0, float(front * f)))
    except Exception:
        return None
import re
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level threshold constants (\u00a710 of methodology)
# ---------------------------------------------------------------------------

# Zone 1 \u2014 supply channel
VUF_WATCH: float = 1.0          # % sustained VUF \u2192 watch
VUF_CRITICAL: float = 2.0       # % sustained VUF \u2192 critical

# Zone 2/3 signal A \u2014 current unbalance
IUF_WATCH: float = 5.0          # % mean IUF \u2192 imbalance present / watch
IUF_CRITICAL: float = 10.0      # % mean IUF \u2192 critical

# Zone 2/3 signal B \u2014 PF drift
PF_DRIFT_WATCH: float = -0.01   # absolute downward drift \u2192 watch
PF_DRIFT_ALERT: float = -0.02   # \u2192 alert
PF_DRIFT_ACTION: float = -0.03  # \u2192 action

# Data cleaning
LOAD_PRECONDITION_FRACTION: float = 0.20  # fraction of p_rated_elec minimum — v0.8
_DIAG_VERSION = "v0.8-no-iqr"  # bump this to force Streamlit module reload
IQR_MULTIPLIER: float = 1.5               # standard Tukey fence

# Multi-band PF comparison
PF_BAND_TARGET_BINS: int   = 100             # bins = range / (1% of range) = 100
PF_BAND_MIN_SAMPLES: int   = 5               # minimum samples per band (baseline and assessment)
PF_BAND_MIN_COUNT: int = 3               # minimum qualifying bands

# Cleaning sufficiency
CLEANING_MIN_SAMPLES: int = 50

# Zone 4 driven equipment
ZONE4_SIGNIFICANCE_PCT: float = 3.0

# Baseline PF check \u2014 most-loaded band must be at or above this fraction of rated
BASELINE_PF_CHECK_LOAD_GATE: float = 0.95

# HVAC chiller application_type value (suppresses Zone 4)
HVAC_CHILLER_APP_TYPE: str = "hvac_chiller"

# Running mask: sample is "running" when P_total exceeds this fraction of rated
RUNNING_THRESHOLD_FRACTION: float = 0.05

# Start transient exclusion: cold start defined when P_total crosses from
# below this fraction of P_rated_elec to above it (§4.1 Step 2)
COLD_START_THRESHOLD_FRACTION: float = 0.01   # 1 % of rated electrical input
COLD_START_TRANSIENT_SAMPLES: int = 2          # samples to drop after cold start


# ---------------------------------------------------------------------------
# Data classes for structured outputs
# ---------------------------------------------------------------------------

@dataclass
class IntegrityResult:
    passed: bool
    failing_check: str | None = None
    reason: str | None = None


@dataclass
class CleaningReport:
    """Sample counts at each of the four cleaning steps (§4.1).

    Step 1 – Load precondition  : n_after_load_precondition
    Step 2 – Start transient    : n_after_start_transient
    Step 3 – User filter        : n_after_user_filter
    Step 4 – IQR rejection      : n_after_iqr  (= n_cleaned)
    """
    n_raw: int = 0
    n_after_load_precondition: int = 0   # Step 1
    n_after_start_transient: int = 0     # Step 2
    n_after_user_filter: int = 0         # Step 3
    n_after_iqr: int = 0                 # Step 4

    @property
    def n_cleaned(self) -> int:
        return self.n_after_iqr

    @property
    def fraction_retained(self) -> float:
        return self.n_cleaned / self.n_raw if self.n_raw > 0 else 0.0


@dataclass
class BandRecord:
    centre_kw: float
    low_kw: float
    high_kw: float
    n_baseline: int
    mean_pf_baseline: float
    std_pf_baseline: float = 0.0           # std dev of PF in baseline — for Welch's t-test
    n_recent: int = 0
    mean_pf_recent: float | None = None
    std_pf_recent: float | None = None     # std dev of PF in recent period
    pf_drift: float | None = None
    suppressed: bool = False
    suppression_reason: str | None = None
    p_value: float | None = None           # Welch's t-test p-value
    drift_significant: bool | None = None  # True if p < 0.05


@dataclass
class BaselineState:
    iuf_mean: float | None = None
    iuf_message: str | None = None
    iuf_tier: str | None = None           # "watch" | "critical" | None
    pf_check_performed: bool = False
    pf_check_band_centre_pct: float | None = None
    pf_drift_vs_nameplate: float | None = None
    pf_tier: str | None = None            # "watch" | "alert" | "action" | None
    pf_message: str | None = None


@dataclass
class BaselineMetadata:
    """All reference data produced at baseline ingestion.
    Stored and reloaded for every subsequent assessment."""
    timestamp_start: Any = None
    timestamp_end: Any = None
    p_baseline_avg_kw: float | None = None
    bands: list[BandRecord] = field(default_factory=list)
    n_qualifying_bands: int = 0
    cleaning_report: CleaningReport | None = None
    baseline_state: BaselineState | None = None
    user_filter_expr: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class SupplyAlarm:
    vuf_pct: float
    tier: str | None = None   # None | "watch" | "critical"

    @property
    def fired(self) -> bool:
        return self.tier is not None


@dataclass
class MotorSideResult:
    iuf_mean_pct: float
    iuf_tier: str | None             # None | "watch" | "critical"
    outlier_phase: str | None        # "1" | "2" | "3" or None
    outlier_direction: str | None    # "low" | "high" or None
    pf_drift_aggregated: float | None
    pf_drift_tier: str | None        # None | "watch" | "alert" | "action"
    pf_drift_suppressed: bool
    pf_drift_suppression_reason: str | None
    bands: list[BandRecord] = field(default_factory=list)
    cell: int = 1                    # 1=Healthy 2=Motor 3=Panel-to-motor 4=Both
    cell_label: str = "Healthy"
    alarm_tier: str | None = None    # overall tier for this zone
    inspection_refs: list[str] = field(default_factory=list)
    message: str = ""


@dataclass
class Zone4Result:
    suppressed: bool = False
    suppression_reason: str | None = None
    p_baseline_avg_kw: float | None = None
    p_recent_avg_kw: float | None = None
    delta_p_pct: float | None = None
    direction: str | None = None     # "positive" | "negative" | None
    finding_state: str | None = None # None | "positive" | "negative" | "suppressed"
    message: str = ""


@dataclass
class AssessmentRecord:
    """Structured output for one user-initiated assessment (§8)."""
    integrity_status: str = "not_run"      # "passed" | "failed" | "suppressed"
    suppressed: bool = False
    suppression_reason: str | None = None
    cleaning_report: CleaningReport | None = None
    supply_alarm: SupplyAlarm | None = None
    motor_side: MotorSideResult | None = None
    zone4: Zone4Result | None = None
    messages: list[str] = field(default_factory=list)
    baseline_bands: list[BandRecord] = field(default_factory=list)  # all bins from baseline


# ---------------------------------------------------------------------------
# Running mask helper
# ---------------------------------------------------------------------------

def _running_mask(df: pd.DataFrame, p_rated_elec_kw: float) -> pd.Series:
    """Return a boolean mask of running (non-shutdown) samples.

    A sample is considered running when P_total exceeds
    RUNNING_THRESHOLD_FRACTION * p_rated_elec_kw.  This is intentionally
    generous \u2014 it flags startup/shutdown transients and true shutdowns only.
    All diagnostic rules operate exclusively on running samples.
    """
    p_total = df["phase_1_active_power"] + df["phase_2_active_power"] + df["phase_3_active_power"]
    threshold_w = RUNNING_THRESHOLD_FRACTION * p_rated_elec_kw * 1000.0
    return p_total > threshold_w


# ---------------------------------------------------------------------------
# \u00a73  Measurement integrity gate
# ---------------------------------------------------------------------------

def integrity_gate(row: pd.Series, meta: dict) -> IntegrityResult:
    """Per-sample integrity check (\u00a73.1).

    Returns IntegrityResult(passed=True) if all five checks pass.
    First failing check is reported; subsequent checks are not evaluated.

    Parameters
    ----------
    row  : one row of the measurement DataFrame (named Series)
    meta : machine metadata dict
    """
    v_nom = float(meta["v_nominal_phase"])
    i_rated = float(meta["i_rated"])
    p1, p2, p3 = float(row["phase_1_active_power"]), float(row["phase_2_active_power"]), float(row["phase_3_active_power"])
    v1, v2, v3 = float(row["phase_1_voltage"]), float(row["phase_2_voltage"]), float(row["phase_3_voltage"])
    i1, i2, i3 = float(row["phase_1_current"]), float(row["phase_2_current"]), float(row["phase_3_current"])
    p_total = p1 + p2 + p3
    # meta["p_rated_shaft_kw"] guaranteed non-zero by resolve_effective_meta (§2.5)
    _p_shaft_ig = float(meta.get("p_rated_shaft_kw", 0))
    _eta_ig     = float(meta.get("eta_rated", 0.90))
    p_rated_elec = (_p_shaft_ig / _eta_ig) if (_p_shaft_ig > 0 and _eta_ig > 0) else 1.0
    running = p_total > RUNNING_THRESHOLD_FRACTION * p_rated_elec * 1000.0

    # --- Check 1: Voltage plausibility ---
    v_lo = 0.85 * v_nom
    v_hi = 1.15 * v_nom
    v_max_abs = 1.5 * v_nom
    for ph, v, i_x in zip(("1", "2", "3"), (v1, v2, v3), (i1, i2, i3)):
        # V=0 with I=0 is a powered-down machine, not a sensor fault — skip
        if v == 0.0 and i_x == 0.0:
            continue
        if v < 50.0:
            return IntegrityResult(
                passed=False,
                failing_check="check_1_voltage_plausibility",
                reason=(
                    f"Phase {ph} voltage {v:.1f} V below 50 V floor "
                    f"(channel failure or short suspected)"
                )
            )
        if v > v_max_abs:
            return IntegrityResult(
                passed=False,
                failing_check="check_1_voltage_plausibility",
                reason=(
                    f"Phase {ph} voltage {v:.1f} V above 1.5 \u00d7 V_nominal "
                    f"({v_max_abs:.1f} V) \u2014 reference lead may be landed on phase conductor"
                )
            )
        if not (v_lo <= v <= v_hi):
            return IntegrityResult(
                passed=False,
                failing_check="check_1_voltage_plausibility",
                reason=(
                    f"Phase {ph} voltage {v:.1f} V outside "
                    f"\u00b115% nominal band [{v_lo:.1f}, {v_hi:.1f}] V"
                )
            )

    # --- Check 2: Current plausibility (running samples only) ---
    if running:
        i_lo = 0.005 * i_rated
        i_hi = 1.5 * i_rated
        for ph, i_x in zip(("1", "2", "3"), (i1, i2, i3)):
            if not (i_lo <= i_x <= i_hi):
                return IntegrityResult(
                    passed=False,
                    failing_check="check_2_current_plausibility",
                    reason=(
                        f"Phase {ph} current {i_x:.2f} A outside "
                        f"[{i_lo:.3f}, {i_hi:.1f}] A "
                        f"(CT failure or scaling error suspected)"
                    )
                )
        # 5% cross-phase check
        currents = {"1": i1, "2": i2, "3": i3}
        for ph, i_x in currents.items():
            others = [v for k, v in currents.items() if k != ph]
            avg_others = sum(others) / 2.0
            if avg_others > 0 and i_x < 0.05 * avg_others:
                return IntegrityResult(
                    passed=False,
                    failing_check="check_2_current_plausibility",
                    reason=(
                        f"Phase {ph} current {i_x:.2f} A is less than 5% of average "
                        f"of other two phases ({avg_others:.2f} A) \u2014 "
                        f"CT may be clamped to wrong conductor"
                    )
                )

    # --- Check 3: Power sign and sum coherence ---
    mag_sum = abs(p1) + abs(p2) + abs(p3)
    signed_sum = abs(p1 + p2 + p3)
    if mag_sum > 5.0 and signed_sum < 0.30 * mag_sum:
        return IntegrityResult(
            passed=False,
            failing_check="check_3_power_sign_coherence",
            reason=(
                f"V-I pairing error suspected: |P_sum|/mag_sum = "
                f"{signed_sum / mag_sum:.2f} < 0.30. "
                f"Verify CT channel assignment."
            )
        )

    # --- Check 4: Per-phase PF plausibility (running only) ---
    if running:
        for ph, p_x, v_x, i_x in zip(
            ("1", "2", "3"),
            (p1, p2, p3),
            (v1, v2, v3),
            (i1, i2, i3),
        ):
            s_x = v_x * i_x
            if s_x > 0:
                pf_x = p_x / s_x
                if not (0.30 <= pf_x <= 1.00):
                    return IntegrityResult(
                        passed=False,
                        failing_check="check_4_pf_plausibility",
                        reason=(
                            f"Phase {ph} PF {pf_x:.3f} outside [0.30, 1.00] "
                            f"(channel pairing or polarity error suspected)"
                        )
                    )

    # --- Check 5: Per-phase PF spread consistency (running only) ---
    if running:
        pfs = []
        for p_x, v_x, i_x in zip(
            (p1, p2, p3), (v1, v2, v3), (i1, i2, i3)
        ):
            s_x = v_x * i_x
            if s_x > 0:
                pfs.append(p_x / s_x)
        if len(pfs) == 3:
            spread = max(pfs) - min(pfs)
            if spread > 0.15:
                return IntegrityResult(
                    passed=False,
                    failing_check="check_5_pf_consistency",
                    reason=(
                        f"Per-phase PF spread {spread:.3f} exceeds 0.15 limit "
                        f"(channel pairing or polarity error suspected)"
                    )
                )

    return IntegrityResult(passed=True)


# ---------------------------------------------------------------------------
# \u00a74  Data cleaning procedure
# ---------------------------------------------------------------------------

def clean_samples(
    raw: pd.DataFrame,
    meta: dict,
    user_filter: str | None = None,
    load_precondition_fraction: float | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Four-step data cleaning procedure (§4.1).

    Integrity checks (§3.1) are applied at data upload in the Data tab and
    are NOT repeated here.  By the time data reaches this function all samples
    have already been screened for wiring errors, CT faults, and physically
    impossible values.

    Step order
    ----------
    1. Load precondition (≥ load_precondition_fraction × P_rated_elec)
       Removes shutdown / stopped samples AND low-load samples where CT class
       tolerance errors become significant relative to the small active current
       component, producing apparent IUF on a healthy motor.
       Defaults to LOAD_PRECONDITION_FRACTION (20 %).

    2. Start transient exclusion
       Removes the first COLD_START_TRANSIENT_SAMPLES (2) samples immediately
       following any cold start from shutdown.  A cold start is identified when
       P_total in sample n is below COLD_START_THRESHOLD_FRACTION (1 %) of
       P_rated_elec and P_total in sample n+1 is at or above that threshold.
       The removal is unconditional — no check on the peak value reached is
       required.  Only cold starts from below 1 % are excluded; normal
       load/unload cycling (~15 % floor) does not trigger this step.

    3. User operating-condition filter (optional)
       If a pandas query string was stored at baseline ingestion it is applied
       here identically at assessment time.

    Parameters
    ----------
    raw                        : raw measurement DataFrame
    meta                       : machine metadata dict
    user_filter                : optional pandas query string applied at step 3
    load_precondition_fraction : override for the minimum load fraction (0–1).
                                 Defaults to LOAD_PRECONDITION_FRACTION (0.20).
    """
    report = CleaningReport(n_raw=len(raw))
    _load_frac = load_precondition_fraction if load_precondition_fraction is not None \
                 else LOAD_PRECONDITION_FRACTION
    # meta["p_rated_shaft_kw"] is guaranteed non-zero by resolve_effective_meta (§2.5)
    # which runs once after data ingestion and fills in data-derived estimates.
    p_shaft = float(meta.get("p_rated_shaft_kw", 0))
    eta     = float(meta.get("eta_rated", 0.90))
    if p_shaft > 0 and eta > 0:
        p_rated_elec = p_shaft / eta
    else:
        # Fallback only if called without resolved meta (e.g. unit tests)
        _pt_fb = (raw["phase_1_active_power"] + raw["phase_2_active_power"]
                  + raw["phase_3_active_power"])
        _p95fb = float(_pt_fb[_pt_fb > 0].quantile(0.95)) / 1000.0 if (_pt_fb > 0).any() else 1.0
        p_rated_elec = _p95fb / 0.95
    load_min_w   = _load_frac * p_rated_elec * 1000.0
    cold_min_w   = COLD_START_THRESHOLD_FRACTION * p_rated_elec * 1000.0

    # ── Step 1: Load precondition (≥ load_precondition_fraction of rated electrical input) ──
    df = raw.copy()
    if len(df) > 0:
        p_total = (df["phase_1_active_power"] + df["phase_2_active_power"]
                   + df["phase_3_active_power"])
        df = df[p_total >= load_min_w].copy()
    report.n_after_load_precondition = len(df)

    # ── Step 2: Start transient exclusion ───────────────────────────────────
    # Detect cold starts on the ORIGINAL (pre-step-1) data so we can identify
    # the crossing point even though those near-zero rows were already removed.
    if len(df) > 0 and "timestamp" in raw.columns:
        raw_p = (raw["phase_1_active_power"] + raw["phase_2_active_power"]
                 + raw["phase_3_active_power"])
        # Boolean: was previous sample below cold-start threshold?
        prev_below = raw_p.shift(1, fill_value=0.0) < cold_min_w
        # Boolean: current sample is at or above cold-start threshold?
        curr_above = raw_p >= cold_min_w
        # Crossing rows (first sample after cold start) — index in raw
        crossing_idx = raw.index[prev_below & curr_above]

        # Build set of timestamps to exclude: crossing + next N-1 rows in raw
        transient_ts: set = set()
        for ci in crossing_idx:
            loc = raw.index.get_loc(ci)
            for offset in range(COLD_START_TRANSIENT_SAMPLES):
                if loc + offset < len(raw):
                    transient_ts.add(raw.iloc[loc + offset]["timestamp"])

        if transient_ts:
            df = df[~df["timestamp"].isin(transient_ts)].copy()

    report.n_after_start_transient = len(df)

    # ── Step 3: User operating-condition filter ──────────────────────────────
    if user_filter and len(df) > 0:
        try:
            df = df.query(user_filter).copy()
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"User filter \u2018{user_filter}\u2019 could not be applied: {exc}. "
                f"Filter step skipped.",
                UserWarning,
                stacklevel=2,
            )
    report.n_after_user_filter = len(df)

    report.n_after_iqr = len(df)   # no IQR step — equals n_after_user_filter

    return df, report


# ---------------------------------------------------------------------------
# \u00a75  Zone 1 \u2014 Supply channel: VUF
# ---------------------------------------------------------------------------

def compute_vuf(df: pd.DataFrame) -> float:
    """Return mean VUF (%) over the supplied samples.

    VUF per sample = max absolute phase deviation from V_avg / V_avg * 100.
    """
    v1 = df["phase_1_voltage"].values
    v2 = df["phase_2_voltage"].values
    v3 = df["phase_3_voltage"].values
    v_avg = (v1 + v2 + v3) / 3.0
    vuf_per_sample = (
        np.maximum(
            np.maximum(np.abs(v1 - v_avg), np.abs(v2 - v_avg)),
            np.abs(v3 - v_avg),
        )
        / v_avg
        * 100.0
    )
    return float(np.mean(vuf_per_sample))


def classify_vuf(vuf_pct: float) -> SupplyAlarm:
    """Return a SupplyAlarm record for the given VUF value."""
    if vuf_pct > VUF_CRITICAL:
        return SupplyAlarm(vuf_pct=vuf_pct, tier="critical")
    elif vuf_pct > VUF_WATCH:
        return SupplyAlarm(vuf_pct=vuf_pct, tier="watch")
    return SupplyAlarm(vuf_pct=vuf_pct, tier=None)


# ---------------------------------------------------------------------------
# \u00a76  Zones 2 & 3 \u2014 Motor-side combined detection
# ---------------------------------------------------------------------------

# --- Signal A: IUF (\u00a76.2) ---

def compute_iuf(cleaned: pd.DataFrame) -> tuple[float, str | None, str | None]:
    """Compute mean IUF over cleaned samples.

    Returns
    -------
    iuf_mean_pct    : mean IUF across all cleaned samples (%)
    outlier_phase   : "1" | "2" | "3" (phase with largest mean deviation)
    outlier_direction : "low" | "high"
    """
    i1 = cleaned["phase_1_current"].values
    i2 = cleaned["phase_2_current"].values
    i3 = cleaned["phase_3_current"].values
    i_avg = (i1 + i2 + i3) / 3.0

    dev_1 = i1 - i_avg
    dev_2 = i2 - i_avg
    dev_3 = i3 - i_avg

    iuf_per_sample = (
        np.maximum(np.maximum(np.abs(dev_1), np.abs(dev_2)), np.abs(dev_3))
        / i_avg
        * 100.0
    )
    iuf_mean = float(np.mean(iuf_per_sample))

    # Identify outlier phase as the one with largest mean absolute deviation
    mean_dev_1 = float(np.mean(np.abs(dev_1)))
    mean_dev_2 = float(np.mean(np.abs(dev_2)))
    mean_dev_3 = float(np.mean(np.abs(dev_3)))

    if iuf_mean < IUF_WATCH:
        return iuf_mean, None, None

    max_dev = max(mean_dev_1, mean_dev_2, mean_dev_3)
    if max_dev == mean_dev_1:
        ph = "1"
        direction = "low" if float(np.mean(dev_1)) < 0 else "high"
    elif max_dev == mean_dev_2:
        ph = "2"
        direction = "low" if float(np.mean(dev_2)) < 0 else "high"
    else:
        ph = "3"
        direction = "low" if float(np.mean(dev_3)) < 0 else "high"

    return iuf_mean, ph, direction


def _iuf_tier(iuf_pct: float) -> str | None:
    if iuf_pct >= IUF_CRITICAL:
        return "critical"
    if iuf_pct >= IUF_WATCH:
        return "watch"
    return None


# --- Signal B: Multi-band PF drift (\u00a76.3) ---

def _pf_machine_series(df: pd.DataFrame) -> pd.Series:
    p_total = (df["phase_1_active_power"] + df["phase_2_active_power"]
               + df["phase_3_active_power"])
    s_sum = (df["phase_1_voltage"] * df["phase_1_current"]
             + df["phase_2_voltage"] * df["phase_2_current"]
             + df["phase_3_voltage"] * df["phase_3_current"])
    return p_total / s_sum.replace(0, np.nan)


def _pf_phase_series(df: pd.DataFrame, phase: int) -> pd.Series:
    """Per-phase PF = P_x / (V_x × I_x)."""
    p = df[f"phase_{phase}_active_power"]
    s = (df[f"phase_{phase}_voltage"] * df[f"phase_{phase}_current"]).replace(0, np.nan)
    return p / s


def _p_phase_series(df: pd.DataFrame, phase: int) -> pd.Series:
    """Per-phase active power."""
    return df[f"phase_{phase}_active_power"]


def _p_total_series(df: pd.DataFrame) -> pd.Series:
    return df["phase_1_active_power"] + df["phase_2_active_power"] + df["phase_3_active_power"]


def select_pf_bands(cleaned_baseline: pd.DataFrame,
                    p_rated_elec_kw: float = 0.0,
                    min_samples: int | None = None) -> list[BandRecord]:
    """Build multi-band structure from cleaned baseline (§6.3.2).

    Bin width = 2% of the actual operating range (P_max − P_min) of the
    cleaned baseline data. This gives 50 equal bins across the full range.
    Since clean_samples has already removed outliers via IQR rejection,
    min/max are sensible boundaries with no further clipping needed.

    min_samples : override PF_BAND_MIN_SAMPLES. Pass 0 to return all bins
                  regardless of count (used for baseline reporting).
    p_rated_elec_kw is retained for backward-compatibility but ignored.
    """
    p_total    = _p_total_series(cleaned_baseline)
    pf_machine = _pf_machine_series(cleaned_baseline)

    _min = min_samples if min_samples is not None else PF_BAND_MIN_SAMPLES

    if len(p_total) < PF_BAND_MIN_SAMPLES:
        return []

    p_min   = float(p_total.min())
    p_max   = float(p_total.max())
    p_range = max(p_max - p_min, 1.0)

    # bin_width = 1% of actual operating range → always 100 bins
    bin_width = 0.01 * p_range
    edges = np.arange(p_min, p_max + bin_width, bin_width)
    if len(edges) < 2:
        return []

    bands: list[BandRecord] = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        mask = (p_total >= lo) & (p_total < hi)
        n = int(mask.sum())
        if n >= _min:
            mean_pf = float(pf_machine[mask].mean()) if n > 0 else 0.0
            std_pf  = float(pf_machine[mask].std(ddof=1)) if n > 1 else 0.0
            centre  = (lo + hi) / 2.0
            bands.append(BandRecord(
                centre_kw=round(centre, 3),
                low_kw=round(lo, 3),
                high_kw=round(hi, 3),
                n_baseline=n,
                mean_pf_baseline=round(mean_pf, 5) if n > 0 else 0.0,
                std_pf_baseline=round(std_pf, 6),
            ))

    return bands


def select_pf_bands_phase(
    cleaned_baseline: pd.DataFrame,
    phase: int,
    min_samples: int | None = None,
    cleaned_recent: pd.DataFrame | None = None,
) -> list[BandRecord]:
    """Build per-phase band structure from cleaned baseline.

    Bin boundaries span the union of baseline AND assessment operating ranges
    so no assessment samples fall outside the bins. Bin width = 1% of the
    combined range → 100 bins. Only samples with positive phase power used.
    PF = P_x / (V_x × I_x). Fully independent of other phases.
    """
    p_phase  = _p_phase_series(cleaned_baseline, phase)
    pf_phase = _pf_phase_series(cleaned_baseline, phase)
    _min     = min_samples if min_samples is not None else PF_BAND_MIN_SAMPLES

    # Keep only samples with positive phase power
    _valid = p_phase > 0
    p_phase  = p_phase[_valid]
    pf_phase = pf_phase[_valid]

    if len(p_phase) < PF_BAND_MIN_SAMPLES:
        return []

    p_min = float(p_phase.min())
    p_max = float(p_phase.max())

    # Extend range to cover assessment data if provided
    if cleaned_recent is not None and len(cleaned_recent) > 0:
        _p_rec = _p_phase_series(cleaned_recent, phase)
        _p_rec = _p_rec[_p_rec > 0]
        if len(_p_rec) > 0:
            p_min = min(p_min, float(_p_rec.min()))
            p_max = max(p_max, float(_p_rec.max()))

    if len(p_phase) < PF_BAND_MIN_SAMPLES:
        return []

    p_min   = float(p_phase.min())
    p_max   = float(p_phase.max())
    p_range = max(p_max - p_min, 1.0)
    bin_width = 0.01 * p_range   # 1% of phase operating range → 100 bins
    edges = np.arange(p_min, p_max + bin_width, bin_width)
    if len(edges) < 2:
        return []

    bands: list[BandRecord] = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        mask = (p_phase >= lo) & (p_phase < hi)
        n = int(mask.sum())
        if n >= _min:
            pf_vals  = pf_phase[mask].dropna()
            mean_pf  = float(pf_vals.mean()) if len(pf_vals) > 0 else 0.0
            std_pf   = float(pf_vals.std(ddof=1)) if len(pf_vals) > 1 else 0.0
            centre   = (lo + hi) / 2.0
            bands.append(BandRecord(
                centre_kw=round(centre, 3),
                low_kw=round(lo, 3),
                high_kw=round(hi, 3),
                n_baseline=n,
                mean_pf_baseline=round(mean_pf, 5),
                std_pf_baseline=round(std_pf, 6),
            ))

    return bands


def compute_pf_drift(
    cleaned_recent: pd.DataFrame,
    stored_bands: list[BandRecord],
    cleaned_baseline: pd.DataFrame | None = None,
) -> tuple[float | None, list[BandRecord], bool, str | None]:
    """Compute per-band PF drift (§6.3.3).

    cleaned_baseline : if provided, baseline std is recomputed from raw data
                       rather than read from stored BandRecord (which may be
                       zero if the baseline was ingested before the std field
                       was added).
    """
    if not stored_bands:
        return None, stored_bands, True, "No baseline bands available"

    p_total    = _p_total_series(cleaned_recent)
    pf_machine = _pf_machine_series(cleaned_recent)

    # Precompute baseline PF series if raw baseline provided
    _bl_p_total  = _p_total_series(cleaned_baseline)    if cleaned_baseline is not None else None
    _bl_pf_mach  = _pf_machine_series(cleaned_baseline) if cleaned_baseline is not None else None

    updated: list[BandRecord] = []
    active_weights: list[float] = []
    active_drifts: list[float] = []

    for band in stored_bands:
        mask = (p_total >= band.low_kw) & (p_total < band.high_kw)
        n_recent = int(mask.sum())

        # Recompute baseline std from raw data if available (avoids stale stored values)
        if _bl_p_total is not None and _bl_pf_mach is not None:
            bl_mask = (_bl_p_total >= band.low_kw) & (_bl_p_total < band.high_kw)
            _bl_n = int(bl_mask.sum())
            _std_bl = float(_bl_pf_mach[bl_mask].std(ddof=1)) if _bl_n > 1 else 0.0
        else:
            _std_bl = band.std_pf_baseline

        b = BandRecord(
            centre_kw=band.centre_kw,
            low_kw=band.low_kw,
            high_kw=band.high_kw,
            n_baseline=band.n_baseline,
            mean_pf_baseline=band.mean_pf_baseline,
            std_pf_baseline=round(_std_bl, 6),
        )
        b.n_recent = n_recent
        if n_recent < PF_BAND_MIN_SAMPLES:
            b.suppressed = True
            b.suppression_reason = (
                f"Only {n_recent} recent samples (minimum {PF_BAND_MIN_SAMPLES})"
            )
        else:
            pf_recent_vals = pf_machine[mask]
            b.mean_pf_recent = round(float(pf_recent_vals.mean()), 5)
            b.std_pf_recent  = round(float(pf_recent_vals.std(ddof=1)) if n_recent > 1 else 0.0, 6)
            b.pf_drift       = round(b.mean_pf_recent - band.mean_pf_baseline, 5)
            active_drifts.append(b.pf_drift)
            active_weights.append(float(n_recent))

            # Welch's t-test using module-level _welch_p
            _p = _welch_p(b.pf_drift, _std_bl, band.n_baseline,
                          b.std_pf_recent, n_recent)
            if _p is not None:
                b.p_value = round(_p, 4)
                b.drift_significant = _p < 0.05
            else:
                b.p_value = None
                b.drift_significant = None
        updated.append(b)

    if len(active_drifts) < PF_BAND_MIN_COUNT:
        reason = (
            f"Only {len(active_drifts)} bands have sufficient recent samples "
            f"(minimum {PF_BAND_MIN_COUNT} required)"
        )
        return None, updated, True, reason

    total_weight = sum(active_weights)
    aggregated = sum(d * w for d, w in zip(active_drifts, active_weights)) / total_weight
    return round(aggregated, 5), updated, False, None


def compute_pf_drift_phase(
    cleaned_recent: pd.DataFrame,
    stored_bands: list[BandRecord],
    phase: int,
    cleaned_baseline: pd.DataFrame | None = None,
) -> list[BandRecord]:
    """Compute per-band PF drift for a single phase using phase-specific bins.

    Bins are defined by phase power P_x (not P_total) — the stored_bands
    passed here should be the phase-specific bands from select_pf_bands_phase.
    PF = P_x / (V_x × I_x). Welch's t-test per band.
    """
    if not stored_bands:
        return []

    p_phase  = _p_phase_series(cleaned_recent, phase)
    pf_phase = _pf_phase_series(cleaned_recent, phase)

    # Filter to positive phase power only — same as select_pf_bands_phase
    _valid_r  = p_phase > 0
    p_phase   = p_phase[_valid_r]
    pf_phase  = pf_phase[_valid_r]

    if cleaned_baseline is not None:
        _bl_p_raw  = _p_phase_series(cleaned_baseline, phase)
        _bl_pf_raw = _pf_phase_series(cleaned_baseline, phase)
        _valid_bl  = _bl_p_raw > 0
        _bl_p      = _bl_p_raw[_valid_bl]
        _bl_pf     = _bl_pf_raw[_valid_bl]
    else:
        _bl_p = _bl_pf = None

    updated: list[BandRecord] = []

    for band in stored_bands:
        mask     = (p_phase >= band.low_kw) & (p_phase < band.high_kw)
        n_recent = int(mask.sum())

        # Baseline std from raw data
        if _bl_p is not None and _bl_pf is not None:
            bl_mask  = (_bl_p >= band.low_kw) & (_bl_p < band.high_kw)
            _bl_n    = int(bl_mask.sum())
            bl_vals  = _bl_pf[bl_mask].dropna()
            _bl_mean = float(bl_vals.mean()) if len(bl_vals) > 0 else 0.0
            _bl_std  = float(bl_vals.std(ddof=1)) if len(bl_vals) > 1 else 0.0
        else:
            _bl_n, _bl_mean, _bl_std = band.n_baseline, band.mean_pf_baseline, band.std_pf_baseline

        b = BandRecord(
            centre_kw=band.centre_kw,
            low_kw=band.low_kw,
            high_kw=band.high_kw,
            n_baseline=_bl_n,
            mean_pf_baseline=round(_bl_mean, 5),
            std_pf_baseline=round(_bl_std, 6),
        )
        b.n_recent = n_recent

        if n_recent < PF_BAND_MIN_SAMPLES:
            b.suppressed = True
            b.suppression_reason = f"Only {n_recent} recent samples"
        else:
            rc_vals          = pf_phase[mask].dropna()
            b.mean_pf_recent = round(float(rc_vals.mean()), 5) if len(rc_vals) > 0 else None
            b.std_pf_recent  = round(float(rc_vals.std(ddof=1)), 6) if len(rc_vals) > 1 else 0.0
            b.pf_drift       = round(b.mean_pf_recent - _bl_mean, 5) if b.mean_pf_recent is not None else None

            # Welch's t-test using module-level _welch_p
            if b.pf_drift is not None:
                _p = _welch_p(b.pf_drift, _bl_std, _bl_n,
                              b.std_pf_recent or 0.0, n_recent)
                if _p is not None:
                    b.p_value = round(_p, 4)
                    b.drift_significant = _p < 0.05
                else:
                    b.p_value = None
                    b.drift_significant = None
            else:
                b.p_value = None
                b.drift_significant = None

        updated.append(b)
    return updated


def _pf_drift_tier(drift: float) -> str | None:
    if drift <= PF_DRIFT_ACTION:
        return "action"
    if drift <= PF_DRIFT_ALERT:
        return "alert"
    if drift <= PF_DRIFT_WATCH:
        return "watch"
    return None


# --- \u00a76.5 Baseline state validation ---

def validate_baseline_state(
    cleaned_baseline: pd.DataFrame,
    bands: list[BandRecord],
    meta: dict,
) -> BaselineState:
    """Validate the baseline at acceptance time (\u00a76.5).

    Issues user-facing messages if baseline IUF or PF are already elevated.
    """
    state = BaselineState()
    # meta["p_rated_shaft_kw"] guaranteed non-zero by resolve_effective_meta (§2.5)
    _p_shaft_vbs = float(meta.get("p_rated_shaft_kw", 0))
    _eta_vbs     = float(meta.get("eta_rated", 0.90))
    p_rated_elec = (_p_shaft_vbs / _eta_vbs) if (_p_shaft_vbs > 0 and _eta_vbs > 0) else 1.0
    pf_nameplate = float(meta.get("pf_rated", 0))

    # 6.5.1 Baseline IUF check
    if len(cleaned_baseline) > 0:
        iuf_b, _, _ = compute_iuf(cleaned_baseline)
        state.iuf_mean = round(iuf_b, 2)
        if iuf_b >= IUF_CRITICAL:
            state.iuf_tier = "critical"
            state.iuf_message = (
                f"The baseline period shows mean IUF of {iuf_b:.1f}%, at or above "
                f"the {IUF_CRITICAL:.0f}% critical threshold. The motor was already "
                f"operating in an inappropriate balance state during the baseline period. "
                f"Recommended actions: (a) verify CT channel assignment and panel wiring; "
                f"(b) inspect panel-to-motor path on affected phase; "
                f"(c) confirm baseline window was not selected during degraded operation; "
                f"(d) re-select a different baseline period if a healthy reference is available."
            )
        elif iuf_b >= IUF_WATCH:
            state.iuf_tier = "watch"
            state.iuf_message = (
                f"The baseline period shows mean IUF of {iuf_b:.1f}%, at or above "
                f"the {IUF_WATCH:.0f}% watch threshold. The motor was already operating "
                f"with elevated current unbalance during the baseline period. "
                f"Inspection of the panel-to-motor electrical path on the affected phase "
                f"is recommended."
            )

    # 6.5.2 Baseline PF check (only if most-loaded band >= 95% of rated)
    if bands:
        most_loaded = max(bands, key=lambda b: b.centre_kw)
        pct_of_rated = most_loaded.centre_kw / p_rated_elec
        state.pf_check_band_centre_pct = round(pct_of_rated * 100, 1)

        if pct_of_rated >= BASELINE_PF_CHECK_LOAD_GATE:
            state.pf_check_performed = True
            drift_vs_nameplate = most_loaded.mean_pf_baseline - pf_nameplate
            state.pf_drift_vs_nameplate = round(drift_vs_nameplate, 5)
            tier = _pf_drift_tier(drift_vs_nameplate)
            state.pf_tier = tier
            if tier:
                state.pf_message = (
                    f"Baseline PF in the most-loaded band ({most_loaded.centre_kw:.1f} kW, "
                    f"{pct_of_rated*100:.0f}% of rated) is {most_loaded.mean_pf_baseline:.3f} "
                    f"vs nameplate {pf_nameplate:.2f} "
                    f"(drift {drift_vs_nameplate:+.3f}, {tier} tier). "
                    f"The motor may have been operating in a degraded electromagnetic state "
                    f"during the baseline period."
                )
        else:
            state.pf_check_performed = False
            state.pf_message = (
                f"Baseline PF validation against nameplate was not performed because the "
                f"baseline data does not contain operation near rated load "
                f"(most-loaded band is at {pct_of_rated*100:.0f}% of rated, below the "
                f"{BASELINE_PF_CHECK_LOAD_GATE*100:.0f}% threshold). "
                f"This is normal for machines that operate well below rated capacity. "
                f"Subsequent assessments will use the baseline data itself as the reference."
            )

    return state


# --- \u00a76.6 Four-cell decision matrix ---

def classify_motor_side(
    iuf_mean_pct: float,
    iuf_outlier_phase: str | None,
    iuf_outlier_direction: str | None,
    pf_drift: float | None,
    pf_drift_suppressed: bool,
    pf_drift_suppression_reason: str | None,
    bands: list[BandRecord],
) -> MotorSideResult:
    """Four-cell classification and diagnostic narrative (\u00a76.6).

    Cell mapping
    ------------
    PF stable + No imbalance    -> Cell 1: Healthy
    PF degraded + No imbalance  -> Cell 2: Motor zone issue
    PF stable + Imbalance       -> Cell 3: Panel-to-motor zone issue
    PF degraded + Imbalance     -> Cell 4: Both zones
    """
    iuf_tier = _iuf_tier(iuf_mean_pct)
    imbalance_present = iuf_tier is not None

    if pf_drift_suppressed or pf_drift is None:
        pf_drift_tier_val = None
        pf_degraded = False
    else:
        pf_drift_tier_val = _pf_drift_tier(pf_drift)
        pf_degraded = pf_drift_tier_val is not None

    # Determine cell
    if not imbalance_present and not pf_degraded:
        cell, label = 1, "Healthy"
    elif not imbalance_present and pf_degraded:
        cell, label = 2, "Motor zone issue"
    elif imbalance_present and not pf_degraded:
        cell, label = 3, "Panel-to-motor zone issue"
    else:
        cell, label = 4, "Both zones"

    # Overall tier = highest of individual tiers
    tier_order = {None: 0, "watch": 1, "alert": 2, "action": 3, "critical": 4}

    def _tier_max(t1: str | None, t2: str | None) -> str | None:
        if tier_order.get(t1, 0) >= tier_order.get(t2, 0):
            return t1
        return t2

    alarm_tier = _tier_max(iuf_tier, pf_drift_tier_val)

    # Inspection checklist references
    inspection_refs: list[str] = []
    if cell in (3, 4):
        inspection_refs.append("\u00a76.8.1 Panel-to-motor zone inspection")
    if cell in (2, 4):
        inspection_refs.append("\u00a76.8.2 Motor zone inspection")
    if cell == 4:
        inspection_refs.append("\u00a76.8.3 Sequencing for Cell 4")

    # Human-readable message
    pf_drift_str = (
        f"PF drift {pf_drift:+.3f}"
        if pf_drift is not None
        else f"PF drift suppressed ({pf_drift_suppression_reason})"
    )
    iuf_str = f"IUF {iuf_mean_pct:.1f}%"
    if imbalance_present and iuf_outlier_phase:
        iuf_str += f" (outlier phase {iuf_outlier_phase} {iuf_outlier_direction})"

    tier_suffix = f" at {alarm_tier} tier" if alarm_tier else ""
    if cell == 1:
        message = (
            f"Motor-side diagnosis: Healthy. {iuf_str} (below {IUF_WATCH:.0f}% threshold). "
            f"{pf_drift_str}. No action required."
        )
    elif cell == 2:
        message = (
            f"Motor-side diagnosis: Motor zone issue{tier_suffix}. "
            f"{iuf_str} (no imbalance). {pf_drift_str} (degraded). "
            f"The motor\u2019s electromagnetic state has shifted from baseline. "
            f"Inspection of the motor interior is recommended. "
            f"See {inspection_refs[0]}."
        )
    elif cell == 3:
        message = (
            f"Motor-side diagnosis: Panel-to-motor zone issue{tier_suffix}. "
            f"{iuf_str}. {pf_drift_str}. "
            f"Issue somewhere in the electrical path between the panel and motor. "
            f"Focus inspection on phase {iuf_outlier_phase} of the panel-to-motor path. "
            f"See {inspection_refs[0]}."
        )
    else:
        message = (
            f"Motor-side diagnosis: Both panel-to-motor and motor zone issues{tier_suffix}. "
            f"{iuf_str}. {pf_drift_str} (degraded). "
            f"Inspect panel-to-motor path on phase {iuf_outlier_phase} first, then "
            f"re-evaluate the motor signature. "
            f"See {inspection_refs[0]} and {inspection_refs[1]}."
        )

    return MotorSideResult(
        iuf_mean_pct=round(iuf_mean_pct, 2),
        iuf_tier=iuf_tier,
        outlier_phase=iuf_outlier_phase,
        outlier_direction=iuf_outlier_direction,
        pf_drift_aggregated=pf_drift,
        pf_drift_tier=pf_drift_tier_val,
        pf_drift_suppressed=pf_drift_suppressed,
        pf_drift_suppression_reason=pf_drift_suppression_reason,
        bands=bands,
        cell=cell,
        cell_label=label,
        alarm_tier=alarm_tier,
        inspection_refs=inspection_refs,
        message=message,
    )


# ---------------------------------------------------------------------------
# \u00a77  Zone 4 \u2014 Driven equipment indication
# ---------------------------------------------------------------------------

_ZONE4_DISABLED_APP_TYPES = {HVAC_CHILLER_APP_TYPE}


def compute_zone4(
    p_baseline_avg_kw: float,
    cleaned_recent: pd.DataFrame,
    meta: dict,
) -> Zone4Result:
    """Zone 4 driven equipment indication (\u00a77.3).

    Computes the percentage change in average motor power consumption between
    baseline and recent periods and reports a finding when the change exceeds
    the significance threshold.  No user confirmation is required \u2014 the finding
    is reported as an informational observation for engineer review and inclusion
    in the assessment report.

    Parameters
    ----------
    p_baseline_avg_kw : pre-computed mean P_total from cleaned baseline (kW)
    cleaned_recent    : cleaned recent DataFrame
    meta              : machine metadata (must include application_type)
    """
    app_type = str(meta.get("application_type", "")).lower()

    # Applicability check \u2014 HVAC chillers suppressed (temperature-dependent load)
    if app_type in _ZONE4_DISABLED_APP_TYPES:
        return Zone4Result(
            suppressed=True,
            finding_state="suppressed",
            suppression_reason=(
                "Zone 4 detection suppressed: HVAC chiller application. "
                "Power consumption varies with ambient temperature and cooling load. "
                "Temperature-corrected methodology will be developed in a future revision."
            ),
            p_baseline_avg_kw=p_baseline_avg_kw,
            message=(
                "Zone 4 detection suppressed for HVAC chiller. "
                "Temperature-corrected methodology pending."
            ),
        )

    # Compute recent average power (DataFrame stores W; convert to kW)
    p_total_recent = _p_total_series(cleaned_recent)
    p_recent_avg = float(p_total_recent.mean()) / 1000.0
    delta_p_pct = (p_recent_avg - p_baseline_avg_kw) / p_baseline_avg_kw * 100.0

    result = Zone4Result(
        p_baseline_avg_kw=round(p_baseline_avg_kw, 2),
        p_recent_avg_kw=round(p_recent_avg, 2),
        delta_p_pct=round(delta_p_pct, 2),
    )

    # Significance check
    if abs(delta_p_pct) < ZONE4_SIGNIFICANCE_PCT:
        result.finding_state = None
        result.direction = None
        result.message = (
            f"Zone 4: No finding. Average power change {delta_p_pct:+.1f}% "
            f"is within the \u00b1{ZONE4_SIGNIFICANCE_PCT:.0f}% significance threshold "
            f"(baseline {p_baseline_avg_kw:.1f} kW \u2192 recent {p_recent_avg:.1f} kW)."
        )
        return result

    direction = "positive" if delta_p_pct >= 0 else "negative"
    result.direction = direction

    if direction == "positive":
        result.finding_state = "positive"
        result.message = (
            f"Zone 4 finding: average motor power has increased {delta_p_pct:+.1f}% "
            f"since baseline (baseline {p_baseline_avg_kw:.1f} kW \u2192 "
            f"recent {p_recent_avg:.1f} kW). "
            f"Possible causes: increase in production load, driven equipment degradation "
            f"(higher friction, wear, fouling), or both. "
            f"Engineer review recommended \u2014 confirm whether production load has "
            f"changed since the baseline period. If load is unchanged, inspect driven "
            f"equipment internals per \u00a77.7 checklist."
        )
    else:
        result.finding_state = "negative"
        result.message = (
            f"Zone 4 finding: average motor power has decreased {abs(delta_p_pct):.1f}% "
            f"since baseline (baseline {p_baseline_avg_kw:.1f} kW \u2192 "
            f"recent {p_recent_avg:.1f} kW). "
            f"Possible explanations: reduced production load, recent maintenance benefit, "
            f"or measurement chain change. "
            f"Engineer review recommended \u2014 confirm whether production load or "
            f"equipment condition has changed since the baseline period."
        )

    return result


# ---------------------------------------------------------------------------
    """Zone 4 two-stage detection (\u00a77.3-7.5).

    Parameters
    ----------
    p_baseline_avg_kw : pre-computed mean P_total from cleaned baseline
    cleaned_recent    : cleaned recent DataFrame
    meta              : machine metadata (must include application_type)
    user_response     : None | "load_increased" | "load_unchanged"
                        | "load_decreased" (for positive/negative cases)

    Returns
    -------
    Zone4Result with finding_state set appropriately.
    """
    app_type = str(meta.get("application_type", "")).lower()

    # Applicability check
    if app_type in _ZONE4_DISABLED_APP_TYPES:
        return Zone4Result(
            suppressed=True,
            suppression_reason=(
                "Zone 4 detection suppressed: application type is hvac_chiller. "
                "Temperature-corrected methodology for HVAC chillers will be developed "
                "in a subsequent revision."
            ),
            p_baseline_avg_kw=p_baseline_avg_kw,
        )

    # Compute recent average power (DataFrame stores W; convert to kW for comparison)
    p_total_recent = _p_total_series(cleaned_recent)
    p_recent_avg = float(p_total_recent.mean()) / 1000.0  # kW
    delta_p_pct = (p_recent_avg - p_baseline_avg_kw) / p_baseline_avg_kw * 100.0

    result = Zone4Result(
        p_baseline_avg_kw=round(p_baseline_avg_kw, 2),
        p_recent_avg_kw=round(p_recent_avg, 2),
        delta_p_pct=round(delta_p_pct, 2),
    )

    # Significance check
    if abs(delta_p_pct) < ZONE4_SIGNIFICANCE_PCT:
        result.finding_state = None
        result.message = (
            f"Zone 4: No finding. Average power change {delta_p_pct:+.1f}% "
            f"is within the {ZONE4_SIGNIFICANCE_PCT:.0f}% significance threshold."
        )
        return result

    direction = "positive" if delta_p_pct >= 0 else "negative"
    result.direction = direction

    # Handle user response
    if user_response is None:
        result.finding_state = "pending"
        if direction == "positive":
            result.message = (
                f"Zone 4 tentative finding: average power consumption has increased "
                f"by {delta_p_pct:+.1f}% since baseline "
                f"(from {p_baseline_avg_kw:.1f} kW to {p_recent_avg:.1f} kW). "
                f"Please confirm whether the actual production load on the driven "
                f"equipment has changed since the baseline period. "
                f"Reply \u2018load_increased\u2019 if load explains the change, or "
                f"\u2018load_unchanged\u2019 if production has been steady."
            )
        else:
            result.message = (
                f"Zone 4 tentative finding: average power consumption has decreased "
                f"by {abs(delta_p_pct):.1f}% since baseline "
                f"(from {p_baseline_avg_kw:.1f} kW to {p_recent_avg:.1f} kW). "
                f"Please confirm whether the actual production load on the driven "
                f"equipment has changed. "
                f"Reply \u2018load_decreased\u2019 if load explains the change, or "
                f"\u2018load_unchanged\u2019 if production has been steady."
            )
        return result

    # Process response
    resp = re.sub(r"\s+", "_", user_response.strip().lower())

    if direction == "positive":
        if re.search(r"\bload_increased\b", resp):
            result.finding_state = "dismissed"
            result.user_response = user_response
            result.message = (
                f"Zone 4 tentative finding dismissed (load change confirmed by user). "
                f"Consider re-establishing the baseline to reflect the new operating "
                f"profile if the load change is permanent."
            )
        elif re.search(r"\bload_unchanged\b", resp):
            result.finding_state = "alarm"
            result.user_response = user_response
            result.message = (
                f"Zone 4 performance degradation alarm. Driven equipment efficiency "
                f"loss is suspected: power has increased {delta_p_pct:+.1f}% without "
                f"a corresponding load change. Inspection of driven equipment internals "
                f"is recommended. See \u00a77.7 inspection checklist."
            )
        else:
            result.finding_state = "pending"
            result.message = (
                f"Zone 4 user response \u2018{user_response}\u2019 not recognised. "
                f"Expected \u2018load_increased\u2019 or \u2018load_unchanged\u2019."
            )

    else:  # negative
        if re.search(r"\bload_decreased\b", resp):
            result.finding_state = "dismissed"
            result.user_response = user_response
            result.message = (
                f"Zone 4 tentative finding dismissed (load change confirmed by user)."
            )
        elif re.search(r"\bload_unchanged\b", resp):
            result.finding_state = "investigation"
            result.user_response = user_response
            result.message = (
                f"Zone 4 unexplained power decrease ({delta_p_pct:+.1f}%). "
                f"Possible causes: "
                f"(a) recent maintenance has improved driven equipment efficiency \u2014 "
                f"consider re-establishing baseline; "
                f"(b) measurement chain change (e.g. CT replacement) has shifted "
                f"calibration \u2014 inspect measurement chain; "
                f"(c) reduced motor output causing process underperformance \u2014 "
                f"inspect driven equipment."
            )
        else:
            result.finding_state = "pending"
            result.message = (
                f"Zone 4 user response \u2018{user_response}\u2019 not recognised. "
                f"Expected \u2018load_decreased\u2019 or \u2018load_unchanged\u2019."
            )

    return result


# ---------------------------------------------------------------------------
# \u00a78  Orchestration: baseline ingestion
# ---------------------------------------------------------------------------

def ingest_baseline(
    raw_baseline: pd.DataFrame,
    meta: dict,
    user_filter: str | None = None,
) -> BaselineMetadata:
    """Process a user-designated baseline window (\u00a76.3.1 + \u00a77.3).

    Steps
    -----
    1. Clean baseline samples through the five-step procedure.
    2. Select multi-band PF structure from cleaned samples.
    3. Compute baseline average power for Zone 4.
    4. Validate the baseline state (\u00a76.5).
    5. Return BaselineMetadata for storage.
    """
    # meta["p_rated_shaft_kw"] is guaranteed non-zero by resolve_effective_meta (§2.5).
    p_shaft = float(meta.get("p_rated_shaft_kw", 0))
    eta     = float(meta.get("eta_rated", 0.90))
    if p_shaft > 0 and eta > 0:
        p_rated_elec = p_shaft / eta
    else:
        # Fallback only if called without resolved meta
        _pt_bl = (raw_baseline["phase_1_active_power"] +
                  raw_baseline["phase_2_active_power"] +
                  raw_baseline["phase_3_active_power"])
        _p95   = float(_pt_bl[_pt_bl > 0].quantile(0.95)) / 1000.0 if len(_pt_bl[_pt_bl > 0]) > 0 else 0.0
        p_rated_elec = _p95 / 0.95 if _p95 > 0 else 1.0
    bm = BaselineMetadata(
        timestamp_start=raw_baseline["timestamp"].min() if "timestamp" in raw_baseline.columns else None,
        timestamp_end=raw_baseline["timestamp"].max() if "timestamp" in raw_baseline.columns else None,
        user_filter_expr=user_filter,
    )

    cleaned, report = clean_samples(raw_baseline, meta, user_filter)
    bm.cleaning_report = report

    if len(cleaned) < CLEANING_MIN_SAMPLES:
        bm.warnings.append(
            f"Insufficient cleaned baseline samples: {len(cleaned)} < {CLEANING_MIN_SAMPLES}. "
            f"Baseline not accepted."
        )
        return bm

    # Band selection is deferred to analysis time (run_assessment) so it always
    # uses the current rated power from meta, not the power at ingest time.
    # BaselineMetadata.bands and n_qualifying_bands remain empty here.

    # Zone 4 baseline average power (stored in kW for readability)
    bm.p_baseline_avg_kw = round(
        float((_p_total_series(cleaned)).mean()) / 1000.0, 3
    )

    # Baseline state validation (bands passed as empty — IUF check still runs)
    bm.baseline_state = validate_baseline_state(cleaned, [], meta)

    return bm


# ---------------------------------------------------------------------------
# \u00a78  Orchestration: run_assessment
# ---------------------------------------------------------------------------

def run_assessment(
    raw_recent: pd.DataFrame,
    baseline: BaselineMetadata,
    meta: dict,
    raw_baseline: pd.DataFrame | None = None,
    load_precondition_fraction: float | None = None,
) -> AssessmentRecord:
    """Run a full user-initiated assessment (§8.1).

    Parameters
    ----------
    raw_recent                 : raw measurement DataFrame for the recent window
    baseline                   : BaselineMetadata from ingest_baseline()
    meta                       : machine metadata dict (current effective meta)
    raw_baseline               : raw baseline DataFrame — used to compute PF bands at
                                 analysis time with current rated power (§6.3.2)
    load_precondition_fraction : override for the minimum load fraction passed to
                                 clean_samples (default: LOAD_PRECONDITION_FRACTION).
    """
    record = AssessmentRecord()

    # -- Check 1: Data cleaning --
    user_filter = baseline.user_filter_expr if baseline else None
    cleaned, cleaning_report = clean_samples(
        raw_recent, meta, user_filter,
        load_precondition_fraction=load_precondition_fraction,
    )
    record.cleaning_report = cleaning_report

    if cleaning_report.n_cleaned < CLEANING_MIN_SAMPLES:
        record.suppressed = True
        record.suppression_reason = (
            f"Insufficient cleaned samples after data cleaning: "
            f"{cleaning_report.n_cleaned} < {CLEANING_MIN_SAMPLES}. "
            f"Assessment suppressed for this cycle."
        )
        record.messages.append(record.suppression_reason)
        return record

    record.integrity_status = "passed"

    # -- Compute PF bands from baseline at analysis time using actual data range --
    _cleaned_bl: pd.DataFrame | None = None   # kept in scope for std recomputation
    if raw_baseline is not None and len(raw_baseline) > 0:
        _cleaned_bl, _ = clean_samples(
            raw_baseline, meta, user_filter,
            load_precondition_fraction=load_precondition_fraction,
        )
        if len(_cleaned_bl) >= CLEANING_MIN_SAMPLES:
            all_bins = select_pf_bands(_cleaned_bl, min_samples=0)   # all 100 bins
            bands    = [b for b in all_bins if b.n_baseline >= PF_BAND_MIN_SAMPLES]
            record.baseline_bands = all_bins   # full histogram for reporting
            baseline = BaselineMetadata(
                timestamp_start=baseline.timestamp_start,
                timestamp_end=baseline.timestamp_end,
                user_filter_expr=baseline.user_filter_expr,
                bands=bands,
                n_qualifying_bands=len(bands),
                p_baseline_avg_kw=baseline.p_baseline_avg_kw,
                cleaning_report=baseline.cleaning_report,
                warnings=list(baseline.warnings),
                baseline_state=baseline.baseline_state,
            )

    # -- Check 2: Supply channel (Zone 1) --
    vuf = compute_vuf(cleaned)
    supply = classify_vuf(vuf)
    record.supply_alarm = supply
    if supply.fired:
        record.messages.append(
            f"Supply quality {supply.tier}: VUF {supply.vuf_pct:.2f}%. "
            + (
                f"Investigate upstream installation; motor derating per NEMA MG-1 "
                f"recommended."
                if supply.tier == "critical"
                else "Logged in health record."
            )
        )

    # -- Check 3: Motor-side combined detection (Zones 2 & 3) --
    iuf_mean, outlier_ph, outlier_dir = compute_iuf(cleaned)

    if baseline.bands:
        agg_drift, updated_bands, suppressed, supp_reason = compute_pf_drift(
            cleaned, baseline.bands, cleaned_baseline=_cleaned_bl
        )
    else:
        agg_drift, updated_bands = None, []
        suppressed = True
        supp_reason = "No baseline bands available"

    motor_result = classify_motor_side(
        iuf_mean_pct=iuf_mean,
        iuf_outlier_phase=outlier_ph,
        iuf_outlier_direction=outlier_dir,
        pf_drift=agg_drift,
        pf_drift_suppressed=suppressed,
        pf_drift_suppression_reason=supp_reason,
        bands=updated_bands,
    )
    record.motor_side = motor_result
    if motor_result.alarm_tier:
        record.messages.append(motor_result.message)

    # -- Check 4: Zone 4 --
    if baseline.p_baseline_avg_kw is not None:
        z4 = compute_zone4(
            p_baseline_avg_kw=baseline.p_baseline_avg_kw,
            cleaned_recent=cleaned,
            meta=meta,
        )
    else:
        z4 = Zone4Result(
            suppressed=True,
            suppression_reason="Baseline average power not available (baseline ingestion incomplete)",
        )
    record.zone4 = z4
    if z4.message:
        record.messages.append(z4.message)

    return record


# ---------------------------------------------------------------------------
# Convenience: human-readable summary
# ---------------------------------------------------------------------------

def assessment_summary(record: AssessmentRecord) -> str:
    """Return a concise text summary of an AssessmentRecord."""
    lines: list[str] = []

    if record.suppressed:
        return f"Assessment suppressed: {record.suppression_reason}"

    r = record.cleaning_report
    if r:
        lines.append(
            f"Data cleaning: {r.n_raw} raw"
            f" \u2192 {r.n_after_load_precondition} load \u226520%"
            f" \u2192 {r.n_after_start_transient} start-transient"
            f" \u2192 {r.n_after_user_filter} user-filter"
            f" \u2192 {r.n_cleaned} cleaned"
            f" ({r.fraction_retained*100:.0f}% retained)"
        )

    s = record.supply_alarm
    if s:
        tier_str = s.tier if s.tier else "none"
        lines.append(f"Zone 1 (supply): VUF {s.vuf_pct:.2f}%  |  alarm tier: {tier_str}")

    m = record.motor_side
    if m:
        pf_str = (
            f"{m.pf_drift_aggregated:+.3f}" if m.pf_drift_aggregated is not None
            else f"suppressed ({m.pf_drift_suppression_reason})"
        )
        lines.append(
            f"Zones 2&3 (motor-side): Cell {m.cell} \u2014 {m.cell_label}  |  "
            f"IUF {m.iuf_mean_pct:.1f}%  |  PF drift {pf_str}  |  "
            f"tier: {m.alarm_tier}"
        )

    z = record.zone4
    if z:
        if z.suppressed:
            lines.append(f"Zone 4: suppressed ({z.suppression_reason})")
        elif z.delta_p_pct is not None:
            lines.append(
                f"Zone 4: \u0394P {z.delta_p_pct:+.1f}%  |  state: {z.finding_state}"
            )
        else:
            lines.append("Zone 4: no data")

    return "\n".join(lines)
