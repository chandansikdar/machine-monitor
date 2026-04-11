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
LOAD_PRECONDITION_FRACTION: float = 0.40  # fraction of p_rated_elec minimum
IQR_MULTIPLIER: float = 1.5               # standard Tukey fence

# Multi-band PF comparison
PF_BAND_BIN_WIDTH_FRACTION: float = 0.02  # 2% of p_rated_elec per bin
PF_BAND_MIN_SAMPLES: int = 20             # minimum samples per band
PF_BAND_MIN_COUNT: int = 3               # minimum qualifying bands

# Cleaning sufficiency
CLEANING_MIN_SAMPLES: int = 100

# Zone 4 driven equipment
ZONE4_SIGNIFICANCE_PCT: float = 3.0

# Baseline PF check \u2014 most-loaded band must be at or above this fraction of rated
BASELINE_PF_CHECK_LOAD_GATE: float = 0.95

# HVAC chiller application_type value (suppresses Zone 4)
HVAC_CHILLER_APP_TYPE: str = "hvac_chiller"

# Running mask: sample is "running" when P_total exceeds this fraction of rated
RUNNING_THRESHOLD_FRACTION: float = 0.05


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
    n_raw: int = 0
    n_after_load_precondition: int = 0
    n_after_iqr: int = 0
    n_after_running_mask: int = 0
    n_after_integrity: int = 0
    n_after_user_filter: int = 0

    @property
    def n_cleaned(self) -> int:
        return self.n_after_user_filter

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
    n_recent: int = 0
    mean_pf_recent: float | None = None
    pf_drift: float | None = None
    suppressed: bool = False
    suppression_reason: str | None = None


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
    """Structured output for one user-initiated assessment (\u00a78)."""
    integrity_status: str = "not_run"      # "passed" | "failed" | "suppressed"
    suppressed: bool = False
    suppression_reason: str | None = None
    cleaning_report: CleaningReport | None = None
    supply_alarm: SupplyAlarm | None = None
    motor_side: MotorSideResult | None = None
    zone4: Zone4Result | None = None
    messages: list[str] = field(default_factory=list)


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
    p_rated_elec = float(meta["p_rated_shaft_kw"]) / float(meta["eta_rated"])
    running = p_total > RUNNING_THRESHOLD_FRACTION * p_rated_elec * 1000.0

    # --- Check 1: Voltage plausibility ---
    v_lo = 0.85 * v_nom
    v_hi = 1.15 * v_nom
    v_max_abs = 1.5 * v_nom
    for ph, v in zip(("1", "2", "3"), (v1, v2, v3)):
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
) -> tuple[pd.DataFrame, CleaningReport]:
    """Five-step data cleaning procedure (\u00a74.1) \u2014 reordered for physical sense.

    Step order
    ----------
    1. Load precondition (\u226540% rated) \u2014 discard low-load / shutdown samples first
    2. Running mask          \u2014 exclude transient start/stop periods
    3. Integrity gate        \u2014 validate measurement quality on loaded running samples
    4. User filter           \u2014 apply any operating-condition filter
    5. IQR outlier rejection \u2014 remove statistical outliers from the clean working set

    Parameters
    ----------
    raw         : raw measurement DataFrame
    meta        : machine metadata dict
    user_filter : optional pandas query string applied at step 4
    """
    report = CleaningReport(n_raw=len(raw))
    p_rated_elec = float(meta["p_rated_shaft_kw"]) / float(meta["eta_rated"])

    # -- Step 1: Load precondition (>= 40% of rated electrical input) --
    df = raw.copy()
    if len(df) > 0:
        p_total = (df["phase_1_active_power"] + df["phase_2_active_power"]
                   + df["phase_3_active_power"])
        load_min_w = LOAD_PRECONDITION_FRACTION * p_rated_elec * 1000.0
        df = df[(p_total >= load_min_w)].copy()
    report.n_after_load_precondition = len(df)

    # -- Step 2: IQR outlier rejection --
    if len(df) >= 4:
        p_total = (df["phase_1_active_power"] + df["phase_2_active_power"]
                   + df["phase_3_active_power"])
        i_avg = (df["phase_1_current"] + df["phase_2_current"]
                 + df["phase_3_current"]) / 3.0
        s_sum = (df["phase_1_voltage"] * df["phase_1_current"]
                 + df["phase_2_voltage"] * df["phase_2_current"]
                 + df["phase_3_voltage"] * df["phase_3_current"])
        pf_machine = p_total / s_sum.replace(0, np.nan)

        keep = pd.Series(True, index=df.index)
        for signal in (p_total, i_avg, pf_machine):
            q25 = signal.quantile(0.25)
            q75 = signal.quantile(0.75)
            iqr = q75 - q25
            lo = q25 - IQR_MULTIPLIER * iqr
            hi = q75 + IQR_MULTIPLIER * iqr
            keep &= signal.between(lo, hi, inclusive="both")
        df = df[keep].copy()
    report.n_after_iqr = len(df)

    # -- Step 3: Running mask --
    if len(df) > 0:
        running = _running_mask(df, p_rated_elec)
        df = df[running].copy()
    report.n_after_running_mask = len(df)

    # -- Step 4: Integrity gate --
    if len(df) > 0:
        integrity_mask = df.apply(
            lambda row: integrity_gate(row, meta).passed, axis=1
        )
        df = df[integrity_mask].copy()
    report.n_after_integrity = len(df)

    # -- Step 5: User operating-condition filter --
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


def _p_total_series(df: pd.DataFrame) -> pd.Series:
    return df["phase_1_active_power"] + df["phase_2_active_power"] + df["phase_3_active_power"]


def select_pf_bands(cleaned_baseline: pd.DataFrame, p_rated_elec_kw: float) -> list[BandRecord]:
    """Build multi-band structure from cleaned baseline (\u00a76.3.2).

    Bin width = 2% of rated electrical input.
    Only bins with >= PF_BAND_MIN_SAMPLES samples are selected.
    """
    p_total = _p_total_series(cleaned_baseline)
    pf_machine = _pf_machine_series(cleaned_baseline)

    bin_width = PF_BAND_BIN_WIDTH_FRACTION * p_rated_elec_kw * 1000.0  # Watts, matching DataFrame
    p_min = float(p_total.min())
    p_max = float(p_total.max())

    # Build edges that cover [p_min, p_max] in steps of bin_width
    edges = np.arange(p_min, p_max + bin_width, bin_width)
    if len(edges) < 2:
        return []

    bands: list[BandRecord] = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        mask = (p_total >= lo) & (p_total < hi)
        n = int(mask.sum())
        if n >= PF_BAND_MIN_SAMPLES:
            mean_pf = float(pf_machine[mask].mean())
            centre = (lo + hi) / 2.0
            bands.append(BandRecord(
                centre_kw=round(centre, 3),
                low_kw=round(lo, 3),
                high_kw=round(hi, 3),
                n_baseline=n,
                mean_pf_baseline=round(mean_pf, 5),
            ))

    return bands


def compute_pf_drift(
    cleaned_recent: pd.DataFrame,
    stored_bands: list[BandRecord],
) -> tuple[float | None, list[BandRecord], bool, str | None]:
    """Compute per-band and aggregated PF drift (\u00a76.3.3).

    Returns
    -------
    aggregated_drift : weighted mean PF drift (None if suppressed)
    updated_bands    : BandRecord list with recent counts and drift filled in
    suppressed       : True if fewer than PF_BAND_MIN_COUNT bands have sufficient data
    suppression_reason : message when suppressed
    """
    if not stored_bands:
        return None, stored_bands, True, "No baseline bands available"

    p_total = _p_total_series(cleaned_recent)
    pf_machine = _pf_machine_series(cleaned_recent)

    updated: list[BandRecord] = []
    active_weights: list[float] = []
    active_drifts: list[float] = []

    for band in stored_bands:
        mask = (p_total >= band.low_kw) & (p_total < band.high_kw)
        n_recent = int(mask.sum())
        b = BandRecord(
            centre_kw=band.centre_kw,
            low_kw=band.low_kw,
            high_kw=band.high_kw,
            n_baseline=band.n_baseline,
            mean_pf_baseline=band.mean_pf_baseline,
        )
        b.n_recent = n_recent
        if n_recent < PF_BAND_MIN_SAMPLES:
            b.suppressed = True
            b.suppression_reason = (
                f"Only {n_recent} recent samples (minimum {PF_BAND_MIN_SAMPLES})"
            )
        else:
            b.mean_pf_recent = round(float(pf_machine[mask].mean()), 5)
            b.pf_drift = round(b.mean_pf_recent - band.mean_pf_baseline, 5)
            active_drifts.append(b.pf_drift)
            active_weights.append(float(n_recent))
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
    p_rated_elec = float(meta["p_rated_shaft_kw"]) / float(meta["eta_rated"])
    pf_nameplate = float(meta["pf_rated"])

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
    p_rated_elec = float(meta["p_rated_shaft_kw"]) / float(meta["eta_rated"])
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

    # Band selection
    bands = select_pf_bands(cleaned, p_rated_elec)
    bm.bands = bands
    bm.n_qualifying_bands = len(bands)

    if bm.n_qualifying_bands < PF_BAND_MIN_COUNT:
        bm.warnings.append(
            f"Only {bm.n_qualifying_bands} PF bands qualified "
            f"(minimum {PF_BAND_MIN_COUNT}). Multi-band PF drift will be suppressed "
            f"for assessments that reference this baseline. "
            f"Consider extending the baseline window."
        )

    # Zone 4 baseline average power (stored in kW for readability)
    bm.p_baseline_avg_kw = round(
        float((_p_total_series(cleaned)).mean()) / 1000.0, 3
    )

    # Baseline state validation
    bm.baseline_state = validate_baseline_state(cleaned, bands, meta)

    return bm


# ---------------------------------------------------------------------------
# \u00a78  Orchestration: run_assessment
# ---------------------------------------------------------------------------

def run_assessment(
    raw_recent: pd.DataFrame,
    baseline: BaselineMetadata,
    meta: dict,
) -> AssessmentRecord:
    """Run a full user-initiated assessment (\u00a78.1).

    Parameters
    ----------
    raw_recent           : raw measurement DataFrame for the recent window
    baseline             : BaselineMetadata from ingest_baseline()
    meta                 : machine metadata dict
    zone4_user_response  : removed — Zone 4 is now informational only

    Returns
    -------
    AssessmentRecord with all zone findings populated.
    """
    record = AssessmentRecord()

    # -- Check 1: Data cleaning --
    user_filter = baseline.user_filter_expr if baseline else None
    cleaned, cleaning_report = clean_samples(raw_recent, meta, user_filter)
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
            cleaned, baseline.bands
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
            f"Data cleaning: {r.n_raw} raw \u2192 {r.n_after_integrity} integrity "
            f"\u2192 {r.n_after_running_mask} running \u2192 {r.n_after_user_filter} filtered "
            f"\u2192 {r.n_after_load_precondition} load \u2192 {r.n_cleaned} cleaned "
            f"({r.fraction_retained*100:.0f}% retained)"
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
