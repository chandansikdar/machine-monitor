"""
chiller_physics.py
==================
Phase 1 physics-based analytics for vapour compression chillers.

Phase 1 requires only electrical data at the chiller panel level:
  - Total active power (kW)          ← or derived from phase currents + voltage
  - Phase voltages (V_a, V_b, V_c)   ← optional, enables imbalance analysis
  - Phase currents (A_a, A_b, A_c)   ← optional, enables imbalance analysis
  - Energy accumulator (kWh)         ← optional, used if power column absent
  - Ambient temperature (°C)         ← optional, enables temperature correction

No thermal measurements, flow rates, or refrigerant pressures are needed
for Phase 1. If any of those are present the higher phases (2–5) will also
activate; they are handled in separate modules.

Public API
----------
  detect_phase_1(df)          → dict of detected column names and flags
  parse_nameplate(desc)       → dict of nameplate values from description text
  run_phase1(df, np_data, ...) → dict with findings, metrics, plots, summary
"""

from __future__ import annotations

import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — COLUMN DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

# Keyword maps: (column_role) → list of substrings to match in column names
# Matching is case-insensitive and checks if any keyword is a substring of the
# column name (after lowercasing and stripping non-alphanumeric characters).

_KW = {
    # ── Power / energy ────────────────────────────────────────────────────────
    "power":     ["power", "kw", "active_power", "real_power", "watt"],
    "energy":    ["kwh", "kw_h", "energy", "consumption", "wh"],

    # ── Phase voltages ────────────────────────────────────────────────────────
    # Voltage detection: look for 'v' or 'volt' plus a phase identifier (a/b/c/1/2/3)
    # or line-to-line designators (ab, bc, ca, l1, l2, l3)
    "volt_a":    ["voltage_a", "v_a", "ua", "va", "u_a", "volt_a",
                  "u_ab", "v_ab", "l1", "phase_a_v", "v1", "phase1_v"],
    "volt_b":    ["voltage_b", "v_b", "ub", "vb", "u_b", "volt_b",
                  "u_bc", "v_bc", "l2", "phase_b_v", "v2", "phase2_v"],
    "volt_c":    ["voltage_c", "v_c", "uc", "vc", "u_c", "volt_c",
                  "u_ca", "v_ca", "l3", "phase_c_v", "v3", "phase3_v"],

    # ── Phase currents ────────────────────────────────────────────────────────
    "curr_a":    ["current_a", "ia", "amp_a", "amps_a", "phase_a_i",
                  "i_a", "i1", "phase1_i", "l1_i", "curr_a"],
    "curr_b":    ["current_b", "ib", "amp_b", "amps_b", "phase_b_i",
                  "i_b", "i2", "phase2_i", "l2_i", "curr_b"],
    "curr_c":    ["current_c", "ic", "amp_c", "amps_c", "phase_c_i",
                  "i_c", "i3", "phase3_i", "l3_i", "curr_c"],

    # ── Average / total current (single-phase or already-averaged) ────────────
    "curr_avg":  ["current", "amps", "ampere", "motor_current", "avg_current",
                  "mean_current", "total_current"],

    # ── Ambient temperature ───────────────────────────────────────────────────
    "amb_temp":  ["ambient_temp", "outdoor_temp", "oat", "dry_bulb",
                  "outside_temp", "amb_t", "t_ambient", "t_amb",
                  "external_temp", "weather_temp"],
}


def _norm(s: str) -> str:
    """Lowercase and strip non-alphanumeric characters for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _find_col(df: pd.DataFrame, role: str) -> Optional[str]:
    """
    Return the first DataFrame column whose normalised name contains any of
    the keywords for *role*.  Returns None if no match found.

    Priority: exact keyword substring match wins.  If multiple columns match
    the same role the first match (in DataFrame column order) is returned.
    """
    keywords = _KW.get(role, [])
    norm_kws = [_norm(k) for k in keywords]
    for col in df.columns:
        nc = _norm(col)
        for kw in norm_kws:
            if kw in nc:
                return col
    return None


def detect_phase_1(df: pd.DataFrame) -> dict:
    """
    Inspect a DataFrame and return a dictionary describing what Phase 1
    inputs are available.

    Parameters
    ----------
    df : pd.DataFrame
        Uploaded sensor data.  Must have a DatetimeIndex or a 'timestamp'
        column.

    Returns
    -------
    dict with keys:
      cols         : dict mapping role → column name (or None)
      has_power    : bool  — direct power column found
      has_energy   : bool  — energy accumulator found (power can be derived)
      has_3v       : bool  — all three phase voltages found
      has_3i       : bool  — all three phase currents found
      has_avg_i    : bool  — single average/total current column found
      has_amb_temp : bool  — ambient temperature column found
      viable       : bool  — at least power or (3 currents + voltage) present
      missing_msg  : str   — human-readable explanation if not viable
    """
    cols = {role: _find_col(df, role) for role in _KW}

    has_power    = cols["power"] is not None
    has_energy   = cols["energy"] is not None
    has_3v       = all(cols[r] is not None for r in ("volt_a", "volt_b", "volt_c"))
    has_3i       = all(cols[r] is not None for r in ("curr_a", "curr_b", "curr_c"))
    has_avg_i    = cols["curr_avg"] is not None
    has_amb_temp = cols["amb_temp"] is not None

    # Phase 1 is viable if we have at least one power signal
    viable = has_power or has_energy or has_3i

    missing_parts = []
    if not viable:
        missing_parts.append(
            "Phase 1 requires at least one of: (a) a total power column "
            "(kW), (b) an energy accumulator column (kWh), or (c) all three "
            "phase current columns (A_a, A_b, A_c).  "
            "None of these were detected in the uploaded data."
        )

    return dict(
        cols=cols,
        has_power=has_power,
        has_energy=has_energy,
        has_3v=has_3v,
        has_3i=has_3i,
        has_avg_i=has_avg_i,
        has_amb_temp=has_amb_temp,
        viable=viable,
        missing_msg=" ".join(missing_parts),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — NAMEPLATE PARSING
# ═══════════════════════════════════════════════════════════════════════════════

# Default nameplate values used when the engineer has not supplied a value.
# These are conservative mid-range assumptions; supplying real values always
# produces more accurate results.
_NP_DEFAULTS = {
    "rated_cooling_kw":    None,     # kW — MUST be supplied for SPI
    "rated_power_kw":      None,     # kW — MUST be supplied for load factor
    "rated_cop":           None,     # dimensionless — for SPI reference
    "rated_voltage_v":     415.0,    # V line-to-line — European default
    "fla_a":               None,     # A — full load amps
    "power_factor_fl":     0.85,     # dimensionless at full load
    "ie_class":            "IE3",    # sets voltage imbalance limit
    "n_circuits":          1,        # number of refrigerant circuits
    "compressor_type":     "unknown",# centrifugal/screw/scroll/reciprocating
    "drive_type":          "direct", # direct / VFD
    "refrigerant":         "unknown",
    "commissioning_power": None,     # kW — measured at first stable run
}

# Regex patterns for parsing free-text nameplate descriptions
_NP_PATTERNS = {
    "rated_cooling_kw": [
        r"cooling\s*capacity[:\s=]+([0-9.]+)\s*kw",
        r"rated\s*cooling[:\s=]+([0-9.]+)\s*kw",
        r"capacity[:\s=]+([0-9.]+)\s*kw",
        r"([0-9.]+)\s*kw\s*cooling",
        r"([0-9.]+)\s*kw\s*rated\s*output",
        r"([0-9.]+)\s*tr\b",            # refrigeration tons → convert ×3.517
        r"([0-9.]+)\s*ton[sr]?\b",
    ],
    "rated_power_kw": [
        r"rated\s*power\s*input[:\s=]+([0-9.]+)\s*kw",
        r"power\s*input[:\s=]+([0-9.]+)\s*kw",
        r"compressor\s*power[:\s=]+([0-9.]+)\s*kw",
        r"input\s*power[:\s=]+([0-9.]+)\s*kw",
        r"rated\s*kw[:\s=]+([0-9.]+)",
        r"([0-9.]+)\s*kw\s*input",
    ],
    "rated_cop": [
        r"cop[:\s=]+([0-9.]+)",
        r"rated\s*cop[:\s=]+([0-9.]+)",
        r"coefficient\s*of\s*performance[:\s=]+([0-9.]+)",
    ],
    "rated_voltage_v": [
        r"([0-9]+)\s*v\b",
        r"voltage[:\s=]+([0-9]+)\s*v",
        r"supply[:\s=]+([0-9]+)\s*v",
    ],
    "fla_a": [
        r"fla[:\s=]+([0-9.]+)\s*a",
        r"full\s*load\s*amp[s]?[:\s=]+([0-9.]+)",
        r"rated\s*current[:\s=]+([0-9.]+)\s*a",
        r"([0-9.]+)\s*a\s*fla",
    ],
    "power_factor_fl": [
        r"power\s*factor[:\s=]+([0-9.]+)",
        r"pf[:\s=]+([0-9.]+)",
        r"cos\s*phi[:\s=]+([0-9.]+)",
    ],
    "n_circuits": [
        r"([0-9])\s*circuit",
        r"([0-9])\s*compressor",
    ],
    "commissioning_power": [
        r"commissioning\s*power[:\s=]+([0-9.]+)\s*kw",
        r"baseline\s*power[:\s=]+([0-9.]+)\s*kw",
        r"reference\s*power[:\s=]+([0-9.]+)\s*kw",
    ],
}

_IE_KEYWORDS = {
    "IE4": ["ie4", "premium efficiency plus", "ie 4"],
    "IE3": ["ie3", "premium efficiency", "ie 3"],
    "IE2": ["ie2", "high efficiency", "ie 2"],
    "IE1": ["ie1", "ie 1"],
}

_COMP_KEYWORDS = {
    "centrifugal": ["centrifugal", "turbo", "turbocor"],
    "screw":       ["screw", "twin screw", "helical"],
    "scroll":      ["scroll"],
    "reciprocating": ["reciprocating", "piston", "recip"],
}

_DRIVE_KEYWORDS = {
    "VFD":    ["vfd", "inverter", "variable speed", "variable frequency", "vsd", "ec drive"],
    "direct": ["direct", "fixed speed", "dol", "star delta"],
}


def parse_nameplate(desc: str) -> dict:
    """
    Extract nameplate values from a free-text machine description string.

    Handles the structured format written by the platform's nameplate form
    as well as free-text entries.  Returns a dict of extracted values;
    missing values are set to their defaults from _NP_DEFAULTS.

    Parameters
    ----------
    desc : str
        The machine description text stored in the database.

    Returns
    -------
    dict with all keys from _NP_DEFAULTS, populated where found.
    """
    np_d = dict(_NP_DEFAULTS)
    if not desc:
        return np_d

    text = desc.lower()

    # ── Numeric fields ────────────────────────────────────────────────────────
    for field, patterns in _NP_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                val = float(m.group(1))
                # Convert tons to kW if matched a tonnage pattern
                if field == "rated_cooling_kw" and ("tr" in pat or "ton" in pat):
                    val = val * 3.517
                np_d[field] = val
                break   # first match wins per field

    # Power factor sanity: expressed as percentage? → convert
    if np_d["power_factor_fl"] and np_d["power_factor_fl"] > 1.0:
        np_d["power_factor_fl"] = np_d["power_factor_fl"] / 100.0

    # ── IE class ──────────────────────────────────────────────────────────────
    for cls, kws in _IE_KEYWORDS.items():
        if any(k in text for k in kws):
            np_d["ie_class"] = cls
            break

    # ── Compressor type ───────────────────────────────────────────────────────
    for ctype, kws in _COMP_KEYWORDS.items():
        if any(k in text for k in kws):
            np_d["compressor_type"] = ctype
            break

    # ── Drive type ────────────────────────────────────────────────────────────
    for dtype, kws in _DRIVE_KEYWORDS.items():
        if any(k in text for k in kws):
            np_d["drive_type"] = dtype
            break

    # ── Refrigerant ───────────────────────────────────────────────────────────
    m = re.search(r"\b(r[\-]?[0-9]{2,4}[a-z]?)\b", text)
    if m:
        np_d["refrigerant"] = m.group(1).upper()

    # ── Derive COP from rated power + cooling capacity if not explicit ────────
    if (np_d["rated_cop"] is None
            and np_d["rated_cooling_kw"] is not None
            and np_d["rated_power_kw"] is not None
            and np_d["rated_power_kw"] > 0):
        np_d["rated_cop"] = round(np_d["rated_cooling_kw"] / np_d["rated_power_kw"], 3)

    return np_d


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RUNNING MASK
# ═══════════════════════════════════════════════════════════════════════════════

def _running_mask(
    power: pd.Series,
    rated_power_kw: Optional[float],
    threshold_pct: float = 10.0,
    exclude_transient_intervals: int = 1,
) -> pd.Series:
    """
    Return a boolean Series marking intervals where the compressor is running.

    Rules (from the manual):
      1. An interval is RUNNING if power > threshold_pct% of rated power.
         If rated_power_kw is unknown, a data-driven threshold of 10% of the
         95th percentile of the power series is used instead.
      2. The first exclude_transient_intervals interval(s) after a OFF→ON
         transition are marked TRANSIENT (excluded).
      3. The last exclude_transient_intervals interval(s) before an ON→OFF
         transition are marked TRANSIENT (excluded).

    Parameters
    ----------
    power : pd.Series
        Active power values (kW).  NaN treated as off.
    rated_power_kw : float or None
        Nameplate rated power input.  If None, data-driven threshold is used.
    threshold_pct : float
        Running threshold as % of rated power (default 10%).
    exclude_transient_intervals : int
        Number of intervals to exclude at each start/stop transition.

    Returns
    -------
    pd.Series[bool]
        True = running and steady-state (safe to use in efficiency calcs).
    """
    p = power.fillna(0.0)

    # ── Determine threshold ───────────────────────────────────────────────────
    if rated_power_kw is not None and rated_power_kw > 0:
        threshold = rated_power_kw * threshold_pct / 100.0
    else:
        # Data-driven: 10% of 95th percentile
        p95 = float(np.nanpercentile(p.values, 95))
        threshold = p95 * threshold_pct / 100.0 if p95 > 0 else 0.5

    # ── Raw on/off ────────────────────────────────────────────────────────────
    raw_on = p > threshold

    # ── Identify transition points ────────────────────────────────────────────
    transitions = raw_on.astype(int).diff().fillna(0)
    starts = transitions[transitions == 1].index   # OFF→ON
    stops  = transitions[transitions == -1].index  # ON→OFF

    transient = pd.Series(False, index=p.index)

    n = exclude_transient_intervals
    all_idx = p.index.tolist()
    pos_map = {idx: i for i, idx in enumerate(all_idx)}

    for ts in starts:
        pos = pos_map[ts]
        for k in range(n):
            if pos + k < len(all_idx):
                transient.iloc[pos + k] = True

    for ts in stops:
        pos = pos_map[ts]
        for k in range(n):
            if pos - k - 1 >= 0:
                transient.iloc[pos - k - 1] = True

    running = raw_on & ~transient
    return running


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3B — SHUTDOWN PERIOD DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_shutdown_periods(
    power: pd.Series,
    running: pd.Series,
    rated_power_kw: Optional[float],
    voltages: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Identify and characterise standby / shutdown periods.

    A shutdown period is defined as any contiguous block of intervals where
    the compressor is not running (power below threshold) but the supply
    voltage is present (voltages at normal level).  This is a valid operating
    state — not a data quality issue — and must be excluded from all
    efficiency metric calculations.

    Parameters
    ----------
    power : pd.Series
        Power signal (kW).
    running : pd.Series[bool]
        Running mask from _running_mask().  Shutdown = ~running.
    rated_power_kw : float or None
        Used for context in the summary.
    voltages : pd.DataFrame or None
        DataFrame of voltage columns (V_a, V_b, V_c).  If provided, used to
        confirm supply is energised during zero-kW periods.

    Returns
    -------
    dict with:
      n_shutdown_intervals   : int
      pct_shutdown           : float  — % of total intervals that are shutdown
      shutdown_kwh_avoided   : float  — always 0 (no energy during shutdown)
      longest_shutdown_h     : float  — duration of longest single shutdown block
      n_shutdown_blocks      : int    — number of separate shutdown episodes
      voltage_confirmed      : bool   — True if voltages confirmed energised
      info_messages          : list[str]  — human-readable messages for UI
    """
    standby = ~running
    n_total    = len(power)
    n_standby  = int(standby.sum())
    pct        = float(n_standby / n_total * 100) if n_total > 0 else 0.0

    # Identify contiguous shutdown blocks
    blocks = []
    in_block = False
    start_idx = None
    for i, (ts, is_standby) in enumerate(standby.items()):
        if is_standby and not in_block:
            in_block = True
            start_idx = ts
        elif not is_standby and in_block:
            in_block = False
            blocks.append((start_idx, ts))
    if in_block:
        blocks.append((start_idx, power.index[-1]))

    # Duration of each block
    durations_h = []
    for start, end in blocks:
        try:
            dt = (end - start).total_seconds() / 3600.0
            durations_h.append(dt)
        except Exception:
            pass

    longest_h = float(max(durations_h)) if durations_h else 0.0

    # Check if voltages are energised during standby periods
    voltage_confirmed = False
    if voltages is not None and not voltages.empty:
        # For any voltage column, check median during standby vs overall median
        for col in voltages.columns:
            v = pd.to_numeric(voltages[col], errors="coerce")
            overall_med = float(v.median())
            if overall_med > 10:   # clearly a voltage column (>10 V median)
                standby_v = v[standby].median()
                if pd.notna(standby_v) and standby_v > 0.5 * overall_med:
                    voltage_confirmed = True
                    break

    # Build informational messages
    msgs = []
    if n_standby == 0:
        msgs.append("No shutdown periods detected — chiller ran continuously throughout the dataset.")
    else:
        msgs.append(
            f"Shutdown / standby periods detected: {n_standby:,} intervals "
            f"({pct:.1f}% of dataset, {len(blocks)} separate episode(s)). "
            f"Longest single shutdown: {longest_h:.1f} hours."
        )
        if voltage_confirmed:
            msgs.append(
                "Supply voltage confirmed present during all shutdown periods — "
                "these are valid scheduled shutdowns, not power failures or data errors."
            )
        msgs.append(
            "All shutdown intervals are excluded from efficiency metric calculations "
            "(load factor, SPI, power factor). Only running intervals are used. "
            "Energy totals include zero-kW shutdown periods as this represents true site consumption."
        )
        if pct > 50:
            msgs.append(
                f"Note: the chiller is in standby for {pct:.0f}% of the dataset. "
                "This may indicate seasonal operation, demand response, or an oversized unit. "
                "Consider filtering the dataset to the operational season for trend analysis."
            )

    return {
        "n_shutdown_intervals": n_standby,
        "pct_shutdown":         pct,
        "n_running_intervals":  int(running.sum()),
        "pct_running":          float(running.mean() * 100),
        "longest_shutdown_h":   longest_h,
        "n_shutdown_blocks":    len(blocks),
        "voltage_confirmed":    voltage_confirmed,
        "info_messages":        msgs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — POWER DERIVATION
# ═══════════════════════════════════════════════════════════════════════════════

def _derive_power(df: pd.DataFrame, cols: dict, np_d: dict) -> pd.Series:
    """
    Return a power series (kW) using the best available signal.

    Priority:
      1. Direct power column (kW)
      2. Derived from energy column via first-difference
      3. Derived from 3-phase currents + voltage + power factor
         P = √3 × V_avg × I_avg × PF / 1000
      4. Derived from single average current column
         P = √3 × V_rated × I_avg × PF / 1000

    Returns a pd.Series of power in kW, NaN where not computable.
    """
    # ── 1. Direct power column ────────────────────────────────────────────────
    if cols["power"]:
        return pd.to_numeric(df[cols["power"]], errors="coerce")

    # ── 2. Energy accumulator → first difference ──────────────────────────────
    if cols["energy"]:
        e = pd.to_numeric(df[cols["energy"]], errors="coerce")
        # Detect interval in hours
        if hasattr(df.index, "freq") and df.index.freq is not None:
            dt_h = pd.tseries.frequencies.to_offset(df.index.freq).nanos / 3.6e12
        else:
            dt_h = df.index.to_series().diff().dt.total_seconds().median() / 3600.0
        if dt_h and dt_h > 0:
            pwr = e.diff() / dt_h  # kWh / h = kW
            pwr = pwr.clip(lower=0)  # diff can be negative on meter rollover
            return pwr

    # ── 3. Three-phase currents + voltage ────────────────────────────────────
    pf = np_d.get("power_factor_fl") or 0.85
    v_rated = np_d.get("rated_voltage_v") or 415.0

    if cols["curr_a"] and cols["curr_b"] and cols["curr_c"]:
        ia = pd.to_numeric(df[cols["curr_a"]], errors="coerce")
        ib = pd.to_numeric(df[cols["curr_b"]], errors="coerce")
        ic = pd.to_numeric(df[cols["curr_c"]], errors="coerce")
        i_avg = (ia + ib + ic) / 3.0

        if cols["volt_a"] and cols["volt_b"] and cols["volt_c"]:
            va = pd.to_numeric(df[cols["volt_a"]], errors="coerce")
            vb = pd.to_numeric(df[cols["volt_b"]], errors="coerce")
            vc = pd.to_numeric(df[cols["volt_c"]], errors="coerce")
            v_avg = (va + vb + vc) / 3.0
        else:
            v_avg = pd.Series(v_rated, index=df.index)

        return np.sqrt(3) * v_avg * i_avg * pf / 1000.0

    # ── 4. Single average current column ────────────────────────────────────
    if cols["curr_avg"]:
        i = pd.to_numeric(df[cols["curr_avg"]], errors="coerce")
        return np.sqrt(3) * v_rated * i * pf / 1000.0

    return pd.Series(np.nan, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — METRIC CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _load_factor(power_running: pd.Series, rated_power_kw: Optional[float]) -> dict:
    """
    Calculate load factor statistics from running-interval power values.

    Returns dict with: mean, median, p25, p75, p90, pct_above_100,
    pct_below_30, pct_30_to_50 — all as floats (0–100 percent).
    None if rated_power_kw is not available.
    """
    if rated_power_kw is None or rated_power_kw <= 0:
        return {
            "available": False,
            "reason": (
                "Load factor requires rated_power_kw from nameplate. "
                "Please enter the rated power input (kW) in the machine "
                "specifications to enable this metric."
            ),
        }
    lf = (power_running / rated_power_kw * 100.0).dropna()
    if len(lf) < 5:
        return {"available": False, "reason": "Insufficient running intervals."}
    return {
        "available":      True,
        "mean":           float(lf.mean()),
        "median":         float(lf.median()),
        "p25":            float(lf.quantile(0.25)),
        "p75":            float(lf.quantile(0.75)),
        "p90":            float(lf.quantile(0.90)),
        "pct_above_100":  float((lf > 100).mean() * 100),
        "pct_below_30":   float((lf < 30).mean() * 100),
        "pct_30_to_50":   float(((lf >= 30) & (lf < 50)).mean() * 100),
    }


def _spi(power_running: pd.Series, rated_cooling_kw: Optional[float]) -> dict:
    """
    Calculate Specific Power Index (kW input / kW rated cooling capacity).

    Returns dict with mean SPI, trend statistics, and availability flag.
    Requires rated_cooling_kw from nameplate.
    """
    if rated_cooling_kw is None or rated_cooling_kw <= 0:
        return {
            "available": False,
            "reason": (
                "SPI requires rated_cooling_kw from nameplate. "
                "Please enter the rated cooling capacity (kW or tons) in the "
                "machine specifications to enable this metric."
            ),
        }
    spi = (power_running / rated_cooling_kw).dropna()
    if len(spi) < 5:
        return {"available": False, "reason": "Insufficient running intervals."}
    return {
        "available":  True,
        "mean":       float(spi.mean()),
        "median":     float(spi.median()),
        "p10":        float(spi.quantile(0.10)),
        "p90":        float(spi.quantile(0.90)),
        "series":     spi,
    }


def _voltage_imbalance(df: pd.DataFrame, cols: dict) -> dict:
    """
    Calculate per-interval voltage imbalance (%) per IEC 60034-26.

    Formula: max |V_i - V_mean| / V_mean × 100

    Returns dict with mean, max, worst_phase, and per-phase deviations.
    Returns available=False if any voltage column is missing.
    """
    if not all(cols[r] for r in ("volt_a", "volt_b", "volt_c")):
        return {
            "available": False,
            "reason": (
                "Voltage imbalance requires all three phase voltage columns "
                "(V_a, V_b, V_c). "
                f"Detected: V_a={'yes' if cols['volt_a'] else 'NO'}, "
                f"V_b={'yes' if cols['volt_b'] else 'NO'}, "
                f"V_c={'yes' if cols['volt_c'] else 'NO'}. "
                "Add the missing phase voltage column(s) to enable this check."
            ),
        }
    va = pd.to_numeric(df[cols["volt_a"]], errors="coerce")
    vb = pd.to_numeric(df[cols["volt_b"]], errors="coerce")
    vc = pd.to_numeric(df[cols["volt_c"]], errors="coerce")
    v_mean = (va + vb + vc) / 3.0
    safe = v_mean.replace(0, np.nan)

    imbal_a = (va - v_mean).abs() / safe * 100.0
    imbal_b = (vb - v_mean).abs() / safe * 100.0
    imbal_c = (vc - v_mean).abs() / safe * 100.0
    imbal   = pd.concat([imbal_a, imbal_b, imbal_c], axis=1).max(axis=1)

    # Worst phase identification
    worst_map = pd.concat([imbal_a, imbal_b, imbal_c], axis=1)
    worst_map.columns = ["A", "B", "C"]
    worst_phase_series = worst_map.idxmax(axis=1)

    return {
        "available":   True,
        "mean":        float(imbal.mean()),
        "p95":         float(imbal.quantile(0.95)),
        "max":         float(imbal.max()),
        "worst_phase": worst_phase_series.value_counts().idxmax(),
        "phase_dev_v": {
            "A": float((va - v_mean).mean()),
            "B": float((vb - v_mean).mean()),
            "C": float((vc - v_mean).mean()),
        },
        "series": imbal,
    }


def _current_imbalance(df: pd.DataFrame, cols: dict) -> dict:
    """
    Calculate per-interval current imbalance (%) per NEMA MG 1-14.35.

    Formula: max |I_i - I_mean| / I_mean × 100
    """
    if not all(cols[r] for r in ("curr_a", "curr_b", "curr_c")):
        return {
            "available": False,
            "reason": (
                "Current imbalance requires all three phase current columns "
                "(I_a, I_b, I_c). "
                f"Detected: I_a={'yes' if cols['curr_a'] else 'NO'}, "
                f"I_b={'yes' if cols['curr_b'] else 'NO'}, "
                f"I_c={'yes' if cols['curr_c'] else 'NO'}. "
                "Add the missing phase current column(s) to enable this check."
            ),
        }
    ia = pd.to_numeric(df[cols["curr_a"]], errors="coerce")
    ib = pd.to_numeric(df[cols["curr_b"]], errors="coerce")
    ic = pd.to_numeric(df[cols["curr_c"]], errors="coerce")
    i_mean = (ia + ib + ic) / 3.0
    safe = i_mean.replace(0, np.nan)

    imbal_a = (ia - i_mean).abs() / safe * 100.0
    imbal_b = (ib - i_mean).abs() / safe * 100.0
    imbal_c = (ic - i_mean).abs() / safe * 100.0
    imbal   = pd.concat([imbal_a, imbal_b, imbal_c], axis=1).max(axis=1)

    return {
        "available":   True,
        "mean":        float(imbal.mean()),
        "p95":         float(imbal.quantile(0.95)),
        "max":         float(imbal.max()),
        "series":      imbal,
    }


def _power_factor(
    df: pd.DataFrame,
    cols: dict,
    power_running: pd.Series,
    np_d: dict,
) -> dict:
    """
    Calculate power factor = kW / kVA.

    Requires either (a) 3-phase voltage + current columns or (b) rated voltage
    + 3-phase currents.  Falls back to stated nameplate PF if no current data.
    """
    pf_available = (
        all(cols[r] for r in ("curr_a", "curr_b", "curr_c"))
    )
    if not pf_available:
        return {
            "available": False,
            "reason": (
                "Power factor calculation requires three-phase current columns. "
                "Without these, only the nameplate power factor value is available. "
                "Add I_a, I_b, I_c columns to compute measured power factor."
            ),
        }

    ia = pd.to_numeric(df[cols["curr_a"]], errors="coerce")
    ib = pd.to_numeric(df[cols["curr_b"]], errors="coerce")
    ic = pd.to_numeric(df[cols["curr_c"]], errors="coerce")
    i_avg = (ia + ib + ic) / 3.0

    v_rated = np_d.get("rated_voltage_v") or 415.0
    if all(cols[r] for r in ("volt_a", "volt_b", "volt_c")):
        va = pd.to_numeric(df[cols["volt_a"]], errors="coerce")
        vb = pd.to_numeric(df[cols["volt_b"]], errors="coerce")
        vc = pd.to_numeric(df[cols["volt_c"]], errors="coerce")
        v_avg = (va + vb + vc) / 3.0
    else:
        v_avg = pd.Series(v_rated, index=df.index)

    kva = np.sqrt(3) * v_avg * i_avg / 1000.0

    # Only compute for running intervals
    pwr = power_running.reindex(df.index)
    pf  = (pwr / kva).where(kva > 0.1)
    pf  = pf.clip(0, 1.0)

    pf_run = pf[power_running.index.isin(pf.index) if not power_running.empty else pd.Series(dtype=bool)]

    return {
        "available": True,
        "mean":      float(pf.dropna().mean()),
        "p10":       float(pf.dropna().quantile(0.10)),
        "series":    pf,
    }


def _energy_and_pattern(
    df: pd.DataFrame,
    power: pd.Series,
    running: pd.Series,
) -> dict:
    """
    Compute daily energy (kWh), run hours, start count, and mean running kW.

    Handles irregular time intervals automatically.
    """
    if not hasattr(df.index, "floor"):
        return {"available": False, "reason": "No DatetimeIndex."}

    # Interval width in hours
    dt_h = df.index.to_series().diff().dt.total_seconds().bfill() / 3600.0
    dt_h = dt_h.clip(lower=1/3600, upper=24)

    # Per-interval energy
    kwh = (power * dt_h).fillna(0.0)

    # Daily aggregates
    kwh_daily = kwh.groupby(df.index.floor("D")).sum()
    run_h_daily = (
        (running.astype(float) * dt_h)
        .groupby(df.index.floor("D")).sum()
    )

    # Start count per day = number of OFF→ON transitions
    starts = running.astype(int).diff().fillna(0) == 1
    starts_daily = starts.astype(int).groupby(df.index.floor("D")).sum()

    # Mean running power per day
    pwr_run = power.where(running)
    mean_run_daily = pwr_run.groupby(df.index.floor("D")).mean()

    return {
        "available":       True,
        "total_kwh":       float(kwh.sum()),
        "mean_daily_kwh":  float(kwh_daily.mean()),
        "run_hours_total": float((running.astype(float) * dt_h).sum()),
        "mean_run_kw":     float(power.where(running).mean()),
        "mean_starts_day": float(starts_daily.mean()),
        "max_starts_day":  int(starts_daily.max()),
        "kwh_daily":       kwh_daily,
        "run_h_daily":     run_h_daily,
        "starts_daily":    starts_daily,
        "mean_run_daily":  mean_run_daily,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — TEMPERATURE CORRECTION (OPTIONAL)
# ═══════════════════════════════════════════════════════════════════════════════

_TC_MIN_INTERVALS = 200    # minimum running intervals for regression
_TC_MIN_TEMP_RANGE = 5.0   # minimum °C spread required
_TC_MIN_R2 = 0.70          # minimum R² to accept regression


def _fit_alpha(
    metric_running: pd.Series,
    temp_running: pd.Series,
) -> dict:
    """
    Fit OLS regression of metric vs temperature to estimate α.

    Returns dict with alpha, intercept, t_ref, r2, ci_95, n, viable, reason.
    """
    valid = metric_running.notna() & temp_running.notna()
    M = metric_running[valid].values
    T = temp_running[valid].values

    n = len(M)
    t_range = float(T.max() - T.min()) if n > 1 else 0.0

    if n < _TC_MIN_INTERVALS:
        return {
            "viable": False,
            "reason": (
                f"Temperature correction requires at least "
                f"{_TC_MIN_INTERVALS} running intervals with temperature data. "
                f"Only {n} valid intervals found. "
                "Extend the baseline period or provide more data to enable correction."
            ),
        }
    if t_range < _TC_MIN_TEMP_RANGE:
        return {
            "viable": False,
            "reason": (
                f"Temperature range across running intervals is only {t_range:.1f}°C. "
                f"A minimum of {_TC_MIN_TEMP_RANGE}°C spread is required for a reliable "
                "regression. This may indicate a climate-controlled plant room or "
                "insufficient baseline duration. Correction disabled."
            ),
        }

    slope, intercept, r, _, se = stats.linregress(T, M)
    r2 = r ** 2
    from scipy.stats import t as _t_dist
    t_crit = float(_t_dist.ppf(0.975, df=n - 2))
    ci_95 = t_crit * se
    t_ref = float(T.mean())

    if r2 < _TC_MIN_R2:
        return {
            "viable": False,
            "alpha": float(slope),
            "r2": r2,
            "reason": (
                f"R² = {r2:.3f} is below the minimum threshold of {_TC_MIN_R2}. "
                "The linear relationship between this metric and ambient temperature "
                "is too weak for reliable correction. This may indicate good automatic "
                "capacity control, an indoor installation, or insufficient temperature "
                "variation. Correction disabled for this metric."
            ),
        }

    return {
        "viable":    True,
        "alpha":     float(slope),
        "intercept": float(intercept),
        "t_ref":     t_ref,
        "r2":        r2,
        "ci_95":     ci_95,
        "n":         n,
        "t_range":   t_range,
    }


def _apply_correction(
    metric: pd.Series,
    temp: pd.Series,
    alpha_result: dict,
) -> pd.Series:
    """Apply temperature correction: metric_corr = metric - α × (T - T_ref)."""
    if not alpha_result.get("viable"):
        return metric
    alpha = alpha_result["alpha"]
    t_ref = alpha_result["t_ref"]
    return metric - alpha * (temp - t_ref)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ROLLING WINDOW ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def _rolling_windows(
    metric: pd.Series,
    baseline_days: int = 14,
    window_7d: int = 7,
    window_30d: int = 30,
) -> dict:
    """
    Compute baseline mean and rolling window means for a metric series.

    Parameters
    ----------
    metric : pd.Series with DatetimeIndex
        The (optionally temperature-corrected) metric at running intervals.
    baseline_days : int
        Number of days at the start of the series to use as the fixed baseline.
    window_7d, window_30d : int
        Rolling window lengths in days.

    Returns
    -------
    dict with:
      baseline_mean  : float
      roll_7d        : pd.Series
      roll_30d       : pd.Series
      pct_change_7d  : latest 7-day mean vs baseline (%)
      pct_change_30d : latest 30-day mean vs baseline (%)
    """
    if metric.empty or metric.dropna().empty:
        return {"available": False}

    metric = metric.dropna().sort_index()

    # Baseline: first baseline_days days
    baseline_end = metric.index[0] + pd.Timedelta(days=baseline_days)
    baseline_vals = metric[metric.index < baseline_end]
    if len(baseline_vals) < 20:
        # Extend to 30 days if not enough data
        baseline_end = metric.index[0] + pd.Timedelta(days=30)
        baseline_vals = metric[metric.index < baseline_end]

    baseline_mean = float(baseline_vals.mean()) if not baseline_vals.empty else float(metric.mean())

    # Rolling means — require at least 30% of window to have data
    def roll(ser, days):
        freq = pd.tseries.frequencies.to_offset(
            pd.infer_freq(ser.index) or "15min"
        )
        win = f"{days}D"
        return ser.rolling(win, min_periods=max(1, int(days * 24 * 0.3))).mean()

    roll_7  = roll(metric, window_7d)
    roll_30 = roll(metric, window_30d)

    def pct_vs_baseline(val):
        if baseline_mean == 0 or pd.isna(val):
            return None
        return float((val - baseline_mean) / abs(baseline_mean) * 100)

    return {
        "available":       True,
        "baseline_mean":   baseline_mean,
        "baseline_days":   len(baseline_vals),
        "roll_7d":         roll_7,
        "roll_30d":        roll_30,
        "pct_change_7d":   pct_vs_baseline(roll_7.dropna().iloc[-1] if not roll_7.dropna().empty else None),
        "pct_change_30d":  pct_vs_baseline(roll_30.dropna().iloc[-1] if not roll_30.dropna().empty else None),
    }


def _daily_profile(metric: pd.Series) -> dict:
    """
    Compute the mean hourly profile (hour 0–23) across the full dataset
    and per-day mean for comparison.

    Returns dict with:
      hourly_baseline : pd.Series indexed 0–23
      daily_mean      : pd.Series indexed by date
    """
    if metric.empty:
        return {"available": False}
    metric = metric.dropna()
    hourly = metric.groupby(metric.index.hour).mean()
    daily  = metric.resample("D").mean()
    return {
        "available":        True,
        "hourly_baseline":  hourly,
        "daily_mean":       daily,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FINDINGS GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

# Thresholds from Section 3.5 of the manual
_THRESHOLDS = {
    # Voltage imbalance (IE3 motor)
    "volt_imbal_critical":  3.0,   # %
    "volt_imbal_warning":   1.0,   # %
    # Current imbalance (NEMA)
    "curr_imbal_critical":  10.0,  # %
    "curr_imbal_warning":    5.0,  # %
    # Load factor
    "lf_high_critical":    105.0,  # %
    "lf_high_warning":      90.0,  # %
    "lf_low_critical":      20.0,  # %
    "lf_low_warning":       30.0,  # %
    # SPI trend vs baseline
    "spi_trend_critical":   15.0,  # % rise
    "spi_trend_warning":     7.0,  # % rise
    "spi_trend_info":        3.0,  # % rise
    # Power factor
    "pf_critical":           0.70,
    "pf_warning":            0.80,
    "pf_info":               0.88,
    # Start count per day
    "starts_critical":       12,
    "starts_warning":         6,
    "starts_info":            4,
    # IEC voltage imbalance limits by IE class
    "volt_imbal_ie_limit": {"IE4": 0.5, "IE3": 1.0, "IE2": 2.0, "IE1": 3.0},
}


def _make_finding(
    metric: str,
    severity: str,   # "critical" | "warning" | "info"
    value: float,
    threshold: float,
    description: str,
    recommendation: str,
) -> dict:
    return dict(
        metric=metric,
        severity=severity,
        value=round(value, 4),
        threshold=round(threshold, 4),
        description=description,
        recommendation=recommendation,
    )


def _findings_voltage_imbalance(vi: dict, ie_class: str) -> list:
    if not vi.get("available"):
        return []
    findings = []
    val = vi["mean"]
    limit = _THRESHOLDS["volt_imbal_ie_limit"].get(ie_class, 1.0)
    worst = vi.get("worst_phase", "unknown")
    dev = vi.get("phase_dev_v", {})
    dev_str = ", ".join(f"Phase {ph}: {v:+.1f} V" for ph, v in dev.items())

    if val >= _THRESHOLDS["volt_imbal_critical"]:
        findings.append(_make_finding(
            "Voltage Imbalance", "critical", val,
            _THRESHOLDS["volt_imbal_critical"],
            f"Mean voltage imbalance {val:.2f}% — severely exceeds IEC limit. "
            f"Worst phase: {worst}. Deviations: {dev_str}.",
            "Immediate investigation of supply voltage. Do not run at full load. "
            "Contact utility if persistent.",
        ))
    elif val >= _THRESHOLDS["volt_imbal_warning"] or val >= limit:
        findings.append(_make_finding(
            "Voltage Imbalance", "warning", val, limit,
            f"Mean voltage imbalance {val:.2f}% exceeds {ie_class} limit "
            f"({limit:.1f}%). Worst phase: {worst}. Deviations: {dev_str}.",
            "Investigate supply quality. Check transformer taps and load balance "
            "across phases. Voltage imbalance causes motor derating and winding stress.",
        ))
    return findings


def _findings_current_imbalance(ci: dict) -> list:
    if not ci.get("available"):
        return []
    findings = []
    val = ci["mean"]
    if val >= _THRESHOLDS["curr_imbal_critical"]:
        findings.append(_make_finding(
            "Current Imbalance", "critical", val,
            _THRESHOLDS["curr_imbal_critical"],
            f"Mean current imbalance {val:.1f}% exceeds NEMA limit of 10%. "
            "Motor derating required.",
            "Urgent investigation. Check supply voltage imbalance first. "
            "If voltage is balanced, suspect winding fault or connection issue.",
        ))
    elif val >= _THRESHOLDS["curr_imbal_warning"]:
        findings.append(_make_finding(
            "Current Imbalance", "warning", val,
            _THRESHOLDS["curr_imbal_warning"],
            f"Mean current imbalance {val:.1f}% is elevated (NEMA limit: 10%).",
            "Investigate supply phase balance. Compare with voltage imbalance. "
            "Monitor for progression.",
        ))
    return findings


def _findings_load_factor(lf: dict) -> list:
    if not lf.get("available"):
        return []
    findings = []
    val = lf["mean"]
    if val > _THRESHOLDS["lf_high_critical"]:
        findings.append(_make_finding(
            "Load Factor", "critical", val,
            _THRESHOLDS["lf_high_critical"],
            f"Mean load factor {val:.1f}% — chiller is overloaded relative to nameplate.",
            "Investigate root cause: high ambient, condenser fouling, or load growth. "
            "Avoid sustained operation above 105% without consulting manufacturer.",
        ))
    elif val > _THRESHOLDS["lf_high_warning"]:
        findings.append(_make_finding(
            "Load Factor", "warning", val,
            _THRESHOLDS["lf_high_warning"],
            f"Mean load factor {val:.1f}% — chiller is near or at rated capacity.",
            "Monitor closely. Check condenser conditions and ambient temperature. "
            "Consider peak load management.",
        ))
    if val < _THRESHOLDS["lf_low_critical"]:
        findings.append(_make_finding(
            "Load Factor", "warning", val,
            _THRESHOLDS["lf_low_critical"],
            f"Mean load factor {val:.1f}% — chiller is severely underloaded.",
            "Check for short cycling (see start count finding). Consider oversizing "
            "assessment. Evaluate staging or setpoint optimisation.",
        ))
    elif val < _THRESHOLDS["lf_low_warning"]:
        findings.append(_make_finding(
            "Load Factor", "info", val,
            _THRESHOLDS["lf_low_warning"],
            f"Mean load factor {val:.1f}% — chiller is operating at low part load.",
            "Compare to IPLV rating. Evaluate whether a smaller unit or staging "
            "would improve seasonal efficiency.",
        ))
    return findings


def _findings_spi_trend(windows: dict, rated_cop: Optional[float]) -> list:
    if not windows.get("available"):
        return []
    findings = []
    pct_7  = windows.get("pct_change_7d")
    pct_30 = windows.get("pct_change_30d")

    for pct, label in [(pct_30, "30-day trend"), (pct_7, "7-day change")]:
        if pct is None:
            continue
        if pct >= _THRESHOLDS["spi_trend_critical"]:
            findings.append(_make_finding(
                f"SPI ({label})", "critical", pct,
                _THRESHOLDS["spi_trend_critical"],
                f"SPI has risen {pct:.1f}% vs baseline — critical degradation signal.",
                "Inspect condenser and evaporator condition. Check refrigerant charge. "
                "Schedule maintenance. Review recent operational changes.",
            ))
        elif pct >= _THRESHOLDS["spi_trend_warning"]:
            findings.append(_make_finding(
                f"SPI ({label})", "warning", pct,
                _THRESHOLDS["spi_trend_warning"],
                f"SPI has risen {pct:.1f}% vs baseline — early degradation signal.",
                "Monitor closely. Consider proactive condenser inspection. "
                "Track trend rate to schedule maintenance before critical threshold.",
            ))
        elif pct >= _THRESHOLDS["spi_trend_info"]:
            findings.append(_make_finding(
                f"SPI ({label})", "info", pct,
                _THRESHOLDS["spi_trend_info"],
                f"SPI has risen {pct:.1f}% vs baseline — marginal change, within noise.",
                "Continue monitoring. No immediate action required.",
            ))
    return findings


def _findings_power_factor(pf: dict) -> list:
    if not pf.get("available"):
        return []
    findings = []
    val = pf["mean"]
    if val < _THRESHOLDS["pf_critical"]:
        findings.append(_make_finding(
            "Power Factor", "critical", val,
            _THRESHOLDS["pf_critical"],
            f"Mean power factor {val:.3f} is very poor.",
            "Investigate capacitor bank condition and winding insulation. "
            "Check for severe underloading. Utility penalty likely.",
        ))
    elif val < _THRESHOLDS["pf_warning"]:
        findings.append(_make_finding(
            "Power Factor", "warning", val,
            _THRESHOLDS["pf_warning"],
            f"Mean power factor {val:.3f} is below recommended level.",
            "Inspect capacitor bank. Evaluate reactive power compensation. "
            "Check for persistent underloading.",
        ))
    return findings


def _findings_short_cycling(pattern: dict) -> list:
    if not pattern.get("available"):
        return []
    findings = []
    val = pattern["mean_starts_day"]
    max_val = pattern["max_starts_day"]
    if val >= _THRESHOLDS["starts_critical"]:
        findings.append(_make_finding(
            "Start Count", "critical", val,
            _THRESHOLDS["starts_critical"],
            f"Mean {val:.1f} starts/day (max {max_val}) — severe short cycling.",
            "Investigate control system and BMS setpoints. Check chilled water "
            "buffer volume. Short cycling causes motor overheating and contact wear.",
        ))
    elif val >= _THRESHOLDS["starts_warning"]:
        findings.append(_make_finding(
            "Start Count", "warning", val,
            _THRESHOLDS["starts_warning"],
            f"Mean {val:.1f} starts/day (max {max_val}) — elevated cycling frequency.",
            "Review deadband settings. Consider increasing buffer tank volume or "
            "adjusting setpoint differential.",
        ))
    return findings


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase1(
    df: pd.DataFrame,
    np_data: Optional[dict] = None,
    description: str = "",
    temp_correction_enabled: bool = False,
    daily_temp_profile: Optional[list] = None,
    baseline_days: int = 14,
    running_threshold_pct: float = 10.0,
) -> dict:
    """
    Run all Phase 1 calculations on a chiller dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Sensor data with DatetimeIndex (or 'timestamp' column which will be
        set as index).  Must contain at least one power-related column.

    np_data : dict or None
        Nameplate values from parse_nameplate() or the UI form.
        If None, parse_nameplate(description) is called to attempt extraction.
        Keys:
          rated_cooling_kw   : float — rated cooling output in kW
          rated_power_kw     : float — rated electrical input in kW
          rated_cop          : float — rated COP (dimensionless)
          rated_voltage_v    : float — line-to-line supply voltage (V)
          fla_a              : float — full load amps (A)
          power_factor_fl    : float — power factor at full load (0–1)
          ie_class           : str   — "IE4", "IE3", "IE2", or "IE1"
          n_circuits         : int   — number of refrigerant circuits
          compressor_type    : str   — centrifugal/screw/scroll/reciprocating
          drive_type         : str   — direct/VFD
          commissioning_power: float — measured kW at first stable run

    description : str
        Machine description text (used if np_data is None).

    temp_correction_enabled : bool
        If True and an ambient temperature column is present (or a daily
        profile is supplied), fit α and apply correction to SPI and LF trends.
        Default False — correction is optional.

    daily_temp_profile : list[float] or None
        24 values (index 0 = 00:00, index 23 = 23:00) representing a typical
        hourly temperature profile (°C) to use when no measured temperature
        column is available.  Used only if temp_correction_enabled=True.

    baseline_days : int
        Number of days at the start of the data to use as the fixed baseline
        reference period (default 14).  Minimum effective is 14 days.

    running_threshold_pct : float
        Percentage of rated power below which an interval is classified as
        standby.  Default 10%.  Raise this for machines with high auxiliary
        draws (e.g. large oil-heater screw chillers).

    Returns
    -------
    dict with keys:
      phase_info         : dict   — detected columns, viability, data summary
      nameplate          : dict   — parsed nameplate values used
      running_mask       : pd.Series[bool]
      power              : pd.Series  — derived power (kW)
      metrics            : dict   — load_factor, spi, volt_imbal, curr_imbal,
                                    power_factor, energy_pattern
      windows            : dict   — baseline, roll_7d, roll_30d, daily_profile
      temp_correction    : dict   — alpha result per metric (or None)
      findings           : list[dict] — all findings sorted by severity
      summary            : str    — plain-text summary for Claude context
      data_coverage      : dict   — date range, n_intervals, n_running
    """
    # ── 0. Prepare index ──────────────────────────────────────────────────────
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"]))
        else:
            df = df.copy()
            df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    # ── 1. Detect columns and viability ───────────────────────────────────────
    phase_info = detect_phase_1(df)
    if not phase_info["viable"]:
        return {
            "phase_info": phase_info,
            "findings": [],
            "summary": f"Phase 1 not viable: {phase_info['missing_msg']}",
            "nameplate": {},
        }

    cols = phase_info["cols"]

    # ── 2. Nameplate ──────────────────────────────────────────────────────────
    if np_data is None:
        np_data = parse_nameplate(description)

    # ── 3. Derive power series ────────────────────────────────────────────────
    power = _derive_power(df, cols, np_data)

    # ── 4. Running mask ───────────────────────────────────────────────────────
    running = _running_mask(
        power,
        rated_power_kw=np_data.get("rated_power_kw"),
        threshold_pct=running_threshold_pct,
    )
    power_running = power.where(running)

    # ── 4b. Detect and characterise shutdown periods ──────────────────────────
    # Build voltage DataFrame if phase voltage columns are available
    _v_cols = {r: cols[r] for r in ("volt_a", "volt_b", "volt_c") if cols.get(r)}
    _volt_df = df[[c for c in _v_cols.values() if c]] if _v_cols else None
    shutdown_info = _detect_shutdown_periods(
        power=power,
        running=running,
        rated_power_kw=np_data.get("rated_power_kw"),
        voltages=_volt_df,
    )

    # ── 5. Ambient temperature series ────────────────────────────────────────
    temp: Optional[pd.Series] = None
    if cols["amb_temp"]:
        temp = pd.to_numeric(df[cols["amb_temp"]], errors="coerce")
    elif daily_temp_profile is not None and len(daily_temp_profile) == 24:
        # Assign profile to every day
        temp = pd.Series(
            [daily_temp_profile[h % 24] for h in df.index.hour],
            index=df.index,
        )

    # ── 6. Compute all metrics ────────────────────────────────────────────────
    lf      = _load_factor(power_running.dropna(), np_data.get("rated_power_kw"))
    spi_raw = _spi(power_running.dropna(), np_data.get("rated_cooling_kw"))
    vi      = _voltage_imbalance(df, cols)
    ci      = _current_imbalance(df, cols)
    pf      = _power_factor(df, cols, power_running, np_data)
    pattern = _energy_and_pattern(df, power, running)

    # ── 7. Temperature correction (optional) ─────────────────────────────────
    tc_results: dict = {}
    spi_series_for_windows = spi_raw.get("series") if spi_raw.get("available") else None

    if temp_correction_enabled and temp is not None and spi_series_for_windows is not None:
        temp_running = temp.where(running).dropna()
        spi_running  = spi_series_for_windows

        # Align
        common = spi_running.index.intersection(temp_running.index)
        alpha_spi = _fit_alpha(spi_running.loc[common], temp_running.loc[common])
        tc_results["SPI"] = alpha_spi

        if alpha_spi.get("viable"):
            spi_corr = _apply_correction(spi_running, temp.reindex(spi_running.index), alpha_spi)
            spi_series_for_windows = spi_corr

    # ── 8. Rolling windows ─────────────────────────────────────────────────────
    # Phase 1 note: SPI = Actual Power ÷ Rated Cooling Capacity and
    # Load Factor = Actual Power ÷ Rated Power Input are perfectly correlated
    # (SPI = LF / Rated COP). They carry identical trend information in Phase 1.
    # Only Load Factor is used for window analysis here. SPI is retained as a
    # reference value (kW/kW and kW/TR) for cross-site comparison and industry
    # benchmarking, but NOT run through a separate rolling window.
    # SPI becomes an independent metric in Phase 2 when actual cooling output
    # replaces rated cooling capacity in the denominator.
    windows = {}
    if lf.get("available"):
        lf_series = (power_running / np_data["rated_power_kw"] * 100.0).dropna()
        windows["LF"] = _rolling_windows(lf_series, baseline_days=baseline_days)
        windows["LF"]["daily_profile"] = _daily_profile(lf_series)
        # Attach SPI reference values to the LF window output for reporting
        # (computed from LF using the rated COP relationship)
        if spi_raw.get("available") and np_data.get("rated_cop"):
            windows["LF"]["spi_mean_kw_kw"]  = spi_raw.get("mean")
            windows["LF"]["spi_mean_kw_tr"]  = (
                spi_raw["mean"] / 3.51685 if spi_raw.get("mean") else None
            )
    elif spi_series_for_windows is not None:
        # LF not available (no rated_power_kw) but SPI is — use SPI as fallback
        windows["SPI_fallback"] = _rolling_windows(
            spi_series_for_windows,
            baseline_days=baseline_days,
        )
        windows["SPI_fallback"]["daily_profile"] = _daily_profile(spi_series_for_windows)

    # ── 9. Generate findings ──────────────────────────────────────────────────
    ie_class = np_data.get("ie_class", "IE3")
    # Note: SPI trend findings are removed in Phase 1 — SPI and Load Factor
    # are perfectly correlated (SPI = LF / rated COP). Only LF window
    # findings are generated. SPI findings activate in Phase 2+ when actual
    # cooling output makes SPI independent of Load Factor.
    all_findings = (
        _findings_voltage_imbalance(vi, ie_class)
        + _findings_current_imbalance(ci)
        + _findings_load_factor(lf)
        + _findings_power_factor(pf)
        + _findings_short_cycling(pattern)
    )

    # Sort: critical first, then warning, then info
    _sev_order = {"critical": 0, "warning": 1, "info": 2}
    all_findings.sort(key=lambda f: _sev_order.get(f["severity"], 3))

    # ── 10. Plain-text summary for Claude context ─────────────────────────────
    summary_lines = [
        f"Chiller Phase 1 analysis: "
        f"{running.sum()} running intervals of {len(df)} total "
        f"({running.mean()*100:.0f}% utilisation).",
    ]
    # Always include shutdown period summary
    for _msg in shutdown_info.get("info_messages", []):
        summary_lines.append(_msg)

    if lf.get("available"):
        summary_lines.append(
            f"Load factor: mean {lf['mean']:.1f}%, "
            f"P25={lf['p25']:.0f}%, P75={lf['p75']:.0f}%."
        )
    if spi_raw.get("available"):
        spi_kw_tr = spi_raw['mean'] / 3.51685
        summary_lines.append(
            f"SPI (reference): {spi_raw['mean']:.4f} kW/kW "
            f"({spi_kw_tr:.4f} kW/TR). "
            f"Note: in Phase 1, SPI and Load Factor carry identical trend "
            f"information. SPI is reported for benchmarking only."
        )
    if vi.get("available"):
        summary_lines.append(
            f"Voltage imbalance: mean {vi['mean']:.2f}%, max {vi['max']:.2f}%."
        )
    if ci.get("available"):
        summary_lines.append(
            f"Current imbalance: mean {ci['mean']:.1f}%, max {ci['max']:.1f}%."
        )
    if pf.get("available"):
        summary_lines.append(f"Power factor: mean {pf['mean']:.3f}.")
    if pattern.get("available"):
        summary_lines.append(
            f"Energy: {pattern['total_kwh']:.0f} kWh total, "
            f"{pattern['run_hours_total']:.0f} run hours, "
            f"mean {pattern['mean_starts_day']:.1f} starts/day."
        )
    if all_findings:
        crit = sum(1 for f in all_findings if f["severity"] == "critical")
        warn = sum(1 for f in all_findings if f["severity"] == "warning")
        info = sum(1 for f in all_findings if f["severity"] == "info")
        summary_lines.append(
            f"Findings: {crit} critical, {warn} warning, {info} info."
        )
        for f in all_findings[:3]:   # top 3 in summary
            summary_lines.append(
                f"  [{f['severity'].upper()}] {f['metric']}: {f['description']}"
            )
    else:
        summary_lines.append("No findings triggered. Performance within normal limits.")

    # ── 11. Data coverage ─────────────────────────────────────────────────────
    data_coverage = {
        "start":       str(df.index.min().date()),
        "end":         str(df.index.max().date()),
        "n_intervals": len(df),
        "n_running":   int(running.sum()),
        "utilisation_pct": float(running.mean() * 100),
        "interval_minutes": round(
            df.index.to_series().diff().dt.total_seconds().median() / 60, 1
        ),
    }

    return {
        "phase_info":      phase_info,
        "nameplate":       np_data,
        "running_mask":    running,
        "power":           power,
        "shutdown_info":   shutdown_info,
        "metrics": {
            "load_factor":    lf,
            "spi":            spi_raw,
            "volt_imbalance": vi,
            "curr_imbalance": ci,
            "power_factor":   pf,
            "energy_pattern": pattern,
        },
        "windows":         windows,
        "temp_correction": tc_results,
        "findings":        all_findings,
        "summary":         "\n".join(summary_lines),
        "data_coverage":   data_coverage,
    }
