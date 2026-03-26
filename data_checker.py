"""
data_checker.py — Data quality module for Machine Analytics

Runs before any statistical analysis. Detects issues where sensor data
is likely invalid, corrupted, or untrustworthy — NOT performance issues.

Checks (7):
  1. Duplicate timestamps      — same timestamp more than once
  2. Missing data gaps         — timestamp gaps larger than expected interval
  3. High null percentage      — column mostly missing
  4. Physical impossibilities  — values below known physical minimums (e.g. negative pressure)
  5. Flatline / stuck sensor   — rolling std dev near zero over sustained window
  6. Frozen value runs         — N consecutive identical readings
  7. Sudden spike / step       — single-step jump > N x IQR (sensor glitch)

NOT included (these are performance findings, not data quality issues):
  - Gradual mean shift or trend over time → handled by Trend & Drift Analysis
  - Efficiency degradation → handled by physics modules (Phase 2/3)
  - Process upsets → handled by AI anomaly detection

Returns a structured report used by the UI and passed to Claude as context.
"""

from __future__ import annotations
from typing import Optional
import re
import numpy as np
import pandas as pd


# ── Tuneable constants ────────────────────────────────────────────────────────
FLATLINE_WINDOW      = 20      # rolling window size for std dev check
FLATLINE_STD_THRESH  = 1e-6    # std dev below this = flatline
FROZEN_RUN_MIN       = 10      # consecutive identical readings = frozen
STEP_IQR_MULTIPLIER  = 10      # step > N x IQR = spike
NULL_PCT_WARN        = 10.0    # % nulls above which column is flagged
NULL_PCT_CRITICAL    = 30.0    # % nulls above which column is critical
GAP_MULTIPLIER       = 5       # gap > N x median interval = missing data

MIN_ROWS_FOR_CHECKS  = 10      # minimum rows needed to run checks


# ── Severity levels ───────────────────────────────────────────────────────────
INFO     = "info"
WARNING  = "warning"
CRITICAL = "critical"


def check_timestamp_format(raw_file_path: str) -> dict:
    """
    Check timestamp column format before ingestion.
    Returns:
        {
            "parseable":      bool,
            "col":            str,
            "sample_raw":     list,   # first 5 raw values
            "sample_parsed":  list,   # first 5 parsed values
            "n_failed":       int,
            "n_total":        int,
            "suggestion":     str,
            "corrected_df":   pd.DataFrame or None,
        }
    """
    result = {
        "parseable": True, "col": None, "sample_raw": [],
        "sample_parsed": [], "n_failed": 0, "n_total": 0,
        "suggestion": "", "corrected_df": None,
    }
    try:
        ext = str(raw_file_path).lower()
        if ext.endswith((".xlsx", ".xls")):
            df_raw = pd.read_excel(raw_file_path, nrows=5000)
        else:
            df_raw = pd.read_csv(raw_file_path, nrows=5000)

        ts_col = next(
            (c for c in df_raw.columns
             if any(kw in c.lower() for kw in ["time","date","timestamp","ts"])),
            None
        )
        if not ts_col:
            result["suggestion"] = (
                "No timestamp column detected. Rename your date/time column to include "
                "'timestamp', 'time', or 'date' in the name."
            )
            result["parseable"] = False
            return result

        result["col"]        = ts_col
        result["n_total"]    = len(df_raw)
        result["sample_raw"] = df_raw[ts_col].dropna().head(5).astype(str).tolist()

        # Try parsing
        parsed = pd.to_datetime(df_raw[ts_col], dayfirst=True,
                                format="mixed", errors="coerce")
        n_failed = int(parsed.isna().sum() - df_raw[ts_col].isna().sum())
        result["n_failed"]      = max(0, n_failed)
        result["sample_parsed"] = [str(v) for v in parsed.dropna().head(5).tolist()]

        if n_failed == 0:
            result["parseable"]  = True
            result["suggestion"] = ""
            return result

        # Diagnose the format issue
        _sample = result["sample_raw"][0] if result["sample_raw"] else ""
        _suggestion = _diagnose_date_format(_sample, n_failed, result["n_total"])
        result["parseable"]  = False
        result["suggestion"] = _suggestion

        # Attempt auto-correction: try common formats
        _fixed = _try_fix_timestamps(df_raw, ts_col)
        if _fixed is not None:
            result["corrected_df"] = _fixed
            result["suggestion"]  += "  Auto-correction succeeded — corrected file available for download."

    except Exception as e:
        result["parseable"]  = False
        result["suggestion"] = f"Could not read file for timestamp check: {e}"

    return result


def _diagnose_date_format(sample: str, n_failed: int, n_total: int) -> str:
    """Give a specific suggestion based on the sample value."""
    s = sample.strip()

    # Excel serial number
    if s.isdigit() and 30000 < int(s) < 60000:
        return (
            f"{n_failed}/{n_total} timestamps could not be parsed. "
            "Timestamps appear to be Excel serial numbers (e.g. 45123). "
            "In Excel, select the column → Format Cells → Date → choose a date format → Save as CSV."
        )
    # Ambiguous DD/MM vs MM/DD
    parts = s.replace("-","/").replace(".","/").split("/")
    if len(parts) == 3:
        try:
            a, b = int(parts[0]), int(parts[1])
            if a <= 12 and b <= 12:
                return (
                    f"{n_failed}/{n_total} timestamps could not be parsed. "
                    f"Sample: '{sample}'. "
                    "Date is ambiguous — both day-first and month-first interpretations are valid. "
                    "Recommended format: YYYY-MM-DD HH:MM:SS (e.g. 2024-10-01 08:00:00)."
                )
            if a > 12:
                return (
                    f"{n_failed}/{n_total} timestamps could not be parsed. "
                    f"Sample: '{sample}'. Appears to be DD/MM/YYYY format. "
                    "Recommended: convert to YYYY-MM-DD HH:MM:SS for unambiguous parsing."
                )
        except ValueError:
            pass
    # Text month
    if re.search(r"[a-zA-Z]{3}", s):
        return (
            f"{n_failed}/{n_total} timestamps could not be parsed. "
            f"Sample: '{sample}'. "
            "Text month format detected (e.g. 01-Jan-2024). "
            "Recommended: convert to YYYY-MM-DD HH:MM:SS."
        )
    return (
        f"{n_failed}/{n_total} timestamps could not be parsed. "
        f"Sample value: '{sample}'. "
        "Recommended format: YYYY-MM-DD HH:MM:SS (ISO 8601). "
        "This format is always unambiguous and parses reliably."
    )


def _try_fix_timestamps(df: pd.DataFrame, ts_col: str) -> Optional[pd.DataFrame]:
    """Try a series of common formats and return corrected df if successful."""
    FORMATS = [
        "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y",
        "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y",
        "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%d-%m-%Y",
        "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y/%m/%d",
        "%d.%m.%Y %H:%M:%S", "%d.%m.%Y",
        "%d-%b-%Y %H:%M:%S", "%d-%b-%Y",
        "%b %d %Y %H:%M:%S",
    ]
    for fmt in FORMATS:
        try:
            parsed = pd.to_datetime(df[ts_col], format=fmt, errors="coerce")
            n_ok   = parsed.notna().sum()
            n_orig = df[ts_col].notna().sum()
            if n_ok >= n_orig * 0.95:  # 95%+ parsed successfully
                df_out = df.copy()
                df_out[ts_col] = parsed
                return df_out
        except Exception:
            continue
    # Last resort: mixed
    try:
        parsed = pd.to_datetime(df[ts_col], dayfirst=True, format="mixed", errors="coerce")
        if parsed.notna().sum() >= df[ts_col].notna().sum() * 0.95:
            df_out = df.copy()
            df_out[ts_col] = parsed
            return df_out
    except Exception:
        pass
    return None


def run_data_quality_checks(
    df: pd.DataFrame,
    physical_minimums: Optional[dict] = None,
) -> dict:
    """
    Run all data quality checks on a DataFrame with a DatetimeIndex.

    Args:
        df                 : DataFrame with DatetimeIndex and numeric columns
        physical_minimums  : optional dict of {col: min_valid_value}
                             e.g. {"pressure_bar": 0, "current_A": 0}
                             Defaults applied for common parameter patterns.

    Returns:
        {
            "issues":  [ {col, check, severity, detail, affected_rows, affected_pct,
                           start_ts, end_ts, point_ts} ],
            "summary": { "total": int, "critical": int, "warning": int, "info": int },
            "passed":  bool,   # True = no critical issues
            "score":   int,    # 0-100 data quality score
        }
    """
    issues = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(df) < MIN_ROWS_FOR_CHECKS:
        return _empty_report("Insufficient rows for data quality checks")

    # ── Auto physical minimums ────────────────────────────────────────────────
    phys_min = _default_physical_minimums(numeric_cols)
    if physical_minimums:
        phys_min.update(physical_minimums)

    # ── 1. Duplicate timestamps ───────────────────────────────────────────────
    dup_count = df.index.duplicated().sum()
    if dup_count > 0:
        issues.append(_issue(
            col="[timestamp]",
            check="Duplicate timestamps",
            severity=WARNING,
            detail=f"{dup_count} duplicate timestamps found. "
                   f"Data may have been merged incorrectly or logger restarted.",
            affected_rows=int(dup_count),
            total_rows=len(df),
        ))

    # ── 2. Missing data gaps ──────────────────────────────────────────────────
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        diffs       = pd.Series(df.index).diff().dropna()
        median_int  = diffs.median()
        if median_int.total_seconds() > 0:
            gap_thresh  = median_int * GAP_MULTIPLIER
            large_gaps  = diffs[diffs > gap_thresh]
            if len(large_gaps) > 0:
                worst_gap   = large_gaps.max()
                worst_start = df.index[diffs[diffs == worst_gap].index[0] - 1]
                issues.append(_issue(
                    col="[timestamp]",
                    check="Missing data gaps",
                    severity=WARNING,
                    detail=f"{len(large_gaps)} gap(s) larger than "
                           f"{_fmt_duration(gap_thresh)} detected. "
                           f"Largest gap: {_fmt_duration(worst_gap)} "
                           f"starting at {worst_start.strftime('%Y-%m-%d %H:%M')}.",
                    affected_rows=int(len(large_gaps)),
                    total_rows=len(df),
                    start_ts=worst_start,
                    end_ts=worst_start + worst_gap,
                ))

    # ── Pre-compute frozen runs for coincidence check (Check 6) ────────────
    # If multiple columns freeze over the same period, it's a real operating
    # condition (e.g. planned stop, constant load) — not a sensor fault.
    # A genuine stuck sensor freezes alone while others keep varying.
    _all_frozen = {}  # col -> list of run dicts
    for _c in numeric_cols:
        _sc = df[_c].dropna()
        if len(_sc) >= FROZEN_RUN_MIN:
            _all_frozen[_c] = _all_runs_of_identical(_sc, FROZEN_RUN_MIN)

    # ── Pre-compute shutdown mask for Check 5/7 ──────────────────────────────
    # Detect confirmed shutdowns: rows where 2+ numeric columns are
    # simultaneously near-zero (below 5% of column running median).
    # Shutdown transitions cause huge step changes that are normal
    # operating events, not data quality issues.
    _near_zero = pd.DataFrame(index=df.index)
    for _c in numeric_cols:
        _s = df[_c].fillna(0).abs()
        _median = float(_s[_s > 0].median()) if (_s > 0).any() else 1.0
        _near_zero[_c] = _s < 0.05 * _median
    _shutdown_mask = _near_zero.sum(axis=1) >= min(2, len(numeric_cols))

    # ── Extended shutdown detection for chiller/HVAC pattern ─────────────────
    # A chiller shutdown shows kW = 0 while phase voltages remain at rated level.
    # Only the power column is near-zero; voltage columns are NOT near-zero.
    # This does NOT trigger the 2-column rule above, so we add a supplementary
    # check: if a power/kW column is near-zero AND at least one voltage column
    # is clearly non-zero (above 50% of its median), mark those rows as shutdown.
    _pwr_cols  = [c for c in numeric_cols if any(k in c.lower() for k in
                  ["kw", "power", "watt", "kwh", "energy"])]
    _volt_cols = [c for c in numeric_cols if any(k in c.lower() for k in
                  ["volt", "voltage", "_v", "ua", "ub", "uc", "va", "vb", "vc",
                   "u_a", "u_b", "u_c", "v_a", "v_b", "v_c", "l1", "l2", "l3"])]
    if _pwr_cols and _volt_cols:
        _pwr_zero = pd.Series(False, index=df.index)
        for _pc in _pwr_cols:
            _ps = df[_pc].fillna(0).abs()
            _pm = float(_ps[_ps > 0].median()) if (_ps > 0).any() else 1.0
            _pwr_zero |= (_ps < 0.05 * _pm)
        _volt_live = pd.Series(False, index=df.index)
        for _vc in _volt_cols:
            _vs = df[_vc].fillna(0).abs()
            _vm = float(_vs[_vs > 0].median()) if (_vs > 0).any() else 1.0
            _volt_live |= (_vs > 0.50 * _vm)
        # Power zero + voltage live = chiller standby / scheduled shutdown
        _chiller_standby = _pwr_zero & _volt_live
        _shutdown_mask   = _shutdown_mask | _chiller_standby

    # Also mark the first running row after each shutdown block (transition edge)
    _shutdown_edge = _shutdown_mask & ~_shutdown_mask.shift(1, fill_value=False)
    _startup_edge  = ~_shutdown_mask & _shutdown_mask.shift(1, fill_value=False)
    _transition_mask = _shutdown_mask | _startup_edge

    # ── Per-column checks ─────────────────────────────────────────────────────
    for col in numeric_cols:
        s = df[col]

        # ── 3. Null percentage ────────────────────────────────────────────────
        null_pct = s.isna().mean() * 100
        if null_pct >= NULL_PCT_CRITICAL:
            issues.append(_issue(col, "High null percentage", CRITICAL,
                f"{null_pct:.1f}% of readings are missing. "
                f"Column may be non-functional or incorrectly mapped.",
                int(s.isna().sum()), len(s)))
        elif null_pct >= NULL_PCT_WARN:
            issues.append(_issue(col, "Elevated null percentage", WARNING,
                f"{null_pct:.1f}% of readings are missing. "
                f"Check sensor connectivity for this period.",
                int(s.isna().sum()), len(s)))

        s_clean = s.dropna()
        if len(s_clean) < MIN_ROWS_FOR_CHECKS:
            continue

        # ── 4. Physical impossibilities ───────────────────────────────────────
        if col in phys_min:
            min_val     = phys_min[col]
            impossible  = s_clean[s_clean < min_val]
            if len(impossible) > 0:
                issues.append(_issue(col, "Physically impossible values", CRITICAL,
                    f"{len(impossible)} readings below physical minimum of {min_val} "
                    f"(min observed: {impossible.min():.4f}). "
                    f"Sensor fault, wiring issue or unit mismatch suspected.",
                    len(impossible), len(s_clean)))

        # ── 5. Flatline detection (rolling std dev) ───────────────────────────
        if len(s_clean) >= FLATLINE_WINDOW:
            roll_std    = s_clean.rolling(FLATLINE_WINDOW).std()
            flat_mask   = roll_std < FLATLINE_STD_THRESH
            flat_count  = int(flat_mask.sum())
            flat_pct    = flat_count / len(s_clean) * 100
            if flat_pct > 5:
                # Find the first flatline start
                flat_starts = flat_mask[flat_mask].index
                first_flat  = flat_starts[0]
                sev = CRITICAL if flat_pct > 20 else WARNING
                flat_end = flat_starts[-1] if len(flat_starts) > 0 else None
                issues.append(_issue(col, "Flatline / stuck sensor", sev,
                    f"{flat_pct:.1f}% of readings show near-zero variation "
                    f"(rolling std < {FLATLINE_STD_THRESH:.0e} over {FLATLINE_WINDOW} readings). "
                    f"First detected at {_fmt_ts(first_flat)}. "
                    f"Sensor may be stuck, frozen, or disconnected.",
                    flat_count, len(s_clean),
                    start_ts=first_flat, end_ts=flat_end))

        # ── 6. Frozen value runs (consecutive identical readings) ──────────────
        # Only flag if this column freezes ALONE.  If other columns also have
        # a frozen run overlapping the same period, this is a real operating
        # condition (shutdown, constant load, idle) — not a stuck sensor.
        runs = _longest_run_of_identical(s_clean)
        if runs["length"] >= FROZEN_RUN_MIN:
            run_start = runs["start"]
            run_end   = runs["end"]

            # Check coincidence: does ANY other column also freeze in this window?
            coincident = False
            for other_col, other_runs in _all_frozen.items():
                if other_col == col:
                    continue
                for orun in other_runs:
                    # Overlapping if one starts before the other ends and vice versa
                    if orun["start"] <= run_end and orun["end"] >= run_start:
                        coincident = True
                        break
                if coincident:
                    break

            if not coincident:
                sev = CRITICAL if runs["length"] > 50 else WARNING
                issues.append(_issue(col, "Frozen value run", sev,
                    f"{runs['length']} consecutive identical readings of "
                    f"{runs['value']:.4f} starting at {_fmt_ts(runs['start'])}. "
                    f"Sensor output appears frozen \u2014 not a valid process reading.",
                    runs["length"], len(s_clean),
                    start_ts=run_start, end_ts=run_end))

        # ── 7. Spike / step change ────────────────────────────────────────────
        # Exclude shutdown transitions — the step from running to off (and back)
        # is a normal operating event, not a data quality issue.
        s_running = s_clean.copy()
        s_running[_transition_mask.reindex(s_running.index, fill_value=False)] = np.nan
        s_running = s_running.dropna()
        if len(s_running) < MIN_ROWS_FOR_CHECKS:
            continue
        q1, q3  = s_running.quantile(0.25), s_running.quantile(0.75)
        iqr     = q3 - q1
        if iqr > 0:
            step_thresh = iqr * STEP_IQR_MULTIPLIER
            diffs_s     = s_running.diff().abs()
            spikes      = diffs_s[diffs_s > step_thresh]
            if len(spikes) > 0:
                worst_spike = spikes.idxmax()
                sev = CRITICAL if float(spikes.max()) > iqr * 20 else WARNING
                issues.append(_issue(col, "Sudden step change / spike", sev,
                    f"{len(spikes)} step change(s) larger than "
                    f"{STEP_IQR_MULTIPLIER}x IQR ({step_thresh:.4f} units). "
                    f"Largest: {spikes.max():.4f} units at {_fmt_ts(worst_spike)}. "
                    f"May indicate sensor glitch, process upset, or data logging error.",
                    len(spikes), len(s_running),
                    point_ts=worst_spike,
                    start_ts=spikes.index.min() if len(spikes) > 1 else worst_spike,
                    end_ts=spikes.index.max() if len(spikes) > 1 else worst_spike))

    # ── Summary ───────────────────────────────────────────────────────────────
    n_critical = sum(1 for i in issues if i["severity"] == CRITICAL)
    n_warning  = sum(1 for i in issues if i["severity"] == WARNING)
    n_info     = sum(1 for i in issues if i["severity"] == INFO)

    # Score: start at 100, deduct per issue
    score = 100
    score -= n_critical * 25
    score -= n_warning  * 10
    score -= n_info     * 3
    score  = max(0, score)

    return {
        "issues":  issues,
        "summary": {
            "total":    len(issues),
            "critical": n_critical,
            "warning":  n_warning,
            "info":     n_info,
        },
        "passed": n_critical == 0,
        "score":  score,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _issue(col, check, severity, detail, affected_rows, total_rows,
           start_ts=None, end_ts=None, point_ts=None):
    """
    start_ts : start of affected period (pandas Timestamp) — for range issues
    end_ts   : end of affected period — None means open-ended to dataset end
    point_ts : single event timestamp — for spike / step change issues
    """
    return {
        "col":          col,
        "check":        check,
        "severity":     severity,
        "detail":       detail,
        "affected_rows": int(affected_rows),
        "affected_pct": round(affected_rows / total_rows * 100, 1) if total_rows else 0,
        "start_ts":     start_ts,
        "end_ts":       end_ts,
        "point_ts":     point_ts,
    }



def _empty_report(reason: str) -> dict:
    return {
        "issues":  [_issue("[data]", "Insufficient data", INFO, reason, 0, 0)],
        "summary": {"total": 1, "critical": 0, "warning": 0, "info": 1},
        "passed":  True,
        "score":   100,
    }


def _longest_run_of_identical(s: pd.Series) -> dict:
    """Find the longest consecutive run of identical values."""
    best = {"length": 0, "value": None, "start": None, "end": None}
    if len(s) == 0:
        return best
    current_val   = s.iloc[0]
    current_start = s.index[0]
    current_end   = s.index[0]
    current_len   = 1
    for idx, val in zip(s.index[1:], s.iloc[1:]):
        if val == current_val:
            current_len += 1
            current_end  = idx
        else:
            if current_len > best["length"]:
                best = {"length": current_len, "value": current_val,
                        "start": current_start, "end": current_end}
            current_val   = val
            current_start = idx
            current_end   = idx
            current_len   = 1
    if current_len > best["length"]:
        best = {"length": current_len, "value": current_val,
                "start": current_start, "end": current_end}
    return best


def _all_runs_of_identical(s: pd.Series, min_length: int) -> list:
    """Return ALL consecutive runs of identical values with length >= min_length.

    Each run: {"length": int, "value": float, "start": Timestamp, "end": Timestamp}
    """
    runs = []
    if len(s) == 0:
        return runs
    current_val   = s.iloc[0]
    current_start = s.index[0]
    current_end   = s.index[0]
    current_len   = 1
    for idx, val in zip(s.index[1:], s.iloc[1:]):
        if val == current_val:
            current_len += 1
            current_end  = idx
        else:
            if current_len >= min_length:
                runs.append({"length": current_len, "value": current_val,
                             "start": current_start, "end": current_end})
            current_val   = val
            current_start = idx
            current_end   = idx
            current_len   = 1
    if current_len >= min_length:
        runs.append({"length": current_len, "value": current_val,
                     "start": current_start, "end": current_end})
    return runs


def _default_physical_minimums(cols: list) -> dict:
    """Auto-assign physical minimum = 0 for common parameter types."""
    mins = {}
    keywords_zero = ["pressure", "current", "amp", "flow", "speed", "rpm",
                     "vibration", "power", "kwh", "kw", "frequency"]
    for col in cols:
        cl = col.lower()
        if any(kw in cl for kw in keywords_zero):
            mins[col] = 0.0
    return mins


def _fmt_ts(ts) -> str:
    try:
        return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def _fmt_duration(td) -> str:
    total_seconds = int(pd.Timedelta(td).total_seconds())
    h, m = divmod(total_seconds // 60, 60)
    d, h = divmod(h, 24)
    if d:
        return f"{d}d {h}h"
    elif h:
        return f"{h}h {m}m"
    else:
        return f"{m}m"


def format_quality_report_for_claude(report: dict) -> str:
    """Format the quality report as a compact string for inclusion in Claude prompts."""
    if not report["issues"]:
        return "DATA QUALITY: All checks passed. No issues detected."
    lines = [
        f"DATA QUALITY REPORT (score: {report['score']}/100):",
        f"  Critical: {report['summary']['critical']}  "
        f"Warning: {report['summary']['warning']}  "
        f"Info: {report['summary']['info']}",
    ]
    for iss in report["issues"]:
        sev  = iss["severity"].upper()
        col  = iss["col"]
        chk  = iss["check"]
        pct  = iss["affected_pct"]
        det  = iss["detail"]
        lines.append(f"  [{sev}] {col} — {chk} ({pct}% affected): {det}")
    return "\n".join(lines)
