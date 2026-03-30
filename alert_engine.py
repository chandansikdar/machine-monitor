"""
alert_engine.py \u2014 Deterministic alert generation engine

Converts pre-computed ML signals + raw sensor data into structured alert objects.
Operates ONLY for Anomaly Detection analysis type.

\u2500\u2500 Alarm generation \u2014 two mutually exclusive scenarios \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

  Scenario A (no user thresholds):
    Per-parameter method selection:
      \u2022 |skewness| > 2.0 OR no control chart available (tier 1)
          \u2192 IQR two-tier method (robust to non-normal distributions)
             Warning band : Q1 \u2212 1.5\u00d7IQR / Q3 + 1.5\u00d7IQR
             Critical band: Q1 \u2212 3.0\u00d7IQR / Q3 + 3.0\u00d7IQR
      \u2022 |skewness| \u2264 2.0 AND control chart available (tier \u2265 2)
          \u2192 Control limit method (UCL/LCL/UWL/LWL)
    Transient vs Recurring:
      < 3 consecutive readings \u2192 Warning or Advisory
      \u2265 3 consecutive readings \u2192 Critical or Warning

  Scenario B (user thresholds present):
    User thresholds are the SOLE alarm trigger.
    Control limits still run in ml_engine for trend context but do NOT generate alarms.
    Critical threshold breach \u2192 Critical alarm
    Warning threshold breach  \u2192 Warning alarm

\u2500\u2500 Predictive breach \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

  Linear regression fitted to raw series. Projection only when:
    \u2022 Alert consistency is "recurring" (transient spikes are not projected)
    \u2022 R\u00b2 \u2265 0.60 (trend is sufficiently directional)
    \u2022 Slope direction is moving toward the relevant limit
  days_to_breach = None when these conditions are not met.

\u2500\u2500 Confidence scoring \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

  HIGH   \u2014 recurring + tier \u2265 2 + control_limit or threshold trigger
  MEDIUM \u2014 recurring + IQR trigger, OR transient + tier \u2265 2
  LOW    \u2014 advisory level, OR any alert at tier 1 (< 30 days data)

\u2500\u2500 Public API \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

  run(data, ml_signals, thresholds) \u2192 list[dict]
  summarise(alerts)                 \u2192 dict  (compact, for Claude prompt injection)
"""

import numpy as np
import pandas as pd
from typing import Optional


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 #
# Configuration constants
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 #

# Skewness magnitude above which IQR replaces control limits (Scenario A)
_SKEW_THRESHOLD = 2.0

# Minimum consecutive readings in breach to classify as "recurring"
_RECURRING_MIN = 3

# Minimum R\u00b2 for a linear trend to support a days_to_breach projection
_R2_MIN = 0.60

# IQR multipliers for two-tier limits
_IQR_WARN = 1.5   # Warning band boundary
_IQR_CRIT = 3.0   # Critical band boundary

# Default sampling interval (minutes) — used if index frequency cannot be inferred
_DEFAULT_INTERVAL_MIN = 15


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 #
# Public interface
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 #

def run(
    data: pd.DataFrame,
    ml_signals: dict,
    thresholds: Optional[dict] = None,
) -> list:
    """
    Generate alert objects from raw sensor data and pre-computed ML signals.

    Args:
        data        : DataFrame indexed by timestamp, running-condition rows only.
        ml_signals  : Output of ml_engine.run().
        thresholds  : {col: {"warning": float, "critical": float}} or None.

    Returns:
        List of alert dicts, sorted Critical \u2192 Warning \u2192 Advisory.
        Each alert contains:
            parameter, level, consistency, trigger, breach_side,
            breach_count, max_consecutive, limit_value, confidence,
            days_to_breach, message
    """
    numeric = data.select_dtypes(include="number")
    if numeric.empty:
        return []

    has_thresholds = bool(thresholds)
    tier = ml_signals.get("tier", 1)
    stat = ml_signals.get("statistical", {})
    cc   = ml_signals.get("control_charts", {})

    alerts = []

    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s) < 4:
            continue

        col_stat = stat.get(col, {})
        col_cc   = cc.get(col, {})
        skew     = abs(col_stat.get("skewness", 0.0))

        # IQR method: non-normal data (|skew| > threshold) OR no control charts yet (tier 1)
        use_iqr = (skew > _SKEW_THRESHOLD) or (not col_cc)

        if has_thresholds and col in thresholds:
            col_alerts = _scenario_b(s, col, thresholds[col])
        elif use_iqr:
            col_alerts = _scenario_a_iqr(s, col, col_stat)
        else:
            col_alerts = _scenario_a_cc(s, col, col_cc)

        for alert in col_alerts:
            alert["days_to_breach"] = _days_to_breach(
                s, alert,
                thresholds.get(col) if thresholds else None,
                col_cc,
            )
            alert["confidence"] = _confidence(alert, tier)

        alerts.extend(col_alerts)

    _order = {"Critical": 0, "Warning": 1, "Advisory": 2}
    alerts.sort(key=lambda a: _order.get(a["level"], 3))
    return alerts


def summarise(alerts: list) -> dict:
    """
    Compact summary dict for injection into the Claude prompt.

    Returns:
        {total, critical, warning, advisory, alerts: [{...}, ...]}
    """
    if not alerts:
        return {"total": 0, "critical": 0, "warning": 0, "advisory": 0, "alerts": []}

    critical = [a for a in alerts if a["level"] == "Critical"]
    warning  = [a for a in alerts if a["level"] == "Warning"]
    advisory = [a for a in alerts if a["level"] == "Advisory"]

    return {
        "total":    len(alerts),
        "critical": len(critical),
        "warning":  len(warning),
        "advisory": len(advisory),
        "alerts": [
            {
                "parameter":      a["parameter"],
                "level":          a["level"],
                "trigger":        a["trigger"],
                "consistency":    a["consistency"],
                "confidence":     a["confidence"],
                "days_to_breach": a.get("days_to_breach"),
                "message":        a["message"],
            }
            for a in alerts
        ],
    }


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 #
# Scenario A \u2014 Control limit alarms (normal / near-normal data)
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 #

def _scenario_a_cc(s: pd.Series, col: str, cc: dict) -> list:
    """
    Generate alarms from UCL/LCL (Critical band) and UWL/LWL (Warning band).

    Alarm matrix:
      UCL/LCL breach, recurring (\u2265 3 consecutive) \u2192 Critical
      UCL/LCL breach, transient (< 3 consecutive)  \u2192 Warning
      UWL/LWL breach, recurring                    \u2192 Warning
      UWL/LWL breach, transient                    \u2192 Advisory
    """
    alerts = []
    ucl = cc.get("ucl_3sigma")
    lcl = cc.get("lcl_3sigma")
    uwl = cc.get("ucl_2sigma")
    lwl = cc.get("lcl_2sigma")

    if ucl is None:
        return alerts

    # \u2014 UCL band (3\u03c3 critical) \u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014
    above_ucl = (s > ucl).astype(int)
    total_ucl = int(above_ucl.sum())
    if total_ucl > 0:
        max_c = _max_consecutive(above_ucl)
        recurring = max_c >= _RECURRING_MIN
        level = "Critical" if recurring else "Warning"
        alerts.append(_make_alert(
            col, level, "recurring" if recurring else "transient",
            "control_limit", "high", total_ucl, max_c, ucl,
            f"{col}: {total_ucl} reading(s) exceeded UCL ({ucl:.4g}). "
            f"Max consecutive run: {max_c}. "
            f"{'Recurring \u2014 sustained process upset, action required.'  if recurring else 'Transient spike \u2014 monitor for recurrence.'}",
        ))

    # \u2014 LCL band (3\u03c3 critical) \u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014
    if lcl is not None:
        below_lcl = (s < lcl).astype(int)
        total_lcl = int(below_lcl.sum())
        if total_lcl > 0:
            max_c = _max_consecutive(below_lcl)
            recurring = max_c >= _RECURRING_MIN
            level = "Critical" if recurring else "Warning"
            alerts.append(_make_alert(
                col, level, "recurring" if recurring else "transient",
                "control_limit", "low", total_lcl, max_c, lcl,
                f"{col}: {total_lcl} reading(s) fell below LCL ({lcl:.4g}). "
                f"Max consecutive run: {max_c}. "
                f"{'Recurring \u2014 sustained low-side excursion, action required.' if recurring else 'Transient dip \u2014 monitor for recurrence.'}",
            ))

    # \u2014 UWL/LWL band (2\u03c3 warning \u2014 readings between 2\u03c3 and 3\u03c3 only) \u2014\u2014\u2014\u2014\u2014\u2014
    if uwl is not None:
        above_uwl = ((s > uwl) & (s <= ucl)).astype(int)
        total_uwl = int(above_uwl.sum())
        if total_uwl > 0:
            max_c = _max_consecutive(above_uwl)
            recurring = max_c >= _RECURRING_MIN
            level = "Warning" if recurring else "Advisory"
            alerts.append(_make_alert(
                col, level, "recurring" if recurring else "transient",
                "control_limit", "high_warn", total_uwl, max_c, uwl,
                f"{col}: {total_uwl} reading(s) in upper warning zone "
                f"(UWL={uwl:.4g}, below UCL={ucl:.4g}). "
                f"Max consecutive: {max_c}. "
                f"{'Recurring \u2014 approaching critical limit.' if recurring else 'Transient \u2014 early warning signal.'}",
            ))

    if lwl is not None and lcl is not None:
        below_lwl = ((s < lwl) & (s >= lcl)).astype(int)
        total_lwl = int(below_lwl.sum())
        if total_lwl > 0:
            max_c = _max_consecutive(below_lwl)
            recurring = max_c >= _RECURRING_MIN
            level = "Warning" if recurring else "Advisory"
            alerts.append(_make_alert(
                col, level, "recurring" if recurring else "transient",
                "control_limit", "low_warn", total_lwl, max_c, lwl,
                f"{col}: {total_lwl} reading(s) in lower warning zone "
                f"(LWL={lwl:.4g}, above LCL={lcl:.4g}). "
                f"Max consecutive: {max_c}. "
                f"{'Recurring \u2014 approaching critical limit.' if recurring else 'Transient \u2014 early warning signal.'}",
            ))

    return alerts


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 #
# Scenario A \u2014 IQR alarms (skewed / non-normal data, or tier 1)
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 #

def _scenario_a_iqr(s: pd.Series, col: str, stat: dict) -> list:
    """
    Generate alarms using two-tier IQR limits.

    Limits computed fresh from raw data (ml_engine uses 1.5\u00d7 only):
      Warning band : Q3 + 1.5\u00d7IQR  /  Q1 \u2212 1.5\u00d7IQR
      Critical band: Q3 + 3.0\u00d7IQR  /  Q1 \u2212 3.0\u00d7IQR

    Readings in the critical band do NOT appear in warning band counts.

    Alarm matrix mirrors the control limit structure:
      Critical band breach, recurring \u2192 Critical
      Critical band breach, transient \u2192 Warning
      Warning band breach, recurring  \u2192 Warning
      Warning band breach, transient  \u2192 Advisory
    """
    alerts = []
    q1  = float(s.quantile(0.25))
    q3  = float(s.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        return alerts

    warn_high = q3 + _IQR_WARN * iqr
    warn_low  = q1 - _IQR_WARN * iqr
    crit_high = q3 + _IQR_CRIT * iqr
    crit_low  = q1 - _IQR_CRIT * iqr
    skew      = stat.get("skewness", 0.0)

    # Reason for IQR selection
    if abs(skew) > _SKEW_THRESHOLD:
        iqr_reason = f"skewness={skew:.2f} exceeds \u00b1{_SKEW_THRESHOLD} \u2014 IQR selected over control limits"
    else:
        iqr_reason = "insufficient data for control charts \u2014 IQR applied as fallback"

    # Critical band (beyond 3.0\u00d7IQR)
    above_crit = (s > crit_high).astype(int)
    below_crit = (s < crit_low).astype(int)

    # Warning band (between 1.5\u00d7 and 3.0\u00d7IQR \u2014 exclusive of critical band)
    above_warn = ((s > warn_high) & (s <= crit_high)).astype(int)
    below_warn = ((s < warn_low) & (s >= crit_low)).astype(int)

    for mask, limit, side, band in [
        (above_crit, crit_high, "high", "critical"),
        (below_crit, crit_low,  "low",  "critical"),
        (above_warn, warn_high, "high", "warning"),
        (below_warn, warn_low,  "low",  "warning"),
    ]:
        count = int(mask.sum())
        if count == 0:
            continue

        max_c     = _max_consecutive(mask)
        recurring = max_c >= _RECURRING_MIN

        if band == "critical":
            level = "Critical" if recurring else "Warning"
        else:
            level = "Warning" if recurring else "Advisory"

        alerts.append(_make_alert(
            col, level, "recurring" if recurring else "transient",
            "iqr", f"{side}_{band}", count, max_c, round(limit, 4),
            f"{col} ({iqr_reason}): {count} reading(s) beyond "
            f"{band} {'upper' if side == 'high' else 'lower'} IQR limit "
            f"({'above' if side == 'high' else 'below'} {limit:.4g}, "
            f"{_IQR_CRIT if band == 'critical' else _IQR_WARN}\u00d7IQR). "
            f"Max consecutive: {max_c}. "
            f"{'Recurring \u2014 action required.' if recurring else 'Transient \u2014 monitor.'}",
        ))

    return alerts


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 #
# Scenario B \u2014 User threshold alarms
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 #

def _scenario_b(s: pd.Series, col: str, limits: dict) -> list:
    """
    Generate alarms from user-defined thresholds (sole trigger in Scenario B).

    High-side thresholds: critical > warning (temperature, vibration, etc.)
    Low-side thresholds : critical < warning (pressure, flow, etc.)
    Direction is auto-detected.
    """
    alerts = []
    warn_val = limits.get("warning")
    crit_val = limits.get("critical")

    # Auto-detect direction
    low_side = (
        warn_val is not None and crit_val is not None and crit_val < warn_val
    )
    side = "low" if low_side else "high"

    for level_name, thr_val in [("Critical", crit_val), ("Warning", warn_val)]:
        if thr_val is None:
            continue

        if low_side:
            mask = (s < thr_val).astype(int)
        else:
            mask = (s > thr_val).astype(int)

        count = int(mask.sum())
        if count == 0:
            continue

        max_c     = _max_consecutive(mask)
        recurring = max_c >= _RECURRING_MIN
        direction = "below" if low_side else "above"

        alerts.append(_make_alert(
            col, level_name, "recurring" if recurring else "transient",
            "threshold", side, count, max_c, thr_val,
            f"{col}: {count} reading(s) {direction} user-defined "
            f"{level_name.lower()} threshold ({thr_val:.4g}). "
            f"Max consecutive: {max_c}. "
            f"{'Recurring \u2014 sustained breach, action required.' if recurring else 'Transient \u2014 monitor for recurrence.'}",
        ))

    return alerts


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 #
# Predictive breach projection
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 #

def _days_to_breach(
    s: pd.Series,
    alert: dict,
    thresholds: Optional[dict],
    cc: dict,
) -> Optional[float]:
    """
    Project a linear trend forward to the nearest relevant limit.

    Returns:
        float  \u2014 estimated days until breach (> 0)
        None   \u2014 trend not clear enough, moving away, or already in breach
    """
    # Only project for recurring alerts with enough data
    if alert.get("consistency") != "recurring":
        return None
    if len(s) < 10:
        return None

    # Fit linear trend
    x = np.arange(len(s), dtype=float)
    y = s.values.astype(float)
    try:
        coeffs = np.polyfit(x, y, 1)
        slope  = coeffs[0]
        y_fit  = np.polyval(coeffs, x)
        ss_res = float(np.sum((y - y_fit) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except Exception:
        return None

    if r2 < _R2_MIN or abs(slope) < 1e-9:
        return None

    current_val = float(s.iloc[-1])
    trigger     = alert.get("trigger", "")
    side        = alert.get("breach_side", "")

    # Resolve the relevant limit to project towards
    target = _resolve_projection_target(trigger, side, thresholds, cc, alert)
    if target is None:
        return None

    # If already at or past target, no projection (breach active)
    high_side = "high" in side or side == "high"
    low_side  = "low" in side  or side == "low"

    if high_side and current_val >= target:
        return None
    if low_side and current_val <= target:
        return None

    # Slope must be moving toward the limit
    if high_side and slope <= 0:
        return None
    if low_side and slope >= 0:
        return None

    steps_to_breach = abs(target - current_val) / abs(slope)
    freq_min = _infer_interval_minutes(s)
    days_to_breach = steps_to_breach * freq_min / (60.0 * 24.0)
    return round(float(days_to_breach), 1)


def _resolve_projection_target(
    trigger: str,
    side: str,
    thresholds: Optional[dict],
    cc: dict,
    alert: dict,
) -> Optional[float]:
    """Choose the limit to project the trend towards."""
    if trigger == "threshold" and thresholds:
        # Project toward the critical threshold (most conservative)
        return thresholds.get("critical") or thresholds.get("warning")

    if trigger == "control_limit":
        if "high" in side:
            return cc.get("ucl_3sigma")
        if "low" in side:
            return cc.get("lcl_3sigma")

    if trigger == "iqr":
        # Project toward the IQR critical band limit
        return alert.get("limit_value")

    return None


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 #
# Confidence scoring
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 #

def _confidence(alert: dict, tier: int) -> str:
    """
    Assign confidence label.

    HIGH   \u2014 recurring + tier \u2265 2 + control_limit or threshold trigger
    MEDIUM \u2014 recurring + IQR trigger, OR transient + tier \u2265 2
    LOW    \u2014 advisory level, OR any alert at tier 1 (< 30 days data)
    """
    level       = alert.get("level")
    trigger     = alert.get("trigger")
    consistency = alert.get("consistency")

    if level == "Advisory":
        return "LOW"
    if tier < 2:
        return "LOW"
    if consistency == "transient":
        return "MEDIUM"
    # Recurring from here
    if trigger in ("control_limit", "threshold"):
        return "HIGH"
    if trigger == "iqr":
        return "MEDIUM"
    return "MEDIUM"


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 #
# Helpers
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 #

def _make_alert(
    parameter: str,
    level: str,
    consistency: str,
    trigger: str,
    breach_side: str,
    breach_count: int,
    max_consecutive: int,
    limit_value: float,
    message: str,
) -> dict:
    """Construct a standardised alert dict."""
    return {
        "parameter":      parameter,
        "level":          level,
        "consistency":    consistency,
        "trigger":        trigger,
        "breach_side":    breach_side,
        "breach_count":   breach_count,
        "max_consecutive": max_consecutive,
        "limit_value":    limit_value,
        "message":        message,
        # Filled in by run() after creation:
        "days_to_breach": None,
        "confidence":     None,
    }


def _max_consecutive(mask: pd.Series) -> int:
    """Return the maximum run of consecutive 1s in a binary integer Series."""
    if int(mask.sum()) == 0:
        return 0
    max_run = current = 0
    for v in mask:
        if v:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return max_run


def _infer_interval_minutes(s: pd.Series) -> float:
    """
    Infer the median sampling interval in minutes from the Series index.
    Falls back to _DEFAULT_INTERVAL_MIN if the index is not datetime-like.
    """
    try:
        diffs = pd.Series(s.index).diff().dropna()
        if diffs.empty:
            return _DEFAULT_INTERVAL_MIN
        median_td = diffs.median()
        return float(median_td.total_seconds()) / 60.0
    except Exception:
        return _DEFAULT_INTERVAL_MIN
