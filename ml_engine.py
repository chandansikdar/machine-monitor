"""
ml_engine.py — Adaptive pre-processing engine

Automatically selects the right analytical method based on data volume
and feeds pre-computed signals to Claude for interpretation.

Tiers:
  TIER 1 — < 30 days   : Statistical only (Z-score, IQR, rolling mean)
  TIER 2 — 30–180 days : Tier 1 + control charts (Western Electric rules)
  TIER 3 — 180+ days   : Tier 2 + Isolation Forest unsupervised anomaly scoring
  TIER 4 — labelled    : Tier 3 + supervised threshold breach scoring

Claude always receives the pre-computed signals as structured context and
provides the final interpretation. The signals make Claude's answers
richer and more data-precise regardless of tier.
"""

import json
from typing import Optional
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #

def run(data: pd.DataFrame, thresholds: Optional[dict] = None) -> dict:
    """
    Entry point. Returns a structured dict that gets injected into the
    Claude prompt as === PRE-COMPUTED SIGNALS ===

    Args:
        data       : DataFrame indexed by timestamp, numeric sensor columns
        thresholds : {col: {"warning": float, "critical": float}} or None
    """
    numeric = data.select_dtypes(include="number")
    if numeric.empty:
        return {"tier": 0, "message": "No numeric columns found.", "signals": {}}

    days = (data.index.max() - data.index.min()).days
    tier, tier_label = _select_tier(days, thresholds)

    result = {
        "tier": tier,
        "tier_label": tier_label,
        "data_days": days,
        "columns_analysed": numeric.columns.tolist(),
        "statistical": _statistical(numeric, thresholds),
    }

    if tier >= 2:
        result["control_charts"] = _control_charts(numeric)

    if tier >= 3:
        result["isolation_forest"] = _isolation_forest(numeric)

    if tier >= 4 and thresholds:
        result["threshold_breaches"] = _threshold_breaches(numeric, thresholds)

    result["guidance"] = _guidance(tier, days)
    return result


# --------------------------------------------------------------------------- #
# Tier selection
# --------------------------------------------------------------------------- #

def _select_tier(days: int, thresholds: Optional[dict]) -> tuple:
    has_thresholds = bool(thresholds)
    if days >= 180:
        tier = 4 if has_thresholds else 3
        label = ("Statistical + Control Charts + Isolation Forest + "
                 "Threshold Scoring" if has_thresholds else
                 "Statistical + Control Charts + Isolation Forest")
    elif days >= 30:
        tier = 4 if has_thresholds else 2
        label = ("Statistical + Control Charts + Threshold Scoring"
                 if has_thresholds else "Statistical + Control Charts")
    else:
        tier = 4 if has_thresholds else 1
        label = ("Statistical + Threshold Scoring"
                 if has_thresholds else
                 "Statistical only (insufficient data for ML models)")
    return tier, label


# --------------------------------------------------------------------------- #
# Tier 1 — Statistical signals
# --------------------------------------------------------------------------- #

def _statistical(numeric: pd.DataFrame, thresholds: Optional[dict]) -> dict:
    signals = {}
    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s) < 4:
            continue

        mean, std = s.mean(), s.std()
        q1, q3    = s.quantile(0.25), s.quantile(0.75)
        iqr       = q3 - q1
        skew      = float(s.skew())

        # Z-score outliers
        z = (s - mean) / std if std > 0 else pd.Series(0, index=s.index)
        outlier_mask  = z.abs() > 3
        outlier_count = int(outlier_mask.sum())
        outlier_pct   = round(100 * outlier_count / len(s), 2)

        # IQR outliers (more robust)
        iqr_low  = q1 - 1.5 * iqr
        iqr_high = q3 + 1.5 * iqr
        iqr_outliers = int(((s < iqr_low) | (s > iqr_high)).sum())

        # Rolling trend (last 20% vs first 20%)
        n20 = max(1, len(s) // 5)
        trend_delta = round(float(s.iloc[-n20:].mean() - s.iloc[:n20].mean()), 4)
        trend_pct   = round(100 * trend_delta / mean, 2) if mean != 0 else 0

        # Recent spike (last 5% vs overall)
        n5 = max(1, len(s) // 20)
        recent_max   = float(s.iloc[-n5:].max())
        recent_z     = round((recent_max - mean) / std, 2) if std > 0 else 0

        sig = {
            "mean":           round(float(mean), 4),
            "std":            round(float(std), 4),
            "min":            round(float(s.min()), 4),
            "max":            round(float(s.max()), 4),
            "skewness":       round(skew, 3),
            "z_outliers":     outlier_count,
            "z_outlier_pct":  outlier_pct,
            "iqr_outliers":   iqr_outliers,
            "trend_delta":    trend_delta,
            "trend_pct":      trend_pct,
            "recent_max_z":   recent_z,
        }

        # Flag summary
        flags = []
        if outlier_pct > 1:
            flags.append(f"{outlier_pct}% readings are Z-score outliers (>3σ)")
        if iqr_outliers > 0:
            flags.append(f"{iqr_outliers} IQR outliers detected")
        if abs(trend_pct) > 5:
            direction = "upward" if trend_pct > 0 else "downward"
            flags.append(f"{abs(trend_pct):.1f}% {direction} trend (recent vs early period)")
        if abs(recent_z) > 2.5:
            flags.append(f"Recent maximum is {recent_z}σ from mean — potential spike")
        if abs(skew) > 2:
            flags.append(f"Distribution is highly skewed ({skew:.2f}) — suggests outlier pull")

        sig["flags"] = flags
        signals[col] = sig

    return signals


# --------------------------------------------------------------------------- #
# Tier 2 — Control charts (Western Electric rules)
# --------------------------------------------------------------------------- #

def _control_charts(numeric: pd.DataFrame) -> dict:
    """
    Apply 4 Western Electric rules to each parameter:
      Rule 1: 1 point beyond 3σ
      Rule 2: 9 consecutive points on same side of mean
      Rule 3: 6 consecutive points steadily increasing or decreasing
      Rule 4: 2 of 3 consecutive points beyond 2σ on same side
    """
    results = {}
    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s) < 20:
            continue

        mean, std = s.mean(), s.std()
        if std == 0:
            continue

        z = (s - mean) / std
        violations = []

        # Rule 1
        r1 = (z.abs() > 3).sum()
        if r1:
            violations.append(f"Rule 1: {int(r1)} point(s) beyond 3σ control limit")

        # Rule 2: 9 consecutive same side
        side = (z > 0).astype(int).values
        for i in range(len(side) - 8):
            window = side[i:i+9]
            if window.sum() == 9 or window.sum() == 0:
                violations.append(f"Rule 2: 9 consecutive points on same side of mean at index {i}")
                break

        # Rule 3: 6 consecutive increasing or decreasing
        vals = s.values
        for i in range(len(vals) - 5):
            window = vals[i:i+6]
            diffs = np.diff(window)
            if (diffs > 0).all() or (diffs < 0).all():
                direction = "increasing" if (diffs > 0).all() else "decreasing"
                violations.append(f"Rule 3: 6 consecutive {direction} points at index {i}")
                break

        # Rule 4: 2 of 3 beyond 2σ same side
        z_vals = z.values
        for i in range(len(z_vals) - 2):
            w = z_vals[i:i+3]
            if ((w > 2).sum() >= 2) or ((w < -2).sum() >= 2):
                violations.append(f"Rule 4: 2 of 3 consecutive points beyond 2σ at index {i}")
                break

        results[col] = {
            "ucl_3sigma": round(float(mean + 3*std), 4),
            "lcl_3sigma": round(float(mean - 3*std), 4),
            "ucl_2sigma": round(float(mean + 2*std), 4),
            "lcl_2sigma": round(float(mean - 2*std), 4),
            "violations": violations,
            "in_control": len(violations) == 0,
        }

    return results


# --------------------------------------------------------------------------- #
# Tier 3 — Isolation Forest
# --------------------------------------------------------------------------- #

def _isolation_forest(numeric: pd.DataFrame) -> dict:
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        # Use up to 10 columns, drop any with NaN
        cols = numeric.dropna(axis=1).columns.tolist()[:10]
        if len(cols) < 2:
            return {"error": "Need at least 2 complete columns for Isolation Forest"}

        X = numeric[cols].dropna()
        if len(X) < 50:
            return {"skipped": "Need at least 50 rows for Isolation Forest"}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # contamination=auto lets sklearn decide
        iso = IsolationForest(
            n_estimators=100,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )
        scores = iso.fit_predict(X_scaled)
        anomaly_scores = iso.score_samples(X_scaled)  # more negative = more anomalous

        anomaly_mask = scores == -1
        anomaly_count = int(anomaly_mask.sum())
        anomaly_pct   = round(100 * anomaly_count / len(X), 2)

        # Find worst anomaly periods
        worst_idx = np.argsort(anomaly_scores)[:5]
        worst_times = [str(X.index[i]) for i in worst_idx if i < len(X)]

        # Which columns drive anomalies most
        anomaly_rows = X[anomaly_mask]
        normal_rows  = X[~anomaly_mask]
        if not anomaly_rows.empty and not normal_rows.empty:
            contribution = {}
            for col in cols:
                delta = abs(anomaly_rows[col].mean() - normal_rows[col].mean())
                contribution[col] = round(float(delta), 4)
            top_drivers = sorted(contribution, key=contribution.get, reverse=True)[:3]
        else:
            top_drivers = []

        return {
            "columns_used":    cols,
            "total_rows":      len(X),
            "anomaly_count":   anomaly_count,
            "anomaly_pct":     anomaly_pct,
            "worst_timestamps": worst_times,
            "top_anomaly_drivers": top_drivers,
            "interpretation": (
                f"Isolation Forest flagged {anomaly_count} readings ({anomaly_pct}%) as anomalous. "
                f"Top contributing parameters: {', '.join(top_drivers) if top_drivers else 'undetermined'}."
            ),
        }

    except ImportError:
        return {"error": "scikit-learn not installed — run: pip install scikit-learn"}
    except Exception as exc:
        return {"error": str(exc)}


# --------------------------------------------------------------------------- #
# Tier 4 — Threshold breach scoring
# --------------------------------------------------------------------------- #

def _threshold_breaches(numeric: pd.DataFrame, thresholds: dict) -> dict:
    results = {}
    for col, limits in thresholds.items():
        if col not in numeric.columns:
            continue
        s = numeric[col].dropna()
        warn_val = limits.get("warning")
        crit_val = limits.get("critical")

        breaches = {}
        if warn_val is not None:
            # Handle both high and low thresholds
            if crit_val is not None and crit_val < warn_val:
                # Low-side thresholds (e.g. pressure dropping)
                warn_breach = int((s < warn_val).sum())
                crit_breach = int((s < crit_val).sum())
            else:
                # High-side thresholds (e.g. temperature rising)
                warn_breach = int((s > warn_val).sum()) if warn_val else 0
                crit_breach = int((s > crit_val).sum()) if crit_val else 0

            total = len(s)
            breaches = {
                "warning_breaches":  warn_breach,
                "critical_breaches": crit_breach,
                "warning_pct":       round(100 * warn_breach / total, 2),
                "critical_pct":      round(100 * crit_breach / total, 2),
                "current_value":     round(float(s.iloc[-1]), 4),
                "current_status": (
                    "critical" if crit_breach and s.iloc[-1] > (crit_val or 9e9)
                    else "warning" if warn_breach and s.iloc[-1] > (warn_val or 9e9)
                    else "normal"
                ),
            }
        results[col] = breaches
    return results


# --------------------------------------------------------------------------- #
# Guidance message for Claude
# --------------------------------------------------------------------------- #

def _guidance(tier: int, days: int) -> str:
    if tier == 1 and days < 30:
        return (
            f"Only {days} days of data available. Statistical pre-processing applied. "
            "Avoid making strong predictive claims. Focus on current state and early indicators. "
            "Note that patterns may not yet be representative of normal operation."
        )
    elif tier == 2:
        return (
            f"{days} days of data. Control chart analysis applied in addition to statistics. "
            "Western Electric rule violations indicate process instability worth investigating."
        )
    elif tier >= 3:
        return (
            f"{days} days of data. Full pre-processing applied including Isolation Forest. "
            "Isolation Forest results highlight multivariate anomalies that univariate stats may miss. "
            "Use all signals together for a comprehensive assessment."
        )
    return ""
