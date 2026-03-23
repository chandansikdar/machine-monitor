"""
data_corrector.py — Auto-correction methods for data quality issues.

Each corrector takes a DataFrame and returns:
  {
    "corrected_df": pd.DataFrame,
    "changes":      int,          # number of rows/values changed
    "description":  str,          # human-readable summary
    "method":       str,          # method name
  }
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional


# ── Public correction functions ───────────────────────────────────────────────

def fix_duplicate_timestamps(df: pd.DataFrame) -> dict:
    """Keep first occurrence of each timestamp, drop duplicates."""
    n_before = len(df)
    df_out   = df[~df.index.duplicated(keep="first")].copy()
    n_fixed  = n_before - len(df_out)
    return {
        "corrected_df": df_out,
        "changes":      n_fixed,
        "description":  f"Removed {n_fixed} duplicate timestamp(s). Kept first occurrence of each.",
        "method":       "Drop duplicate timestamps",
    }


def fix_missing_gaps(
    df: pd.DataFrame,
    max_gap_interpolate_minutes: int = 60,
) -> dict:
    """
    Interpolate short gaps (up to max_gap_interpolate_minutes) linearly.
    Long gaps are left as-is with NaN.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    # Infer median interval
    diffs      = pd.Series(df.index).diff().dropna()
    median_int = diffs.median()
    if median_int.total_seconds() <= 0:
        return {"corrected_df": df, "changes": 0,
                "description": "Cannot determine interval — no correction applied.",
                "method": "Interpolate gaps"}

    # Reindex to fill gaps
    full_idx   = pd.date_range(df.index.min(), df.index.max(), freq=median_int)
    df_reindex = df.reindex(full_idx)
    n_gaps     = df_reindex.isna().any(axis=1).sum() - df.isna().any(axis=1).sum()

    # Only interpolate gaps shorter than threshold
    max_gap_periods = int(pd.Timedelta(minutes=max_gap_interpolate_minutes) / median_int)
    numeric_cols    = df_reindex.select_dtypes(include="number").columns

    n_interpolated = 0
    for col in numeric_cols:
        s = df_reindex[col].copy()
        # Mark positions that were NaN in reindexed but not originally long gaps
        null_mask = s.isna()
        # Use limit= to only fill short runs
        filled = s.interpolate(method="time", limit=max_gap_periods, limit_direction="both")
        n_interpolated += int((filled.notna() & null_mask).sum())
        df_reindex[col] = filled

    return {
        "corrected_df": df_reindex,
        "changes":      n_interpolated,
        "description":  (
            f"Detected {n_gaps} missing timestamp(s). "
            f"Interpolated {n_interpolated} value(s) for gaps up to {max_gap_interpolate_minutes} min. "
            f"Longer gaps left as NaN."
        ),
        "method":       f"Linear time interpolation (max gap {max_gap_interpolate_minutes} min)",
    }


def fix_isolated_spikes(
    df: pd.DataFrame,
    col: str,
    iqr_multiplier: float = 10.0,
) -> dict:
    """
    Replace single-reading spikes (|diff| > iqr_multiplier x IQR) with
    the mean of the two neighbouring readings.
    Only fixes isolated spikes (1 reading), not sustained excursions.
    """
    s   = df[col].copy()
    q1, q3  = s.quantile(0.25), s.quantile(0.75)
    iqr     = q3 - q1
    if iqr == 0:
        return {"corrected_df": df, "changes": 0,
                "description": "IQR is zero — cannot detect spikes.", "method": "Spike removal"}

    thresh  = iqr * iqr_multiplier
    diffs   = s.diff().abs()
    diffs_b = s.diff(-1).abs()  # diff with next reading

    # Isolated spike: large jump IN and large jump OUT
    spike_mask = (diffs > thresh) & (diffs_b > thresh)
    n_spikes   = int(spike_mask.sum())

    if n_spikes == 0:
        return {"corrected_df": df, "changes": 0,
                "description": f"No isolated spikes found in {col}.",
                "method": "Spike removal"}

    df_out = df.copy()
    for idx in s[spike_mask].index:
        loc = s.index.get_loc(idx)
        if 0 < loc < len(s) - 1:
            prev_val = s.iloc[loc - 1]
            next_val = s.iloc[loc + 1]
            df_out.loc[idx, col] = (prev_val + next_val) / 2

    return {
        "corrected_df": df_out,
        "changes":      n_spikes,
        "description":  (
            f"Replaced {n_spikes} isolated spike(s) in `{col}` with the "
            f"mean of neighbouring readings. "
            f"Only single-reading spikes were corrected — sustained excursions were left unchanged."
        ),
        "method":       f"Isolated spike interpolation ({iqr_multiplier}x IQR threshold)",
    }


def fix_physical_impossibles(
    df: pd.DataFrame,
    col: str,
    min_value: float = 0.0,
) -> dict:
    """Replace physically impossible values (below min_value) with NaN."""
    s          = df[col].copy()
    impossible = s < min_value
    n_fixed    = int(impossible.sum())

    if n_fixed == 0:
        return {"corrected_df": df, "changes": 0,
                "description": f"No physically impossible values found in {col}.",
                "method": "Physical range clamp"}

    df_out          = df.copy()
    df_out.loc[impossible, col] = np.nan
    return {
        "corrected_df": df_out,
        "changes":      n_fixed,
        "description":  (
            f"Replaced {n_fixed} reading(s) below physical minimum ({min_value}) "
            f"in `{col}` with NaN. These should be reviewed and either interpolated "
            f"or excluded from analysis."
        ),
        "method":       f"Set to NaN where value < {min_value}",
    }


def fix_flatline(
    df: pd.DataFrame,
    col: str,
    flatline_window: int = 20,
    std_thresh: float = 1e-6,
) -> dict:
    """
    Identify contiguous flatline periods and replace them with NaN.
    Returns the corrected DataFrame and a description.
    NOTE: This is a destructive operation — the user should inspect carefully.
    """
    s        = df[col].copy()
    roll_std = s.rolling(flatline_window).std()
    flat     = roll_std < std_thresh

    # Expand backwards to catch the full flat block
    flat_expanded = flat | flat.shift(-flatline_window + 1).fillna(False)
    n_flat = int(flat_expanded.sum())

    if n_flat == 0:
        return {"corrected_df": df, "changes": 0,
                "description": f"No flatline periods found in {col}.",
                "method": "Flatline removal"}

    df_out = df.copy()
    df_out.loc[flat_expanded, col] = np.nan

    return {
        "corrected_df": df_out,
        "changes":      n_flat,
        "description":  (
            f"Replaced {n_flat} reading(s) in `{col}` identified as flatline "
            f"(rolling std < {std_thresh:.0e} over {flatline_window} readings) with NaN.  \n"
            f"These NaN values will be excluded from statistical calculations. "
            f"Consider whether the flatline represents a sensor fault or genuine steady-state operation."
        ),
        "method":       "Flatline → NaN",
    }


# ── Suggestion map ────────────────────────────────────────────────────────────
# For each check type: what correction is available and what to suggest

CORRECTION_SUGGESTIONS = {
    "Duplicate timestamps": {
        "auto":       True,
        "fn":         "fix_duplicate_timestamps",
        "args":       {},
        "suggestion": "Safe to auto-fix. Duplicate timestamps corrupt time-series statistics.",
    },
    "Missing data gaps": {
        "auto":       True,
        "fn":         "fix_missing_gaps",
        "args":       {"max_gap_interpolate_minutes": 60},
        "suggestion": "Short gaps (< 60 min) can be interpolated. Longer gaps need manual review.",
    },
    "Sudden step change / spike": {
        "auto":       True,
        "fn":         "fix_isolated_spikes",
        "args":       {},   # col added dynamically
        "suggestion": "Isolated single-reading spikes can be auto-corrected by interpolating neighbours.",
    },
    "Physically impossible values": {
        "auto":       True,
        "fn":         "fix_physical_impossibles",
        "args":       {},   # col + min_value added dynamically
        "suggestion": "Impossible values can be replaced with NaN and optionally interpolated.",
    },
    "Flatline / stuck sensor": {
        "auto":       True,
        "fn":         "fix_flatline",
        "args":       {},   # col added dynamically
        "suggestion": (
            "The flatline period can be replaced with NaN to exclude it from calculations. "
            "Verify whether this is a sensor fault or real steady-state before applying."
        ),
    },
    "Frozen value run": {
        "auto":       True,
        "fn":         "fix_flatline",
        "args":       {},
        "suggestion": (
            "The frozen run can be replaced with NaN. "
            "Confirm this is a sensor fault and not genuine constant output."
        ),
    },
    "Baseline shift": {
        "auto":       False,
        "fn":         None,
        "args":       {},
        "suggestion": (
            "A baseline shift may indicate sensor recalibration, component replacement, "
            "or a genuine process change. Cannot be auto-corrected — "
            "review the timestamp of the shift against maintenance records "
            "and either accept the new baseline or re-calibrate the sensor."
        ),
    },
    "High null percentage": {
        "auto":       False,
        "fn":         None,
        "args":       {},
        "suggestion": (
            "A high proportion of missing values usually indicates a sensor offline period. "
            "Check the sensor connection and data logger for that timeframe. "
            "If the sensor was offline, the column should be excluded from analysis for that period."
        ),
    },
    "Elevated null percentage": {
        "auto":       False,
        "fn":         None,
        "args":       {},
        "suggestion": (
            "Some missing values detected. Check sensor connectivity. "
            "For short outages, interpolation may be acceptable."
        ),
    },
}
