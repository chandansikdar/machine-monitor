"""
pump_physics.py — Physics-based analytics for centrifugal pumps.

Phases activated automatically based on detected sensor columns:
  Phase 1: Power / current only
  Phase 2: Phase 1 + flow
  Phase 3: Phase 2 + differential pressure  (+ mandatory H-Q curve)
  Phase 4: Power + differential pressure + fluid temperatures (NO flow required)
  Phase 5: Vibration + shaft speed
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

# ── Physical constants ────────────────────────────────────────────────────────
G = 9.81          # m/s²
RHO_WATER = 998.0  # kg/m³ at 20°C
CP_WATER  = 4182.0 # J/(kg·°C) at 20°C

# ── Column keyword detection ──────────────────────────────────────────────────
FLOW_KW     = ["flow", "flowrate", "m3h", "m3_h", "gpm", "flow_rate"]
PRESSURE_IN = ["suction_press", "inlet_press", "p_in", "p_suction", "suction_pressure"]
PRESSURE_OUT= ["discharge_press", "outlet_press", "p_out", "p_discharge", "discharge_pressure"]
TEMP_IN     = ["temp_in", "inlet_temp", "t_in", "t_suction", "fluid_temp_in"]
TEMP_OUT    = ["temp_out", "outlet_temp", "t_out", "t_discharge", "fluid_temp_out"]
POWER_KW    = ["power", "kw", "power_kw", "active_power", "shaft_power"]
CURRENT     = ["current", "amp", "_a", "motor_current", "phase_current"]
VIBRATION   = ["vibration", "vibr", "velocity", "mm_s", "mm/s"]
SPEED       = ["speed", "rpm", "rps", "shaft_speed"]


def _find_col(df: pd.DataFrame, keywords: list) -> Optional[str]:
    """Return first column whose name contains any keyword (case-insensitive)."""
    for col in df.columns:
        cl = col.lower()
        if any(kw in cl for kw in keywords):
            return col
    return None


def detect_phase(df: pd.DataFrame) -> dict:
    """
    Detect which pump physics phases are available based on column names.
    Returns a dict of detected columns and the highest available phase.
    """
    cols = {
        "power":    _find_col(df, POWER_KW),
        "current":  _find_col(df, CURRENT),
        "flow":     _find_col(df, FLOW_KW),
        "p_in":     _find_col(df, PRESSURE_IN),
        "p_out":    _find_col(df, PRESSURE_OUT),
        "t_in":     _find_col(df, TEMP_IN),
        "t_out":    _find_col(df, TEMP_OUT),
        "vibration":_find_col(df, VIBRATION),
        "speed":    _find_col(df, SPEED),
    }

    has_power    = cols["power"] or cols["current"]
    has_flow     = bool(cols["flow"])
    has_pressure = bool(cols["p_in"] and cols["p_out"])
    has_temp     = bool(cols["t_in"] and cols["t_out"])
    has_vibr     = bool(cols["vibration"])
    has_speed    = bool(cols["speed"])

    phases = []
    if has_power:
        phases.append(1)
    if has_power and has_flow:
        phases.append(2)
    if has_power and has_flow and has_pressure:
        phases.append(3)
    if has_power and has_pressure and has_temp:
        phases.append(4)
    if has_vibr and has_speed:
        phases.append(5)

    return {
        "cols":           cols,
        "phases":         phases,
        "highest_phase":  max(phases) if phases else 0,
        "has_power":      has_power,
        "has_flow":       has_flow,
        "has_pressure":   has_pressure,
        "has_temp":       has_temp,
        "has_vibr":       has_vibr,
        "has_speed":      has_speed,
    }


# ── Phase 4 viability check ───────────────────────────────────────────────────

def check_phase4_viability(
    df: pd.DataFrame,
    cols: dict,
    rated_efficiency: float = 0.78,
    rho: float = RHO_WATER,
    cp: float = CP_WATER,
) -> dict:
    """
    Check whether the thermodynamic efficiency method (Phase 4) is viable
    for this pump based on expected temperature rise.

    Returns:
        {
            "viable":           bool,
            "level":            "viable" | "marginal" | "not_viable",
            "expected_delta_t": float,   # °C
            "mean_delta_p":     float,   # bar
            "min_sensor_accuracy": float, # °C required
            "message":          str,
            "recommendation":   str,
        }
    """
    result = {
        "viable": False,
        "level": "not_viable",
        "expected_delta_t": 0.0,
        "mean_delta_p": 0.0,
        "min_sensor_accuracy": 0.02,
        "message": "",
        "recommendation": "",
    }

    p_in_col  = cols.get("p_in")
    p_out_col = cols.get("p_out")

    if not p_in_col or not p_out_col:
        result["message"] = (
            "Differential pressure cannot be calculated — "
            "inlet and/or outlet pressure columns not detected."
        )
        result["recommendation"] = (
            "Add inlet pressure (suction) and outlet pressure (discharge) "
            "sensor columns to enable Phase 4 viability assessment."
        )
        return result

    # Calculate mean differential pressure
    dp = (df[p_out_col] - df[p_in_col]).dropna()
    dp = dp[dp > 0]
    if len(dp) < 10:
        result["message"] = "Insufficient valid differential pressure readings."
        return result

    mean_dp = float(dp.mean())  # bar

    # Expected temperature rise using thermodynamic equation
    # ΔT = (1 - η) × ΔP × 10⁵ / (ρ × Cp)
    expected_dt = (1.0 - rated_efficiency) * mean_dp * 1e5 / (rho * cp)

    result["mean_delta_p"]     = round(mean_dp, 3)
    result["expected_delta_t"] = round(expected_dt, 4)

    # Viability thresholds
    THRESHOLD_NOT_VIABLE = 0.05   # below this: not viable even with best sensors
    THRESHOLD_MARGINAL   = 0.10   # below this: marginal, needs PT1000 ±0.02°C
    THRESHOLD_VIABLE     = 0.10   # above this: viable

    if expected_dt < THRESHOLD_NOT_VIABLE:
        result["level"]   = "not_viable"
        result["viable"]  = False
        result["message"] = (
            f"Thermodynamic method is NOT viable for this pump.  \n"
            f"Mean differential pressure: **{mean_dp:.2f} bar**  \n"
            f"Expected temperature rise at {rated_efficiency*100:.0f}% efficiency: "
            f"**{expected_dt*1000:.1f} m°C** ({expected_dt:.4f}°C)  \n"
            f"This is far below the minimum measurable threshold of **0.05°C** "
            f"for even the best PT1000 sensors (±0.02°C accuracy)."
        )
        result["recommendation"] = (
            "Use **Phase 3** (power + flow + pressure) for efficiency monitoring. "
            "The thermodynamic method is only viable for high-pressure pumps "
            "(typically above 15–20 bar differential pressure)."
        )

    elif expected_dt < THRESHOLD_MARGINAL:
        result["level"]   = "marginal"
        result["viable"]  = False
        result["message"] = (
            f"Thermodynamic method has **marginal viability** for this pump.  \n"
            f"Mean differential pressure: **{mean_dp:.2f} bar**  \n"
            f"Expected temperature rise at {rated_efficiency*100:.0f}% efficiency: "
            f"**{expected_dt*1000:.1f} m°C** ({expected_dt:.4f}°C)  \n"
            f"This is above the absolute minimum (0.05°C) but below the reliable "
            f"threshold (0.10°C). Results would have high uncertainty."
        )
        result["recommendation"] = (
            "Only proceed with Phase 4 if you have **calibrated PT1000 RTDs** "
            "with ±0.02°C accuracy installed at the pump inlet and outlet. "
            "Cross-check all thermodynamic results against Phase 3 hydraulic "
            "efficiency. For routine monitoring, Phase 3 is more reliable."
        )

    else:
        result["level"]   = "viable"
        result["viable"]  = True
        result["message"] = (
            f"Thermodynamic method is **viable** for this pump.  \n"
            f"Mean differential pressure: **{mean_dp:.2f} bar**  \n"
            f"Expected temperature rise at {rated_efficiency*100:.0f}% efficiency: "
            f"**{expected_dt*1000:.1f} m°C** ({expected_dt:.4f}°C)  \n"
            f"This exceeds the 0.10°C reliable threshold."
        )
        result["recommendation"] = (
            "Phase 4 thermodynamic analysis can proceed. "
            "Ensure PT100 or PT1000 RTDs with ±0.02°C accuracy are used. "
            "Thermocouples are not sufficiently accurate for this method."
        )

    return result


# ── Phase 1 calculations ──────────────────────────────────────────────────────

def phase1_power_baseline(
    df: pd.DataFrame,
    cols: dict,
    rated_power_kw: float,
    commissioning_power_kw: float,
    voltage_v: float = 415.0,
    power_factor: float = 0.85,
    fla_amps: float = 0.0,
) -> dict:
    """
    Phase 1: Power baseline analysis.
    Returns load factor, specific power index, trend rate, energy consumed.
    """
    # Derive power series
    if cols["power"]:
        power = df[cols["power"]].dropna()
    elif cols["current"]:
        i = df[cols["current"]].dropna()
        power = np.sqrt(3) * voltage_v * i * power_factor / 1000.0
    else:
        return {"error": "No power or current column found."}

    # Running mask (> 5% of rated power)
    running = power[power > 0.05 * rated_power_kw]
    if len(running) < 10:
        return {"error": "Insufficient running data for Phase 1 analysis."}

    mean_power      = float(running.mean())
    load_factor_pct = round(mean_power / rated_power_kw * 100, 1)
    spi             = round(mean_power / commissioning_power_kw, 3) \
                      if commissioning_power_kw > 0 else None

    # Trend: linear regression on 7-day rolling mean
    roll = running.resample("1h").mean().dropna()
    trend_pct = 0.0
    if len(roll) >= 14:
        x = np.arange(len(roll))
        m, _ = np.polyfit(x, roll.values, 1)
        trend_pct = round(m / roll.mean() * 100 * 24 * 30, 2)  # % per month

    # Energy
    dt_hours = _infer_interval_hours(df)
    energy_kwh = float((power * dt_hours).sum())

    return {
        "mean_power_kw":    round(mean_power, 2),
        "rated_power_kw":   rated_power_kw,
        "load_factor_pct":  load_factor_pct,
        "spi":              spi,
        "spi_trend_pct_per_month": trend_pct,
        "energy_kwh":       round(energy_kwh, 1),
        "running_hours":    round(len(running) * dt_hours, 1),
    }


# ── Phase 2 calculations ──────────────────────────────────────────────────────

def phase2_hydraulic_efficiency(
    df: pd.DataFrame,
    cols: dict,
    rated_power_kw: float,
    rated_head_m: float,
    rated_flow_m3h: float,
    bep_flow_m3h: float,
    rated_efficiency: float = 0.78,
    rho: float = RHO_WATER,
    voltage_v: float = 415.0,
    power_factor: float = 0.85,
) -> dict:
    """Phase 2: Specific energy and approximate efficiency."""

    # Power
    if cols["power"]:
        power = df[cols["power"]].dropna()
    elif cols["current"]:
        i = df[cols["current"]].dropna()
        power = np.sqrt(3) * voltage_v * i * power_factor / 1000.0
    else:
        return {"error": "No power or current column."}

    flow = df[cols["flow"]].dropna()

    # Align
    common = power.index.intersection(flow.index)
    if len(common) < 10:
        return {"error": "Insufficient aligned power and flow data."}

    p = power.loc[common]
    q = flow.loc[common]

    # Running mask
    mask = (p > 0.05 * rated_power_kw) & (q > 0.05 * rated_flow_m3h)
    p, q = p[mask], q[mask]
    if len(p) < 10:
        return {"error": "Insufficient running data for Phase 2."}

    # Specific energy kWh/m³
    sp_energy = (p / q).replace([np.inf, -np.inf], np.nan).dropna()
    mean_sp_energy = float(sp_energy.mean())

    # Commissioning specific energy (approximate)
    comm_sp_energy = rated_power_kw / rated_flow_m3h

    sp_energy_rise_pct = round(
        (mean_sp_energy - comm_sp_energy) / comm_sp_energy * 100, 1
    ) if comm_sp_energy > 0 else None

    # Approximate wire-to-water efficiency
    q_m3s = q / 3600.0
    eta_approx = (rho * G * rated_head_m * q_m3s) / (p * 1000.0)
    mean_eta = float(eta_approx.mean())
    eta_drop_pct = round((rated_efficiency - mean_eta) / rated_efficiency * 100, 1)

    # BEP deviation
    mean_flow = float(q.mean())
    bep_dev_pct = round((mean_flow - bep_flow_m3h) / bep_flow_m3h * 100, 1)

    # Flow-power correlation
    corr = float(p.corr(q))

    return {
        "mean_specific_energy_kwh_m3": round(mean_sp_energy, 4),
        "commissioning_specific_energy": round(comm_sp_energy, 4),
        "specific_energy_rise_pct":    sp_energy_rise_pct,
        "approx_efficiency":           round(mean_eta, 3),
        "rated_efficiency":            rated_efficiency,
        "efficiency_drop_pct":         eta_drop_pct,
        "mean_flow_m3h":               round(mean_flow, 1),
        "bep_flow_m3h":                bep_flow_m3h,
        "bep_deviation_pct":           bep_dev_pct,
        "flow_power_correlation":      round(corr, 3),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_interval_hours(df: pd.DataFrame) -> float:
    """Infer median sampling interval in hours."""
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
        return 0.25  # default 15 min
    diffs = pd.Series(df.index).diff().dropna()
    median_sec = diffs.median().total_seconds()
    return median_sec / 3600.0
