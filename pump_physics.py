"""
pump_physics.py — Centrifugal Pump Physics-Based Analytics
============================================================
Five-phase progressive analysis for centrifugal pumps driven by induction motors.

Phase 1 — Power baseline       : Power / current only
Phase 2 — Hydraulic efficiency : Phase 1 + flow
Phase 3 — Full duty point      : Phase 2 + pressures + H-Q curve (mandatory)
Phase 4 — Thermodynamic        : Power + pressures + fluid temperatures
Phase 5 — Mechanical signature : Vibration + shaft speed
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, List
import re

G         = 9.81
RHO_WATER = 998.0
CP_WATER  = 4182.0
HIGH      = "HIGH"
MEDIUM    = "MEDIUM"
LOW       = "LOW"

KW_POWER   = ["power_kw","power","kw","active_power","shaft_power"]
KW_CURRENT = ["motor_current","current_a","current","_a","amp","amps"]
KW_FLOW    = ["flow_m3h","flowrate","flow","m3h","m3_h","gpm","flow_rate"]
KW_P_IN    = ["suction_pressure","suction_press","inlet_press","p_in","p_suction"]
KW_P_OUT   = ["discharge_pressure","discharge_press","outlet_press","p_out","p_discharge"]
KW_T_IN    = ["fluid_temp_in","temp_in","inlet_temp","t_in","t_suction","suction_temp"]
KW_T_OUT   = ["fluid_temp_out","temp_out","outlet_temp","t_out","t_discharge","discharge_temp"]
KW_VIBR    = ["vibration","vibr","mm_s","velocity_mm","rms_velocity"]
KW_SPEED   = ["speed_rpm","shaft_speed","speed","rpm","rps"]
KW_V_A     = ["voltage_a","v_a","u_ab"]
KW_V_B     = ["voltage_b","v_b","u_bc"]
KW_V_C     = ["voltage_c","v_c","u_ca"]
KW_I_A     = ["current_a","ia","phase_a_current"]
KW_I_B     = ["current_b","ib","phase_b_current"]
KW_I_C     = ["current_c","ic","phase_c_current"]


def _find(df, kws):
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in kws):
            return c
    return None


def _interval_h(df):
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
        return 0.25
    return pd.Series(df.index).diff().dropna().median().total_seconds() / 3600.0


def _trend_pm(s):
    if len(s) < 10:
        return 0.0
    x = np.arange(len(s))
    try:
        m, _ = np.polyfit(x, s.values.astype(float), 1)
        mean  = float(s.mean())
        dt_h  = (_interval_h_s(s))
        return float(m / mean * 100 * 720 / dt_h) if mean != 0 else 0.0
    except Exception:
        return 0.0


def _interval_h_s(s):
    if isinstance(s.index, pd.DatetimeIndex) and len(s) > 1:
        med = pd.Series(s.index).diff().dropna().median().total_seconds() / 3600.0
        return max(med, 0.0167)
    return 0.25


def detect_phase(df):
    cols = {
        "power":   _find(df, KW_POWER),
        "current": _find(df, KW_CURRENT),
        "flow":    _find(df, KW_FLOW),
        "p_in":    _find(df, KW_P_IN),
        "p_out":   _find(df, KW_P_OUT),
        "t_in":    _find(df, KW_T_IN),
        "t_out":   _find(df, KW_T_OUT),
        "vibr":    _find(df, KW_VIBR),
        "speed":   _find(df, KW_SPEED),
        "v_a":     _find(df, KW_V_A),
        "v_b":     _find(df, KW_V_B),
        "v_c":     _find(df, KW_V_C),
        "i_a":     _find(df, KW_I_A),
        "i_b":     _find(df, KW_I_B),
        "i_c":     _find(df, KW_I_C),
    }
    hp = bool(cols["power"] or cols["current"])
    hf = bool(cols["flow"])
    hpr= bool(cols["p_in"] and cols["p_out"])
    ht = bool(cols["t_in"] and cols["t_out"])
    hv = bool(cols["vibr"])
    hs = bool(cols["speed"])
    h3v= bool(cols["v_a"] and cols["v_b"] and cols["v_c"])
    h3i= bool(cols["i_a"] and cols["i_b"] and cols["i_c"])

    phases = []
    if hp:             phases.append(1)
    if hp and hf:      phases.append(2)
    if hp and hf and hpr: phases.append(3)
    if hp and hpr and ht: phases.append(4)
    if hv and hs:      phases.append(5)

    return {"cols": cols, "phases": phases,
            "highest_phase": max(phases) if phases else 0,
            "has_power": hp, "has_flow": hf, "has_pressure": hpr,
            "has_temp": ht, "has_vibr": hv, "has_speed": hs,
            "has_3v": h3v, "has_3i": h3i}


def parse_nameplate(desc):
    np_d = {
        "rated_power_kw": None, "rated_speed_rpm": None,
        "fla_amps": None, "voltage_v": None, "power_factor": None,
        "motor_efficiency": None, "ie_class": None, "poles": None,
        "frequency_hz": 50.0, "insulation_class": None,
        "commissioning_power_kw": None,
        "rated_flow_m3h": None, "rated_head_m": None,
        "pump_efficiency": None, "bep_flow_m3h": None,
        "pump_speed_rpm": None, "npsh_r_m": None,
        "impeller_diameter_mm": None, "n_vanes": None,
        "drive_type": "direct", "vfd_min_rpm": None, "vfd_max_rpm": None,
        "gear_ratio": None, "coupling_efficiency": 100.0,
        "rho": RHO_WATER, "cp": CP_WATER,
    }
    def n(pat, t):
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            try: return float(m.group(1).replace(",","."))
            except: return None
        return None
    t = desc
    np_d["rated_power_kw"]         = n(r"rated\s+power[:\s]+([\d.,]+)\s*kw",t) or n(r"power[:\s]+([\d.,]+)\s*kw",t)
    np_d["rated_speed_rpm"]        = n(r"rated\s+speed[:\s]+([\d.,]+)\s*rpm",t) or n(r"motor\s+speed[:\s]+([\d.,]+)",t)
    np_d["fla_amps"]               = n(r"fla[:\s]+([\d.,]+)",t) or n(r"full\s+load\s+amps?[:\s]+([\d.,]+)",t)
    np_d["voltage_v"]              = n(r"voltage[:\s]+([\d.,]+)\s*v",t) or n(r"supply\s*volt[a-z]*[:\s]+([\d.,]+)",t)
    np_d["power_factor"]           = n(r"power\s+factor[:\s]+([\d.,]+)",t) or n(r"cos\s*[φϕ][:\s]+([\d.,]+)",t)
    np_d["motor_efficiency"]       = n(r"motor\s+eff[a-z]*[:\s]+([\d.,]+)\s*%?",t)
    np_d["commissioning_power_kw"] = n(r"commission[a-z]*\s+power[:\s]+([\d.,]+)",t)
    np_d["frequency_hz"]           = n(r"freq[a-z]*[:\s]+([\d.,]+)\s*hz",t) or 50.0
    np_d["poles"]                  = n(r"(\d)\s*[-]?\s*pole",t)
    _ie = re.search(r"\bie\s*([1-4])\b",t,re.IGNORECASE)
    if _ie: np_d["ie_class"] = int(_ie.group(1))
    np_d["rated_flow_m3h"]         = n(r"rated\s+flow[:\s]+([\d.,]+)\s*m3",t) or n(r"flow[:\s]+([\d.,]+)\s*m3",t)
    np_d["rated_head_m"]           = n(r"rated\s+head[:\s]+([\d.,]+)\s*m\b",t) or n(r"head[:\s]+([\d.,]+)\s*m\b",t)
    np_d["pump_efficiency"]        = n(r"pump\s+eff[a-z]*[:\s]+([\d.,]+)\s*%?",t) or n(r"efficiency[:\s]+([\d.,]+)\s*%?",t)
    np_d["bep_flow_m3h"]           = n(r"bep\s+flow[:\s]+([\d.,]+)",t)
    np_d["pump_speed_rpm"]         = n(r"pump\s+speed[:\s]+([\d.,]+)",t)
    np_d["npsh_r_m"]               = n(r"npsh[_r]?[:\s]+([\d.,]+)\s*m?",t)
    np_d["impeller_diameter_mm"]   = n(r"impeller[:\s]+([\d.,]+)\s*mm",t) or n(r"diameter[:\s]+([\d.,]+)\s*mm",t)
    np_d["n_vanes"]                = n(r"vanes?[:\s]+(\d+)",t) or n(r"blades?[:\s]+(\d+)",t)
    if "vfd" in t.lower():
        np_d["drive_type"] = "VFD"
        np_d["vfd_min_rpm"] = n(r"min\s+speed[:\s]+([\d.,]+)\s*rpm",t)
        np_d["vfd_max_rpm"] = n(r"max\s+speed[:\s]+([\d.,]+)\s*rpm",t)
    elif "belt" in t.lower(): np_d["drive_type"] = "belt"
    elif "gearbox" in t.lower():
        np_d["drive_type"] = "gearbox"
        np_d["gear_ratio"] = n(r"gear\s+ratio[:\s]+([\d.,]+)",t)
        np_d["coupling_efficiency"] = n(r"gearbox\s+eff[a-z]*[:\s]+([\d.,]+)\s*%?",t) or 98.0
    rho = n(r"density[:\s]+([\d.,]+)\s*kg",t)
    if rho: np_d["rho"] = rho
    cp = n(r"(?:specific\s+heat|cp)[:\s]+([\d.,]+)",t)
    if cp: np_d["cp"] = cp
    return np_d


def check_phase4_viability(df, cols, rated_efficiency=0.78, rho=RHO_WATER, cp=CP_WATER):
    r = {"viable": False, "level": "not_viable", "expected_delta_t": 0.0,
         "mean_delta_p": 0.0, "message": "", "recommendation": ""}
    if not cols.get("p_in") or not cols.get("p_out"):
        r["message"] = "Differential pressure not available."
        return r
    dp = (df[cols["p_out"]] - df[cols["p_in"]]).dropna()
    dp = dp[dp > 0]
    if len(dp) < 10:
        r["message"] = "Insufficient pressure data."
        return r
    mean_dp = float(dp.mean())
    eta = min(max(rated_efficiency, 0.5), 0.95)
    expected_dt = (1.0 - eta) * mean_dp * 1e5 / (rho * cp)
    r["mean_delta_p"] = round(mean_dp, 3)
    r["expected_delta_t"] = round(expected_dt, 4)
    if expected_dt < 0.05:
        r["level"] = "not_viable"
        r["message"] = (f"Thermodynamic method **not viable**. Mean \u0394P: **{mean_dp:.2f} bar** \u2192 "
                        f"Expected \u0394T: **{expected_dt*1000:.1f} m\u00b0C** ({expected_dt:.4f}\u00b0C). "
                        f"Below minimum measurable threshold of 0.05\u00b0C.")
        r["recommendation"] = "Use Phase 3 (pressure + flow) for efficiency monitoring."
    elif expected_dt < 0.10:
        r["level"] = "marginal"
        r["message"] = (f"Thermodynamic method **marginal**. Expected \u0394T: **{expected_dt*1000:.1f} m\u00b0C**. "
                        f"Above minimum but below reliable threshold (0.10\u00b0C).")
        r["recommendation"] = "Use calibrated PT1000 RTDs (\u00b10.02\u00b0C). Cross-check with Phase 3."
    else:
        r["level"] = "viable"
        r["viable"] = True
        r["message"] = (f"Thermodynamic method **viable**. Expected \u0394T: **{expected_dt*1000:.1f} m\u00b0C** "
                        f"exceeds 0.10\u00b0C reliable threshold.")
        r["recommendation"] = "Use PT100/PT1000 RTDs with \u00b10.02\u00b0C accuracy."
    return r


def run_phase1(df, cols, np_d):
    res = {"phase": 1, "name": "Power Baseline", "metrics": {}, "findings": [], "warnings": []}
    rated_kw = np_d.get("rated_power_kw") or 0
    comm_kw  = np_d.get("commissioning_power_kw") or 0
    voltage  = np_d.get("voltage_v") or 415.0
    pf       = np_d.get("power_factor") or 0.85
    eta_m    = (np_d.get("motor_efficiency") or 93.0) / 100.0
    ie       = np_d.get("ie_class") or 3
    dt_h     = _interval_h(df)

    if cols.get("power"):
        power = df[cols["power"]].dropna()
    elif cols.get("current"):
        power = np.sqrt(3) * voltage * df[cols["current"]].dropna() * pf / 1000.0
    else:
        res["warnings"].append("No power or current column found."); return res

    if rated_kw <= 0:
        res["warnings"].append("Rated motor power not entered — load factor cannot be calculated.")
        rated_kw = float(power.median()) * 1.2 if len(power) > 0 else 1.0

    running = power[power > 0.05 * rated_kw]
    if len(running) < 10:
        res["warnings"].append("Insufficient running data."); return res

    mean_p   = float(running.mean())
    lf       = round(mean_p / rated_kw * 100, 1)
    shaft_p  = round(mean_p * eta_m, 2)
    spi      = round(mean_p / comm_kw, 3) if comm_kw > 0 else None
    energy   = round(float((power * dt_h).sum()), 1)
    run_h    = round(len(running) * dt_h, 1)

    roll = (running.resample("1h").mean().dropna()
            if isinstance(running.index, pd.DatetimeIndex) else running)
    trend = round(_trend_pm(roll), 2) if len(roll) >= 14 else None

    res["metrics"]["Mean input power (kW)"]     = round(mean_p, 2)
    res["metrics"]["Rated power (kW)"]           = rated_kw
    res["metrics"]["Estimated shaft power (kW)"] = shaft_p
    res["metrics"]["Load factor (%)"]            = lf
    if spi is not None: res["metrics"]["Specific power index"] = spi
    if trend is not None: res["metrics"]["Power trend (%/month)"] = trend
    res["metrics"]["Total energy (kWh)"]         = energy
    res["metrics"]["Running hours"]              = run_h

    # Load factor
    if lf > 100:
        res["findings"].append({"finding": "Motor overloaded",
            "detail": f"Load factor {lf}% exceeds rated. Risk of winding damage.",
            "severity": "critical", "confidence": HIGH})
    elif lf > 90:
        res["findings"].append({"finding": "High load factor",
            "detail": f"Load factor {lf}% near rated limit. Monitor winding temperature.",
            "severity": "warning", "confidence": HIGH})
    elif lf < 40:
        res["findings"].append({"finding": "Low load factor — possible oversizing",
            "detail": f"Load factor only {lf}%. Consider impeller trim or speed reduction.",
            "severity": "info", "confidence": MEDIUM})

    # SPI
    if spi is not None:
        if spi > 1.20:
            res["findings"].append({"finding": "High specific power index",
                "detail": f"SPI {spi:.3f} — {(spi-1)*100:.0f}% more power than at commissioning.",
                "severity": "critical", "confidence": MEDIUM})
        elif spi > 1.10:
            res["findings"].append({"finding": "Elevated specific power index",
                "detail": f"SPI {spi:.3f} — 10%+ above commissioning baseline. Investigate impeller.",
                "severity": "warning", "confidence": MEDIUM})

    # Trend
    if trend is not None and abs(trend) > 1.0:
        d = "rising" if trend > 0 else "falling"
        res["findings"].append({"finding": f"Power trend {d}",
            "detail": f"Power {d} at {abs(trend):.1f}%/month.",
            "severity": "warning" if trend > 1.0 else "info",
            "confidence": MEDIUM if abs(trend) > 2.0 else LOW})

    # Voltage imbalance (IEC 60034-26)
    if cols.get("v_a") and cols.get("v_b") and cols.get("v_c"):
        va = df[cols["v_a"]].dropna(); vb = df[cols["v_b"]].dropna(); vc = df[cols["v_c"]].dropna()
        ci = va.index.intersection(vb.index).intersection(vc.index)
        if len(ci) >= 10:
            va_m, vb_m, vc_m = float(va.loc[ci].mean()), float(vb.loc[ci].mean()), float(vc.loc[ci].mean())
            v_avg = (va_m + vb_m + vc_m) / 3.0
            vi = max(abs(va_m-v_avg), abs(vb_m-v_avg), abs(vc_m-v_avg)) / v_avg * 100
            vi = round(vi, 2)
            res["metrics"]["Voltage imbalance V\u2082/V\u2081 (%)"] = vi
            res["metrics"]["Mean line voltage (V)"] = round(v_avg, 1)
            limits = {1: 2.0, 2: 1.5, 3: 1.0, 4: 0.5}
            lim = limits.get(ie, 1.0)
            if vi > lim * 2:
                res["findings"].append({"finding": "Severe voltage imbalance",
                    "detail": f"V\u2082/V\u2081 = {vi}% exceeds IE{ie} limit {lim}% by {vi/lim:.1f}\u00d7. Risk of motor damage. Investigate supply.",
                    "severity": "critical", "confidence": HIGH})
            elif vi > lim:
                res["findings"].append({"finding": "Voltage imbalance above IE class limit",
                    "detail": f"V\u2082/V\u2081 = {vi}% exceeds IE{ie} limit {lim}% per IEC 60034-26.",
                    "severity": "warning", "confidence": HIGH})

    # Current imbalance
    if cols.get("i_a") and cols.get("i_b") and cols.get("i_c"):
        ia = df[cols["i_a"]].dropna(); ib = df[cols["i_b"]].dropna(); ic = df[cols["i_c"]].dropna()
        ci = ia.index.intersection(ib.index).intersection(ic.index)
        if len(ci) >= 10:
            ia_m, ib_m, ic_m = float(ia.loc[ci].mean()), float(ib.loc[ci].mean()), float(ic.loc[ci].mean())
            i_avg = (ia_m + ib_m + ic_m) / 3.0
            ii = max(abs(ia_m-i_avg), abs(ib_m-i_avg), abs(ic_m-i_avg)) / i_avg * 100
            res["metrics"]["Current imbalance (%)"] = round(ii, 2)
            if ii > 10:
                res["findings"].append({"finding": "High current imbalance",
                    "detail": f"Current imbalance {ii:.1f}%. Check supply voltage balance and motor winding.",
                    "severity": "warning", "confidence": MEDIUM})

    return res


def run_phase2(df, cols, np_d):
    res = {"phase": 2, "name": "Hydraulic Efficiency", "metrics": {}, "findings": [], "warnings": []}
    rated_kw  = np_d.get("rated_power_kw") or 0
    rated_q   = np_d.get("rated_flow_m3h") or 0
    rated_h   = np_d.get("rated_head_m") or 0
    bep_q     = np_d.get("bep_flow_m3h") or (rated_q * 0.9 if rated_q else 0)
    eta_pump  = (np_d.get("pump_efficiency") or 78.0) / 100.0
    eta_m     = (np_d.get("motor_efficiency") or 93.0) / 100.0
    eta_c     = (np_d.get("coupling_efficiency") or 100.0) / 100.0
    rho       = np_d.get("rho") or RHO_WATER
    voltage   = np_d.get("voltage_v") or 415.0
    pf        = np_d.get("power_factor") or 0.85

    if cols.get("power"):
        power = df[cols["power"]].dropna()
    elif cols.get("current"):
        power = np.sqrt(3) * voltage * df[cols["current"]].dropna() * pf / 1000.0
    else:
        res["warnings"].append("No power column."); return res

    flow = df[cols["flow"]].dropna()
    ci   = power.index.intersection(flow.index)
    if len(ci) < 10:
        res["warnings"].append("Insufficient aligned power + flow data."); return res

    p = power.loc[ci]; q = flow.loc[ci]
    minp = 0.05 * rated_kw if rated_kw > 0 else float(p.quantile(0.1))
    minq = 0.05 * rated_q  if rated_q  > 0 else float(q.quantile(0.1))
    mask = (p > minp) & (q > minq)
    p, q = p[mask], q[mask]
    if len(p) < 10:
        res["warnings"].append("Insufficient running data."); return res

    sp_e = (p / q).replace([np.inf,-np.inf], np.nan).dropna()
    mean_spe  = float(sp_e.mean())
    comm_spe  = (rated_kw / rated_q) if rated_q > 0 else None
    spe_rise  = round((mean_spe - comm_spe) / comm_spe * 100, 1) if comm_spe else None

    q_m3s = q / 3600.0
    if rated_h > 0:
        eta_w2w      = rho * G * rated_h * q_m3s / (p * 1000.0)
        mean_w2w     = float(eta_w2w.mean())
        mean_eta_act = mean_w2w / (eta_m * eta_c)
    else:
        mean_w2w = mean_eta_act = None

    mean_q    = float(q.mean())
    bep_dev   = round((mean_q - bep_q) / bep_q * 100, 1) if bep_q > 0 else None
    corr      = float(p.corr(q))
    shaft_pw  = float((p * eta_m * eta_c).mean())

    roll = (sp_e.resample("1h").mean().dropna()
            if isinstance(sp_e.index, pd.DatetimeIndex) else sp_e)
    spe_trend = round(_trend_pm(roll), 2) if len(roll) >= 14 else 0.0

    res["metrics"]["Mean flow (m\u00b3/h)"]                = round(mean_q, 1)
    res["metrics"]["Mean input power (kW)"]              = round(float(p.mean()), 2)
    res["metrics"]["Estimated shaft power (kW)"]         = round(shaft_pw, 2)
    res["metrics"]["Specific energy (kWh/m\u00b3)"]       = round(mean_spe, 4)
    if comm_spe:
        res["metrics"]["Commissioning specific energy (kWh/m\u00b3)"] = round(comm_spe, 4)
        res["metrics"]["Specific energy vs commissioning (%)"] = spe_rise
    res["metrics"]["Specific energy trend (%/month)"]    = spe_trend
    if mean_w2w is not None:
        res["metrics"]["Wire-to-water efficiency (%)"]   = round(mean_w2w * 100, 1)
    if mean_eta_act is not None:
        res["metrics"]["Estimated pump efficiency (%)"]  = round(mean_eta_act * 100, 1)
        res["metrics"]["Rated pump efficiency (%)"]      = round(eta_pump * 100, 1)
    if bep_dev is not None:
        res["metrics"]["BEP deviation (%)"]              = bep_dev
        res["metrics"]["BEP flow (m\u00b3/h)"]           = bep_q
    res["metrics"]["Flow-power correlation (r)"]         = round(corr, 3)

    if spe_rise is not None:
        if spe_rise > 15:
            res["findings"].append({"finding": "Specific energy significantly elevated",
                "detail": f"Specific energy {spe_rise:.1f}% above commissioning. Significant hydraulic deterioration.",
                "severity": "critical", "confidence": HIGH})
        elif spe_rise > 8:
            res["findings"].append({"finding": "Specific energy rising",
                "detail": f"Specific energy {spe_rise:.1f}% above commissioning. Investigate impeller.",
                "severity": "warning", "confidence": HIGH})

    if spe_trend > 2.0:
        res["findings"].append({"finding": "Specific energy trend rising",
            "detail": f"Rising at {spe_trend:.1f}%/month. Progressive efficiency loss.",
            "severity": "warning", "confidence": MEDIUM})

    if mean_eta_act is not None:
        drop = (eta_pump - mean_eta_act) / eta_pump * 100
        if drop > 10:
            res["findings"].append({"finding": "Pump efficiency significantly below rated",
                "detail": f"Estimated {mean_eta_act*100:.1f}% vs rated {eta_pump*100:.0f}%. Drop {drop:.1f}%.",
                "severity": "warning", "confidence": MEDIUM})

    if bep_dev is not None and abs(bep_dev) > 25:
        res["findings"].append({"finding": "Operating far from BEP",
            "detail": f"Flow {bep_dev:+.1f}% from BEP ({bep_q} m\u00b3/h). Risk of recirculation and accelerated wear.",
            "severity": "warning", "confidence": HIGH})

    if corr < 0.70:
        res["findings"].append({"finding": "Low flow-power correlation",
            "detail": f"r = {corr:.2f}. Possible internal bypass, worn wear ring, or valve cycling.",
            "severity": "warning", "confidence": LOW})

    return res


def run_phase3(df, cols, np_d, hq_curve=None):
    res = {"phase": 3, "name": "Full Duty Point", "metrics": {}, "findings": [], "warnings": []}
    rated_kw  = np_d.get("rated_power_kw") or 0
    rated_q   = np_d.get("rated_flow_m3h") or 0
    eta_pump  = (np_d.get("pump_efficiency") or 78.0) / 100.0
    eta_m     = (np_d.get("motor_efficiency") or 93.0) / 100.0
    eta_c     = (np_d.get("coupling_efficiency") or 100.0) / 100.0
    rho       = np_d.get("rho") or RHO_WATER
    npsh_r    = np_d.get("npsh_r_m")
    voltage   = np_d.get("voltage_v") or 415.0
    pf        = np_d.get("power_factor") or 0.85

    if cols.get("power"):
        power = df[cols["power"]].dropna()
    elif cols.get("current"):
        power = np.sqrt(3) * voltage * df[cols["current"]].dropna() * pf / 1000.0
    else:
        res["warnings"].append("No power column."); return res

    flow  = df[cols["flow"]].dropna()
    p_in  = df[cols["p_in"]].dropna()
    p_out = df[cols["p_out"]].dropna()
    ci    = power.index.intersection(flow.index).intersection(p_in.index).intersection(p_out.index)
    if len(ci) < 10:
        res["warnings"].append("Insufficient aligned data."); return res

    pw, q, pi, po = power.loc[ci], flow.loc[ci], p_in.loc[ci], p_out.loc[ci]
    minp = 0.05 * rated_kw if rated_kw > 0 else float(pw.quantile(0.1))
    minq = 0.05 * rated_q  if rated_q  > 0 else float(q.quantile(0.1))
    mask = (pw > minp) & (q > minq) & ((po - pi) > 0)
    pw, q, pi, po = pw[mask], q[mask], pi[mask], po[mask]
    if len(pw) < 10:
        res["warnings"].append("Insufficient running data."); return res

    dp   = po - pi
    head = dp * 1e5 / (rho * G)
    q_s  = q / 3600.0

    mean_h  = float(head.mean())
    mean_q  = float(q.mean())
    mean_dp = float(dp.mean())

    eta_hyd = rho * G * head * q_s / (pw * 1000.0)
    eta_hyd = eta_hyd[(eta_hyd > 0.1) & (eta_hyd < 1.0)]
    mean_eta_hyd = float(eta_hyd.mean()) if len(eta_hyd) > 0 else None

    shaft_pw = float((pw * eta_m * eta_c).mean())
    k_vals   = (dp * 1e5 / (q_s ** 2)).replace([np.inf,-np.inf], np.nan).dropna()
    mean_k   = float(k_vals.mean()) if len(k_vals) > 0 else 0.0
    npsh_a   = float((pi * 1e5 / (rho * G)).mean())

    wear_idx = None; duty_dev = None
    if hq_curve and len(hq_curve) >= 3:
        try:
            qc = np.array([pt["q"] for pt in hq_curve])
            hc = np.array([pt["h"] for pt in hq_curve])
            idx = np.argsort(qc); qc, hc = qc[idx], hc[idx]
            exp_h = float(np.interp(mean_q, qc, hc))
            if exp_h > 0:
                wear_idx = round((exp_h - mean_h) / exp_h * 100, 1)
                duty_dev = round((mean_h - exp_h) / exp_h * 100, 1)
        except Exception:
            pass

    res["metrics"]["Mean flow (m\u00b3/h)"]                 = round(mean_q, 1)
    res["metrics"]["Mean differential pressure (bar)"]    = round(mean_dp, 3)
    res["metrics"]["Mean total head (m)"]                  = round(mean_h, 1)
    res["metrics"]["Estimated shaft power (kW)"]           = round(shaft_pw, 2)
    if mean_eta_hyd is not None:
        res["metrics"]["True hydraulic efficiency (%)"]    = round(mean_eta_hyd * 100, 1)
        res["metrics"]["Rated pump efficiency (%)"]        = round(eta_pump * 100, 1)
        drop = round((eta_pump - mean_eta_hyd) / eta_pump * 100, 1)
        res["metrics"]["Efficiency vs rated (%)"]          = -drop
    if wear_idx is not None: res["metrics"]["Impeller wear index (%)"]       = wear_idx
    if duty_dev is not None: res["metrics"]["Duty point head deviation (%)"] = duty_dev
    res["metrics"]["System resistance K (Pa\u00b7s\u00b2/m\u00b3)"]          = round(mean_k, 0)
    res["metrics"]["Inlet pressure head (m)"]              = round(npsh_a, 1)
    if npsh_r:
        res["metrics"]["NPSH required (m)"]                = npsh_r
        res["metrics"]["NPSH margin (m)"]                  = round(npsh_a - npsh_r, 2)

    if mean_eta_hyd is not None:
        drop = (eta_pump - mean_eta_hyd) / eta_pump * 100
        if drop > 12:
            res["findings"].append({"finding": "True hydraulic efficiency significantly below rated",
                "detail": f"Actual {mean_eta_hyd*100:.1f}% vs rated {eta_pump*100:.0f}%. Drop {drop:.1f}%. Impeller wear or increased clearances likely.",
                "severity": "critical", "confidence": HIGH})
        elif drop > 6:
            res["findings"].append({"finding": "Hydraulic efficiency below rated",
                "detail": f"Efficiency {mean_eta_hyd*100:.1f}% ({drop:.1f}% below rated).",
                "severity": "warning", "confidence": HIGH})

    if wear_idx is not None:
        if wear_idx > 10:
            res["findings"].append({"finding": "Significant impeller wear",
                "detail": f"Wear index {wear_idx:.1f}%. Actual head {wear_idx:.1f}% below H-Q curve. Plan impeller replacement.",
                "severity": "critical", "confidence": HIGH})
        elif wear_idx > 5:
            res["findings"].append({"finding": "Impeller wear developing",
                "detail": f"Wear index {wear_idx:.1f}%. Duty point shifting below manufacturer curve.",
                "severity": "warning", "confidence": HIGH})
    elif hq_curve is None or len(hq_curve) < 3:
        res["warnings"].append("No H-Q curve provided — impeller wear index cannot be calculated. H-Q curve is mandatory for Phase 3.")

    if npsh_r:
        margin = npsh_a - npsh_r
        if margin < 1.0:
            res["findings"].append({"finding": "Cavitation risk — NPSH margin critical",
                "detail": f"NPSH available {npsh_a:.1f} m, required {npsh_r:.1f} m. Margin {margin:.2f} m below 1 m safe minimum.",
                "severity": "critical", "confidence": HIGH})
        elif margin < 3.0:
            res["findings"].append({"finding": "Reduced NPSH margin",
                "detail": f"NPSH margin {margin:.2f} m. Below recommended 3 m. Monitor suction conditions.",
                "severity": "warning", "confidence": HIGH})

    return res


def run_phase4(df, cols, np_d):
    res = {"phase": 4, "name": "Thermodynamic Efficiency", "metrics": {}, "findings": [], "warnings": []}
    eta_pump = (np_d.get("pump_efficiency") or 78.0) / 100.0
    rho      = np_d.get("rho") or RHO_WATER
    cp       = np_d.get("cp") or CP_WATER

    viab = check_phase4_viability(df, cols, eta_pump, rho, cp)
    res["metrics"]["Viability"] = viab["level"]
    res["metrics"]["Expected \u0394T (\u00b0C)"] = viab["expected_delta_t"]
    if not viab["viable"]:
        res["warnings"].append(viab["message"])
        if viab.get("recommendation"):
            res["warnings"].append(f"Recommendation: {viab['recommendation']}")
        return res

    p_in  = df[cols["p_in"]].dropna(); p_out = df[cols["p_out"]].dropna()
    t_in  = df[cols["t_in"]].dropna(); t_out = df[cols["t_out"]].dropna()
    ci    = p_in.index.intersection(p_out.index).intersection(t_in.index).intersection(t_out.index)
    if len(ci) < 10:
        res["warnings"].append("Insufficient aligned data."); return res

    dp = (p_out.loc[ci] - p_in.loc[ci]);  dp = dp[dp > 0]
    dt = (t_out.loc[ci] - t_in.loc[ci]).loc[dp.index]
    eta_td = 1.0 - (cp * rho * dt) / (dp * 1e5)
    eta_td = eta_td[(eta_td > 0.3) & (eta_td < 1.0)]
    if len(eta_td) < 10:
        res["warnings"].append("Insufficient valid thermodynamic efficiency readings."); return res

    mean_eta = float(eta_td.mean())
    mean_dp  = float(dp.mean())
    mean_dt  = float(dt.mean())
    dt_th    = (1.0 - eta_pump) * mean_dp * 1e5 / (rho * cp)
    recirc   = round((mean_dt - dt_th) / dt_th * 100, 1) if dt_th > 0 else None

    res["metrics"]["Thermodynamic efficiency (%)"] = round(mean_eta * 100, 1)
    res["metrics"]["Rated pump efficiency (%)"]    = round(eta_pump * 100, 1)
    res["metrics"]["Mean \u0394P (bar)"]           = round(mean_dp, 3)
    res["metrics"]["Mean \u0394T (\u00b0C)"]       = round(mean_dt, 4)
    res["metrics"]["Theoretical \u0394T (\u00b0C)"]= round(dt_th, 4)
    if recirc is not None: res["metrics"]["Recirculation index (%)"] = recirc

    drop = (eta_pump - mean_eta) / eta_pump * 100
    if drop > 10:
        res["findings"].append({"finding": "Thermodynamic efficiency significantly below rated",
            "detail": f"Measured {mean_eta*100:.1f}% vs rated {eta_pump*100:.0f}%. Drop {drop:.1f}% confirmed independently of flow meter.",
            "severity": "critical", "confidence": HIGH})
    elif drop > 5:
        res["findings"].append({"finding": "Thermodynamic efficiency below rated",
            "detail": f"Efficiency {mean_eta*100:.1f}% ({drop:.1f}% below rated).",
            "severity": "warning", "confidence": HIGH})
    if recirc is not None and recirc > 25:
        res["findings"].append({"finding": "Internal recirculation detected",
            "detail": f"Recirculation index {recirc:.0f}%. Operating far from BEP causing excess fluid heating.",
            "severity": "warning", "confidence": HIGH})
    return res


def run_phase5(df, cols, np_d, bearing_freqs=None):
    res = {"phase": 5, "name": "Mechanical Signature", "metrics": {}, "findings": [], "warnings": []}
    n_vanes   = int(np_d.get("n_vanes") or 0)
    rated_spd = np_d.get("rated_speed_rpm") or np_d.get("pump_speed_rpm") or 0

    vibr  = df[cols["vibr"]].dropna()
    speed = df[cols["speed"]].dropna()
    if len(vibr) < 10:
        res["warnings"].append("Insufficient vibration data."); return res

    ci      = vibr.index.intersection(speed.index) if len(speed) > 0 else vibr.index
    vibr_c  = vibr.loc[ci] if len(ci) > 5 else vibr
    spd_c   = speed.loc[ci] if len(ci) > 5 and len(speed) > 0 else None
    mean_v  = float(vibr_c.mean()); max_v = float(vibr_c.max())
    mean_s  = float(spd_c.mean()) if spd_c is not None and len(spd_c) > 0 else (rated_spd or 0)
    roll_v  = (vibr_c.resample("1h").mean().dropna()
               if isinstance(vibr_c.index, pd.DatetimeIndex) else vibr_c)
    vtrend  = round(_trend_pm(roll_v), 2) if len(roll_v) >= 14 else 0.0
    bpf     = round(mean_s / 60.0 * n_vanes, 1) if mean_s > 0 and n_vanes > 0 else None

    affl_dev = None
    if np_d.get("drive_type") == "VFD" and spd_c is not None and rated_spd > 0:
        if cols.get("power"):
            pw_c = df[cols["power"]].dropna().loc[ci] if ci is not None else None
            if pw_c is not None and len(pw_c) > 20:
                n_r = (spd_c / rated_spd).clip(0.3, 1.1)
                expected = pw_c.mean() * (n_r ** 3)
                dev = ((pw_c - expected) / expected).abs().mean() * 100
                affl_dev = round(float(dev), 1)

    iso_warn, iso_alarm = 4.5, 7.1
    res["metrics"]["Mean vibration (mm/s RMS)"]      = round(mean_v, 3)
    res["metrics"]["Peak vibration (mm/s RMS)"]      = round(max_v, 3)
    res["metrics"]["Vibration trend (%/month)"]      = vtrend
    res["metrics"]["Mean shaft speed (RPM)"]         = round(mean_s, 0)
    res["metrics"]["ISO 10816 warning (mm/s)"]       = iso_warn
    res["metrics"]["ISO 10816 alarm (mm/s)"]         = iso_alarm
    if bpf: res["metrics"]["Blade pass frequency (Hz)"] = bpf
    if affl_dev is not None: res["metrics"]["Affinity law deviation (%)"] = affl_dev
    if bearing_freqs:
        for k, v in bearing_freqs.items():
            res["metrics"][f"Bearing freq {k} (Hz)"] = round(v, 2)

    if mean_v > iso_alarm:
        res["findings"].append({"finding": "Vibration above ISO alarm level",
            "detail": f"Mean {mean_v:.2f} mm/s exceeds ISO 10816 Class II alarm {iso_alarm} mm/s. Immediate action.",
            "severity": "critical", "confidence": HIGH})
    elif mean_v > iso_warn:
        res["findings"].append({"finding": "Vibration above ISO warning level",
            "detail": f"Mean {mean_v:.2f} mm/s exceeds ISO 10816 warning {iso_warn} mm/s. Schedule inspection.",
            "severity": "warning", "confidence": HIGH})
    if vtrend > 5.0:
        res["findings"].append({"finding": "Vibration rising rapidly",
            "detail": f"+{vtrend:.1f}%/month. Investigate imbalance, misalignment, or bearing deterioration.",
            "severity": "warning", "confidence": MEDIUM})
    elif vtrend > 2.0:
        res["findings"].append({"finding": "Vibration rising",
            "detail": f"+{vtrend:.1f}%/month. Monitor closely.",
            "severity": "info", "confidence": MEDIUM})
    if affl_dev is not None and affl_dev > 10:
        res["findings"].append({"finding": "Affinity law deviation",
            "detail": f"Power deviates {affl_dev:.1f}% from N\u00b3 prediction. Internal hydraulic deterioration possible.",
            "severity": "warning", "confidence": MEDIUM})
    return res


def run_all_phases(df, machine_description, hq_curve=None, bearing_freqs=None):
    """Run all applicable phases and return combined structured results."""
    phase_info = detect_phase(df)
    np_d       = parse_nameplate(machine_description)
    phases_run = {}

    if 1 in phase_info["phases"]:
        phases_run[1] = run_phase1(df, phase_info["cols"], np_d)
    if 2 in phase_info["phases"]:
        phases_run[2] = run_phase2(df, phase_info["cols"], np_d)
    if 3 in phase_info["phases"]:
        phases_run[3] = run_phase3(df, phase_info["cols"], np_d, hq_curve)
    if 4 in phase_info["phases"]:
        viab = check_phase4_viability(df, phase_info["cols"],
                                       (np_d.get("pump_efficiency") or 78.0) / 100.0,
                                       np_d.get("rho") or RHO_WATER,
                                       np_d.get("cp") or CP_WATER)
        if viab["viable"]:
            phases_run[4] = run_phase4(df, phase_info["cols"], np_d)
    if 5 in phase_info["phases"]:
        phases_run[5] = run_phase5(df, phase_info["cols"], np_d, bearing_freqs)

    all_findings = []
    for ph_num, ph_res in phases_run.items():
        for f in ph_res.get("findings", []):
            all_findings.append({**f, "phase": ph_num})
    sev = {"critical": 0, "warning": 1, "info": 2}
    all_findings.sort(key=lambda x: sev.get(x.get("severity","info"), 2))

    lines = ["=== CENTRIFUGAL PUMP PHYSICS ANALYSIS ===",
             f"Phases activated: {sorted(phases_run.keys())}",
             f"Drive type: {np_d.get('drive_type','unknown')}",
             f"Rated power: {np_d.get('rated_power_kw','not entered')} kW",
             ""]
    for ph_num, ph_res in sorted(phases_run.items()):
        lines.append(f"--- Phase {ph_num}: {ph_res['name']} ---")
        for w in ph_res.get("warnings", []):
            lines.append(f"  WARNING: {w}")
        for k, v in ph_res.get("metrics", {}).items():
            lines.append(f"  {k}: {v}")
        for f in ph_res.get("findings", []):
            lines.append(f"  FINDING [{f['severity'].upper()}/{f['confidence']}]: {f['finding']} — {f['detail']}")
        lines.append("")

    return {"phase_info": phase_info, "np_data": np_d, "phases": phases_run,
            "all_findings": all_findings, "summary": "\n".join(lines)}
