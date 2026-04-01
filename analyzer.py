"""
analyzer.py — Claude analysis engine with adaptive ML pre-processing

Flow:
  1. ml_engine.run() selects tier and computes pre-processed signals
  2. Signals injected into Claude prompt as structured context
  3. Claude interprets signals + raw stats + machine profile
  4. Returns structured JSON insights
"""

import json
import os
from typing import Optional

import anthropic
import pandas as pd

import ml_engine
import alert_engine

SYSTEM_PROMPT = """You are an expert industrial machine health analyst with deep knowledge of rotating equipment, process machinery, and condition monitoring.

Your job is to analyse time-series sensor data from industrial machines and return ONLY a valid JSON object — no preamble, no markdown code blocks, no explanation outside the JSON.

The JSON must exactly follow this schema:

{
  "health_score": <integer 0–100, where 100 = perfect condition>,
  "score_breakdown": [
    {
      "factor": "<short label for what drove the score, e.g. 'Vibration critical breach' or 'All parameters normal'>",
      "impact": "positive" | "negative" | "neutral",
      "detail": "<one sentence explaining the specific signal and magnitude, e.g. 'vibration_mm_s exceeded critical threshold (4.5 mm/s) in 23 readings (0.27% of period)'>",
      "weight": "high" | "medium" | "low"
    }
  ],
  "kpis": [
    {
      "label": "<short parameter name>",
      "value": "<value with unit>",
      "status": "normal" | "warning" | "critical"
    }
  ],
  "insights": [
    "<one concrete, actionable observation per string>"
  ],
  "anomalies": [
    {
      "parameter": "<column name or pattern name for schedule compliance>",
      "description": "<what is anomalous — pattern, frequency, cumulative hours/impact>",
      "corrective_action": "<specific recommended action to resolve this anomaly>",
      "potential_impact": "<quantified benefit of fixing this — hours saved, cost reduction, risk reduction>"
    }
  ],
  "chart_recommendations": [
    {
      "type": "time_series" | "histogram" | "correlation" | "scatter",
      "title": "<descriptive chart title>",
      "parameters": ["<column_name_1>", "<column_name_2>"],
      "note": "<optional interpretation hint>"
    }
  ],
  "narrative": "<2–3 sentence plain-language summary an engineer can act on immediately>"
}

Rules:
- health_score must reflect the actual data, not be optimistic by default.
- Only reference column names that appear in the dataset.
- kpis should cover the 3–5 most critical parameters for this machine type.
- insights should be specific (include values, trends, percentages) not generic.
- anomalies array may be empty [] if none are detected.
- Always recommend at least one chart_recommendation.
- narrative must name the machine type and highlight the single most important finding.
- FOCUS: You must ONLY perform the analysis type specified in the REQUESTED ANALYSIS section.
  Do not mix analysis types. If asked for Schedule Compliance, report only schedule findings.
  If asked for Anomaly Detection, report only anomalies. Do not include unrelated findings.
- SCORE BREAKDOWN: Always include a score_breakdown array explaining what drove the health score.
  List 3–6 factors maximum. Each factor must reference specific numbers from the pre-computed signals.
  Use weight "high" for threshold breaches and critical violations, "medium" for warning-level findings
  and significant trends, "low" for minor statistical signals. Always include at least one factor even
  if all parameters are normal (e.g. "All parameters within normal range" with impact "positive").
  The score_breakdown must be honest — if the score is 65, the breakdown must show what prevented it
  from being higher. Never list vague factors like "overall condition" — always cite specific parameters
  and values.
- THRESHOLDS: If the machine profile contains a PARAMETER THRESHOLDS section, use those
  exact warning and critical limits to classify KPI status and flag anomalies. For parameters
  WITHOUT defined thresholds, use the pre-computed statistical signals. Always state whether
  a finding is based on defined thresholds or statistical inference.
- PRE-COMPUTED SIGNALS: The prompt includes signals from an adaptive pre-processing engine.
  Use these signals as primary evidence. Reference specific numbers (outlier counts, trend
  percentages, control chart violations, Isolation Forest results) in your insights.
- DATA VOLUME: Calibrate your confidence to the amount of data available. With less than
  30 days, avoid strong predictive claims. With more data, be more definitive.
- SCHEDULE COMPLIANCE: When analysis type is Operational Schedule Compliance:
  The SCHEDULE COMPLIANCE DATA section contains a pre-computed field "off_schedule_compliance_pct".
  You MUST use this exact value as the health_score. Do not calculate your own compliance figure.
  You MUST use the term "off-schedule compliance" (not "schedule compliance") throughout your narrative.
  Definition: off-schedule compliance = % of off-schedule hours the machine was correctly NOT running.
  100% = machine never ran outside permitted schedule. 0% = machine ran during all off-schedule hours.
  KPIs must use PERCENTAGES not raw counts. Keep values SHORT (under 8 chars). Use these exact labels and formats:
    "Off-Schedule Compliance" — value taken directly from off_schedule_compliance_pct field
    "Off-Schedule Running" — value like "70.0%"
    "Weekend" — value like "28.1%"
    "After-Hours" — value like "41.9%"
  Never use words like "of total", "runtime", "running" in the value field. Percentage only.
  Never use raw reading counts like "5,985 readings" in KPI values.
  ANOMALIES RULE FOR SCHEDULE COMPLIANCE:
  The prompt contains an "off_schedule_patterns" list. You MUST create EXACTLY ONE anomaly
  entry per item in that list. If the list has 2 items, return exactly 2 anomalies.
  If it has 1 item, return 1. If it is empty, return []. Never merge two entries into one.
  CRITICAL: Each sub-field must contain ONLY its own content:
    "description"       — ONLY the pattern observation (what, when, frequency, cumulative hours).
    "corrective_action" — ONLY the specific fix recommendation.
    "potential_impact"  — ONLY the quantified benefit (hours, kWh, cost saving with numbers).
  Set "parameter" to the "pattern" value from off_schedule_patterns (e.g. "Weekday (Mon-Fri)").
  Order anomalies by total_hours descending (highest hours first).
  INSIGHTS RULE: For Schedule Compliance, return an empty insights array [].
"""

ANALYSIS_DESCRIPTIONS = {
    # ── Primary analyses ──────────────────────────────────────────────────────
    "Overall Health Assessment": (
        "Provide a comprehensive health assessment covering all parameters. "
        "Your assessment must include: "
        "(1) Current parameter status against specs and thresholds. "
        "(2) Any notable correlations or relationships between parameters — e.g. if two parameters "
        "are both degrading together, or if a parameter that should correlate with another has decoupled. "
        "(3) Cross-parameter ratios or balance issues — e.g. efficiency (output vs input energy), "
        "pressure ratios, temperature differentials, phase balance. "
        "(4) Overall degradation trajectory. "
        "Be specific — cite actual values, percentages and parameter names."
    ),
    "Trend & Drift Analysis": (
        "Focus exclusively on gradual trends and drift over time. "
        "For each parameter: report the drift percentage (recent vs early period), "
        "direction (rising or falling), and rate. "
        "Identify which parameters are trending towards warning or critical thresholds "
        "and estimate time to breach where possible. "
        "Flag any parameters showing accelerating drift. "
        "Do not report on current anomalies or spikes — focus only on sustained directional change."
    ),
    "Anomaly Detection": (
        "Identify and catalogue all anomalous events. "
        "For each anomaly: name the parameter, state the anomalous value, "
        "give the timestamp or approximate period, state the Z-score or IQR deviation, "
        "and explain why it is significant. "
        "Report Isolation Forest multivariate anomalies if available. "
        "Distinguish between isolated spikes and sustained out-of-range periods. "
        "Do not report on trends — focus on sudden, unexpected deviations."
    ),
    "Operational Schedule Compliance": (
        "Analyse whether the machine ran outside its permitted schedule. "
        "Report only off-schedule compliance findings. "
        "Do NOT report on sensor health, vibration, temperature, pressure, or any parameter conditions. "
        "KPIs must cover only: off-schedule compliance %, off-schedule running %, weekend %, after-hours %. "
        "Health score = off_schedule_compliance_pct from the AUTHORITATIVE COMPLIANCE VALUE section. "
        "Always use the term 'off-schedule compliance' not 'schedule compliance'. "
        "ANOMALIES: Do NOT list every individual off-schedule block as a separate anomaly. "
        "Instead identify the PATTERNS: recurring after-hours running, systematic weekend operation, "
        "repeated overnight starts, etc. Only report patterns clearly evidenced in the data. "
        "If only one pattern exists return one entry; if none, return []. "
        "Do NOT invent patterns to fill a quota. Each anomaly must include: the pattern name, "
        "frequency (e.g. 'every weekday evening'), total duration impact, and a specific "
        "corrective action recommendation. "
        "INSIGHTS: Return an empty insights array []. "
        "All actionable content is captured in corrective_action and potential_impact fields."
    ),
    # ── Additional analytics ──────────────────────────────────────────────────
    "Correlation Analysis": (
        "Analyse the statistical relationships between all parameters. "
        "Report the strongest positive and negative correlations with their coefficients. "
        "Identify any parameter pairs that should be correlated but show unexpectedly low correlation "
        "(possible sensor fault or developing fault). "
        "Note any parameters that appear to be co-degrading. "
        "Recommend a scatter or correlation chart for the most interesting pair."
    ),
    "Parameter Distribution": (
        "Analyse the statistical distribution of each parameter. "
        "Report mean, standard deviation, skewness, min, max and IQR for each. "
        "Identify parameters with unusual distribution shapes: highly skewed, bimodal, "
        "or with extreme outliers pulling the distribution. "
        "Compare actual operating ranges against design specification ranges where available. "
        "Flag parameters operating significantly away from their design point."
    ),
    "Cross-Parameter Comparison": (
        "Compare physically related parameters against each other. "
        "Calculate key ratios and differentials: pressure ratio, temperature differential, "
        "specific energy (power per unit output), efficiency proxies. "
        "Identify imbalances, unexpected ratios, or parameters that have drifted out of their "
        "expected relationship with each other. "
        "Use the machine specifications to determine what the expected ratios should be."
    ),
}


def _parse_thresholds(description: str) -> Optional[dict]:
    """Extract threshold dict from machine description text."""
    if "=== PARAMETER THRESHOLDS ===" not in description:
        return None
    thresh_text = description.split("=== PARAMETER THRESHOLDS ===")[-1].strip()
    thresholds = {}
    for line in thresh_text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        try:
            param, rest = line.split(":", 1)
            param = param.strip()
            limits = {}
            for part in rest.split(","):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    limits[k.strip()] = float(v.strip())
            if limits:
                thresholds[param] = limits
        except Exception:
            continue
    return thresholds if thresholds else None


class Analyzer:
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.client = anthropic.Anthropic(api_key=key)

    def analyze(
        self,
        machine_info: dict,
        data: pd.DataFrame,
        analysis_type: str,
        date_range: Optional[tuple] = None,
        baseline_period: Optional[tuple] = None,
        current_period: Optional[tuple] = None,
        extra_context: str = "",
        schedule: Optional[dict] = None,
        logs_text: str = "",
        data_quality_context: str = "",
    ) -> dict:
        try:
            filtered  = self._filter_by_date(data, date_range)
            dq_context = data_quality_context or ""

            # ── Baseline period for control limits ──────────────────
            if baseline_period and baseline_period[0] != baseline_period[1]:
                baseline_data = self._filter_by_date(data, baseline_period)
                baseline_note = (
                    f"Control chart limits (UCL/LCL) are calculated from the specified "
                    f"normal operation baseline period: "
                    f"{baseline_period[0]} to {baseline_period[1]} "
                    f"({len(baseline_data)} readings). "
                    f"This baseline was defined by the engineer as representing normal operation."
                )
            else:
                # Default: first 20% of full dataset
                _total = len(data)
                _n_baseline = max(int(_total * 0.20), min(500, _total))
                baseline_data = data.iloc[:_n_baseline]
                _bl_start = data.index.min().date()
                _bl_end   = baseline_data.index.max().date()
                baseline_note = (
                    f"No normal operation baseline was specified by the engineer. "
                    f"Control chart limits (UCL/LCL) are calculated from the first 20% "
                    f"of available data ({_bl_start} to {_bl_end}, {_n_baseline} readings) "
                    f"as the default baseline. For more accurate limits, the engineer should "
                    f"specify a known normal operation period."
                )
            if dq_context:
                dq_context = baseline_note + "\n\n" + dq_context
            else:
                dq_context = baseline_note

            # Parse thresholds from machine description
            thresholds = _parse_thresholds(machine_info.get("description", ""))

            # Run ML pre-processing — scope depends on analysis type
            # Anomaly Detection: full ML (control charts, isolation forest, thresholds)
            # Trend & Drift: statistical only (no anomaly detection)
            # Other: minimal statistical
            # Schedule Compliance: skip entirely
            if analysis_type == "Operational Schedule Compliance":
                ml_signals = {"tier": 0, "tier_label": "", "data_days": 0, "guidance": "", "statistical": {}}
            elif analysis_type == "Trend & Drift Analysis":
                # Statistical trends only — running-state rows only (stopped zeros skew drift)
                _running_only = self._filter_running_only(filtered, schedule)
                _trend_data   = _running_only if not _running_only.empty else filtered
                _days = (filtered.index.max() - filtered.index.min()).days
                _numeric = _trend_data.select_dtypes(include="number")
                _dropped = len(filtered) - len(_trend_data)
                _running_note = (
                    f"Trend computed on running-state rows only ({len(_trend_data):,} of "
                    f"{len(filtered):,} rows). {_dropped:,} stopped/idle rows excluded to prevent "
                    f"zero-state readings from inflating drift."
                ) if _dropped > 0 else "All rows used (no stopped-state rows identified)."

                # Baseline mean — from baseline_period if set, else first 20% of running rows
                if baseline_period and baseline_period[0] != baseline_period[1]:
                    _bl_data = self._filter_by_date(_trend_data, baseline_period)
                    _bl_data = self._filter_running_only(_bl_data, schedule)
                    _bl_label = f"{baseline_period[0]} to {baseline_period[1]}"
                else:
                    _n_bl = max(1, int(len(_trend_data) * 0.20))
                    _bl_data = _trend_data.iloc[:_n_bl]
                    _bl_label = f"first 20% of running data ({_n_bl} rows)"

                # Current mean — from current_period if set, else last min(30d, 20%) of running rows
                if current_period and current_period[0] != current_period[1]:
                    _cur_data = self._filter_by_date(_trend_data, current_period)
                    _cur_data = self._filter_running_only(_cur_data, schedule)
                    _cur_label = f"{current_period[0]} to {current_period[1]}"
                else:
                    _interval_days = _days / max(len(_trend_data) - 1, 1) if len(_trend_data) > 1 else 1
                    _cur_n = max(1, min(
                        int(len(_trend_data) * 0.20),
                        int(30 / max(_interval_days, 1e-6)),
                    ))
                    _cur_data = _trend_data.iloc[-_cur_n:]
                    _cur_label = f"last min(30d, 20%) of running data ({_cur_n} rows)"

                # Per-parameter breach-rate severity + mean shift (context only)
                # Severity thresholds (grounded in 3-sigma statistical expectation):
                #   UCL (3-sigma): expected breach rate 0.27% at steady state
                #     Normal   < 1%    | Advisory 1-5%   | Warning 5-20%  | Critical > 20%
                #   UWL (2-sigma): expected breach rate 4.55% at steady state
                #     Normal   < 5%    | Advisory 5-15%  | Warning 15-40% | Critical > 40%
                #   Final severity = max(UCL_severity, UWL_severity)
                #   LCL/LWL: same thresholds applied symmetrically for downward drift
                _SEV_ORDER = {"Normal": 0, "Advisory": 1, "Warning": 2, "Critical": 3}

                def _breach_severity(rate_pct, limit_type):
                    """Return severity string for a given breach rate and limit type."""
                    if limit_type in ("UCL", "LCL"):
                        if rate_pct > 20: return "Critical"
                        if rate_pct > 5:  return "Warning"
                        if rate_pct > 1:  return "Advisory"
                        return "Normal"
                    else:  # UWL / LWL
                        if rate_pct > 40: return "Critical"
                        if rate_pct > 15: return "Warning"
                        if rate_pct > 5:  return "Advisory"
                        return "Normal"

                _drift_summary = {}
                for _c in _numeric.columns:
                    _bl_s  = _bl_data[_c].dropna()  if _c in _bl_data.columns  else pd.Series(dtype=float)
                    _cur_s = _cur_data[_c].dropna() if _c in _cur_data.columns else pd.Series(dtype=float)
                    if len(_bl_s) < 4 or len(_cur_s) < 4:
                        continue

                    _bl_mean = float(_bl_s.mean())
                    _bl_std  = float(_bl_s.std()) or 1e-9
                    _cur_mean = float(_cur_s.mean())

                    # Control limits from baseline
                    _ucl = _bl_mean + 3 * _bl_std
                    _uwl = _bl_mean + 2 * _bl_std
                    _lcl = _bl_mean - 3 * _bl_std
                    _lwl = _bl_mean - 2 * _bl_std

                    # Breach rates on current period data
                    _n = len(_cur_s)
                    _ucl_rate = round(100 * (_cur_s > _ucl).sum() / _n, 2)
                    _uwl_rate = round(100 * ((_cur_s > _uwl) & (_cur_s <= _ucl)).sum() / _n, 2)
                    _lcl_rate = round(100 * (_cur_s < _lcl).sum() / _n, 2)
                    _lwl_rate = round(100 * ((_cur_s < _lwl) & (_cur_s >= _lcl)).sum() / _n, 2)

                    # Severity per limit, then take the worst
                    _sev_ucl = _breach_severity(_ucl_rate, "UCL")
                    _sev_uwl = _breach_severity(_uwl_rate, "UWL")
                    _sev_lcl = _breach_severity(_lcl_rate, "LCL")
                    _sev_lwl = _breach_severity(_lwl_rate, "LWL")

                    _all_sevs = [(_sev_ucl, "UCL", _ucl_rate, _ucl),
                                 (_sev_uwl, "UWL", _uwl_rate, _uwl),
                                 (_sev_lcl, "LCL", _lcl_rate, _lcl),
                                 (_sev_lwl, "LWL", _lwl_rate, _lwl)]
                    _worst = max(_all_sevs, key=lambda x: _SEV_ORDER[x[0]])
                    _worst_sev, _worst_limit, _worst_rate, _worst_val = _worst

                    # Mean shift — contextual, not severity driver
                    _drift_pct = round(100 * (_cur_mean - _bl_mean) / _bl_mean, 2) if _bl_mean != 0 else 0.0

                    _drift_summary[_c] = {
                        "baseline_mean":   round(_bl_mean, 4),
                        "current_mean":    round(_cur_mean, 4),
                        "drift_pct":       _drift_pct,
                        "direction":       "rising" if _drift_pct > 0 else "falling",
                        "severity":        _worst_sev,
                        "driving_limit":   _worst_limit,
                        "driving_limit_value": round(_worst_val, 4),
                        "driving_breach_rate_pct": _worst_rate,
                        "breach_rates": {
                            "UCL": _ucl_rate, "UWL": _uwl_rate,
                            "LCL": _lcl_rate, "LWL": _lwl_rate,
                        },
                    }

                _drift_note = (
                    f"Baseline period: {_bl_label}. "
                    f"Current period: {_cur_label}. "
                    f"Drift = (current mean - baseline mean) / baseline mean × 100."
                )

                ml_signals = {
                    "tier": 1,
                    "tier_label": "Trend analysis — statistical only (running-state rows)",
                    "data_days": _days,
                    "columns_analysed": _numeric.columns.tolist(),
                    "statistical": ml_engine._statistical(_numeric, thresholds),
                    "drift_vs_baseline": _drift_summary,
                    "guidance": ml_engine._guidance(1, _days) + " " + _running_note + " " + _drift_note,
                }
            elif analysis_type == "Anomaly Detection":
                # Full ML pipeline including isolation forest and control charts
                ml_signals    = ml_engine.run(filtered, thresholds, baseline_data=baseline_data)
                _alerts       = alert_engine.run(filtered, ml_signals, thresholds)
                alert_summary = alert_engine.summarise(_alerts)
            else:
                # Correlation, Distribution, Cross-Parameter — statistical only
                ml_signals = ml_engine.run(filtered, thresholds, baseline_data=baseline_data)

            if analysis_type != "Anomaly Detection":
                _alerts       = []
                alert_summary = {"total": 0, "critical": 0, "warning": 0, "advisory": 0, "alerts": []}

            # Schedule compliance stats
            schedule_stats = None
            if analysis_type == "Operational Schedule Compliance" and schedule:
                schedule_stats = self._compute_schedule_stats(filtered, schedule)

            prompt = self._build_prompt(
                machine_info, filtered, analysis_type,
                extra_context, schedule_stats, logs_text, ml_signals,
                dq_context=dq_context,
                alert_summary=alert_summary,
            )

            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

            insights = json.loads(raw)
            # Attach tier info for display
            insights["_ml_tier"]       = ml_signals.get("tier", 0)
            insights["_ml_tier_label"] = ml_signals.get("tier_label", "")
            return {
                "success":  True,
                "insights": insights,
                "alerts":   _alerts,
            }

        except json.JSONDecodeError as exc:
            return {"success": False, "error": f"Claude returned invalid JSON: {exc}"}
        except anthropic.AuthenticationError:
            return {"success": False, "error": "Invalid API key. Please check your ANTHROPIC_API_KEY."}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------ #
    # Schedule compliance
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_schedule_stats(data: pd.DataFrame, schedule: dict) -> dict:
        df = data.copy()
        df.index = pd.to_datetime(df.index)
        work_days  = schedule.get("work_days", list(range(5)))   # list of int 0-6
        ind_col    = schedule.get("indicator_col", "")
        threshold  = schedule.get("running_threshold", 0)

        if ind_col and ind_col in df.columns:
            running_mask = df[ind_col] > threshold
        else:
            num_cols = df.select_dtypes(include="number").columns
            if len(num_cols):
                col = num_cols[0]
                running_mask = df[col] > df[col].quantile(0.10)
            else:
                running_mask = pd.Series(True, index=df.index)

        hour_start   = schedule.get("work_hour_start", 8)
        hour_end     = schedule.get("work_hour_end", 18)
        sched_entries = schedule.get("sched_entries", [])
        sched_per_day = schedule.get("sched_per_day", {})

        # Build in_schedule mask: day + time window
        # Uses sched_per_day (per-day windows) when available, else flat work_days + hour window
        _DAYS_IDX = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
        _any_spd = any(v.get("enabled") for v in sched_per_day.values()) if sched_per_day else False
        if not sched_entries and not _any_spd:
            # No schedule defined — all time treated as permitted
            in_schedule = pd.Series(True, index=df.index)
        elif _any_spd:
            in_schedule = pd.Series(False, index=df.index)
            for _dn, _didx in _DAYS_IDX.items():
                _dcfg = sched_per_day.get(_dn, {})
                if not _dcfg.get("enabled", False):
                    continue
                _day_rows = df.index.dayofweek == _didx
                _wins = _dcfg.get("windows", [{"start": hour_start, "end": hour_end}])
                _hw = pd.Series(
                    [any(w["start"] <= h + n/60 < w["end"]
                         for w in _wins)
                     for h, n in zip(df.index.hour, df.index.minute)],
                    index=df.index,
                )
                in_schedule |= (_day_rows & _hw)
        else:
            _day_mask  = df.index.dayofweek.isin(work_days)
            _hour_mask = pd.Series(
                [hour_start <= h + n/60 < hour_end
                 for h, n in zip(df.index.hour, df.index.minute)],
                index=df.index,
            )
            in_schedule = _day_mask & _hour_mask

        off_schedule_running = running_mask & ~in_schedule
        weekend_running      = running_mask & df.index.dayofweek.isin([5, 6])

        off_blocks = []
        if off_schedule_running.any():
            s = off_schedule_running.astype(int).diff().fillna(0)
            starts = df.index[s == 1].tolist()
            ends   = df.index[s == -1].tolist()
            if off_schedule_running.iloc[0]:
                starts = [df.index[0]] + starts
            if off_schedule_running.iloc[-1]:
                ends = ends + [df.index[-1]]
            for st, en in zip(starts[:10], ends[:10]):
                off_blocks.append({"start": str(st), "end": str(en)})

        numeric_cols     = df.select_dtypes(include="number").columns.tolist()[:5]
        running_rows     = int(running_mask.sum())
        off_rows         = int(off_schedule_running.sum())
        total_off_sched  = int((~in_schedule).sum())
        # Off-schedule compliance: % of off-schedule time machine was correctly NOT running
        # = (total off-schedule slots - ran during off-schedule) / total off-schedule slots
        if total_off_sched > 0:
            off_sched_compliance = round(100 * (total_off_sched - off_rows) / total_off_sched, 1)
        else:
            off_sched_compliance = 100.0

        # Aggregate off-schedule running into two buckets only:
        #   "Weekday" = Mon–Fri (dow 0–4)
        #   "Weekend" = Sat–Sun (dow 5–6)
        # This mirrors the user-defined schedule definition.
        _WD_INTS = {0, 1, 2, 3, 4}   # Mon–Fri
        _WE_INTS = {5, 6}             # Sat–Sun
        _pattern_agg = {}
        if off_schedule_running.any():
            _osr_df = off_schedule_running[off_schedule_running].copy()
            for _ts in _osr_df.index:
                _bucket = "Weekday (Mon–Fri)" if _ts.dayofweek in _WD_INTS else "Weekend (Sat–Sun)"
                if _bucket not in _pattern_agg:
                    _pattern_agg[_bucket] = {
                        "pattern": _bucket,
                        "occurrences": 0,
                        "total_hours": 0.0,
                        "affected_days": set(),
                    }
                _pattern_agg[_bucket]["total_hours"] = round(
                    _pattern_agg[_bucket]["total_hours"] + 0.25, 2)
                _pattern_agg[_bucket]["affected_days"].add(_ts.date())
            # Replace set with count for JSON serialisation
            for _b in _pattern_agg:
                _pattern_agg[_b]["occurrences"] = len(_pattern_agg[_b]["affected_days"])
                del _pattern_agg[_b]["affected_days"]

        # Sort by total_hours descending — most impactful bucket first
        _patterns_sorted = sorted(_pattern_agg.values(), key=lambda x: x["total_hours"], reverse=True)

        # Actual average power during off-schedule running — use power/current column if available
        # Priority: power_kW > motor_current_A (proxy via schedule threshold) > fallback 0
        _avg_power_kw = 0.0
        _power_col = next(
            (c for c in df.select_dtypes(include="number").columns
             if any(k in c.lower() for k in ["power_kw", "power_k", "kw"])), None
        )
        if _power_col is None:
            _power_col = next(
                (c for c in df.select_dtypes(include="number").columns
                 if any(k in c.lower() for k in ["power"])), None
            )
        if _power_col and off_schedule_running.any():
            _osr_power = df.loc[off_schedule_running, _power_col].dropna()
            if len(_osr_power) > 0:
                _avg_power_kw = round(float(_osr_power.mean()), 3)

        _DAY_NAMES_ALL = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

        # Build a per-day-group schedule description that captures all distinct windows
        # e.g. "Mon, Tue: 08:00–18:00; Wed, Thu: 10:00–22:00"
        def _fmt_h(v):
            return f"{int(v):02d}:{int(round((v - int(v)) * 60)):02d}"

        if sched_per_day and any(v.get("enabled") for v in sched_per_day.values()):
            # Group days by their windows signature
            _win_groups = {}
            for _dn in _DAY_NAMES_ALL:
                _dcfg = sched_per_day.get(_dn, {})
                if not _dcfg.get("enabled"):
                    continue
                _wins = _dcfg.get("windows", [{"start": hour_start, "end": hour_end}])
                _sig  = "; ".join(f"{_fmt_h(w['start'])}–{_fmt_h(w['end'])}" for w in _wins)
                if _sig not in _win_groups:
                    _win_groups[_sig] = []
                _win_groups[_sig].append(_dn)
            _schedule_desc = "  |  ".join(
                f"{', '.join(days)}: {sig}" for sig, days in _win_groups.items()
            )
        else:
            _schedule_desc = f"{', '.join([_DAY_NAMES_ALL[d] for d in work_days])}: {_fmt_h(hour_start)}–{_fmt_h(hour_end)}"

        return {
            "permitted_schedule": {
                "work_days":    [_DAY_NAMES_ALL[d] for d in work_days],
                "hours":        _schedule_desc,
                "per_day":      {_dn: sched_per_day[_dn] for _dn in _DAY_NAMES_ALL
                                 if sched_per_day.get(_dn, {}).get("enabled")},
            },
            "total_readings":                len(df),
            "running_readings":              running_rows,
            "total_off_schedule_readings":   total_off_sched,
            "off_schedule_readings":         off_rows,
            "weekend_running_readings":      int(weekend_running.sum()),
            "off_schedule_pct":              round(100 * off_rows / running_rows, 1) if running_rows else 0,
            "off_schedule_compliance_pct":   off_sched_compliance,
            "off_schedule_blocks":           off_blocks,
            "off_schedule_patterns":         _patterns_sorted,
            "data_weeks":                    max(1, round((df.index.max() - df.index.min()).days / 7, 1)),
            "currency_symbol":               schedule.get("currency_symbol", "$"),
            "rate_per_kwh":                  schedule.get("rate_per_kwh", 0.15),
            "avg_power_kw":                  _avg_power_kw,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _filter_running_only(data: pd.DataFrame, schedule: Optional[dict]) -> pd.DataFrame:
        """
        Return only rows where the machine is in a running state.

        Uses indicator_col + running_threshold from the schedule dict when available.
        Falls back to: values above the 10th percentile of the first numeric column.
        This mirrors the logic in _compute_schedule_stats so behaviour is consistent.
        """
        if data is None or data.empty:
            return data
        ind_col   = (schedule or {}).get("indicator_col", "")
        threshold = (schedule or {}).get("running_threshold", 0)
        if ind_col and ind_col in data.columns:
            mask = data[ind_col] > threshold
        else:
            num_cols = data.select_dtypes(include="number").columns
            if len(num_cols):
                col  = num_cols[0]
                mask = data[col] > data[col].quantile(0.10)
            else:
                return data   # no numeric cols — return as-is
        result = data[mask]
        return result if not result.empty else data

    @staticmethod
    def _filter_by_date(data: pd.DataFrame, date_range) -> pd.DataFrame:
        if not date_range or len(date_range) < 2:
            return data
        # Start: beginning of start date (00:00:00)
        # End: end of end date (23:59:59) — makes end date fully inclusive
        start = pd.Timestamp(date_range[0]).normalize()
        end   = pd.Timestamp(date_range[1]).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        mask  = (data.index >= start) & (data.index <= end)
        filtered = data.loc[mask]
        return filtered if not filtered.empty else data

    def _build_prompt(
        self,
        machine_info: dict,
        data: pd.DataFrame,
        analysis_type: str,
        extra_context: str,
        schedule_stats: Optional[dict],
        logs_text: str,
        ml_signals: dict,
        dq_context: str = "",
        alert_summary: Optional[dict] = None,
    ) -> str:
        numeric_cols = data.select_dtypes(include="number").columns.tolist()
        stats        = data[numeric_cols].describe().round(3).to_dict() if numeric_cols else {}
        sample       = data[numeric_cols].tail(10).reset_index().to_dict(orient="records")
        analysis_desc = ANALYSIS_DESCRIPTIONS.get(analysis_type, f"Perform a {analysis_type}.")

        schedule_section = ""
        if schedule_stats:
            schedule_section = f"""
=== SCHEDULE COMPLIANCE DATA ===
{json.dumps(schedule_stats, indent=2, default=str)}
"""
        logs_section = ""
        if logs_text and logs_text.strip():
            logs_section = f"""
=== MAINTENANCE LOG HISTORY ===
{logs_text.strip()}
"""
        # Build alert section for Anomaly Detection
        alert_section = ""
        if alert_summary and alert_summary.get("total", 0) > 0:
            alert_section = f"""
=== ALERT ENGINE OUTPUT ===
Deterministic alerts generated before Claude interpretation.
Critical: {alert_summary['critical']}  Warning: {alert_summary['warning']}  Advisory: {alert_summary['advisory']}

{json.dumps(alert_summary['alerts'], indent=2, default=str)}

IMPORTANT: Incorporate these pre-computed alerts into your anomaly assessment.
Reference specific alert messages, consistency (recurring/transient), and
days_to_breach projections in your insights and score_breakdown.
Where alerts conflict with your statistical interpretation, the alert engine
takes precedence on level classification \u2014 explain any discrepancy.
"""

        # Build compact ML signals — only flags and key numbers, not full raw stats
        stat_flags = {}
        for col, sig in ml_signals.get("statistical", {}).items():
            flags = sig.get("flags", [])
            if flags or abs(sig.get("trend_pct", 0)) > 2 or sig.get("z_outliers", 0) > 0:
                stat_flags[col] = {
                    "mean": sig.get("mean"),
                    "trend_pct": sig.get("trend_pct"),
                    "z_outliers": sig.get("z_outliers"),
                    "recent_max_z": sig.get("recent_max_z"),
                    "flags": flags,
                }

        cc_violations = {}
        for col, cc in ml_signals.get("control_charts", {}).items():
            if cc.get("violations"):
                cc_violations[col] = cc.get("violations")

        ml_section = f"""
=== PRE-COMPUTED SIGNALS (Tier {ml_signals.get('tier', '?')}: {ml_signals.get('tier_label', '')}) ===
Data volume: {ml_signals.get('data_days', '?')} days
Guidance: {ml_signals.get('guidance', '')}

Statistical flags (parameters with anomalies or trends only):
{json.dumps(stat_flags, indent=2, default=str)}
"""
        if cc_violations:
            ml_section += f"""
Control chart violations:
{json.dumps(cc_violations, indent=2, default=str)}
"""
        if "isolation_forest" in ml_signals:
            iso = ml_signals["isolation_forest"]
            ml_section += f"""
Isolation Forest: {iso.get('interpretation', '')}
Worst timestamps: {iso.get('worst_timestamps', [])}
"""
        if "threshold_breaches" in ml_signals:
            ml_section += f"""
Threshold breach counts:
{json.dumps(ml_signals.get('threshold_breaches', {}), indent=2, default=str)}
"""
        if "drift_vs_baseline" in ml_signals and ml_signals["drift_vs_baseline"]:
            ml_section += f"""
=== TREND SEVERITY AND DRIFT VS BASELINE ===
Per-parameter breach-rate severity. Severity is determined solely by breach rate against
baseline control limits (UCL/LCL = 3-sigma, UWL/LWL = 2-sigma).
Mean shift (drift_pct) is CONTEXTUAL INFORMATION ONLY — it explains why breach rates are
elevated but does not determine severity. Always report severity from the "severity" field
and state which limit drove it (driving_limit) and its breach rate (driving_breach_rate_pct).
Then report mean shift as supporting context: "current mean is X% above/below baseline mean."
{json.dumps(ml_signals["drift_vs_baseline"], indent=2, default=str)}
"""

        # For Schedule Compliance — only send schedule data, no sensor stats
        if analysis_type == "Operational Schedule Compliance":
            # Pre-build anomaly scaffold from off_schedule_patterns so Claude cannot skip entries
            _patterns = (schedule_stats or {}).get("off_schedule_patterns", [])
            _rate      = (schedule_stats or {}).get("off_schedule_compliance_pct", "N/A")
            _currency  = (schedule_stats or {}).get("currency_symbol", "$")
            _kwh_rate  = (schedule_stats or {}).get("rate_per_kwh", 0.15)
            _avg_kw    = (schedule_stats or {}).get("avg_power_kw", 0.0)
            if _patterns:
                _scaffold_lines = [
                    f"ANOMALY INSTRUCTIONS (strictly enforced):",
                    f"- Return EXACTLY {len(_patterns)} anomaly entries — one per numbered pattern below.",
                    f"- The 'parameter' and 'description' fields are PRE-WRITTEN. Copy them VERBATIM into your JSON.",
                    f"- Only write your own text in 'corrective_action' and 'potential_impact'.",
                    f"- Do NOT paraphrase, rewrite, or add to the description field.",
                    f"- Do NOT skip any entry. Do NOT merge entries.",
                    "",
                ]
                # Derive permitted window label — now a full grouped description
                _perm      = (schedule_stats or {}).get("permitted_schedule", {})
                _perm_hrs  = _perm.get("hours", "defined hours")
                # _perm_hrs already contains full description e.g. "Mon, Tue: 08:00–18:00  |  Wed, Thu: 10:00–22:00"

                for _i, _p in enumerate(_patterns):
                    _hrs    = _p.get("total_hours", 0)
                    _occ    = _p.get("occurrences", 0)
                    _pname  = _p.get("pattern", "Unknown")
                    _wk_hrs = round(_hrs / max((schedule_stats or {}).get("data_weeks", 13), 1), 1)
                    _kwh    = round(_hrs * 5 * _kwh_rate, 0)  # assume 5kW avg; rate from config
                    _desc = (
                        f"{_pname}: {_hrs} hrs of off-schedule running across {_occ} days "
                        f"(~{_wk_hrs} hrs/week), outside the defined operating boundary "
                        f"of {_perm_hrs}."
                    )
                    _scaffold_lines += [
                        f"ANOMALY {_i+1} OF {len(_patterns)} — copy fields EXACTLY as shown, only fill the two marked fields:",
                        f'  "parameter": "{_pname}"',
                        f'  "description": "{_desc}"   ← COPY THIS VERBATIM. Do NOT rewrite.',
                        f'  "corrective_action": "FILL IN: specific action to stop {_pname} off-schedule running"',
                        f'  "potential_impact":  "FILL IN: ~{_hrs} hrs saved, ~{_kwh:.0f} {_currency}/period at {_kwh_rate} {_currency}/kWh — add operational risk context"',
                        "",
                    ]
                _anomaly_scaffold = "\n".join(_scaffold_lines)
            else:
                _anomaly_scaffold = "No off-schedule patterns detected. Return anomalies: []."

            prompt = f"""
=== MACHINE PROFILE ===
Type        : {machine_info.get('machine_type', 'Unknown')}
ID          : {machine_info.get('machine_id', 'Unknown')}
Time range  : {data.index.min()} → {data.index.max()}
Total rows  : {len(data):,}
{schedule_section}
{f'=== ENGINEER NOTES ==={chr(10)}{extra_context}' if extra_context.strip() else ''}
{f'=== DATA QUALITY ==={chr(10)}{dq_context}{chr(10)}IMPORTANT: Where data quality issues flagged (flatlines, frozen sensors, shifts), factor this into your analysis. Do not treat flatlined values as valid readings. Flag affected parameters explicitly.' if dq_context.strip() else ''}

=== REQUESTED ANALYSIS ===
Type: Operational Schedule Compliance
Task: {analysis_desc}

=== AUTHORITATIVE COMPLIANCE VALUE ===
off_schedule_compliance_pct = {schedule_stats.get("off_schedule_compliance_pct", "N/A") if schedule_stats else "N/A"}%
Definition: % of off-schedule hours the machine was correctly NOT running.
You MUST set health_score to this exact value. Do NOT calculate your own compliance figure.
You MUST use the term "off-schedule compliance" (not "schedule compliance") in your narrative.

IMPORTANT: Report ONLY off-schedule compliance findings. Do NOT report on sensor health,
vibration, temperature, pressure, or any other parameter conditions.
KPIs must cover only: off-schedule compliance %, off-schedule running %, weekend running %, after-hours running %.

{_anomaly_scaffold}

INSIGHTS — EMPTY FOR SCHEDULE COMPLIANCE:
- Return "insights": [] (empty array).

Return your analysis as a single JSON object following the schema in the system prompt.
"""
        else:
            prompt = f"""
=== MACHINE PROFILE ===
Type        : {machine_info.get('machine_type', 'Unknown')}
ID          : {machine_info.get('machine_id', 'Unknown')}
Specs/Notes : {machine_info.get('description', 'Not provided')}

=== DATASET OVERVIEW ===
Total rows  : {len(data):,}
Parameters  : {', '.join(numeric_cols) if numeric_cols else 'None detected'}
Time range  : {data.index.min()} → {data.index.max()}

=== STATISTICAL SUMMARY ===
{json.dumps(stats, indent=2, default=str)}

=== RECENT 10 READINGS ===
{json.dumps(sample, indent=2, default=str)}
{ml_section}{alert_section}{logs_section}
=== REQUESTED ANALYSIS ===
Type: {analysis_type}
Task: {analysis_desc}

{f'=== ENGINEER NOTES ==={chr(10)}{extra_context}' if extra_context.strip() else ''}
{f'=== DATA QUALITY ==={chr(10)}{dq_context}{chr(10)}IMPORTANT: Where data quality issues flagged (flatlines, frozen sensors, shifts), factor this into your analysis. Do not treat flatlined values as valid readings. Flag affected parameters explicitly.' if dq_context.strip() else ''}

Return your analysis as a single JSON object following the schema in the system prompt.
"""
        return prompt
