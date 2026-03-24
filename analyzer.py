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
      "parameter": "<column name>",
      "description": "<what is anomalous, when, and why it matters>"
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
- SCHEDULE COMPLIANCE: When analysis type is Operational Schedule Compliance, the health_score
  should reflect schedule adherence (100 = fully compliant, 0 = running entirely off-schedule).
  Anomalies should only reference off-schedule running events, not sensor anomalies.
  KPIs must use PERCENTAGES not raw counts. Keep values SHORT (under 8 chars). Use these exact labels and formats:
    "Schedule Compliance" — value like "30.0%"
    "Off-Schedule" — value like "70.0%"
    "Weekend" — value like "28.1%"
    "After-Hours" — value like "41.9%"
  Never use words like "of total", "runtime", "running" in the value field. Percentage only.
  Never use raw reading counts like "5,985 readings" in KPI values.
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
        "Report only schedule compliance findings — off-schedule running periods, "
        "weekend running, after-hours running. "
        "Do NOT report on sensor health, vibration, temperature, pressure, or any parameter conditions. "
        "KPIs must cover only: compliance %, off-schedule %, weekend %, after-hours %. "
        "Health score = schedule compliance percentage (100 = fully compliant)."
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

            # Run adaptive ML pre-processing (skip for schedule compliance)
            if analysis_type == "Operational Schedule Compliance":
                ml_signals = {"tier": 0, "tier_label": "", "data_days": 0, "guidance": "", "statistical": {}}
            else:
                ml_signals = ml_engine.run(filtered, thresholds, baseline_data=baseline_data)

            # Schedule compliance stats
            schedule_stats = None
            if analysis_type == "Operational Schedule Compliance" and schedule:
                schedule_stats = self._compute_schedule_stats(filtered, schedule)

            prompt = self._build_prompt(
                machine_info, filtered, analysis_type,
                extra_context, schedule_stats, logs_text, ml_signals,
                dq_context=dq_context
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
            insights["_ml_tier"] = ml_signals.get("tier", 0)
            insights["_ml_tier_label"] = ml_signals.get("tier_label", "")
            return {"success": True, "insights": insights}

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
        work_days  = schedule.get("work_days", list(range(5)))
        hour_start = schedule.get("work_hour_start", 8)
        hour_end   = schedule.get("work_hour_end", 18)
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

        in_schedule = (
            df.index.dayofweek.isin(work_days) &
            (df.index.hour >= hour_start) &
            (df.index.hour < hour_end)
        )
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

        numeric_cols = df.select_dtypes(include="number").columns.tolist()[:5]
        running_rows = int(running_mask.sum())
        off_rows     = int(off_schedule_running.sum())
        return {
            "permitted_schedule": {
                "work_days": [["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d] for d in work_days],
                "hours": f"{hour_start:02d}:00 – {hour_end:02d}:00",
            },
            "total_readings":           len(df),
            "running_readings":         running_rows,
            "off_schedule_readings":    off_rows,
            "weekend_running_readings": int(weekend_running.sum()),
            "off_schedule_pct":         round(100 * off_rows / running_rows, 1) if running_rows else 0,
            "off_schedule_blocks":      off_blocks,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

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

        # For Schedule Compliance — only send schedule data, no sensor stats
        if analysis_type == "Operational Schedule Compliance":
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

IMPORTANT: Report ONLY schedule compliance findings. Do NOT report on sensor health,
vibration, temperature, pressure, or any other parameter conditions.
KPIs must cover only: compliance %, off-schedule hours, weekend running, after-hours running.
Anomalies must reference only off-schedule running events, not sensor readings.
Health score = schedule compliance percentage (100 = fully compliant).

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
{ml_section}{logs_section}
=== REQUESTED ANALYSIS ===
Type: {analysis_type}
Task: {analysis_desc}

{f'=== ENGINEER NOTES ==={chr(10)}{extra_context}' if extra_context.strip() else ''}
{f'=== DATA QUALITY ==={chr(10)}{dq_context}{chr(10)}IMPORTANT: Where data quality issues flagged (flatlines, frozen sensors, shifts), factor this into your analysis. Do not treat flatlined values as valid readings. Flag affected parameters explicitly.' if dq_context.strip() else ''}

Return your analysis as a single JSON object following the schema in the system prompt.
"""
        return prompt
