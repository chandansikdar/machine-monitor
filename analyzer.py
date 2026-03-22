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
    "Overall Health Assessment":        "Provide a comprehensive health assessment covering all parameters.",
    "Trend & Drift Analysis":           "Identify gradual trends, drifts, and degradation patterns over time.",
    "Anomaly Detection":                "Find outliers, spikes, and values outside expected operating ranges.",
    "Correlation Analysis":             "Find relationships and dependencies between parameters.",
    "Parameter Distribution":           "Analyse the statistical distribution and spread of each parameter.",
    "Cross-Parameter Comparison":       "Compare parameters against each other to identify imbalances.",
    "Operational Schedule Compliance":  "Analyse whether the machine ran outside its permitted schedule (weekends, holidays, out-of-hours). Identify unauthorised running periods, estimate energy waste, and flag if sensor readings during off-schedule periods indicate abnormal operation.",
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
        extra_context: str = "",
        schedule: Optional[dict] = None,
        logs_text: str = "",
    ) -> dict:
        try:
            filtered = self._filter_by_date(data, date_range)

            # Parse thresholds from machine description
            thresholds = _parse_thresholds(machine_info.get("description", ""))

            # Run adaptive ML pre-processing (skip for schedule compliance)
            if analysis_type == "Operational Schedule Compliance":
                ml_signals = {"tier": 0, "tier_label": "", "data_days": 0, "guidance": "", "statistical": {}}
            else:
                ml_signals = ml_engine.run(filtered, thresholds)

            # Schedule compliance stats
            schedule_stats = None
            if analysis_type == "Operational Schedule Compliance" and schedule:
                schedule_stats = self._compute_schedule_stats(filtered, schedule)

            prompt = self._build_prompt(
                machine_info, filtered, analysis_type,
                extra_context, schedule_stats, logs_text, ml_signals
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

Return your analysis as a single JSON object following the schema in the system prompt.
"""
        return prompt
