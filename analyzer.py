"""
analyzer.py — Claude analysis engine

Each call to analyze() builds a rich context packet (machine profile +
statistical summary + recent readings) and sends it to Claude with a
structured JSON schema.  Claude returns insights, KPIs, anomalies,
chart recommendations, a health score, and a plain-language narrative.
"""

import json
import os
from typing import Optional

import anthropic
import pandas as pd

# --------------------------------------------------------------------------- #
# Prompt design
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are an expert industrial machine health analyst with deep knowledge of rotating equipment, process machinery, and condition monitoring.

Your job is to analyse time-series sensor data from industrial machines and return ONLY a valid JSON object — no preamble, no markdown code blocks, no explanation outside the JSON.

The JSON must exactly follow this schema:

{
  "health_score": <integer 0–100, where 100 = perfect condition>,
  "kpis": [
    {
      "label": "<short parameter name>",
      "value": "<value with unit, e.g. '74.3 °C'>",
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
      "note": "<optional interpretation hint for the engineer>"
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
"""


ANALYSIS_DESCRIPTIONS = {
    "Overall Health Assessment":   "Provide a comprehensive health assessment covering all parameters.",
    "Trend & Drift Analysis":      "Identify gradual trends, drifts, and degradation patterns over time.",
    "Anomaly Detection":           "Find outliers, spikes, and values outside expected operating ranges.",
    "Correlation Analysis":        "Find relationships and dependencies between parameters.",
    "Parameter Distribution":      "Analyse the statistical distribution and spread of each parameter.",
    "Cross-Parameter Comparison":  "Compare parameters against each other to identify imbalances.",
}


class Analyzer:
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.client = anthropic.Anthropic(api_key=key)

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        machine_info: dict,
        data: pd.DataFrame,
        analysis_type: str,
        date_range: Optional[tuple] = None,
        extra_context: str = "",
    ) -> dict:
        """
        Run a Claude analysis and return:
          {"success": True,  "insights": <parsed dict>}
          {"success": False, "error": <message>}
        """
        try:
            # Optional date filtering
            filtered = self._filter_by_date(data, date_range)
            prompt = self._build_prompt(machine_info, filtered, analysis_type, extra_context)

            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = response.content[0].text.strip()
            # Strip accidental markdown fences
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

            insights = json.loads(raw)
            return {"success": True, "insights": insights}

        except json.JSONDecodeError as exc:
            return {"success": False, "error": f"Claude returned invalid JSON: {exc}"}
        except anthropic.AuthenticationError:
            return {"success": False, "error": "Invalid API key. Please check your ANTHROPIC_API_KEY."}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _filter_by_date(data: pd.DataFrame, date_range) -> pd.DataFrame:
        if not date_range or len(date_range) < 2:
            return data
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        mask = (data.index >= start) & (data.index <= end)
        filtered = data.loc[mask]
        return filtered if not filtered.empty else data

    def _build_prompt(
        self,
        machine_info: dict,
        data: pd.DataFrame,
        analysis_type: str,
        extra_context: str,
    ) -> str:
        numeric_cols = data.select_dtypes(include="number").columns.tolist()

        # Statistical summary (rounded for readability)
        stats = data[numeric_cols].describe().round(3).to_dict() if numeric_cols else {}

        # Last 10 readings as a compact sample
        sample_records = (
            data[numeric_cols].tail(10).reset_index().to_dict(orient="records")
        )
        sample_json = json.dumps(sample_records, indent=2, default=str)

        # Description of the requested analysis
        analysis_desc = ANALYSIS_DESCRIPTIONS.get(
            analysis_type, f"Perform a {analysis_type}."
        )

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
{sample_json}

=== REQUESTED ANALYSIS ===
Type: {analysis_type}
Task: {analysis_desc}

{f'=== ENGINEER NOTES ==={chr(10)}{extra_context}' if extra_context.strip() else ''}

Return your analysis as a single JSON object following the schema in the system prompt.
"""
        return prompt
