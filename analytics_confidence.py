"""
analytics_confidence.py
=======================
Shared confidence and message colour synthesis layer.

Implements the framework defined in the Integration Guide (Section 4):
  - Severity:     Critical | Warning
  - Consistency:  Transient | Recurring
  - Confidence:   HIGH | MEDIUM | LOW

Colour rules (Section 4.2):
  Only Critical + Recurring + HIGH or MEDIUM → RED
  Everything else → YELLOW
  LOW confidence always adds ⚠️ regardless of other criteria

Overall health (Section 5):
  At least one RED message → RED
  No RED, at least one YELLOW → YELLOW
  No messages → GREEN
"""

from __future__ import annotations
from typing import Optional

# ── Colour constants (emoji-safe) ──────────────────────────────────────────
RED    = "\U0001F534"   # 🔴
YELLOW = "\U0001F7E1"   # 🟡
GREEN  = "\U0001F7E2"   # 🟢
WARN   = "\u26a0\ufe0f" # ⚠️

# ── Thresholds ─────────────────────────────────────────────────────────────
DQ_GOOD = 70   # data quality score >= this is "good"


# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE
# ══════════════════════════════════════════════════════════════════════════════

def assign_confidence(
    layer1_detected: bool,
    layer2_detected: bool,
    dq_score: int,
) -> str:
    """
    Return the confidence level for a finding.

    Parameters
    ----------
    layer1_detected : bool
        True if Layer 1 (AI statistical) independently flagged this condition.
    layer2_detected : bool
        True if Layer 2 (physics module) generated a finding.
    dq_score : int
        Data quality score of the involved sensor columns (0–100).

    Returns
    -------
    str — "HIGH", "MEDIUM", or "LOW"
    """
    both   = layer1_detected and layer2_detected
    one    = layer1_detected or layer2_detected
    good   = dq_score >= DQ_GOOD

    if both and good:
        return "HIGH"
    if (both and not good) or (one and good):
        return "MEDIUM"
    # one layer only AND poor data quality
    return "LOW"


# ══════════════════════════════════════════════════════════════════════════════
# MESSAGE PREFIX
# ══════════════════════════════════════════════════════════════════════════════

def assign_message_prefix(
    layer1_detected: bool,
    layer2_detected: bool,
    dq_score: int,
) -> str:
    """
    Return the message prefix text based on the evidence combination.
    Returns empty string for HIGH confidence (confirmed finding — no prefix needed).
    """
    good  = dq_score >= DQ_GOOD
    both  = layer1_detected and layer2_detected

    if both and good:
        return ""   # HIGH — no prefix
    if both and not good:
        return f"{WARN} Poor data quality \u2014 verify sensors before maintenance action."
    if layer1_detected and not layer2_detected and good:
        return "Statistical signal without physics confirmation \u2014 verify sensor and operating conditions."
    if layer2_detected and not layer1_detected and good:
        return "Physics signal without statistical corroboration \u2014 investigate to confirm."
    # LOW — one layer, poor data
    return f"{WARN} Poor data quality \u2014 verify sensors before maintenance action."


# ══════════════════════════════════════════════════════════════════════════════
# MESSAGE COLOUR
# ══════════════════════════════════════════════════════════════════════════════

def assign_message_colour(
    severity: str,
    consistency: str,
    confidence: str,
) -> str:
    """
    Return the message colour string (with optional ⚠️).

    Rules (Section 4.2):
      Critical + Recurring + HIGH  → RED
      Critical + Recurring + MEDIUM → RED
      Everything else              → YELLOW
      LOW confidence always adds ⚠️

    Parameters
    ----------
    severity    : "critical" | "warning"
    consistency : "recurring" | "transient"
    confidence  : "HIGH" | "MEDIUM" | "LOW"

    Returns
    -------
    str — e.g. "🔴", "🟡", "🟡 ⚠️"
    """
    is_red = (
        severity.lower() == "critical"
        and consistency.lower() == "recurring"
        and confidence in ("HIGH", "MEDIUM")
    )

    base = RED if is_red else YELLOW

    if confidence == "LOW":
        return f"{base} {WARN}"
    return base


# ══════════════════════════════════════════════════════════════════════════════
# OVERALL HEALTH
# ══════════════════════════════════════════════════════════════════════════════

def assign_overall_health(message_colours: list[str]) -> str:
    """
    Derive the overall machine health colour from the list of message colours.

    Rules (Section 5):
      At least one RED (with or without ⚠️) → RED
      No RED, at least one YELLOW           → YELLOW
      No messages                           → GREEN

    Parameters
    ----------
    message_colours : list of colour strings as returned by assign_message_colour()

    Returns
    -------
    str — RED, YELLOW, or GREEN symbol
    """
    if not message_colours:
        return GREEN
    if any(RED in c for c in message_colours):
        return RED
    return YELLOW


# ══════════════════════════════════════════════════════════════════════════════
# FINDING ENRICHMENT
# ══════════════════════════════════════════════════════════════════════════════

def enrich_finding(
    finding: dict,
    layer1_detected: bool,
    dq_score: int,
) -> dict:
    """
    Enrich a physics finding dict (from chiller_physics or pump_physics) with
    confidence, message colour, and message prefix.

    The physics module always sets layer2_detected=True (it generated the finding).
    layer1_detected and dq_score come from the synthesis layer in main.py.

    Parameters
    ----------
    finding         : dict from _make_finding() — must have 'severity' and 'consistency'
    layer1_detected : bool — did AI Layer 1 also flag this condition?
    dq_score        : int  — data quality score of the primary sensor column

    Returns
    -------
    dict — the same finding dict with added keys:
        layer1_detected, layer2_detected, confidence,
        message_colour, message_prefix
    """
    severity    = finding.get("severity", "warning")
    consistency = finding.get("consistency", "recurring")

    confidence  = assign_confidence(
        layer1_detected=layer1_detected,
        layer2_detected=True,   # physics always detected it (it generated the finding)
        dq_score=dq_score,
    )
    colour  = assign_message_colour(severity, consistency, confidence)
    prefix  = assign_message_prefix(
        layer1_detected=layer1_detected,
        layer2_detected=True,
        dq_score=dq_score,
    )

    return {
        **finding,
        "layer1_detected": layer1_detected,
        "layer2_detected": True,
        "confidence":      confidence,
        "message_colour":  colour,
        "message_prefix":  prefix,
    }


def enrich_all_findings(
    findings: list[dict],
    layer1_signals: dict[str, bool],
    dq_score: int,
) -> list[dict]:
    """
    Enrich a list of physics findings.

    Parameters
    ----------
    findings       : list of finding dicts from the physics module
    layer1_signals : dict mapping metric name (lowercase) → bool
                     True if AI Layer 1 independently flagged this metric.
                     Use an empty dict {} if Layer 1 results are not yet available.
    dq_score       : overall data quality score

    Returns
    -------
    list of enriched finding dicts
    """
    enriched = []
    for f in findings:
        metric_key = f.get("metric", "").lower().replace(" ", "_")
        l1 = layer1_signals.get(metric_key, False)
        enriched.append(enrich_finding(f, l1, dq_score))
    return enriched
