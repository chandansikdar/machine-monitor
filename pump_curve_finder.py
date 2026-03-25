"""
pump_curve_finder.py — Automatic pump performance curve sourcing.

Strategy:
  Step 1: Try manufacturer's known product selector URL pattern
  Step 2: If not found, fall back to web search via Claude + Anthropic API
  Step 3: Use Claude to extract H-Q data points from retrieved content
  Step 4: Return structured result with source reference for report

Source reference is stored with every curve for traceability.
"""

from __future__ import annotations
import os
import re
import json
import datetime
import requests
from typing import Optional

import anthropic

# ── Known manufacturer product selector URL patterns ─────────────────────────
MANUFACTURER_URLS = {
    "grundfos": {
        "name": "Grundfos",
        "selector_url": "https://product-selection.grundfos.com",
        "search_url":   "https://product-selection.grundfos.com/products/cr?search={model}",
        "datasheet_hint": "Download datasheet or performance curve PDF",
        "notes": "Search by model number. Select product → Downloads → Performance curve.",
    },
    "ksb": {
        "name": "KSB",
        "selector_url": "https://www.ksb.com/en-global/products-services/pumps",
        "search_url":   "https://www.ksb.com/en-global/search?searchTerm={model}",
        "datasheet_hint": "Product page → Downloads → Data sheet",
        "notes": "Enter model number in search. Curves available as PDF.",
    },
    "flowserve": {
        "name": "Flowserve",
        "selector_url": "https://www.flowserve.com/en/products/pumps",
        "search_url":   "https://www.flowserve.com/en/search#q={model}",
        "datasheet_hint": "Product page → Literature → Data sheet",
        "notes": "Registration may be required. Search by model/series.",
    },
    "wilo": {
        "name": "Wilo",
        "selector_url": "https://www.wilo.com/en/Products",
        "search_url":   "https://www.wilo.com/en/search/?q={model}",
        "datasheet_hint": "Product → Downloads → Data sheet / characteristic curve",
        "notes": "Good online curve viewer for HVAC pumps.",
    },
    "sulzer": {
        "name": "Sulzer",
        "selector_url": "https://www.sulzer.com/en/products/pumps",
        "search_url":   "https://www.sulzer.com/en/search?query={model}",
        "datasheet_hint": "Product page → Downloads",
        "notes": "Industrial process pumps. Curves in product datasheets.",
    },
    "goulds": {
        "name": "Goulds / ITT",
        "selector_url": "https://www.goulds.com",
        "search_url":   "https://www.goulds.com/search?q={model}",
        "datasheet_hint": "Product → Literature / Curves",
        "notes": "North American focus. ITT brand.",
    },
    "armstrong": {
        "name": "Armstrong",
        "selector_url": "https://armstrongfluidtechnology.com/en",
        "search_url":   "https://armstrongfluidtechnology.com/en/search?q={model}",
        "datasheet_hint": "Product → Downloads",
        "notes": "HVAC and building services pumps.",
    },
    "ebara": {
        "name": "Ebara",
        "selector_url": "https://www.ebara.com/en/products/pump",
        "search_url":   "https://www.ebara.com/en/search?q={model}",
        "datasheet_hint": "Product → Catalog / Data sheet",
        "notes": "Asian and European range.",
    },
    "xylem": {
        "name": "Xylem",
        "selector_url": "https://www.xylem.com/en-us/products--services/pumps",
        "search_url":   "https://www.xylem.com/en-us/search/?q={model}",
        "datasheet_hint": "Product → Downloads",
        "notes": "Water and wastewater pumps.",
    },
    "lowara": {
        "name": "Lowara (Xylem)",
        "selector_url": "https://www.lowara.com/en-gb/products",
        "search_url":   "https://www.lowara.com/en-gb/search?q={model}",
        "datasheet_hint": "Product → Downloads → Performance curves",
        "notes": "Part of Xylem group.",
    },
    "pedrollo": {
        "name": "Pedrollo",
        "selector_url": "https://www.pedrollo.com/en/products",
        "search_url":   "https://www.pedrollo.com/en/?s={model}",
        "datasheet_hint": "Product → Technical data / Curve",
        "notes": "Italian manufacturer, wide centrifugal range.",
    },
    "calpeda": {
        "name": "Calpeda",
        "selector_url": "https://www.calpeda.com/en/products",
        "search_url":   "https://www.calpeda.com/en/search?q={model}",
        "datasheet_hint": "Product → Documentation → Curve",
        "notes": "Italian manufacturer.",
    },
}


def _normalise_manufacturer(name: str) -> str:
    """Return a normalised key for the manufacturer lookup."""
    n = name.lower().strip()
    # Handle common aliases
    aliases = {
        "itt": "goulds", "goulds pumps": "goulds",
        "xylem goulds": "goulds",
        "lowara xylem": "lowara",
        "ksb pumps": "ksb",
        "grundfos pumps": "grundfos",
        "sulzer pumps": "sulzer",
        "wilo pumps": "wilo",
        "armstrong pumps": "armstrong",
        "armstrong fluid technology": "armstrong",
    }
    for alias, key in aliases.items():
        if alias in n:
            return key
    for key in MANUFACTURER_URLS:
        if key in n:
            return key
    return n


def get_manufacturer_info(manufacturer: str) -> Optional[dict]:
    """Return manufacturer URL info if known."""
    key = _normalise_manufacturer(manufacturer)
    return MANUFACTURER_URLS.get(key)


def build_source_reference(
    method: str,
    url: str,
    manufacturer: str,
    model: str,
    confidence: str,
    notes: str = "",
) -> dict:
    """Build a structured source reference dict for report inclusion."""
    return {
        "method":         method,
        "url":            url,
        "manufacturer":   manufacturer,
        "model":          model,
        "confidence":     confidence,
        "retrieved_date": datetime.date.today().isoformat(),
        "notes":          notes,
    }


def format_source_reference_text(ref: dict) -> str:
    """Format source reference as a readable citation string for reports."""
    method_labels = {
        "manufacturer_site": "Manufacturer product selector",
        "web_search":        "Web search",
        "manual_entry":      "Manual entry by engineer",
        "not_found":         "Not found — manual entry required",
    }
    label = method_labels.get(ref.get("method", ""), ref.get("method", ""))
    return (
        f"Pump curve source: {label}. "
        f"Manufacturer: {ref.get('manufacturer','')}. "
        f"Model: {ref.get('model','')}. "
        f"Retrieved: {ref.get('retrieved_date','')}. "
        f"Confidence: {ref.get('confidence','')}. "
        + (f"URL: {ref.get('url','')}. " if ref.get('url') else "")
        + (f"Notes: {ref.get('notes','')}." if ref.get('notes') else "")
    )


def search_and_extract_curves(
    manufacturer: str,
    model: str,
    rated_flow_m3h: float,
    rated_head_m: float,
    api_key: str,
) -> dict:
    """
    Main entry point. Tries manufacturer site first, then web search.

    Returns:
        {
            "success":          bool,
            "method":           str,   # manufacturer_site | web_search | not_found
            "hq_points":        list,  # [{"q": float, "h": float}, ...]
            "eta_points":       list,  # [{"q": float, "eta": float}, ...]  optional
            "power_points":     list,  # [{"q": float, "p": float}, ...]   optional
            "source_ref":       dict,
            "message":          str,
            "raw_response":     str,
            "needs_review":     bool,
        }
    """
    client = anthropic.Anthropic(api_key=api_key)
    mfr_info = get_manufacturer_info(manufacturer)

    # ── Step 1: Try manufacturer site ────────────────────────────────────────
    step1_result = None
    if mfr_info:
        search_url = mfr_info["search_url"].format(model=model.replace(" ", "+"))
        step1_result = _try_manufacturer_site(
            client, manufacturer, model, search_url,
            mfr_info, rated_flow_m3h, rated_head_m
        )
        if step1_result and step1_result.get("success"):
            return step1_result

    # ── Step 2: Web search fallback ──────────────────────────────────────────
    step2_result = _try_web_search(
        client, manufacturer, model, rated_flow_m3h, rated_head_m
    )
    if step2_result and step2_result.get("success"):
        return step2_result

    # ── Step 3: Not found — return guidance ──────────────────────────────────
    manual_url = mfr_info["selector_url"] if mfr_info else ""
    return {
        "success":      False,
        "method":       "not_found",
        "hq_points":    [],
        "eta_points":   [],
        "power_points": [],
        "source_ref":   build_source_reference(
            "not_found", manual_url, manufacturer, model, "none",
            "Automatic extraction failed. Manual entry required."
        ),
        "message": (
            f"Could not automatically retrieve performance curves for "
            f"{manufacturer} {model}. "
            + (f"Try the manufacturer's product selector at: {manual_url}"
               if manual_url else
               "This manufacturer is not in the known URL library.")
        ),
        "raw_response": "",
        "needs_review": True,
    }


def _try_manufacturer_site(
    client, manufacturer: str, model: str,
    search_url: str, mfr_info: dict,
    rated_flow: float, rated_head: float,
) -> Optional[dict]:
    """Attempt to retrieve and extract curves from manufacturer's product page."""
    try:
        prompt = f"""You are helping extract pump performance curve data.

Manufacturer: {manufacturer}
Model number: {model}
Known rated duty point: Q = {rated_flow} m³/h, H = {rated_head} m

Please search the manufacturer's product selector at:
{search_url}

Look for the product matching model {model} and extract the H-Q performance curve data.

If you find curve data, return a JSON object with this exact structure:
{{
  "found": true,
  "source_url": "exact URL where curve was found",
  "hq_points": [{{"q": 0, "h": 0}}, {{"q": 10, "h": 45}}, ...],
  "eta_points": [{{"q": 0, "eta": 0}}, ...],
  "power_points": [{{"q": 0, "p": 0}}, ...],
  "impeller_diameter_mm": 196,
  "rated_speed_rpm": 2900,
  "notes": "any relevant notes about the data"
}}

If you cannot find the product or the curves, return:
{{"found": false, "reason": "explanation"}}

Return ONLY the JSON, no other text."""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
            }],
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                raw_text += block.text

        data = _parse_json_response(raw_text)
        if not data or not data.get("found"):
            return None

        hq   = data.get("hq_points", [])
        eta  = data.get("eta_points", [])
        pw   = data.get("power_points", [])

        if not hq or len(hq) < 3:
            return None

        source_url = data.get("source_url", mfr_info["selector_url"])
        return {
            "success":      True,
            "method":       "manufacturer_site",
            "hq_points":    hq,
            "eta_points":   eta,
            "power_points": pw,
            "source_ref":   build_source_reference(
                "manufacturer_site", source_url, manufacturer, model,
                "high",
                f"Retrieved from {mfr_info['name']} product selector. "
                + data.get("notes", "")
            ),
            "message": (
                f"Performance curves retrieved from {mfr_info['name']} "
                f"product selector. {len(hq)} H-Q data points extracted. "
                f"Please review and confirm the data is correct before analysis."
            ),
            "raw_response": raw_text,
            "needs_review": True,
        }

    except Exception as e:
        return None


def _try_web_search(
    client, manufacturer: str, model: str,
    rated_flow: float, rated_head: float,
) -> Optional[dict]:
    """Fall back to general web search for pump curve data."""
    try:
        query = (
            f"{manufacturer} {model} centrifugal pump performance curve "
            f"H-Q datasheet specifications"
        )
        prompt = f"""You are helping extract centrifugal pump performance curve data.

Manufacturer: {manufacturer}
Model number: {model}
Known rated duty point: Q = {rated_flow} m³/h, H = {rated_head} m

Search the web for "{query}" and find H-Q performance curve data for this pump.

Look for:
- Manufacturer datasheets (PDF)
- Product pages with curve data
- Technical specifications tables

If you find curve data, return a JSON object:
{{
  "found": true,
  "source_url": "exact URL of source",
  "source_type": "manufacturer_pdf | third_party | specification_table",
  "hq_points": [{{"q": 0, "h": 0}}, {{"q": 10, "h": 45}}, ...],
  "eta_points": [{{"q": 0, "eta": 0}}, ...],
  "power_points": [{{"q": 0, "p": 0}}, ...],
  "impeller_diameter_mm": null,
  "rated_speed_rpm": null,
  "confidence": "high | medium | low",
  "notes": "source description and any caveats"
}}

If not found: {{"found": false, "reason": "explanation"}}

Minimum 3 H-Q points required. Include shutdown head (Q=0) if available.
Return ONLY the JSON."""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
            }],
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                raw_text += block.text

        data = _parse_json_response(raw_text)
        if not data or not data.get("found"):
            return None

        hq   = data.get("hq_points", [])
        eta  = data.get("eta_points", [])
        pw   = data.get("power_points", [])

        if not hq or len(hq) < 3:
            return None

        confidence = data.get("confidence", "medium")
        source_url = data.get("source_url", "")
        source_type = data.get("source_type", "web_search")

        return {
            "success":      True,
            "method":       "web_search",
            "hq_points":    hq,
            "eta_points":   eta,
            "power_points": pw,
            "source_ref":   build_source_reference(
                "web_search", source_url, manufacturer, model,
                confidence,
                f"Found via web search. Source type: {source_type}. "
                + data.get("notes", "")
            ),
            "message": (
                f"Performance curves found via web search (confidence: {confidence}). "
                f"{len(hq)} H-Q data points extracted from: {source_url}. "
                f"**Please review carefully** — web-sourced curves may not match "
                f"your specific impeller diameter or pump variant."
            ),
            "raw_response": raw_text,
            "needs_review": True,
        }

    except Exception as e:
        return None


def _parse_json_response(text: str) -> Optional[dict]:
    """Extract and parse JSON from Claude's response text."""
    if not text:
        return None
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Find JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None


def validate_hq_points(
    hq_points: list,
    rated_flow: float,
    rated_head: float,
    tolerance_pct: float = 20.0,
) -> dict:
    """
    Validate extracted H-Q points against the known rated duty point.
    Returns validation result with warnings.
    """
    if not hq_points or len(hq_points) < 3:
        return {"valid": False, "warnings": ["Fewer than 3 H-Q data points."]}

    flows  = [pt["q"] for pt in hq_points]
    heads  = [pt["h"] for pt in hq_points]
    warnings = []

    # Check head is monotonically decreasing with flow (centrifugal pump characteristic)
    for i in range(len(heads) - 1):
        if heads[i] < heads[i+1] and flows[i] < flows[i+1]:
            warnings.append(
                f"Head increases with flow between Q={flows[i]} and "
                f"Q={flows[i+1]} m³/h — unusual for centrifugal pump."
            )

    # Check rated duty point is within the curve range
    if rated_flow < min(flows) or rated_flow > max(flows):
        warnings.append(
            f"Rated flow {rated_flow} m³/h is outside curve range "
            f"({min(flows)}–{max(flows)} m³/h)."
        )

    # Interpolate head at rated flow and compare to rated head
    try:
        import numpy as np
        h_interp = float(np.interp(rated_flow, flows, heads))
        diff_pct  = abs(h_interp - rated_head) / rated_head * 100
        if diff_pct > tolerance_pct:
            warnings.append(
                f"Curve head at rated flow ({h_interp:.1f} m) differs "
                f"{diff_pct:.1f}% from nameplate rated head ({rated_head} m). "
                f"Verify correct model and impeller diameter."
            )
    except Exception:
        pass

    return {
        "valid":    len(warnings) == 0,
        "warnings": warnings,
        "n_points": len(hq_points),
        "flow_range": (min(flows), max(flows)),
        "head_range": (min(heads), max(heads)),
    }
