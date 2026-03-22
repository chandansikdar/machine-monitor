"""
log_reader.py — Maintenance log ingestion

Supports three file types:
  CSV / Excel  → tabular log, converted to readable text
  PDF          → text extracted with pdfplumber
  Image        → base64-encoded and sent to Claude vision for transcription

Returns a dict:
  {"success": True,  "text": <extracted text>, "method": <how it was read>}
  {"success": False, "error": <message>}
"""

import base64
import io
import os
from typing import Optional

import anthropic
import pandas as pd


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #

def read_log(file, api_key: Optional[str] = None) -> dict:
    """
    Detect file type and extract maintenance log content as plain text.
    Images are sent to Claude vision for transcription.
    """
    name = file.name.lower()

    if name.endswith((".csv", ".xlsx", ".xls")):
        return _read_tabular(file)

    elif name.endswith(".pdf"):
        return _read_pdf(file)

    elif name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif")):
        return _read_image(file, api_key)

    else:
        # Try as plain text (txt, log, etc.)
        return _read_text(file)


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #

def _read_tabular(file) -> dict:
    try:
        name = file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        if df.empty:
            return {"success": False, "error": "File is empty."}

        # Convert to a readable markdown-style table
        lines = ["Maintenance log (tabular format):"]
        lines.append(" | ".join(str(c) for c in df.columns))
        lines.append("-" * 60)
        for _, row in df.iterrows():
            lines.append(" | ".join(str(v) for v in row.values))

        return {
            "success": True,
            "text": "\n".join(lines),
            "method": "csv/excel",
            "rows": len(df),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _read_pdf(file) -> dict:
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(file) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"[Page {i+1}]\n{page_text.strip()}")
                # Also try to extract tables
                tables = page.extract_tables()
                for t in tables:
                    for row in t:
                        if row:
                            text_parts.append(" | ".join(str(c or "") for c in row))

        if not text_parts:
            return {"success": False, "error": "No text found in PDF. It may be a scanned image — try uploading as an image file instead."}

        return {
            "success": True,
            "text": "\n\n".join(text_parts),
            "method": "pdf",
            "pages": len(text_parts),
        }
    except ImportError:
        return {"success": False, "error": "pdfplumber not installed. Run: pip install pdfplumber"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _read_image(file, api_key: Optional[str] = None) -> dict:
    """
    Send image to Claude vision to transcribe handwritten or printed log entries.
    """
    try:
        key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            return {"success": False, "error": "API key required to read image logs."}

        # Read and encode image
        image_bytes = file.read()
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        # Detect media type
        name = file.name.lower()
        if name.endswith(".png"):
            media_type = "image/png"
        elif name.endswith((".jpg", ".jpeg")):
            media_type = "image/jpeg"
        elif name.endswith(".webp"):
            media_type = "image/webp"
        else:
            media_type = "image/jpeg"

        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "This is a maintenance log for an industrial machine. "
                                "Please transcribe all text visible in this image accurately, "
                                "preserving dates, work descriptions, technician names, part numbers, "
                                "and any other maintenance details. "
                                "Format as a structured list of log entries if possible. "
                                "If the image is unclear in parts, note that with [unclear]."
                            ),
                        },
                    ],
                }
            ],
        )

        transcribed = response.content[0].text.strip()
        return {
            "success": True,
            "text": f"Maintenance log (transcribed from image):\n\n{transcribed}",
            "method": "image/vision",
        }

    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _read_text(file) -> dict:
    try:
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        return {
            "success": True,
            "text": content,
            "method": "text",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}
