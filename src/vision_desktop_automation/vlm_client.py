import base64
import json
import logging
import random
import re
import time
from typing import Any

import requests
from PIL import Image

from vision_desktop_automation.config import (
    API_RETRIES,
    GEMINI_API_KEY,
    GEMINI_API_URL,
    VLM_MODEL,
)

RETRYABLE_HTTP_STATUSES = {500, 502, 503, 504}


class NonRetryableGeminiAPIError(RuntimeError):
    """Raised for Gemini HTTP errors that should not be retried locally."""


def _response_error_detail(response: requests.Response) -> str:
    """Extract the useful Gemini error message without dumping a huge body."""
    try:
        data = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:500] if text else (response.reason or "no response body")

    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        status = str(error.get("status", "")).strip()
        message = str(error.get("message", "")).strip()
        if status and message:
            return f"{status}: {message}"[:500]
        if message:
            return message[:500]
        if status:
            return status[:500]

    return json.dumps(data, sort_keys=True)[:500]


def image_to_base64(image: Image.Image, max_width: int = 960) -> tuple[str, str]:
    """Convert a PIL image to base64 for Gemini Vision. Returns (data, mime_type).

    Small images (≤200px wide) are encoded as lossless PNG to avoid JPEG block
    artifacts that can confuse the VLM on icon/crop inputs.
    """
    import io

    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, int(image.height * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    if image.width <= 200 and image.height <= 200:
        image.save(buf, format="PNG")
        mime_type = "image/png"
    else:
        image.save(buf, format="JPEG", quality=92)
        mime_type = "image/jpeg"

    return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), mime_type


def call_gemini_vision(prompt: str, image: Image.Image) -> str:
    """Call Gemini Vision with a prompt and image, then return the text response."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it as an environment variable before running."
        )

    b64, mime_type = image_to_base64(image)
    url = GEMINI_API_URL.format(model=VLM_MODEL)

    payload = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": b64}},
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 3200,
            "temperature": 0.0,
            "topP": 1.0,
            "responseMimeType": "application/json",
        },
    }

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY,
    }

    last_error: Any = None

    for attempt in range(1, API_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=45)

            if response.status_code in RETRYABLE_HTTP_STATUSES:
                wait = min(45, (2 ** attempt) * 3 + random.uniform(1.0, 4.0))
                logging.warning(f"Gemini {response.status_code} — retry in {wait:.1f}s")
                time.sleep(wait)
                continue

            if response.status_code >= 400:
                detail = _response_error_detail(response)
                last_error = f"HTTP {response.status_code}: {detail}"
                logging.warning(f"Gemini request failed: {last_error}")
                raise NonRetryableGeminiAPIError(
                    f"Gemini API failed after {attempt} attempt(s). Last: {last_error}"
                )

            response.raise_for_status()

            data = response.json()
            candidates = data.get("candidates", [])

            if not candidates:
                raise ValueError("Gemini returned empty candidates list")

            return candidates[0]["content"]["parts"][0]["text"].strip()

        except requests.exceptions.Timeout:
            last_error = "timeout"
            logging.warning(f"Gemini timeout attempt {attempt}")
            time.sleep(2 ** attempt)

        except NonRetryableGeminiAPIError:
            raise

        except Exception as e:
            last_error = e

            if "429" in str(e):
                raise

            logging.warning(f"Gemini attempt {attempt} failed: {e}")
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Gemini API failed after {API_RETRIES} attempts. Last: {last_error}")


def parse_vlm_json(response_text: str) -> dict[str, Any]:
    """Parse JSON returned by the VLM, with partial recovery for common fields."""
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", response_text).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    result: dict[str, Any] = {}

    m = re.search(r'"found"\s*:\s*(true|false)', cleaned, flags=re.I)
    if m:
        result["found"] = m.group(1).lower() == "true"

    m = re.search(r'"correct"\s*:\s*(true|false)', cleaned, flags=re.I)
    if m:
        result["correct"] = m.group(1).lower() == "true"

    m = re.search(r'"confidence"\s*:\s*([0-9.]+)', cleaned)
    if m:
        result["confidence"] = float(m.group(1))

    m = re.search(r'"x1_pct"\s*:\s*([0-9.]+)', cleaned)
    if m:
        result["x1_pct"] = float(m.group(1))

    m = re.search(r'"y1_pct"\s*:\s*([0-9.]+)', cleaned)
    if m:
        result["y1_pct"] = float(m.group(1))

    m = re.search(r'"x2_pct"\s*:\s*([0-9.]+)', cleaned)
    if m:
        result["x2_pct"] = float(m.group(1))

    m = re.search(r'"y2_pct"\s*:\s*([0-9.]+)', cleaned)
    if m:
        result["y2_pct"] = float(m.group(1))

    m = re.search(r'"click_x_pct"\s*:\s*([0-9.]+)', cleaned)
    if m:
        result["click_x_pct"] = float(m.group(1))

    m = re.search(r'"click_y_pct"\s*:\s*([0-9.]+)', cleaned)
    if m:
        result["click_y_pct"] = float(m.group(1))

    m = re.search(r'"label_match"\s*:\s*([0-9.]+)', cleaned)
    if m:
        result["label_match"] = float(m.group(1))

    m = re.search(r'"visual_match"\s*:\s*([0-9.]+)', cleaned)
    if m:
        result["visual_match"] = float(m.group(1))

    m = re.search(r'"reason"\s*:\s*"([^"]*)"', cleaned)
    if m:
        result["reason"] = m.group(1)

    if not ("click_x_pct" in result and "click_y_pct" in result) and "box" in cleaned:
        box_match = re.search(
            r'"box"\s*:\s*\{[^}]*"x1_pct"\s*:\s*([0-9.]+)[^}]*"y1_pct"\s*:\s*([0-9.]+)[^}]*"x2_pct"\s*:\s*([0-9.]+)[^}]*"y2_pct"\s*:\s*([0-9.]+)',
            cleaned,
        )
        if box_match:
            result["x1_pct"] = float(result.get("x1_pct", box_match.group(1)))
            result["y1_pct"] = float(result.get("y1_pct", box_match.group(2)))
            result["x2_pct"] = float(result.get("x2_pct", box_match.group(3)))
            result["y2_pct"] = float(result.get("y2_pct", box_match.group(4)))

    if result:
        logging.warning(f"Recovered partial JSON: {result}")
        return result

    raise ValueError(f"Invalid JSON unrecoverable: {cleaned[:180]}")



def recover_planner_regions_from_text(text: str) -> dict[str, Any]:
    """
    Recover planner candidate regions when Gemini returns markdown,
    extra text, or partially malformed output.

    Only complete objects with score and coordinate fields are accepted.
    """
    cleaned = text.replace("```json", "").replace("```", "").strip()

    decoder = json.JSONDecoder()
    regions: list[dict[str, Any]] = []
    idx = 0

    while idx < len(cleaned):
        start = cleaned.find("{", idx)

        if start == -1:
            break

        try:
            obj, end = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            idx = start + 1
            continue

        if end <= 0:
            idx = start + 1
            continue

        if isinstance(obj, dict):
            if isinstance(obj.get("candidate_regions"), list):
                for region in obj["candidate_regions"]:
                    if isinstance(region, dict):
                        regions.append(region)

            elif all(k in obj for k in ("score", "x1_pct", "y1_pct", "x2_pct", "y2_pct")):
                regions.append(obj)

        idx = start + end

    valid_regions: list[dict[str, Any]] = []

    for region in regions:
        try:
            valid_regions.append(
                {
                    "name": str(region.get("name", "recovered_region")),
                    "reason": str(region.get("reason", "recovered partial JSON"))[:80],
                    "score": float(region["score"]),
                    "x1_pct": float(region["x1_pct"]),
                    "y1_pct": float(region["y1_pct"]),
                    "x2_pct": float(region["x2_pct"]),
                    "y2_pct": float(region["y2_pct"]),
                }
            )
        except Exception:
            continue

    if not valid_regions:
        partial = _recover_partial_planner_region(cleaned)
        if partial is not None:
            logging.warning(
                "Recovered one planner region from partial/truncated JSON output"
            )
            return {"candidate_regions": [partial]}
        raise ValueError("Could not recover planner regions from partial JSON")

    logging.warning(f"Recovered {len(valid_regions)} planner region(s) from partial JSON")
    return {"candidate_regions": valid_regions}


def _recover_partial_planner_region(text: str) -> dict[str, Any] | None:
    """Recover a single candidate region from truncated planner output."""
    fields: dict[str, str] = {}
    patterns = {
        "name": r'"name"\s*:\s*"([^"]*)"',
        "reason": r'"reason"\s*:\s*"([^"]*)"',
        "score": r'"score"\s*:\s*([0-9.]+)',
        "x1_pct": r'"x1_pct"\s*:\s*([0-9.]+)',
        "y1_pct": r'"y1_pct"\s*:\s*([0-9.]+)',
        "x2_pct": r'"x2_pct"\s*:\s*([0-9.]+)',
        "y2_pct": r'"y2_pct"\s*:\s*([0-9.]+)',
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            fields[key] = m.group(1)

    required = ["score", "x1_pct", "y1_pct", "x2_pct", "y2_pct"]
    if not all(k in fields for k in required):
        return None

    try:
        return {
            "name": fields.get("name", "recovered_region"),
            "reason": fields.get("reason", "recovered partial JSON")[:80],
            "score": float(fields["score"]),
            "x1_pct": float(fields["x1_pct"]),
            "y1_pct": float(fields["y1_pct"]),
            "x2_pct": float(fields["x2_pct"]),
            "y2_pct": float(fields["y2_pct"]),
        }
    except Exception:
        return None
