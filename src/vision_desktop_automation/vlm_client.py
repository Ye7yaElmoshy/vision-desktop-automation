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


def image_to_base64(image: Image.Image, max_width: int = 1280) -> str:
    """Convert a PIL image to base64 JPEG for Gemini Vision."""
    import io

    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, int(image.height * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def call_gemini_vision(prompt: str, image: Image.Image) -> str:
    """Call Gemini Vision with a prompt and image, then return the text response."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it as an environment variable before running."
        )

    b64 = image_to_base64(image)
    url = GEMINI_API_URL.format(model=VLM_MODEL)

    payload = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 1400,
            "temperature": 0.1,
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

            if response.status_code in (500, 502, 503, 504):
                wait = min(45, (2 ** attempt) * 3 + random.uniform(1.0, 4.0))
                logging.warning(f"Gemini {response.status_code} — retry in {wait:.1f}s")
                time.sleep(wait)
                continue

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

    m = re.search(r'"click_x_pct"\s*:\s*([0-9.]+)', cleaned)
    if m:
        result["click_x_pct"] = float(m.group(1))

    m = re.search(r'"click_y_pct"\s*:\s*([0-9.]+)', cleaned)
    if m:
        result["click_y_pct"] = float(m.group(1))

    if result:
        logging.warning(f"Recovered partial JSON: {result}")
        return result

    raise ValueError(f"Invalid JSON unrecoverable: {cleaned[:180]}")


def validate_grounding_result(result: dict[str, Any]) -> bool:
    """Validate a simple point-based grounding response."""
    if not result.get("found", False):
        return False

    x = float(result.get("click_x_pct", 0))
    y = float(result.get("click_y_pct", 0))

    if x > 1.0:
        x = x / 100.0
        result["click_x_pct"] = x
        logging.warning(f"Auto-normalized click_x_pct from percentage to fraction: {x:.3f}")

    if y > 1.0:
        y = y / 100.0
        result["click_y_pct"] = y
        logging.warning(f"Auto-normalized click_y_pct from percentage to fraction: {y:.3f}")

    if not (0.01 <= x <= 0.99) or not (0.01 <= y <= 0.99):
        logging.warning(f"Coordinates out of range: ({x}, {y})")
        return False

    if x == 0.0 and y == 0.0:
        logging.warning("Hallucination: found=true with (0,0)")
        return False

    return True


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
        raise ValueError("Could not recover planner regions from partial JSON")

    logging.warning(f"Recovered {len(valid_regions)} planner region(s) from partial JSON")
    return {"candidate_regions": valid_regions}