from typing import Any

from vision_desktop_automation.config import (
    MIN_REGION_HEIGHT_PX,
    MIN_REGION_WIDTH_PX,
)


def normalize_pct(value: Any, default: float) -> float:
    """Normalize a model-returned coordinate into a 0.0–1.0 fraction."""
    try:
        v = float(value)
        if v > 1.0:
            v = v / 100.0
        return max(0.0, min(1.0, v))
    except Exception:
        return default


def expand_region_pixels(
    px1: int,
    py1: int,
    px2: int,
    py2: int,
    img_w: int,
    img_h: int,
    min_w: int = MIN_REGION_WIDTH_PX,
    min_h: int = MIN_REGION_HEIGHT_PX,
    pad: int = 80,
) -> tuple[int, int, int, int]:
    """
    Expand tight planner regions instead of rejecting them.

    Gemini often returns a tight box around the icon. A larger crop preserves
    the icon, label, and surrounding context for the grounder.
    """
    cx = (px1 + px2) // 2
    cy = (py1 + py2) // 2

    target_w = max(px2 - px1, min_w) + 2 * pad
    target_h = max(py2 - py1, min_h) + 2 * pad

    new_px1 = max(0, cx - target_w // 2)
    new_py1 = max(0, cy - target_h // 2)
    new_px2 = min(img_w, cx + target_w // 2)
    new_py2 = min(img_h, cy + target_h // 2)

    return new_px1, new_py1, new_px2, new_py2


def box_area(box: dict[str, float]) -> float:
    """Calculate area of a normalized bounding box."""
    return max(0.0, box["x2"] - box["x1"]) * max(0.0, box["y2"] - box["y1"])


def box_iou(a: dict[str, float], b: dict[str, float]) -> float:
    """Calculate Intersection over Union between two normalized boxes."""
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])

    inter = box_area({"x1": ix1, "y1": iy1, "x2": ix2, "y2": iy2})
    union = box_area(a) + box_area(b) - inter

    return inter / union if union > 0 else 0.0


def describe_screen_region(x: int, y: int, screen_w: int, screen_h: int) -> str:
    """Convert screen coordinates into a human-readable region label."""
    horizontal = "left" if x < screen_w / 3 else "right" if x > (2 * screen_w) / 3 else "center"
    vertical = "top" if y < screen_h / 3 else "bottom" if y > (2 * screen_h) / 3 else "middle"

    if horizontal == "center" and vertical == "middle":
        return "center of the screen"
    if horizontal == "center":
        return f"{vertical} center"
    if vertical == "middle":
        return f"middle {horizontal}"

    return f"{vertical} {horizontal}"