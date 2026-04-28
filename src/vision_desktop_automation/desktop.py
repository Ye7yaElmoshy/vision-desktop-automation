import logging
import time
from typing import Any

import numpy as np
import pyautogui
from PIL import Image

from vision_desktop_automation.config import (
    CACHE_STRONG_DIFF_THRESHOLD,
    CACHE_TOLERANT_DIFF_THRESHOLD,
    DESKTOP_WAIT,
    UI_RESET_WAIT,
    USE_VLM_CACHE_CONFIRMATION,
)
from vision_desktop_automation.prompts import CACHE_CHECK_PROMPT
from vision_desktop_automation.vlm_client import call_gemini_vision, parse_vlm_json


_icon_cache: dict[str, Any] = {
    "x": None,
    "y": None,
    "hits": 0,
    "reference_crop": None,
}


def get_active_window_title() -> str:
    try:
        import pygetwindow as gw

        active = gw.getActiveWindow()
        return active.title if active else ""
    except Exception:
        return ""


def get_dpi_scale() -> tuple[float, float]:
    screen_w, screen_h = pyautogui.size()
    shot = pyautogui.screenshot()
    shot_w, shot_h = shot.size

    scale_x = screen_w / shot_w if shot_w > 0 else 1.0
    scale_y = screen_h / shot_h if shot_h > 0 else 1.0

    if scale_x != 1.0 or scale_y != 1.0:
        logging.info(f"DPI scale: ({scale_x:.2f}, {scale_y:.2f})")

    return scale_x, scale_y


def capture_icon_crop(
    x: int,
    y: int,
    scale_x: float,
    scale_y: float,
) -> Image.Image | None:
    try:
        shot = pyautogui.screenshot()
        img_w, img_h = shot.size

        cx = int(x / scale_x)
        cy = int(y / scale_y)

        x1 = max(0, cx - 40)
        y1 = max(0, cy - 40)
        x2 = min(img_w, cx + 40)
        y2 = min(img_h, cy + 40)

        if (x2 - x1) < 40 or (y2 - y1) < 40:
            return None

        return shot.crop((x1, y1, x2, y2))

    except Exception:
        return None


def icon_still_at_cached_location(
    x: int,
    y: int,
    scale_x: float,
    scale_y: float,
) -> bool:
    try:
        current_crop = capture_icon_crop(x, y, scale_x, scale_y)

        if current_crop is None:
            logging.warning("Cache crop too small — invalidating")
            return False

        reference_crop = _icon_cache.get("reference_crop")

        if reference_crop is None:
            logging.warning("No reference crop stored — trusting cache and storing now")
            _icon_cache["reference_crop"] = current_crop
            return True

        ref = reference_crop.resize((80, 80)).convert("RGB")
        cur = current_crop.resize((80, 80)).convert("RGB")

        ref_arr = np.array(ref, dtype=np.float32)
        cur_arr = np.array(cur, dtype=np.float32)

        diff = np.mean(np.abs(ref_arr - cur_arr))
        logging.info(f"Cache pixel diff at ({x},{y}): {diff:.1f}")

        if diff < CACHE_STRONG_DIFF_THRESHOLD:
            logging.info(f"Cache valid by strong pixel match, diff={diff:.1f}")
            return True

        if diff < CACHE_TOLERANT_DIFF_THRESHOLD:
            logging.info(
                f"Cache accepted by tolerant local threshold, diff={diff:.1f}. "
                "Launch validation will confirm result."
            )
            return True

        if USE_VLM_CACHE_CONFIRMATION:
            logging.info(f"Cache diff {diff:.1f} high — confirming with VLM")

            shot = pyautogui.screenshot()
            img_w, img_h = shot.size

            cx = int(x / scale_x)
            cy = int(y / scale_y)

            x1 = max(0, cx - 60)
            y1 = max(0, cy - 60)
            x2 = min(img_w, cx + 60)
            y2 = min(img_h, cy + 60)

            crop = shot.crop((x1, y1, x2, y2))

            response = call_gemini_vision(CACHE_CHECK_PROMPT, crop)
            result = parse_vlm_json(response)

            found = bool(result.get("found", False))
            logging.info(f"VLM cache confirmation: found={found}")

            return found

        logging.warning(
            f"Cache invalid by local threshold, diff={diff:.1f} >= "
            f"{CACHE_TOLERANT_DIFF_THRESHOLD}"
        )
        return False

    except Exception as e:
        logging.warning(f"Cache check failed: {e} — invalidating")
        return False


def get_cached_icon_position() -> tuple[int, int] | None:
    if _icon_cache["x"] is None or _icon_cache["y"] is None:
        return None

    return int(_icon_cache["x"]), int(_icon_cache["y"])


def update_icon_cache(
    x: int,
    y: int,
    scale_x: float,
    scale_y: float,
) -> None:
    _icon_cache["x"] = x
    _icon_cache["y"] = y
    _icon_cache["reference_crop"] = capture_icon_crop(x, y, scale_x, scale_y)
    logging.info("Reference crop stored for icon cache")


def increment_cache_hits() -> None:
    _icon_cache["hits"] += 1
    logging.info(f"Cache hits: {_icon_cache['hits']}")


def invalidate_icon_cache() -> None:
    _icon_cache["x"] = None
    _icon_cache["y"] = None
    _icon_cache["reference_crop"] = None


def show_desktop() -> None:
    """Minimize windows without toggling them back open."""
    logging.info("Minimizing all windows...")
    pyautogui.hotkey("win", "m")
    time.sleep(DESKTOP_WAIT)


def reset_ui_state() -> None:
    screen_width, screen_height = pyautogui.size()
    pyautogui.moveTo(screen_width - 200, screen_height // 2, duration=0.2)
    time.sleep(UI_RESET_WAIT)
    pyautogui.click()
    time.sleep(0.2)


def ensure_desktop_clear() -> bool:
    """
    Make the desktop visible for icon grounding.

    Uses Win+M first because Win+D is a toggle.
    Win+D is used only once as fallback.
    """
    logging.info("Showing desktop for grounding...")

    pyautogui.hotkey("win", "m")
    time.sleep(1.0)

    active_title = get_active_window_title().strip()

    if not active_title or active_title.lower() in {"program manager", "desktop"}:
        logging.info("Desktop likely clear after Win+M")
        return True

    logging.info(f"Active window after Win+M: '{active_title}'")

    pyautogui.hotkey("win", "d")
    time.sleep(1.0)

    active_title = get_active_window_title().strip()

    if not active_title or active_title.lower() in {"program manager", "desktop"}:
        logging.info("Desktop clear after fallback Win+D")
        return True

    logging.warning(f"Could not confirm desktop is clear. Active window: '{active_title}'")
    return False