import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

from vision_desktop_automation.config import (
    TEMPLATE_DIR,
    TEMPLATE_FILE_PATTERNS,
    TEMPLATE_MATCH_SCALES,
    TEMPLATE_MATCH_THRESHOLD,
)


def load_template_images() -> list[tuple[str, np.ndarray]]:
    """Load Notepad template images from TEMPLATE_DIR for local fallback."""
    templates: list[tuple[str, np.ndarray]] = []

    for pattern in TEMPLATE_FILE_PATTERNS:
        for path in TEMPLATE_DIR.glob(pattern):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)

            if img is None:
                logging.warning(f"Could not load template image: {path}")
                continue

            templates.append((path.name, img))

    if not templates:
        logging.warning(
            f"No template images found in {TEMPLATE_DIR}. "
            "Template fallback will be skipped."
        )

    return templates


def template_match_notepad_icon(screenshot: Image.Image) -> tuple[int, int, float] | None:
    """
    Local OpenCV fallback for locating the Notepad desktop shortcut.

    This is not the primary method. It is a graceful-degradation fallback for
    cases where Gemini is unavailable, returns 503, or produces unusable JSON.
    It performs grayscale multi-scale template matching.
    """
    templates = load_template_images()

    if not templates:
        return None

    screenshot_rgb = np.array(screenshot.convert("RGB"))
    screenshot_bgr = cv2.cvtColor(screenshot_rgb, cv2.COLOR_RGB2BGR)
    screenshot_gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)

    best: dict[str, Any] | None = None

    for template_name, template_bgr in templates:
        template_gray_base = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        base_h, base_w = template_gray_base.shape[:2]

        for scale in TEMPLATE_MATCH_SCALES:
            w = int(base_w * scale)
            h = int(base_h * scale)

            if w < 12 or h < 12:
                continue

            if w > screenshot_gray.shape[1] or h > screenshot_gray.shape[0]:
                continue

            resized_template = cv2.resize(
                template_gray_base,
                (w, h),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
            )

            result = cv2.matchTemplate(
                screenshot_gray,
                resized_template,
                cv2.TM_CCOEFF_NORMED,
            )

            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if best is None or max_val > best["score"]:
                best = {
                    "template": template_name,
                    "score": float(max_val),
                    "x": int(max_loc[0] + w / 2),
                    "y": int(max_loc[1] + h / 2),
                    "box": (
                        int(max_loc[0]),
                        int(max_loc[1]),
                        int(max_loc[0] + w),
                        int(max_loc[1] + h),
                    ),
                    "scale": scale,
                }

    if best is None:
        return None

    logging.info(
        f"Best template match: template={best['template']}, "
        f"score={best['score']:.3f}, center=({best['x']},{best['y']}), "
        f"box={best['box']}, scale={best['scale']:.2f}"
    )

    if best["score"] < TEMPLATE_MATCH_THRESHOLD:
        logging.warning(
            f"Template fallback rejected: best score {best['score']:.3f} "
            f"is below threshold {TEMPLATE_MATCH_THRESHOLD:.2f}"
        )
        return None

    return int(best["x"]), int(best["y"]), float(best["score"])