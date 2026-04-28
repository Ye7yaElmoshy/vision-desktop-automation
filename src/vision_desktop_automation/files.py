import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyautogui

from vision_desktop_automation.config import (
    FAILURE_SCREENSHOT_DIR,
    LOG_FILE,
    OUTPUT_DIR,
    UNSAVED_NOTE_COUNTER_FILE,
)


def get_next_unsaved_note_number() -> int:
    try:
        if UNSAVED_NOTE_COUNTER_FILE.exists():
            n = int(UNSAVED_NOTE_COUNTER_FILE.read_text().strip())
        else:
            n = 0
        n += 1
        UNSAVED_NOTE_COUNTER_FILE.write_text(str(n))
        return n
    except Exception:
        return int(time.time())


def setup_logging() -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    try:
        handlers.append(logging.FileHandler(LOG_FILE, encoding="utf-8"))
    except PermissionError:
        fallback = Path(f"automation_{int(time.time())}.log")
        handlers.append(logging.FileHandler(fallback, encoding="utf-8"))
        print(f"WARNING: {LOG_FILE} locked, logging to {fallback}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def ensure_runtime_dirs() -> None:
    FAILURE_SCREENSHOT_DIR.mkdir(exist_ok=True)

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory: {OUTPUT_DIR}")
    except Exception as e:
        raise RuntimeError(f"Cannot create output directory {OUTPUT_DIR}: {e}") from e


def save_debug_screenshot(label: str) -> None:
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = FAILURE_SCREENSHOT_DIR / f"{label}_{timestamp}.png"
        pyautogui.screenshot().save(path)
        logging.info(f"Saved debug screenshot: {path}")
    except Exception as e:
        logging.warning(f"Could not save debug screenshot: {e}")


def save_annotated_screenshot(label: str, click_x: int, click_y: int) -> None:
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = FAILURE_SCREENSHOT_DIR / f"annotated_{label}_{timestamp}.png"

        img = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)

        cv2.circle(img, (click_x, click_y), 20, (0, 255, 0), 3)
        cv2.circle(img, (click_x, click_y), 5, (0, 255, 0), -1)

        cv2.putText(
            img,
            f"Detected: ({click_x}, {click_y})",
            (max(0, click_x - 150), max(30, click_y - 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imwrite(str(path), img)
        logging.info(f"Saved annotated screenshot: {path}")

    except Exception as e:
        logging.warning(f"Could not save annotated screenshot: {e}")


def verify_outputs(posts: list[dict[str, Any]]) -> bool:
    logging.info("=" * 60)
    logging.info("Verifying files")

    all_ok = True

    for post in posts:
        filepath = OUTPUT_DIR / f"post_{post['id']}.txt"
        expected = f"Title: {post['title']}"

        if not filepath.exists():
            logging.error(f"MISSING: {filepath.name}")
            all_ok = False
            continue

        if filepath.stat().st_size == 0:
            logging.error(f"EMPTY: {filepath.name}")
            all_ok = False
            continue

        content = filepath.read_text(encoding="utf-8", errors="replace")

        if expected in content:
            logging.info(f"OK: post_{post['id']}.txt")
        else:
            logging.error(f"WRONG CONTENT: post_{post['id']}.txt")
            all_ok = False

    if all_ok:
        logging.info("All files verified successfully")
    else:
        logging.warning("Some files failed — re-run the script")

    return all_ok