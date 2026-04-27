"""
Vision-Based Desktop Automation with VLM Grounding
====================================================
ScreenSeekeR-inspired implementation for Windows Notepad automation.

Key design:
1. Planner proposes candidate search regions from the full screenshot.
2. Grounder searches inside candidate regions.
3. Icon verifier confirms identity.
4. Fallback uses direct grounding if candidate-region search fails.
5. Automation opens Notepad, pastes JSONPlaceholder posts, saves files.

Important:
- Do NOT hardcode API keys.
- Set GEMINI_API_KEY as an environment variable before running.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyautogui
import pyperclip

from PIL import Image


from vision_desktop_automation.grounding import (
    planner_guided_ground_icon,
    verify_icon_identity,
)

from vision_desktop_automation.template_matching import template_match_notepad_icon

from vision_desktop_automation.desktop import (
    ensure_desktop_clear,
    get_active_window_title,
    get_cached_icon_position,
    get_dpi_scale,
    icon_still_at_cached_location,
    increment_cache_hits,
    invalidate_icon_cache,
    reset_ui_state,
    show_desktop,
    update_icon_cache,
)

from vision_desktop_automation.files import (
    ensure_runtime_dirs,
    get_next_unsaved_note_number,
    save_annotated_screenshot,
    save_debug_screenshot,
    setup_logging,
    verify_outputs,
)

from vision_desktop_automation.vlm_client import (
    call_gemini_vision,
    parse_vlm_json,
    recover_planner_regions_from_text,
    validate_grounding_result,
)

from vision_desktop_automation.api import fetch_posts
from vision_desktop_automation.geometry import (
    box_iou,
    describe_screen_region,
    expand_region_pixels,
    normalize_pct,
)

from vision_desktop_automation.prompts import (
    CACHE_CHECK_PROMPT,
    GROUNDING_PROMPT,
    PLANNER_PROMPT,
    VERIFY_PROMPT,
)

from vision_desktop_automation.config import (
    AFTER_CLOSE_WAIT,
    AFTER_SAVE_WAIT,
    ALLOW_DIRECT_GROUNDING_FALLBACK,
    API_RETRIES,
    API_URL,
    BOX_NMS_IOU_THRESHOLD,
    BOX_SCORE_WEIGHT,
    CACHE_STRONG_DIFF_THRESHOLD,
    CACHE_TOLERANT_DIFF_THRESHOLD,
    DESKTOP_WAIT,
    FAILURE_SCREENSHOT_DIR,
    GEMINI_API_KEY,
    GEMINI_API_URL,
    GROUNDING_CONFIDENCE_WEIGHT,
    ICON_DETECTION_RETRIES,
    LOG_FILE,
    MAX_CANDIDATE_REGIONS,
    MAX_GROUNDING_PROPOSALS,
    MAX_SEARCH_DEPTH,
    MIN_PLANNER_REGION_SCORE,
    MIN_REGION_HEIGHT_PX,
    MIN_REGION_WIDTH_PX,
    NOTEPAD_OPEN_WAIT_MAX,
    OUTPUT_DIR,
    PASTE_WAIT,
    PATCH_THRESHOLD_PX,
    POST_ENTER_WAIT,
    POST_LIMIT,
    RECURSIVE_ACCEPT_CONFIDENCE,
    RECURSIVE_ACCEPT_SCORE,
    RECURSIVE_PLANNER_DEPTH,
    REGION_NMS_IOU_THRESHOLD,
    REGION_SCORE_WEIGHT,
    SAVE_DIALOG_WAIT,
    SAVE_RETRIES,
    SCREENSHOT_PATH,
    SEARCH_CONFIDENCE_THRESHOLD,
    SKIP_VERIFICATION_IF_CONFIDENT,
    TARGET_DESCRIPTION,
    TEMPLATE_DIR,
    TEMPLATE_FILE_PATTERNS,
    TEMPLATE_MATCH_SCALES,
    TEMPLATE_MATCH_THRESHOLD,
    UNSAVED_NOTE_COUNTER_FILE,
    UI_RESET_WAIT,
    USE_PLANNER_CANDIDATE_REGIONS,
    USE_TEMPLATE_MATCHING_FALLBACK,
    USE_VLM_CACHE_CONFIRMATION,
    VERIFICATION_SKIP_CONFIDENCE,
    VERIFICATION_SKIP_REGION_SCORE,
    VLM_MODEL,
)

# =========================
# SAFETY
# =========================
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.15


# =========================
# WINDOW HELPERS
# =========================
def get_notepad_windows():
    try:
        import pygetwindow as gw

        return gw.getWindowsWithTitle("Notepad")
    except Exception:
        return []



def ensure_notepad_focused():
    windows = get_notepad_windows()
    if not windows:
        raise RuntimeError("Notepad window lost")

    try:
        w = windows[0]
        w.activate()
        time.sleep(0.3)
        pyautogui.click(w.left + w.width // 2, w.top + w.height // 2)
        time.sleep(0.2)
    except Exception as e:
        raise RuntimeError(f"Could not focus Notepad: {e}")


def save_notepad_as(save_path: str):
    pyautogui.hotkey("ctrl", "shift", "s")
    time.sleep(2.0)

    title = get_active_window_title()
    if "Notepad" in title and "Save" not in title:
        pyautogui.hotkey("ctrl", "s")
        time.sleep(2.0)

    pyperclip.copy(save_path)
    time.sleep(0.2)
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.2)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.5)
    pyautogui.press("enter")
    time.sleep(1.0)

    active = get_active_window_title()
    if any(k in active for k in ["Confirm", "Replace", "already exists", "Overwrite"]):
        logging.info("Overwrite dialog on save: left+enter")
        pyautogui.press("left")
        time.sleep(0.2)
        pyautogui.press("enter")
        time.sleep(0.8)


def close_all_notepad_windows():
    windows = get_notepad_windows()
    if not windows:
        return

    logging.info(f"Found {len(windows)} leftover Notepad window(s) — closing safely...")

    for w in windows:
        try:
            w.activate()
            time.sleep(0.3)

            title = get_active_window_title()
            if "*" not in title:
                logging.info("Leftover Notepad does not appear unsaved — closing without Save As")
                pyautogui.hotkey("alt", "F4")
                time.sleep(0.8)
                continue

            note_num = get_next_unsaved_note_number()
            save_path = str(OUTPUT_DIR / f"unsaved_note_{note_num}.txt")
            logging.info(f"Saving leftover as: unsaved_note_{note_num}.txt")

            save_notepad_as(save_path)
            pyautogui.hotkey("alt", "F4")
            time.sleep(0.8)

            active = get_active_window_title()
            if any(k in active for k in ["Save", "save", "changes", "Confirm"]):
                logging.info("Discarding remaining unsaved changes dialog")
                pyautogui.press("tab")
                time.sleep(0.2)
                pyautogui.press("enter")
                time.sleep(0.3)

            logging.info(f"Leftover Notepad saved and closed as unsaved_note_{note_num}.txt")
        except Exception as e:
            logging.warning(f"Could not save/close leftover Notepad: {e}")
            try:
                pyautogui.hotkey("alt", "F4")
                time.sleep(0.5)
                pyautogui.press("tab")
                time.sleep(0.2)
                pyautogui.press("enter")
            except Exception:
                pass


def dismiss_unexpected_window(title: str):
    if not title.strip():
        return

    logging.info(f"Dismissing unexpected window: '{title}'")
    is_browser = any(
        k in title
        for k in ["Chrome", "Firefox", "Edge", "Opera", "Brave", "Safari", "Browser", "Google", "Mozilla"]
    )

    if is_browser:
        logging.info("Browser detected — Alt+F4 then confirming close dialog")
        pyautogui.hotkey("alt", "F4")
        time.sleep(1.2)
        active = get_active_window_title()
        if active and any(k in active for k in ["Close", "close", "tabs", "Chrome", "Edge", "Firefox", "Google"]):
            logging.info(f"Browser close dialog: '{active}' — pressing Enter")
            pyautogui.press("enter")
            time.sleep(0.8)
        return

    pyautogui.press("escape")
    time.sleep(0.4)

    try:
        import pygetwindow as gw

        if not gw.getWindowsWithTitle(title):
            logging.info("Dismissed with Escape")
            return
    except Exception:
        pass

    pyautogui.hotkey("alt", "F4")
    time.sleep(0.5)
    for _ in range(4):
        time.sleep(0.3)
        active = get_active_window_title()
        if any(k in active for k in ["Save", "save", "Confirm", "changes", "Close"]):
            pyautogui.press("tab")
            time.sleep(0.2)
            pyautogui.press("enter")
            time.sleep(0.3)
            break


# =========================
# UI WORKFLOW
# =========================



def open_notepad():
    ensure_desktop_clear()
    reset_ui_state()
    close_all_notepad_windows()
    ensure_desktop_clear()
    reset_ui_state()

    scale_x, scale_y = get_dpi_scale()
    last_error = None

    for attempt in range(1, ICON_DETECTION_RETRIES + 1):
        try:
            cached_position = get_cached_icon_position()
            if cached_position is not None:
                x, y = cached_position
                logging.info(f"Checking icon cache: ({x}, {y})")
                if not icon_still_at_cached_location(x, y, scale_x, scale_y):
                    logging.warning("Icon moved or changed — invalidating cache")
                    invalidate_icon_cache()
                    raise RuntimeError("Cache invalidated")
            else:
                logging.info(f"Planner-guided VLM grounding attempt {attempt}")
                ensure_desktop_clear()
                time.sleep(0.5)

                screenshot = pyautogui.screenshot()
                screenshot.save(SCREENSHOT_PATH)
                pil_image = Image.open(SCREENSHOT_PATH)

                detection_method = "planner_guided_vlm"

                try:
                    x_shot, y_shot, confidence = planner_guided_ground_icon(
                        pil_image,
                        TARGET_DESCRIPTION,
                    )
                except Exception as vlm_error:
                    logging.warning(f"Planner-guided VLM grounding failed: {vlm_error}")

                    if not USE_TEMPLATE_MATCHING_FALLBACK:
                        raise

                    logging.info("Trying local template-matching fallback...")
                    template_result = template_match_notepad_icon(pil_image)

                    if template_result is None:
                        raise RuntimeError(
                            f"VLM grounding failed and template fallback found no reliable match. "
                            f"Original VLM error: {vlm_error}"
                        )

                    x_shot, y_shot, confidence = template_result
                    detection_method = "template_matching_fallback"

                    logging.info(
                        f"Template fallback selected Notepad candidate at "
                        f"({x_shot},{y_shot}) with score={confidence:.3f}"
                    )

                if detection_method == "template_matching_fallback":
                    logging.info(
                        "Skipping VLM verifier for template fallback; "
                        "Notepad launch validation will confirm result"
                    )
                elif confidence < VERIFICATION_SKIP_CONFIDENCE:
                    if not verify_icon_identity(
                        pil_image,
                        x_shot / pil_image.width,
                        y_shot / pil_image.height,
                    ):
                        raise ValueError("Wrong icon detected after planner-guided grounding")
                else:
                    logging.info("Skipping outer icon verification due high grounding confidence")
                x = int(x_shot * scale_x)
                y = int(y_shot * scale_y)
                logging.info(f"Grounded: shot=({x_shot},{y_shot}) → screen=({x},{y})")
                save_annotated_screenshot(f"attempt_{attempt}", x, y)

                update_icon_cache(x, y, scale_x, scale_y)

            logging.info(f"Double-clicking Notepad at ({x}, {y})")
            pyautogui.doubleClick(x, y, interval=0.2)

            launched = False
            for tick in range(NOTEPAD_OPEN_WAIT_MAX):
                time.sleep(1.0)
                if get_notepad_windows():
                    logging.info(f"Notepad opened after {tick + 1}s")
                    launched = True
                    break

            if not launched:
                active_title = get_active_window_title()
                if not active_title.strip():
                    logging.warning("Nothing opened — likely missed icon")
                elif "Notepad" not in active_title:
                    logging.warning(f"Wrong app opened: '{active_title}'")
                    dismiss_unexpected_window(active_title)

                invalidate_icon_cache()
                raise RuntimeError(f"Notepad not launched. Active: '{active_title}'")

            increment_cache_hits()
            return

        except Exception as e:
            last_error = e
            logging.warning(f"Open Notepad attempt {attempt} failed: {e}")
            save_debug_screenshot(f"icon_detection_attempt_{attempt}")
            reset_ui_state()
            wait = 10 if "429" in str(e) else 1
            logging.info(f"Waiting {wait}s before retry")
            time.sleep(wait)

    raise RuntimeError(f"Failed to open Notepad after {ICON_DETECTION_RETRIES} retries. Last: {last_error}")


def paste_post_content(post: dict[str, Any]):
    content = f"Title: {post['title']}\n\n{post['body']}"
    pyperclip.copy(content)
    time.sleep(0.2)

    ensure_notepad_focused()
    logging.info(f"Pasting post {post['id']}")
    pyautogui.hotkey("ctrl", "v")
    time.sleep(PASTE_WAIT)

    title = get_active_window_title()
    if "Notepad" in title and "*" not in title:
        logging.warning("Paste may have failed — retrying")
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.3)


def wait_for_save_dialog(timeout: int = 5) -> bool:
    for _ in range(timeout * 2):
        time.sleep(0.5)
        title = get_active_window_title()
        if "Save As" in title or ("Notepad" not in title and title.strip()):
            logging.info(f"Save dialog detected: '{title}'")
            return True
    logging.warning("Save dialog did not appear")
    return False


def save_file(post_id: int):
    filename = f"post_{post_id}.txt"
    full_path = str(OUTPUT_DIR / filename)
    last_error = None

    for attempt in range(1, SAVE_RETRIES + 1):
        try:
            ensure_notepad_focused()
            logging.info(f"Saving {filename}, attempt {attempt}")

            pyautogui.hotkey("ctrl", "s")

            if not wait_for_save_dialog(timeout=5):
                logging.warning("Save dialog did not appear — retrying Ctrl+S")
                ensure_notepad_focused()
                pyautogui.hotkey("ctrl", "s")
                time.sleep(SAVE_DIALOG_WAIT)

            time.sleep(0.5)
            pyperclip.copy(full_path)
            time.sleep(0.2)
            pyautogui.hotkey("ctrl", "a")
            time.sleep(0.2)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.5)
            pyautogui.press("enter")
            time.sleep(POST_ENTER_WAIT)

            active = get_active_window_title()
            if any(k in active for k in ["Confirm", "Replace", "already exists", "Overwrite"]):
                logging.info(f"Overwrite dialog: '{active}' — confirming replace")
                pyautogui.press("left")
                time.sleep(0.2)
                pyautogui.press("enter")
            else:
                logging.info("No overwrite dialog detected")

            time.sleep(AFTER_SAVE_WAIT)

            saved_path = Path(full_path)
            if saved_path.exists() and saved_path.stat().st_size > 0:
                logging.info(f"Verified: {filename} ({saved_path.stat().st_size} bytes)")
                return filename

            raise RuntimeError(f"File not found or empty: {full_path}")

        except Exception as e:
            last_error = e
            logging.warning(f"Save attempt {attempt} failed: {e}")
            save_debug_screenshot(f"save_attempt_{attempt}_post_{post_id}")
            pyautogui.press("escape")
            time.sleep(1)

    raise RuntimeError(f"Failed to save after {SAVE_RETRIES} retries. Last: {last_error}")


def close_notepad():
    logging.info("Closing Notepad")
    windows = get_notepad_windows()
    if windows:
        try:
            windows[0].activate()
            time.sleep(0.2)
        except Exception:
            pass

    pyautogui.hotkey("ctrl", "w")
    time.sleep(AFTER_CLOSE_WAIT)

    if get_notepad_windows():
        logging.warning("Notepad still open after Ctrl+W — checking for dialog")
        active = get_active_window_title()
        if any(k in active for k in ["Save", "save", "changes", "Confirm"]):
            logging.info(f"Save dialog: '{active}' — discarding with Tab+Enter")
            pyautogui.press("tab")
            time.sleep(0.2)
            pyautogui.press("enter")
            time.sleep(0.5)

        if get_notepad_windows():
            logging.warning("Force closing with Alt+F4")
            pyautogui.hotkey("alt", "F4")
            time.sleep(0.5)


def process_post(post: dict[str, Any]):
    logging.info("=" * 60)
    logging.info(f"Processing post {post['id']} / {POST_LIMIT}")
    open_notepad()
    paste_post_content(post)
    save_file(post["id"])
    close_notepad()
    logging.info(f"Finished post {post['id']}")

# =========================
# API
# =========================

def validate_environment():
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it as an environment variable before running."
        )
    logging.info("Environment validated")

# =========================
# MAIN
# =========================
def main():
    setup_logging()
    ensure_runtime_dirs()
    logging.info("Starting in 8 seconds — do not touch mouse or keyboard")
    logging.info("Make sure the Notepad shortcut is visible on the desktop")
    time.sleep(8)

    try:
        validate_environment()
        posts = fetch_posts()
    except Exception as e:
        logging.exception(f"Startup failed: {e}")
        return

    failed_posts: list[int] = []

    for post in posts:
        try:
            process_post(post)
            logging.info("")
        except Exception as e:
            logging.exception(f"Failed on post {post['id']}: {e}")
            save_debug_screenshot(f"fatal_post_{post['id']}")
            failed_posts.append(post["id"])
            try:
                ensure_desktop_clear()
                reset_ui_state()
            except Exception:
                pass

    logging.info("=" * 60)
    if failed_posts:
        logging.warning(f"Failed posts: {failed_posts}")
    else:
        logging.info("All posts processed successfully")

    verify_outputs(posts)

if __name__ == "__main__":
    main()
