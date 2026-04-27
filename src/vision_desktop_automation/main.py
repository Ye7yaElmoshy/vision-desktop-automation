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

from vision_desktop_automation.notepad import process_post

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
