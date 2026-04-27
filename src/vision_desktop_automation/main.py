"""
Vision-Based Desktop Automation with VLM Grounding
====================================================
ScreenSeekeR-inspired implementation for Windows Notepad automation.

Important:
- Do NOT hardcode API keys.
- Set GEMINI_API_KEY as an environment variable before running.
"""

import logging
import time

import pyautogui

from vision_desktop_automation.api import fetch_posts
from vision_desktop_automation.config import GEMINI_API_KEY
from vision_desktop_automation.desktop import ensure_desktop_clear, reset_ui_state
from vision_desktop_automation.files import (
    ensure_runtime_dirs,
    save_debug_screenshot,
    setup_logging,
    verify_outputs,
)
from vision_desktop_automation.notepad import process_post


# =========================
# SAFETY
# =========================
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.15


def validate_environment() -> None:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it as an environment variable before running."
        )

    logging.info("Environment validated")


def main() -> None:
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