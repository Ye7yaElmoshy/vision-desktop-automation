"""
Vision-Based Desktop Automation with VLM Grounding
====================================================
ScreenSeekeR-inspired implementation for Windows Notepad automation.

Important:
- Do NOT hardcode API keys.
- Set GEMINI_API_KEY as an environment variable before running.
"""

import argparse
import logging
import time
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vision-based desktop automation with VLM grounding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m vision_desktop_automation --target "Chrome browser icon"
  python -m vision_desktop_automation --target "VS Code" --posts 5
  python -m vision_desktop_automation --target "Calculator app" --output-dir Desktop/calc-test
        """,
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target icon description (e.g., 'Chrome browser icon'). If not provided, uses config default.",
    )
    parser.add_argument(
        "--posts",
        type=int,
        default=None,
        help="Number of posts to process (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory path (default: Desktop/tjm-project)",
    )
    return parser.parse_args()


def validate_environment() -> None:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it as an environment variable before running."
        )

    logging.info("Environment validated")


def main() -> None:
    args = parse_args()

    # Apply CLI overrides to config before anything else reads those values
    import vision_desktop_automation.config as cfg

    if args.target:
        cfg.TARGET_DESCRIPTION = args.target

    if args.posts:
        cfg.POST_LIMIT = args.posts

    if args.output_dir:
        cfg.OUTPUT_DIR = Path(args.output_dir).expanduser().resolve()

    setup_logging()
    ensure_runtime_dirs()

    if args.target:
        logging.info(f"Target override: {args.target}")
    if args.posts:
        logging.info(f"Posts limit override: {args.posts}")
    if args.output_dir:
        logging.info(f"Output directory override: {cfg.OUTPUT_DIR}")

    logging.info("Starting in 8 seconds — do not touch mouse or keyboard")
    logging.info(f"Target: {cfg.TARGET_DESCRIPTION}")
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
