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
from vision_desktop_automation.notepad import process_post_notepad, process_post_generic


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
    parser.add_argument(
        "--app",
        type=str,
        choices=["notepad", "generic"],
        default=None,
        help="Workflow type: 'notepad' for Notepad automation, 'generic' for just opening an app (default: notepad if --target not set, generic if --target is set)",
    )
    parser.add_argument(
        "--search-mode",
        type=str,
        choices=["fast", "robust"],
        default=None,
        help="Planner search mode: 'fast' for the top candidate only, 'robust' for multiple plausible regions.",
    )
    return parser.parse_args()


def validate_environment() -> None:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it as an environment variable before running."
        )

    logging.info("Environment validated")


def main(force_app_type: str | None = None) -> None:
    args = parse_args()

    # Apply CLI overrides to config before anything else reads those values
    import vision_desktop_automation.config as cfg

    if args.target:
        cfg.TARGET_DESCRIPTION = args.target

    if args.posts:
        cfg.POST_LIMIT = args.posts

    if args.output_dir:
        cfg.OUTPUT_DIR = Path(args.output_dir).expanduser().resolve()

    if args.search_mode:
        cfg.PLANNER_SEARCH_MODE = args.search_mode

    setup_logging()
    ensure_runtime_dirs()

    if args.target:
        logging.info(f"Target override: {args.target}")
    if args.posts:
        logging.info(f"Posts limit override: {args.posts}")
    if args.output_dir:
        logging.info(f"Output directory override: {cfg.OUTPUT_DIR}")
    if args.search_mode:
        logging.info(f"Planner search mode override: {cfg.PLANNER_SEARCH_MODE}")

    logging.info("Starting in 2 seconds — do not touch mouse or keyboard")
    logging.info(f"Target: {cfg.TARGET_DESCRIPTION}")
    time.sleep(2)

    try:
        validate_environment()
    except Exception as e:
        logging.exception(f"Startup failed: {e}")
        return

    # Determine which workflow to use.
    # Priority: force_app_type (GUI launcher) > args.app (CLI --app flag) > auto-detect.
    app_type = force_app_type if force_app_type is not None else args.app
    if app_type is None:
        # Auto-detect based on whether --target was provided
        app_type = "generic" if args.target else "notepad"

    logging.info(f"Using workflow: {app_type}")

    if app_type == "generic":
        logging.info("Generic workflow selected — skipping post fetch")
        try:
            process_post_generic({"id": "generic"})
            logging.info("Generic workflow completed successfully")
        except Exception as e:
            logging.exception(f"Generic workflow failed: {e}")
            save_debug_screenshot("generic_fatal")
        return

    try:
        posts = fetch_posts()
    except Exception as e:
        logging.exception(f"Startup failed: {e}")
        return

    failed_posts: list[int] = []

    for post in posts:
        try:
            if app_type == "notepad":
                process_post_notepad(post)
            else:  # generic
                process_post_generic(post)
            logging.info("")
        except pyautogui.FailSafeException:
            logging.error("PyAutoGUI fail-safe triggered — terminating process immediately")
            raise
        except Exception as e:
            logging.exception(f"Failed on post {post['id']}: {e}")
            save_debug_screenshot(f"fatal_post_{post['id']}")
            failed_posts.append(post["id"])

            try:
                ensure_desktop_clear()
                reset_ui_state()
            except Exception:
                pass

    # Final cleanup — `close_all_notepad_windows` is normally invoked at the
    # start of the next iteration's `open_notepad`, but the last post has no
    # next iteration, so any leftover Notepad would be left running.
    try:
        from vision_desktop_automation.notepad import close_all_notepad_windows
        close_all_notepad_windows()
    except Exception as e:
        logging.warning(f"Final Notepad cleanup failed: {e}")

    logging.info("=" * 60)

    if failed_posts:
        logging.warning(f"Failed posts: {failed_posts}")
    else:
        logging.info("All posts processed successfully")

    verify_outputs(posts)


if __name__ == "__main__":
    main()
