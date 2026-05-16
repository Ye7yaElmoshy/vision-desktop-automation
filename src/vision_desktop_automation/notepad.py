import logging
import time
from pathlib import Path
from typing import Any

import pyautogui
import pyperclip

from vision_desktop_automation.config import (
    AFTER_CLOSE_WAIT,
    AFTER_SAVE_WAIT,
    ICON_DETECTION_RETRIES,
    NOTEPAD_OPEN_WAIT_MAX,
    PASTE_WAIT,
    POST_ENTER_WAIT,
    SAVE_DIALOG_WAIT,
    SAVE_RETRIES,
    SCREENSHOT_PATH,
    USE_TEMPLATE_MATCHING_FALLBACK,
    VERIFICATION_SKIP_CONFIDENCE,
)
import vision_desktop_automation.config as cfg

from vision_desktop_automation.desktop import (
    ensure_desktop_clear,
    get_active_window_title,
    get_cached_icon_position,
    get_dpi_scale,
    icon_still_at_cached_location,
    increment_cache_hits,
    invalidate_icon_cache,
    move_mouse_to_safe_position,
    reset_ui_state,
    update_icon_cache,
)

from vision_desktop_automation.files import (
    get_next_unsaved_note_number,
    save_annotated_screenshot,
    save_debug_screenshot,
)

from vision_desktop_automation.grounding import (
    planner_guided_ground_icon,
    verify_icon_identity,
)

from vision_desktop_automation.template_matching import template_match_icon


def _raise_if_failsafe(e: Exception) -> None:
    if isinstance(e, pyautogui.FailSafeException):
        raise


def is_visible_sane_window(w: Any) -> bool:
    """Return True for normal on-screen Notepad windows."""
    left = int(getattr(w, "left", 0))
    top = int(getattr(w, "top", 0))
    width = int(getattr(w, "width", 0))
    height = int(getattr(w, "height", 0))

    if width <= 100 or height <= 100:
        return False

    if left < -1000 or top < -1000:
        return False

    return True


def get_notepad_windows():
    """
    Return real Notepad app windows only.

    The previous implementation used pygetwindow.getWindowsWithTitle("Notepad"),
    which is a substring match and incorrectly matches anything containing
    "notepad" — e.g. an editor showing the file 'notepad.py' has a title like
    'project – notepad.py' and was being treated as a launched Notepad window.

    Real Notepad windows always have titles of the form:
        "<file or 'Untitled'> - Notepad"
        "*<file> - Notepad"             (unsaved indicator)
    so we filter on that suffix and exclude anything else (Notepad++, editors,
    file explorers, etc.).
    """
    try:
        import pygetwindow as gw

        windows = []
        for w in gw.getAllWindows():
            title = str(getattr(w, "title", "")).strip()
            if not title:
                continue
            # Strip leading "*" used to mark unsaved changes, then check suffix.
            normalized = title[1:] if title.startswith("*") else title
            if normalized.endswith(" - Notepad"):
                windows.append(w)
        return windows
    except Exception:
        return []


def get_visible_notepad_windows():
    return [w for w in get_notepad_windows() if is_visible_sane_window(w)]


def log_notepad_window(idx: int, w: Any) -> None:
    title = str(getattr(w, "title", ""))
    left = int(getattr(w, "left", 0))
    top = int(getattr(w, "top", 0))
    width = int(getattr(w, "width", 0))
    height = int(getattr(w, "height", 0))
    logging.info(
        f"Notepad window #{idx}: title='{title}', "
        f"left={left}, top={top}, width={width}, height={height}"
    )


def ensure_notepad_focused():
    windows = get_notepad_windows()
    if not windows:
        raise RuntimeError("Notepad window lost")

    try:
        sane_windows = []

        for idx, w in enumerate(windows):
            log_notepad_window(idx, w)

            if is_visible_sane_window(w):
                sane_windows.append(w)

        if not sane_windows:
            raise RuntimeError("No visible sane Notepad window found")

        w = sane_windows[0]
        w.activate()
        time.sleep(0.5)

        # Do not click window coordinates here.
        # Clicking raw pygetwindow coordinates can move the cursor to a fail-safe corner.
        logging.info("Notepad activated without mouse-center click")

    except Exception as e:
        raise RuntimeError(f"Could not focus Notepad: {e}")


def save_notepad_as(save_path: str):
    # Ctrl+Shift+S works on Windows 11 Notepad; fall back to Alt+F, A for Windows 10
    pyautogui.hotkey("ctrl", "shift", "s")
    time.sleep(1.5)

    title = get_active_window_title()
    if "Save" not in title:
        pyautogui.hotkey("alt", "f")
        time.sleep(0.5)
        pyautogui.press("a")
        time.sleep(1.5)

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
            if not is_visible_sane_window(w):
                logging.info(
                    f"Closing hidden/minimized leftover Notepad: "
                    f"'{getattr(w, 'title', '')}'"
                )
                w.close()
                time.sleep(0.5)
                continue

            w.activate()
            time.sleep(0.3)

            title = get_active_window_title()
            if "*" not in title:
                logging.info("Leftover Notepad does not appear unsaved — closing without Save As")
                pyautogui.hotkey("alt", "F4")
                time.sleep(0.8)
                continue

            note_num = get_next_unsaved_note_number()
            save_path = str(cfg.OUTPUT_DIR / f"unsaved_note_{note_num}.txt")
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
            _raise_if_failsafe(e)
            logging.warning(f"Could not save/close leftover Notepad: {e}")
            try:
                if hasattr(w, "close"):
                    w.close()
                else:
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

    # Never Alt+F4 the desktop itself — that triggers the Windows Shutdown dialog.
    # If the click missed the icon, focus may land on Program Manager (the desktop).
    if title.lower().strip() in {"program manager", "desktop"}:
        logging.info("Active window is the desktop itself — no dismissal needed (skipping Alt+F4)")
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

    logging.info("Unexpected non-browser window still present — minimizing instead of closing")
    pyautogui.hotkey("win", "m")
    time.sleep(0.5)


# =========================
# UI WORKFLOW
# =========================



def open_notepad():
    # Single desktop-clear at entry — close_all_notepad_windows handles
    # leftover Notepad windows; the desktop is already clear from the
    # previous post's close_notepad call. The ensure_desktop_clear inside
    # the per-attempt loop below handles the cold-start case.
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
                if not icon_still_at_cached_location(x, y, scale_x, scale_y, cfg.TARGET_DESCRIPTION):
                    logging.warning("Icon moved or changed — invalidating cache")
                    invalidate_icon_cache()
                    raise RuntimeError("Cache invalidated")
            else:
                logging.info(f"Planner-guided VLM grounding attempt {attempt}")
                ensure_desktop_clear()
                time.sleep(0.5)

                pil_image = pyautogui.screenshot()
                pil_image.save(SCREENSHOT_PATH)

                detection_method = "planner_guided_vlm"

                try:
                    x_shot, y_shot, confidence = planner_guided_ground_icon(
                        pil_image,
                        cfg.TARGET_DESCRIPTION,
                    )
                except Exception as vlm_error:
                    logging.warning(f"Planner-guided VLM grounding failed: {vlm_error}")

                    if not USE_TEMPLATE_MATCHING_FALLBACK:
                        raise

                    logging.info("Trying local template-matching fallback...")
                    template_result = template_match_icon(pil_image)

                    if template_result is None:
                        raise RuntimeError(
                            f"VLM grounding failed and template fallback found no reliable match. "
                            f"Original VLM error: {vlm_error}"
                        )

                    x_shot, y_shot, confidence = template_result
                    detection_method = "template_matching_fallback"

                    logging.info(
                        f"Template fallback selected candidate at "
                        f"({x_shot},{y_shot}) with score={confidence:.3f}"
                    )

                if detection_method == "template_matching_fallback":
                    logging.info(
                        "Skipping VLM verifier for template fallback; "
                        "launch validation will confirm result"
                    )
                elif confidence < VERIFICATION_SKIP_CONFIDENCE:
                    if not verify_icon_identity(
                        pil_image,
                        x_shot / pil_image.width,
                        y_shot / pil_image.height,
                        cfg.TARGET_DESCRIPTION,
                    ):
                        raise ValueError("Wrong icon detected after planner-guided grounding")
                else:
                    logging.info("Skipping outer icon verification due high grounding confidence")
                x = int(x_shot * scale_x)
                y = int(y_shot * scale_y)
                logging.info(f"Grounded: shot=({x_shot},{y_shot}) → screen=({x},{y})")
                save_annotated_screenshot(f"attempt_{attempt}", x, y, pil_image)

                update_icon_cache(x, y, scale_x, scale_y)

            logging.info(f"Double-clicking Notepad at ({x}, {y})")
            move_mouse_to_safe_position()
            pyautogui.doubleClick(x, y, interval=0.2)
            move_mouse_to_safe_position()

            launched = False
            # Poll every 0.2s instead of every 1.0s — same total timeout,
            # but detects Notepad opening 0.5–0.8s earlier on average.
            poll_ticks = NOTEPAD_OPEN_WAIT_MAX * 5
            for tick in range(poll_ticks):
                time.sleep(0.2)
                if get_visible_notepad_windows():
                    elapsed_s = (tick + 1) * 0.2
                    logging.info(f"Notepad opened after {elapsed_s:.1f}s")
                    launched = True
                    break

            if not launched:
                active_title = get_active_window_title()
                if not active_title.strip():
                    logging.warning("Nothing opened — likely missed icon")
                elif "Notepad" not in active_title:
                    logging.warning(f"Wrong app opened: '{active_title}'")
                    dismiss_unexpected_window(active_title)

                # GUI fallback: try selecting the icon and pressing Enter,
                # then retry with a slight offset double-click.
                try:
                    logging.info("Attempting GUI fallback: select icon and press Enter")
                    move_mouse_to_safe_position()
                    pyautogui.click(x, y)
                    time.sleep(0.2)
                    pyautogui.press("enter")
                    # Poll briefly for Notepad window
                    for tick in range(poll_ticks):
                        time.sleep(0.2)
                        if get_visible_notepad_windows():
                            elapsed_s = (tick + 1) * 0.2
                            logging.info(f"Notepad opened via GUI fallback after {elapsed_s:.1f}s")
                            increment_cache_hits()
                            return
                except Exception as gui_e:
                    _raise_if_failsafe(gui_e)
                    logging.warning(f"GUI fallback failed: {gui_e}")

                try:
                    logging.info("Retrying double-click with slight offset")
                    move_mouse_to_safe_position()
                    pyautogui.doubleClick(x + 5, y + 5, interval=0.25)
                    for tick in range(poll_ticks):
                        time.sleep(0.2)
                        if get_visible_notepad_windows():
                            elapsed_s = (tick + 1) * 0.2
                            logging.info(f"Notepad opened after offset double-click in {elapsed_s:.1f}s")
                            increment_cache_hits()
                            return
                except Exception as dc_e:
                    _raise_if_failsafe(dc_e)
                    logging.warning(f"Offset double-click failed: {dc_e}")

                invalidate_icon_cache()
                raise RuntimeError(f"Notepad not launched. Active: '{active_title}'")

            increment_cache_hits()
            return

        except Exception as e:
            _raise_if_failsafe(e)
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
        if "Save As" in title or "Save as" in title:
            logging.info(f"Save dialog detected: '{title}'")
            return True
    logging.warning("Save dialog did not appear")
    return False


def save_file(post_id: int):
    filename = f"post_{post_id}.txt"
    full_path = str(cfg.OUTPUT_DIR / filename)
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

            time.sleep(0.15)
            pyperclip.copy(full_path)
            time.sleep(0.1)
            pyautogui.hotkey("ctrl", "a")
            time.sleep(0.1)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.2)
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
            _raise_if_failsafe(e)
            last_error = e
            logging.warning(f"Save attempt {attempt} failed: {e}")
            save_debug_screenshot(f"save_attempt_{attempt}_post_{post_id}")
            pyautogui.press("escape")
            time.sleep(1)

    raise RuntimeError(f"Failed to save after {SAVE_RETRIES} retries. Last: {last_error}")


def close_notepad():
    """
    Close the current Notepad window using Ctrl+W.

    Ctrl+W is preferred over Alt+F4 because:
    - Notepad treats it as 'close document' rather than 'kill window'
    - It cannot accidentally trigger the Windows Shutdown dialog
      if focus shifts to the desktop mid-call
    - Works consistently across Windows 10 (classic Notepad) and
      Windows 11 (tabbed Notepad)
    """
    logging.info("Closing Notepad")
    windows = get_visible_notepad_windows()
    if windows:
        try:
            w = windows[0]
            w.activate()
            time.sleep(0.25)
        except Exception:
            pass

    pyautogui.hotkey("ctrl", "w")
    time.sleep(AFTER_CLOSE_WAIT)

    if get_visible_notepad_windows():
        active = get_active_window_title()
        if any(k in active for k in ["Save", "save", "changes", "Confirm", "Notepad"]):
            logging.info(f"Save/confirm dialog: '{active}' — discarding with Tab+Enter")
            pyautogui.press("tab")
            time.sleep(0.2)
            pyautogui.press("enter")
            time.sleep(0.5)

        # Second attempt if still open
        if get_visible_notepad_windows():
            logging.warning("Notepad still open after Ctrl+W — sending second Ctrl+W")
            pyautogui.hotkey("ctrl", "w")
            time.sleep(0.5)

    for w in get_notepad_windows():
        if not is_visible_sane_window(w):
            try:
                logging.info(
                    f"Closing hidden/minimized Notepad after visible close: "
                    f"'{getattr(w, 'title', '')}'"
                )
                w.close()
                time.sleep(0.3)
            except Exception as e:
                _raise_if_failsafe(e)
                logging.warning(f"Could not close hidden Notepad window: {e}")


def process_post_notepad(post: dict[str, Any]):
    """Notepad workflow: open, paste, save, close."""
    logging.info("=" * 60)
    logging.info(f"Processing post {post['id']} / {cfg.POST_LIMIT}")
    open_notepad()
    paste_post_content(post)
    save_file(post["id"])
    close_notepad()
    logging.info(f"Finished post {post['id']}")


def process_post_generic(post: dict[str, Any]):
    """Generic workflow: just find and open the target app."""
    logging.info("=" * 60)
    logging.info("Processing generic app launch — opening target app only")
    ensure_desktop_clear()
    reset_ui_state()

    scale_x, scale_y = get_dpi_scale()
    last_error = None

    for attempt in range(1, ICON_DETECTION_RETRIES + 1):
        try:
            logging.info(f"Finding and opening target app — attempt {attempt}")
            ensure_desktop_clear()
            time.sleep(0.5)

            pil_image = pyautogui.screenshot()
            pil_image.save(SCREENSHOT_PATH)

            try:
                x_shot, y_shot, confidence = planner_guided_ground_icon(
                    pil_image,
                    cfg.TARGET_DESCRIPTION,
                )
            except Exception as vlm_error:
                logging.warning(f"VLM grounding failed: {vlm_error}")

                if not USE_TEMPLATE_MATCHING_FALLBACK:
                    raise

                logging.info("Trying local template-matching fallback...")
                template_result = template_match_icon(pil_image)

                if template_result is None:
                    raise RuntimeError(
                        f"VLM grounding failed and template fallback found no reliable match. "
                        f"Original VLM error: {vlm_error}"
                    )

                x_shot, y_shot, confidence = template_result
                logging.info(
                    f"Template fallback selected candidate at "
                    f"({x_shot},{y_shot}) with score={confidence:.3f}"
                )

            x = int(x_shot * scale_x)
            y = int(y_shot * scale_y)
            logging.info(f"Grounded: shot=({x_shot},{y_shot}) → screen=({x},{y})")
            save_annotated_screenshot(f"generic_attempt_{attempt}", x, y, pil_image)

            logging.info(f"Double-clicking target at ({x}, {y})")
            move_mouse_to_safe_position()
            pyautogui.doubleClick(x, y, interval=0.2)
            move_mouse_to_safe_position()

            logging.info("Target app launched")
            time.sleep(2)
            logging.info(f"Finished post {post['id']}")
            return

        except Exception as e:
            _raise_if_failsafe(e)
            last_error = e
            logging.warning(f"Open app attempt {attempt} failed: {e}")
            save_debug_screenshot(f"generic_attempt_{attempt}")
            reset_ui_state()
            wait = 10 if "429" in str(e) else 1
            logging.info(f"Waiting {wait}s before retry")
            time.sleep(wait)

    raise RuntimeError(f"Failed to open target app after {ICON_DETECTION_RETRIES} retries. Last: {last_error}")

