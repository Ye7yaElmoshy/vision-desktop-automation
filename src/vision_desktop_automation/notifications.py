"""
Windows notification service for Vision Desktop Automation.

Notifications are suppressed while a grounding screenshot is being captured,
to prevent toast popups from appearing in the desktop image sent to Gemini.
"""

import logging
import subprocess
import threading

_grounding_active = False
_lock = threading.Lock()


def set_grounding_active(active: bool) -> None:
    """Mark that a grounding screenshot sequence is starting or ending."""
    global _grounding_active
    with _lock:
        _grounding_active = active


def send_notification(title: str, message: str, is_error: bool = False) -> None:
    """
    Send a Windows Action Center toast notification.

    Silently defers the notification if grounding is active, preventing
    toast popups from contaminating the desktop screenshot sent to Gemini.
    """
    with _lock:
        active = _grounding_active

    if active:
        logging.info(f"Notification deferred (grounding active): [{title}] {message}")
        return

    try:
        _send_windows_toast(title, message, is_error)
        logging.info(f"Notification sent: [{title}] {message}")
    except Exception as e:
        logging.warning(f"Failed to send notification: {e}")


def _send_windows_toast(title: str, message: str, is_error: bool) -> None:
    """Fire a Windows 10/11 Action Center toast via PowerShell WinRT bindings."""
    safe_title = title.replace("'", "''")
    safe_message = message.replace("'", "''")
    ps_lines = [
        "[Windows.UI.Notifications.ToastNotificationManager, "
        "Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null",
        "$t = [Windows.UI.Notifications.ToastNotificationManager]::"
        "GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)",
        f"$t.SelectSingleNode('//text[@id=\"1\"]').InnerText = '{safe_title}'",
        f"$t.SelectSingleNode('//text[@id=\"2\"]').InnerText = '{safe_message}'",
        "$n = [Windows.UI.Notifications.ToastNotification]::new($t)",
        "[Windows.UI.Notifications.ToastNotificationManager]::"
        "$n.SuppressPopup = $true"
        "CreateToastNotifier('Vision Desktop Automation').Show($n)",
    ]
    subprocess.Popen(
        [
            "powershell",
            "-NonInteractive",
            "-WindowStyle", "Hidden",
            "-Command", "; ".join(ps_lines),
        ],
        creationflags=subprocess.CREATE_NO_WINDOW,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
