import sys
from pathlib import Path
from types import SimpleNamespace

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from vision_desktop_automation import notepad


def _window(title: str, left: int, top: int, width: int, height: int):
    return SimpleNamespace(
        title=title,
        left=left,
        top=top,
        width=width,
        height=height,
    )


def test_hidden_minimized_notepad_window_is_not_visible_sane():
    hidden = _window("post_4.txt - Notepad", -32000, -32000, 199, 34)
    visible = _window("Untitled - Notepad", 120, 120, 906, 740)

    assert not notepad.is_visible_sane_window(hidden)
    assert notepad.is_visible_sane_window(visible)


def test_visible_notepad_windows_filters_hidden_leftovers(monkeypatch):
    hidden = _window("post_4.txt - Notepad", -32000, -32000, 199, 34)
    visible = _window("Untitled - Notepad", 120, 120, 906, 740)

    monkeypatch.setattr(notepad, "get_notepad_windows", lambda: [hidden, visible])

    assert notepad.get_visible_notepad_windows() == [visible]

