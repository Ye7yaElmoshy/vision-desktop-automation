import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests
from PIL import Image

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from vision_desktop_automation import vlm_client


def _response(status_code: int, body: bytes, reason: str = "") -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = body
    response.reason = reason
    return response


def test_call_gemini_vision_fails_fast_with_google_error_body(monkeypatch):
    response = _response(
        403,
        b'{"error":{"code":403,"message":"API key not authorized","status":"PERMISSION_DENIED"}}',
        reason="Forbidden",
    )
    post = Mock(return_value=response)

    monkeypatch.setattr(vlm_client, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(vlm_client, "API_RETRIES", 3)
    monkeypatch.setattr(vlm_client.requests, "post", post)
    monkeypatch.setattr(vlm_client.time, "sleep", Mock())

    with pytest.raises(RuntimeError) as exc_info:
        vlm_client.call_gemini_vision("find the icon", Image.new("RGB", (8, 8)))

    message = str(exc_info.value)
    assert "HTTP 403" in message
    assert "PERMISSION_DENIED: API key not authorized" in message
    assert post.call_count == 1
    vlm_client.time.sleep.assert_not_called()

