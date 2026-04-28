import logging
from typing import Any

import requests

from vision_desktop_automation.config import API_URL, POST_LIMIT


def fetch_posts() -> list[dict[str, Any]]:
    """Fetch and validate the first POST_LIMIT posts from JSONPlaceholder."""
    logging.info("Fetching posts")

    response = requests.get(API_URL, timeout=10)
    response.raise_for_status()

    posts = response.json()[:POST_LIMIT]

    if not isinstance(posts, list) or not posts:
        raise ValueError("API returned invalid data")

    for post in posts:
        if not all(k in post for k in ("id", "title", "body")):
            raise ValueError("Post missing required keys")

    logging.info(f"Fetched {len(posts)} posts")
    return posts