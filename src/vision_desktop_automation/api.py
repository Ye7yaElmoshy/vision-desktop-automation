import logging
import time
from typing import Any

import requests

from vision_desktop_automation.config import API_RETRIES, API_URL
import vision_desktop_automation.config as cfg


def fetch_posts() -> list[dict[str, Any]]:
    """Fetch and validate the first POST_LIMIT posts from JSONPlaceholder.

    Reads cfg.POST_LIMIT dynamically so CLI overrides applied in main.py
    are respected.
    """
    logging.info("Fetching posts")

    last_error: Any = None

    for attempt in range(1, API_RETRIES + 1):
        try:
            response = requests.get(API_URL, timeout=10)
            response.raise_for_status()

            posts = response.json()[:cfg.POST_LIMIT]

            if not isinstance(posts, list) or not posts:
                raise ValueError("API returned invalid data")

            for post in posts:
                if not all(k in post for k in ("id", "title", "body")):
                    raise ValueError("Post missing required keys")

            logging.info(f"Fetched {len(posts)} posts")
            return posts

        except Exception as e:
            last_error = e
            logging.warning(f"API fetch attempt {attempt} failed: {e}")
            if attempt < API_RETRIES:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to fetch posts after {API_RETRIES} attempts. Last: {last_error}")