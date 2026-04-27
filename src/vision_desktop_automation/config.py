import os
from pathlib import Path

# =========================
# PATHS
# =========================
SCREENSHOT_PATH = Path("screen.png")
FAILURE_SCREENSHOT_DIR = Path("failure_screenshots")
LOG_FILE = Path("automation.log")
UNSAVED_NOTE_COUNTER_FILE = Path("unsaved_note_counter.txt")
TEMPLATE_DIR = Path("templates")

DESKTOP_PATH = Path(os.path.expanduser("~")) / "Desktop"
OUTPUT_DIR = DESKTOP_PATH / "tjm-project"

# =========================
# VLM SETTINGS
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
VLM_MODEL = "gemini-flash-latest"

# =========================
# GROUNDING HYPERPARAMETERS
# =========================
MAX_SEARCH_DEPTH = 3
PATCH_THRESHOLD_PX = 400
SEARCH_CONFIDENCE_THRESHOLD = 0.85

# =========================
# PLANNER-GUIDED SEARCH SETTINGS
# =========================
USE_PLANNER_CANDIDATE_REGIONS = True
MAX_CANDIDATE_REGIONS = 5
MIN_REGION_WIDTH_PX = 80
MIN_REGION_HEIGHT_PX = 80

REGION_SCORE_WEIGHT = 0.25
GROUNDING_CONFIDENCE_WEIGHT = 0.55
BOX_SCORE_WEIGHT = 0.20

MAX_GROUNDING_PROPOSALS = 4
REGION_NMS_IOU_THRESHOLD = 0.60
BOX_NMS_IOU_THRESHOLD = 0.50

RECURSIVE_ACCEPT_SCORE = 0.82
RECURSIVE_ACCEPT_CONFIDENCE = 0.86
RECURSIVE_PLANNER_DEPTH = 2

MIN_PLANNER_REGION_SCORE = 0.50

# You said you fixed this, so keep it True.
ALLOW_DIRECT_GROUNDING_FALLBACK = True

SKIP_VERIFICATION_IF_CONFIDENT = True
VERIFICATION_SKIP_CONFIDENCE = 0.97
VERIFICATION_SKIP_REGION_SCORE = 0.90

# =========================
# TEMPLATE MATCHING FALLBACK
# =========================
USE_TEMPLATE_MATCHING_FALLBACK = True
TEMPLATE_MATCH_THRESHOLD = 0.82
TEMPLATE_MATCH_SCALES = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.50]
TEMPLATE_FILE_PATTERNS = ["notepad*.png", "notepad*.jpg", "notepad*.jpeg"]

# =========================
# CACHE SETTINGS
# =========================
CACHE_STRONG_DIFF_THRESHOLD = 50
CACHE_TOLERANT_DIFF_THRESHOLD = 100
USE_VLM_CACHE_CONFIRMATION = False

# =========================
# TARGET
# =========================
TARGET_DESCRIPTION = "Notepad desktop shortcut icon"

# =========================
# API
# =========================
API_URL = "https://jsonplaceholder.typicode.com/posts"
POST_LIMIT = 10

# =========================
# TIMING
# =========================
DESKTOP_WAIT = 1.0
UI_RESET_WAIT = 0.3
NOTEPAD_OPEN_WAIT_MAX = 12
PASTE_WAIT = 0.8
SAVE_DIALOG_WAIT = 3.0
POST_ENTER_WAIT = 1.0
AFTER_SAVE_WAIT = 1.0
AFTER_CLOSE_WAIT = 1.5

# =========================
# RETRIES
# =========================
ICON_DETECTION_RETRIES = 3
SAVE_RETRIES = 2
API_RETRIES = 3