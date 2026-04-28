# Vision-Based Desktop Automation with Dynamic Icon Grounding

Python automation project for visually locating the Windows Notepad desktop shortcut, launching it, writing content from an API, and saving the results as text files.

The project uses a planner-guided Vision Language Model grounding flow inspired by ScreenSeekeR-style visual search. The goal is not just to open Notepad, but to demonstrate a flexible approach for locating GUI targets dynamically without relying on fixed screen coordinates.

---

## Project Objective

The application dynamically locates the Notepad desktop shortcut on Windows and automates the following workflow:

1. Fetch the first 10 posts from JSONPlaceholder.
2. Show the Windows desktop.
3. Visually locate the Notepad desktop shortcut.
4. Double-click Notepad to launch it.
5. Paste one post into Notepad using this format:

```
Title: {title}

{body}
```

6. Save the file as `post_{id}.txt` inside `Desktop/tjm-project`.
7. Close Notepad.
8. Repeat the process for all 10 posts.
9. Verify that all expected files were created successfully.

---

## Key Features

- Dynamic desktop icon grounding
- No hardcoded Notepad coordinates
- Planner-guided VLM region search
- Grounder-based precise icon localization
- Optional icon verification
- Template matching fallback using OpenCV
- Icon cache for faster repeated launches
- Annotated screenshots for reporting
- Output verification after execution
- Modular Python project structure
- Gemini API key loaded from environment variables

---

## Project Structure

```
vision-desktop-automation/
│
├── src/
│   └── vision_desktop_automation/
│       ├── __init__.py
│       ├── main.py
│       ├── config.py
│       ├── prompts.py
│       ├── api.py
│       ├── files.py
│       ├── geometry.py
│       ├── vlm_client.py
│       ├── grounding.py
│       ├── template_matching.py
│       ├── desktop.py
│       └── notepad.py
│
├── tests/
│   ├── __init__.py
│   ├── test_geometry.py
│   ├── test_api.py
│   └── test_files.py
│
├── templates/               ← OpenCV template images
│
├── docs/
│   ├── screenshots/
│   └── REFACTOR_NOTES.md
│
├── logs/                    ← runtime logs (gitignored)
├── output/                  ← post_{id}.txt files (gitignored)
├── failure_screenshots/     ← debug and annotated screenshots
│
├── .env.example             ← GEMINI_API_KEY=your_key_here
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Architecture

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `main.py` | Application entrypoint and orchestration |
| `config.py` | Runtime constants and configuration |
| `prompts.py` | VLM planner, grounder, verifier, and cache prompts |
| `api.py` | Fetching and validating JSONPlaceholder posts |
| `files.py` | Logging, screenshots, output verification, runtime folders |
| `geometry.py` | Coordinate normalization, bounding boxes, IoU, region helpers |
| `vlm_client.py` | Gemini Vision API calls, retries, JSON parsing |
| `grounding.py` | Planner-guided VLM grounding logic |
| `template_matching.py` | OpenCV template matching fallback |
| `desktop.py` | Desktop visibility, DPI scale, active window, icon cache |
| `notepad.py` | Notepad launch, paste, save, close, and post workflow |

---

## Primary Detection Method

The primary method is **planner-guided VLM grounding**. Flow:

1. Take a screenshot of the desktop.
2. Ask the planner to propose likely candidate regions.
3. Crop candidate regions.
4. Ask the grounder to locate the target icon inside the crop.
5. Verify the detected icon if needed.
6. Return click coordinates.
7. Launch Notepad.

This is intentionally more general than hardcoded coordinates or image-only template matching.

---

## Fallback Detection Method

The fallback method is **OpenCV template matching**. It is used only when the VLM path fails due to:

- Gemini 503 errors
- Gemini timeout
- Invalid or unusable VLM JSON
- Temporary API instability

Template matching uses images stored inside the `templates/` directory. Expected filenames:

```
templates/notepad*.png
templates/notepad*.jpg
templates/notepad*.jpeg
```

Template matching is less general than VLM grounding, but provides graceful degradation when the external VLM service is unavailable.

---

## Icon Cache

After the icon is successfully detected once, the system stores:

- Screen coordinates
- A small reference crop
- Cache hit count

For later posts, the system checks whether the icon is still at the cached location. If valid, the automation skips the expensive VLM call and opens Notepad directly — improving speed and reducing repeated Gemini API calls.

---

## Requirements

**Target environment:**

- Windows 10/11
- 1920×1080 resolution recommended
- Python 3.10+
- Notepad shortcut visible on the desktop
- Gemini API key set as an environment variable

**Python packages:**

```
opencv-python
numpy
pyautogui
pyperclip
requests
pillow
pygetwindow
```

---

## Setup

### 1. Clone the repository

```powershell
git clone https://github.com/Ye7yaElmoshy/vision-desktop-automation.git
cd vision-desktop-automation
```

### 2. Create and activate a virtual environment

Using regular Python:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Or using `uv`:

```powershell
uv venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -e .
```

If needed, install manually:

```powershell
python -m pip install opencv-python numpy pyautogui pyperclip requests pillow pygetwindow
```

---

## Gemini API Key

The API key must **not** be hardcoded. Set it in PowerShell:

```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

For a permanent user-level environment variable:

```powershell
setx GEMINI_API_KEY "your-api-key-here"
```

After using `setx`, close and reopen the terminal.

---

## Running the Project

Make sure the Notepad shortcut is visible on the desktop, then run:

```powershell
$env:PYTHONPATH="src"
python -m vision_desktop_automation.main
```

> Do not touch the mouse or keyboard while the automation is running.

---

## Output

Generated files are saved to `Desktop/tjm-project`:

```
post_1.txt
post_2.txt
post_3.txt
post_4.txt
post_5.txt
post_6.txt
post_7.txt
post_8.txt
post_9.txt
post_10.txt
```

Each file contains:

```
Title: {post title}

{post body}
```

---

## Screenshots

Debug and annotated screenshots are saved to `failure_screenshots/`. Annotated screenshots show the detected Notepad icon coordinates. Exact generated filenames may include timestamps depending on the run.

---

## Detection Examples

The following screenshots are stored in `docs/screenshots/`.

### Top-left annotated detection
![Top-left annotated detection](docs/screenshots/Top%20left%20ANNOTATED%20screenshot.jpeg)

### Top-left popup confirmation
![Top-left popup confirmation](docs/screenshots/Top%20left%20POP-UP%20screenshot.jpeg)

### Center annotated detection
![Center annotated detection](docs/screenshots/Center%20ANNOTATED%20screenshot.jpeg)

### Center popup confirmation
![Center popup confirmation](docs/screenshots/Center%20POP-UP%20screenshot.jpeg)

### Bottom-left annotated detection 1
![Bottom-left annotated detection 1](docs/screenshots/Bottom%20left%201%20ANNOTATED%20screenshot.jpeg)

### Bottom-left annotated detection 2
![Bottom-left annotated detection 2](docs/screenshots/Bottom%20left%202%20ANNOTATED%20screenshot.jpeg)

### Bottom-right popup confirmation
![Bottom-right popup confirmation](docs/screenshots/Bottom%20Right%20POP-UP%20screenshot.jpeg)

---

## Testing / Verification

Verify imports before running:

```powershell
$env:PYTHONPATH="src"
python -c "import vision_desktop_automation.main; print('Import OK')"
```

Compile all modules:

```powershell
python -m py_compile src\vision_desktop_automation\main.py
python -m py_compile src\vision_desktop_automation\api.py
python -m py_compile src\vision_desktop_automation\config.py
python -m py_compile src\vision_desktop_automation\desktop.py
python -m py_compile src\vision_desktop_automation\files.py
python -m py_compile src\vision_desktop_automation\geometry.py
python -m py_compile src\vision_desktop_automation\grounding.py
python -m py_compile src\vision_desktop_automation\notepad.py
python -m py_compile src\vision_desktop_automation\prompts.py
python -m py_compile src\vision_desktop_automation\template_matching.py
python -m py_compile src\vision_desktop_automation\vlm_client.py
```

Expected result: no output from `py_compile`, and `Import OK` from the import test.

---

## Error Handling

The project handles the following failure cases:

| Failure | Handling |
|---|---|
| Gemini temporary failures | Retries with backoff |
| Gemini timeout | Retries with backoff |
| Invalid VLM JSON | Partial JSON recovery |
| VLM grounding failure | Template matching fallback |
| Template fallback failure | Logged, skips post |
| Notepad launch failure | Window title validation |
| Wrong app opening | Unexpected-window dismissal |
| Save dialog issues | Retries |
| Overwrite confirmation | Auto-confirmed |
| Leftover unsaved Notepad windows | Safe Save As handling |
| Missing or empty output files | Final verification |

---

## Why This Approach

Hardcoded coordinates are fragile — they fail if the icon moves. Pure template matching is faster but less general and depends on having a reference image.

The **planner-guided VLM approach** is more flexible because it can reason about icon appearance, label text, screen layout, similar-looking icons, different desktop positions, and different themes. This makes the solution closer to a scalable GUI grounding system rather than a Notepad-only shortcut script.

---

## Known Limitations

- Requires a visible Notepad desktop shortcut.
- Requires a Gemini API key for the primary VLM grounding method.
- Template matching fallback requires template images in `templates/`.
- Automation depends on Windows UI timing.
- Busy desktops may reduce grounding reliability.
- Different languages or renamed shortcuts may require prompt/config updates.
- Primarily tested on Windows at 1920×1080.

---

## Future Improvements

- Add a dry-run mode that detects the icon without clicking.
- Add CLI arguments for target description, post limit, and output path.
- Add unit tests for `geometry.py`.
- Add screenshot regression tests for different icon positions.
- Add support for arbitrary app/icon targets.
- Add OCR-based label validation as another fallback.
- Add structured JSON logs.
- Add a report generator for annotated screenshots.
- Add better template creation instructions.
- Add configuration profiles for different resolutions and icon sizes.
