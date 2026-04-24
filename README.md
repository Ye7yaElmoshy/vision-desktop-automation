# Vision-Based Desktop Automation with Dynamic Icon Grounding

## Objective

This project automates Windows Notepad using vision-based desktop grounding.

The system dynamically locates the Notepad desktop shortcut regardless of its position on the Windows desktop, opens Notepad, writes posts fetched from the JSONPlaceholder API, saves each post as a text file, closes Notepad, and repeats the process.

The assignment intentionally uses a vision-heavy approach even though Notepad could be launched more simply. The goal is to demonstrate scalable GUI grounding, not task-specific shortcuts.

---

## Features

- Planner-guided VLM grounding using Gemini Vision.
- Candidate-region proposal from full desktop screenshots.
- Grounder refinement inside cropped candidate regions.
- Dynamic icon detection regardless of icon position.
- Handles cluttered desktops with many icons.
- Handles different icon placements such as top-left, center, bottom-left, and bottom-right.
- Expands tight planner regions to preserve icon and label context.
- Uses cached coordinates after the first successful detection to reduce repeated VLM calls.
- Locally validates cached coordinates using pixel similarity.
- Direct full-screen VLM fallback.
- Optional OpenCV template-matching fallback.
- Saves annotated screenshots of detections.
- Verifies all generated output files.

---

## Requirements

- Windows 10/11.
- 1920x1080 resolution recommended.
- Python 3.10+.
- A visible Notepad shortcut on the desktop.
- A Gemini API key stored as an environment variable.

---

## Setup

Install `uv`:

```powershell
pip install uv
```

Install project dependencies:

```powershell
uv sync
```

Set your Gemini API key in PowerShell:

```powershell
$env:GEMINI_API_KEY="your_real_gemini_api_key_here"
```

Run the automation:

```powershell
uv run vision-automation
```

Alternative run command:

```powershell
uv run python -m vision_desktop_automation.main
```

---

## Automation Workflow

1. Show the desktop.
2. Capture a fresh screenshot.
3. Locate the Notepad desktop shortcut using planner-guided VLM grounding.
4. Double-click the detected icon coordinates.
5. Validate that Notepad launched.
6. Fetch the first 10 posts from JSONPlaceholder.
7. Paste each post into Notepad using this format:

```text
Title: {title}

{body}
```

8. Save each post as `post_{id}.txt`.
9. Store the files inside `Desktop/tjm-project`.
10. Close Notepad.
11. Repeat the process for all 10 posts.
12. Verify that all generated files exist and contain the expected title.

---

## Grounding Approach

The primary grounding method is inspired by ScreenSeekeR-style cascaded visual search.

The system separates the visual search process into two main roles: a planner and a grounder.

### 1. Planner

The planner receives the full desktop screenshot and proposes candidate regions where the Notepad icon is likely to exist.

Instead of asking the model to directly click from the full screenshot, the planner narrows the search area first.

### 2. Grounder

The grounder receives a cropped candidate region and returns:

- A bounding box around the likely target.
- A click point at the icon center.
- Confidence values.
- Label and visual match scores.

This makes the system more flexible than hardcoded coordinates or simple template matching.

---

## Why Planner-Guided Grounding?

Hardcoded coordinates fail when the icon moves.

Template matching can fail when:

- The icon size changes.
- The Windows theme changes.
- The desktop background is busy.
- The shortcut label changes.
- The icon is slightly different across Windows versions.

The VLM-based planner-grounder approach is more general because it can reason about both:

- The icon appearance.
- The text label.

This makes the solution closer to a general GUI grounding system rather than a task-specific Notepad launcher.

---

## Fallback Strategy

The fallback hierarchy is:

1. Cached coordinates if locally valid.
2. Planner-guided VLM grounding.
3. Direct full-screen VLM grounding.
4. Template matching fallback if local templates are available.
5. Clean failure with diagnostic screenshot and logs.

Template matching is included only as graceful degradation. The primary method remains planner-guided VLM grounding.

---

## Cache Strategy

After the first successful detection, the system stores the Notepad coordinates and a small reference crop around the icon.

For later posts, the system checks whether the icon is still at the cached location using local pixel similarity.

If the cache is valid, the system reuses the stored coordinates instead of calling Gemini again. This reduces:

- API calls.
- Runtime.
- Risk of Gemini 503 errors.
- Cost and latency.

If the cache is invalid, the system reruns visual grounding.

---

## Error Handling

The system includes handling for:

- Icon not found.
- Gemini API timeout or 503 errors.
- Invalid or partial JSON returned by the VLM.
- Save dialog delays.
- Existing output files.
- Notepad launch validation.
- Leftover Notepad windows.
- Failed saves.
- Missing or empty output files.

The script also saves debug screenshots in `failure_screenshots/` when failures occur.

---

## Detection Examples

The following screenshots are stored in `docs/screenshots/`.

### Top-left annotated detection

![Top-left annotated detection](docs/screenshots/Top%20left%20ANNOTATED%20screenshot.png)

### Top-left popup confirmation

![Top-left popup confirmation](docs/screenshots/Top%20left%20POP-UP%20screenshot.png)

### Center annotated detection

![Center annotated detection](docs/screenshots/Center%20ANNOTATED%20screenshot.png)

### Center popup confirmation

![Center popup confirmation](docs/screenshots/Center%20POP-UP%20screenshot.png)

### Bottom-left annotated detection 1

![Bottom-left annotated detection 1](docs/screenshots/Bottom%20left%201%20ANNOTATED%20screenshot.png)

### Bottom-left annotated detection 2

![Bottom-left annotated detection 2](docs/screenshots/Bottom%20left%202%20ANNOTATED%20screenshot.png)

### Bottom-right popup confirmation

![Bottom-right popup confirmation](docs/screenshots/Bottom%20Right%20POP-UP%20screenshot.png)

---

## Output

Generated files are saved to:

```text
Desktop/tjm-project/
```

Expected output:

```text
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

```text
Title: {title}

{body}
```

---

## Project Structure

```text
vision-desktop-automation/
â
âââ README.md
âââ pyproject.toml
âââ uv.lock
âââ .gitignore
âââ .env.example
â
âââ src/
â   âââ vision_desktop_automation/
â       âââ __init__.py
â       âââ main.py
â
âââ docs/
â   âââ screenshots/
â       âââ Bottom left 1 ANNOTATED screenshot.png
â       âââ Bottom left 2 ANNOTATED screenshot.png
â       âââ Bottom Right POP-UP screenshot.png
â       âââ Center ANNOTATED screenshot.png
â       âââ Center POP-UP screenshot.png
â       âââ Top left ANNOTATED screenshot.png
â       âââ Top left POP-UP screenshot.png
â
âââ templates/
â   âââ .gitkeep
â
âââ failure_screenshots/
    âââ .gitkeep
```

---

## Configuration

The Gemini API key is not hardcoded in the source code.

The code reads it from the environment:

```python
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
```

Example PowerShell setup:

```powershell
$env:GEMINI_API_KEY="your_real_gemini_api_key_here"
```

A `.env.example` file is included only as a safe placeholder:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## Known Limitations

- First-time detection depends on Gemini API availability.
- Gemini may occasionally return `503` or timeout.
- Template matching fallback requires local template images in the `templates/` folder.
- The project is designed for Windows desktop automation.
- The project was mainly tested on 1920x1080 resolution.
- Very cluttered desktops may increase grounding latency.
- If the icon is fully hidden behind another window, the system cannot visually ground it until the desktop is visible.

---

## Future Improvements

- Split the script into separate modules:
  - `grounding.py`
  - `automation.py`
  - `api.py`
  - `utils.py`
- Add CLI arguments for arbitrary target descriptions.
- Add support for grounding any desktop icon or GUI button.
- Add unit tests for JSON parsing and coordinate conversion.
- Improve screenshot annotation placement near screen edges.
- Add automatic screenshot naming by detected region.
- Add more robust local fallback methods for API outages.

---

## Further Notes

This project uses planner-guided VLM grounding because the assignment focuses on scalable visual grounding, not simply opening Notepad.

The system can be extended to other desktop icons or GUI elements by changing the target description. The planner-grounder design is intentionally more general than hardcoded coordinates or template matching.

Template matching is included as a backup, but it is not the primary method because it is less flexible across different themes, icon sizes, and Windows versions.