# Refactor Notes

## Purpose

This project started as a single-file Python automation script. After the initial implementation was working, the codebase was refactored into a clearer module structure to make it easier to review, maintain, test, and explain during the interview.

The main goal of the refactor was to separate responsibilities while preserving the original automation behavior.

---

## High-Level Automation Flow

 1. Fetch the first 10 posts from JSONPlaceholder.
 2. Show the Windows desktop.
 3. Locate the Notepad desktop shortcut using visual grounding.
 4. Launch Notepad.
 5. Paste one post into Notepad.
 6. Save the file as `post_{id}.txt` inside `Desktop/tjm-project`.
 7. Close Notepad.
 8. Repeat the process for all 10 posts.
 9. Verify that all expected output files were created correctly.

---

## Module Structure

```
src/
└── vision_desktop_automation/
    ├── __init__.py
    ├── main.py
    ├── config.py
    ├── prompts.py
    ├── api.py
    ├── files.py
    ├── geometry.py
    ├── vlm_client.py
    ├── grounding.py
    ├── template_matching.py
    ├── desktop.py
    └── notepad.py
```

---

## Module Responsibilities

### `main.py`

The application entrypoint. Intentionally small — it only handles:

- Logging setup
- Runtime directory setup
- Environment validation
- Fetching posts
- Looping over posts
- Calling the Notepad automation workflow
- Final output verification

It does not contain grounding logic, the Notepad workflow, API logic, or filesystem helper implementations.

---

### `config.py`

Central location for constants and runtime configuration. Includes:

- Paths
- Gemini model configuration
- Grounding thresholds
- Retry counts
- Timing values
- Target description
- Template matching settings
- Cache validation thresholds

The Gemini API key is read from the environment:

```python
os.getenv("GEMINI_API_KEY")
```

It is **not** hardcoded.

---

### `prompts.py`

Contains all VLM prompt templates. Keeps long prompt strings out of the main logic and makes it easier to update planner, grounder, cache-check, and verifier prompts independently.

---

### `api.py`

Handles fetching posts from JSONPlaceholder. Validates that each post contains the required fields:

- `id`
- `title`
- `body`

---

### `files.py`

Handles filesystem and logging-related helpers. Responsibilities:

- Creating runtime directories
- Setting up logging
- Saving debug screenshots
- Saving annotated screenshots
- Managing unsaved note numbering
- Verifying output files

---

### `geometry.py`

Contains coordinate and box helpers. Responsibilities:

- Percentage normalization
- Expanding tight planner regions
- Box area calculation
- Intersection-over-union calculation
- Human-readable screen region description

---

### `vlm_client.py`

Handles Gemini Vision API communication. Responsibilities:

- Converting screenshots to base64
- Calling Gemini Vision
- Retrying on temporary Gemini/API failures
- Parsing JSON responses
- Recovering partial planner JSON when possible

---

### `grounding.py`

Contains the planner-guided VLM grounding logic. Responsibilities:

- Asking the planner for likely candidate regions
- Expanding small planner regions
- Non-maximum suppression for overlapping regions
- Asking the grounder for icon proposals
- Parsing grounding proposals
- Verifying icon identity
- Recursive region search
- Optional direct full-screen grounding fallback

The main method exposed to the rest of the app:

```python
planner_guided_ground_icon(...)
```

---

### `template_matching.py`

Contains the local OpenCV fallback for template matching. This is **not** the primary method — it is used for graceful degradation if the VLM path fails due to:

- Gemini 503 errors
- Gemini timeout
- Invalid VLM JSON
- Temporary API instability

The fallback uses grayscale multi-scale template matching against images inside the `templates/` directory.

---

### `desktop.py`

Owns desktop-related helpers and icon cache state. Responsibilities:

- Clearing/minimizing windows to show the desktop
- Reading the active window title
- Calculating DPI scale
- Capturing icon crops
- Validating cached icon position
- Updating icon cache
- Invalidating icon cache
- Tracking cache hits

> **Design note:** `desktop.py` owns `_icon_cache`, and the rest of the code accesses it through helper functions. This avoids direct cache mutation inside the Notepad workflow.

---

### `notepad.py`

Contains the Notepad automation workflow. Responsibilities:

- Locating and opening Notepad
- Handling leftover Notepad windows
- Focusing Notepad
- Pasting post content
- Opening the Save dialog
- Saving files
- Handling overwrite dialogs
- Closing Notepad
- Processing one post end-to-end

The main method exposed to `main.py`:

```python
process_post(post)
```

---

## Primary Detection Method

The primary detection method is **planner-guided VLM grounding**. The flow:

1. Take a screenshot.
2. Ask the planner to suggest candidate regions.
3. Crop the best candidate regions.
4. Ask the grounder to locate the target inside the crop.
5. Verify the detected icon if confidence is not high enough.
6. Return the final click coordinates.

This approach is more flexible than hardcoded coordinates or pure template matching because it can reason about both icon appearance and label text.

---

## Fallback Detection Method

The local fallback is **OpenCV template matching**. It is used only if the VLM-based grounding path fails, providing graceful degradation when the external VLM service is temporarily unavailable.

The template fallback is less general than VLM grounding because it depends on example template images, but it is fast and local.

---

## Icon Cache

After the icon is found once, the coordinates and a reference crop are cached. For later posts, the system first checks whether the cached icon location still appears valid. If valid, it skips the VLM call entirely — improving speed and reducing API usage. If invalid, the system falls back to visual grounding.

---

## Error Handling

The project handles several failure cases:

| Failure | Handling |
|---|---|
| Gemini temporary failures | Retries with backoff |
| Invalid VLM JSON | Partial JSON recovery |
| VLM grounding failure | Template matching fallback |
| Notepad launch failure | Window title validation |
| Wrong app launch | Unexpected-window dismissal |
| Save dialog failure | Retries |
| Existing files | Overwrite confirmation |
| Leftover unsaved Notepad windows | Safe Save As handling |
| Missing or wrong output files | Final verification |

---

## Why This Refactor Helps

- `main.py` is now small and easy to read.
- Each module has a clear, single responsibility.
- Visual grounding code is isolated from Notepad-specific automation.
- The Notepad workflow is easier to debug independently.
- Fallback logic is separated from the primary VLM grounding logic.
- Configuration values are centralized.
- The project is easier to explain during an interview.

---

## What Was Intentionally Preserved

The refactor preserves the original behavior exactly:

- Same automation workflow
- Same output directory
- Same file naming format (`post_{id}.txt`)
- Same VLM grounding approach
- Same template matching fallback
- Same retry logic
- Same final output verification

The only intentional redesign was moving icon cache ownership into `desktop.py` instead of allowing `main.py` or `notepad.py` to directly mutate `_icon_cache`.

---

## Future Improvements

- Add unit tests for pure helper functions in `geometry.py`.
- Add a dry-run mode that performs detection without clicking.
- Add a CLI interface for changing the target icon description.
- Add better template management and template-generation instructions.
- Add structured logging output in JSON format.
- Add support for arbitrary app/icon targets beyond Notepad.
- Add screenshot-based regression tests for top-left, center, and bottom-right icon positions.
- Add a small report generator that collects annotated screenshots and logs.
- Add optional OCR-based label validation as another fallback.
- Add configuration profiles for different screen resolutions and icon sizes.

---

## Testing Checklist

Before running the automation, verify imports:

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

Run the automation only when ready:

```powershell
python -m vision_desktop_automation.main
```

---

## Notes for Interview Discussion

Key points to mention:

- The **primary method is VLM-based grounding**, not template matching.
- Template matching exists only as graceful degradation.
- The approach is intentionally overkill for Notepad — the goal is testing general visual grounding.
- The code avoids hardcoded desktop coordinates.
- The cache reduces repeated VLM calls after the first successful detection.
- The modular structure makes the project easier to extend to other icons or GUI targets.
---
