PLANNER_PROMPT = """You are a GUI search planner.

Target: {target_description}

Return likely search regions for the target desktop icon.

Rules:
- Return ONLY valid JSON.
- No markdown.
- Use fractions from 0.0 to 1.0.
- Keep strings short.
- If the target is clearly visible, return only the best region.
- Do not select Notepad++, Notes, WordPad, VS Code, browsers, or text documents.

JSON format:
{{
  "candidate_regions": [
    {{
      "name": "short_name",
      "reason": "max 8 words",
      "score": 0.95,
      "x1_pct": 0.00,
      "y1_pct": 0.00,
      "x2_pct": 0.25,
      "y2_pct": 1.00
    }}
  ]
}}
"""

GROUNDING_PROMPT = """You are a GUI grounding agent. Locate a specific desktop icon in this screenshot or cropped screenshot.

Target: {target_description}

This desktop may be cluttered and may use either light or dark theme.
Use BOTH visual appearance AND label text to identify the correct icon.

For Windows Notepad specifically:
- LABEL TEXT: "Notepad" or "notepad" exactly or very close.
- ICON APPEARANCE: Simple document/notepad shape with horizontal lines. Color may vary by theme.
- Do NOT select: Notepad++, Notes, WordPad, VS Code, browser icons, or any rich text editor.

Task:
Return up to {max_proposals} possible target boxes. If there is only one strong match, return one proposal.

Respond ONLY with valid JSON:
{{
  "proposals": [
    {{
      "found": true,
      "confidence": 0.95,
      "label_match": 0.95,
      "visual_match": 0.90,
      "x1_pct": 0.38,
      "y1_pct": 0.70,
      "x2_pct": 0.47,
      "y2_pct": 0.84,
      "click_x_pct": 0.425,
      "click_y_pct": 0.77,
      "reason": "brief reason"
    }}
  ]
}}

Rules:
- Percent values must be fractions from 0.0 to 1.0, not percentages like 57.3.
- The box must tightly cover the icon and its label if visible.
- click_x_pct and click_y_pct must be the center of the actual icon, not the whole label.
- Set "proposals": [] if no matching icon is visible in this image.
- Do not include markdown or explanations outside JSON.
"""

CACHE_CHECK_PROMPT = """Look carefully at this small image.
Is there a desktop icon whose TEXT LABEL says "Notepad" or "notepad"?
The icon may appear in light or dark theme.
Do not confuse it with Notepad++, Notes, WordPad, or similar names.
Respond ONLY with: {"found": true} or {"found": false}
"""

VERIFY_PROMPT = """Look at this icon carefully.

Is this the standard Windows Notepad application icon?
- It should be a simple document/notepad shape with horizontal lines.
- It may appear in light or dark theme.
- It is NOT Notepad++.
- It is NOT WordPad, Notes, VS Code, or another app.

Respond ONLY with:
{"correct": true}
or
{"correct": false, "reason": "brief explanation"}
"""