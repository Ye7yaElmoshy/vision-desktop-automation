PLANNER_PROMPT = """You are a GUI search planner.

Help determine which desktop regions to focus on for the target icon, and name nearby desktop items that support this localization.
Target: {target_description}

Return up to {max_regions} likely search regions for the target desktop icon.

Rules:
- Return ONLY valid JSON.
- No markdown.
- Use fractions from 0.0 to 1.0.
- Keep strings short.
- If the target is clearly visible, return only the best region.
- Every region must include coordinate fields and a score.
- Always include neighbor_reasoning for each region. The neighbors list may be empty if no clear neighbors exist on the desktop, but neighbor_reasoning must always be filled in with a brief explanation of how the target's likely position relates to its surroundings.
- Use optional fields: neighbors and neighbor_reasoning.
- neighbor_reasoning should be a short string (≤30 words) using XML-style tags inside the string only: <element>, <area>, <neighbor>.
- Do not emit XML tags outside JSON or break JSON validity.

JSON format:
{{
  "candidate_regions": [
    {{
      "name": "short_name",
      "reason": "Notepad icon visible in left desktop column",
      "score": 0.95,
      "x1_pct": 0.00,
      "y1_pct": 0.00,
      "x2_pct": 0.25,
      "y2_pct": 1.00,
      "neighbors": ["Recycle Bin", "This PC"],
      "neighbor_reasoning": "The <element>Notepad icon</element> is most likely in the <area>left desktop column</area>, near the <neighbor>Recycle Bin</neighbor>."
    }}
  ]
}}
"""

PLANNER_PROMPT_ROBUST = """You are a GUI search planner.

Help determine which desktop regions to focus on for the target icon, and name nearby desktop items that support this localization.
Target: {target_description}

Return up to {max_regions} likely search regions for the target desktop icon.

Rules:
- Return ONLY valid JSON.
- No markdown.
- Use fractions from 0.0 to 1.0.
- Keep strings short.
- If the target may appear in several plausible desktop zones, return multiple reasonable candidate regions.
- Do not collapse distinct likely locations into a single region unless they overlap heavily.
- Every region must include coordinate fields and a score.
- Always include neighbor_reasoning for each region. The neighbors list may be empty if no clear neighbors exist on the desktop, but neighbor_reasoning must always be filled in with a brief explanation of how the target's likely position relates to its surroundings.
- Use optional fields: neighbors and neighbor_reasoning.
- neighbor_reasoning should be a short string (≤30 words) using XML-style tags inside the string only: <element>, <area>, <neighbor>.
- Do not emit XML tags outside JSON or break JSON validity.

JSON format:
{{
  "candidate_regions": [
    {{
      "name": "short_name",
      "reason": "Notepad icon visible in left desktop column",
      "score": 0.95,
      "x1_pct": 0.00,
      "y1_pct": 0.00,
      "x2_pct": 0.25,
      "y2_pct": 1.00,
      "neighbors": ["Recycle Bin", "This PC"],
      "neighbor_reasoning": "The <element>Notepad icon</element> is most likely in the <area>left desktop column</area>, near the <neighbor>Recycle Bin</neighbor>."
    }}
  ]
}}
"""

GROUNDING_PROMPT = """You are a GUI grounding agent. Locate a specific desktop icon in this screenshot or cropped screenshot.

Target: {target_description}

The desktop may be cluttered and may use either a light or dark Windows desktop theme.
Icons may appear at small, medium, or large Windows desktop icon sizes — judge by
proportion, not absolute pixel size. The screenshot may be a cropped region around
the icon, and the text label may be partially visible. Use BOTH visual appearance
AND any visible label text to identify the correct icon, and prefer the icon whose
label most exactly matches the target name (e.g., 'Notepad' over 'Notepad++' or
'Notes').

Task:
Return up to {max_proposals} possible target boxes, one per matching icon if
multiple visually-similar icons are visible. If there is only one strong match,
return one proposal. Rank proposals by how confidently they match the target.

Respond ONLY with valid JSON. Do not return only found/confidence without
coordinates and click coordinates.
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
- Every returned proposal must include x1_pct, y1_pct, x2_pct, y2_pct, click_x_pct, and click_y_pct.
- Set "proposals": [] if no matching icon is visible in this image.
- Do not include markdown or explanations outside JSON.
"""

CACHE_CHECK_PROMPT = """Look carefully at this small image.

Target: {target_description}

Is there a desktop icon in this image that matches the target?
The icon may appear in light or dark theme.
Respond ONLY with: {{"found": true}} or {{"found": false}}
"""

# Result Checking Prompt adapted from ScreenSeekeR Table 8 (arXiv:2504.07981).
VERIFY_PROMPT = """Look at this small image carefully.

Target: {target_description}

Does this cropped image of a desktop icon show the correct target icon?

Respond ONLY with valid JSON:
{{
  "result": "is_target" | "target_elsewhere" | "target_not_found",
  "new_instruction": null,
  "reason": "brief explanation"
}}

Rules:
- Use exactly one of the three allowed result values.
- When result is "target_elsewhere", new_instruction must be a clearer rewritten instruction.
- Otherwise, set new_instruction to null.
- reason must be a brief string of 20 words or fewer.
- Do not add any fields besides result, new_instruction, and reason.
"""