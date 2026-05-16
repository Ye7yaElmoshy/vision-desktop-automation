"""
Unit tests for grounding.py.

Two groups:
  1. Pure helpers (no network): parse_grounding_proposals, nms_proposals, nms_regions.
  2. Higher-level pipeline functions (propose_candidate_regions, vlm_ground_icon,
     search_region_recursive, planner_guided_ground_icon) tested via mocks of
     call_gemini_vision / parse_vlm_json so no real API calls are made.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from vision_desktop_automation.config import (
    BOX_SCORE_WEIGHT,
    DISAMBIGUATION_MARGIN,
    GROUNDING_CONFIDENCE_WEIGHT,
    MIN_PLANNER_REGION_SCORE,
    MIN_REGION_HEIGHT_PX,
    MIN_REGION_WIDTH_PX,
    REGION_SCORE_WEIGHT,
    SEARCH_CONFIDENCE_THRESHOLD,
)
from vision_desktop_automation.grounding import (
    disambiguate_proposals,
    nms_proposals,
    nms_regions,
    parse_grounding_proposals,
    planner_guided_ground_icon,
    propose_candidate_regions,
    search_region_recursive,
    vlm_ground_icon,
)


@pytest.fixture
def fake_screenshot():
    """A blank 1920x1080 RGB image — stand-in for a real desktop screenshot."""
    return Image.new("RGB", (1920, 1080))


def test_parse_grounding_proposals_picks_high_confidence_match():
    """A clean grounder response should round-trip through parsing with the
    expected confidence and click coordinates."""
    data = {
        "proposals": [
            {
                "found": True,
                "confidence": 0.92,
                "label_match": 0.95,
                "visual_match": 0.88,
                "x1_pct": 0.10,
                "y1_pct": 0.20,
                "x2_pct": 0.20,
                "y2_pct": 0.30,
                "click_x_pct": 0.15,
                "click_y_pct": 0.25,
            }
        ]
    }
    parsed = parse_grounding_proposals(data)
    assert len(parsed) == 1
    p = parsed[0]
    assert p["confidence"] == pytest.approx(0.92)
    assert p["click_x_pct"] == pytest.approx(0.15)
    assert p["click_y_pct"] == pytest.approx(0.25)
    # proposal_score = 0.50*0.92 + 0.30*0.95 + 0.20*0.88 = 0.921
    assert p["proposal_score"] == pytest.approx(0.921, abs=1e-3)


def test_parse_grounding_proposals_rejects_invalid_clicks():
    """Proposals with (0,0) clicks, sub-edge clicks, or found=false must be dropped —
    these are common VLM hallucination / no-match failure modes."""
    data = {
        "proposals": [
            {  # (0,0) hallucination — rejected by the (0,0) guard
                "found": True,
                "confidence": 0.9,
                "click_x_pct": 0.0,
                "click_y_pct": 0.0,
                "x1_pct": 0.0,
                "y1_pct": 0.0,
                "x2_pct": 0.1,
                "y2_pct": 0.1,
            },
            {  # click below the 0.01 edge guard — rejected
                "found": True,
                "confidence": 0.8,
                "click_x_pct": 0.005,
                "click_y_pct": 0.5,
                "x1_pct": 0.0,
                "y1_pct": 0.4,
                "x2_pct": 0.02,
                "y2_pct": 0.6,
            },
            {  # explicitly not found — rejected
                "found": False,
                "confidence": 0.7,
            },
        ]
    }
    parsed = parse_grounding_proposals(data)
    assert parsed == []


def test_parse_grounding_proposals_synthesizes_box_from_click_only():
    """When the model returns only click coordinates without a box, parse_grounding_proposals
    should synthesize a small box around the click — keeps NMS working when the
    model is terse."""
    data = {
        "proposals": [
            {
                "found": True,
                "confidence": 0.85,
                "click_x_pct": 0.5,
                "click_y_pct": 0.5,
                # no x1/y1/x2/y2
            }
        ]
    }
    parsed = parse_grounding_proposals(data)
    assert len(parsed) == 1
    box = parsed[0]["box"]
    assert box["x1"] < 0.5 < box["x2"]
    assert box["y1"] < 0.5 < box["y2"]


def test_nms_proposals_dedupes_overlapping_boxes_keeping_best():
    """When multiple proposals point at essentially the same icon (high IOU),
    NMS must keep the highest proposal_score and discard the rest."""
    proposals = [
        {
            "proposal_score": 0.9,
            "box": {"x1": 0.10, "y1": 0.10, "x2": 0.20, "y2": 0.20},
        },
        {
            "proposal_score": 0.7,  # heavily overlapping with the above → drop
            "box": {"x1": 0.11, "y1": 0.11, "x2": 0.21, "y2": 0.21},
        },
        {
            "proposal_score": 0.6,  # disjoint → keep
            "box": {"x1": 0.80, "y1": 0.80, "x2": 0.90, "y2": 0.90},
        },
    ]
    kept = nms_proposals(proposals)
    assert len(kept) == 2
    scores = sorted(p["proposal_score"] for p in kept)
    assert scores == [0.6, 0.9]


def test_nms_regions_keeps_highest_scoring_overlap_among_planner_regions():
    """Planner regions are deduplicated by NMS as well — if the planner returns
    multiple overlapping search regions, only the highest-scoring one survives.
    This is what lets the pipeline correctly handle multiple candidate icons."""
    regions = [
        {
            "name": "top_left_a",
            "score": 0.95,
            "x1_pct": 0.0, "y1_pct": 0.0, "x2_pct": 0.3, "y2_pct": 0.3,
        },
        {
            "name": "top_left_b",  # nearly identical to top_left_a → drop
            "score": 0.80,
            "x1_pct": 0.02, "y1_pct": 0.02, "x2_pct": 0.32, "y2_pct": 0.32,
        },
        {
            "name": "bottom_right",  # disjoint → keep
            "score": 0.70,
            "x1_pct": 0.7, "y1_pct": 0.7, "x2_pct": 1.0, "y2_pct": 1.0,
        },
    ]
    kept = nms_regions(regions)
    names = {r["name"] for r in kept}
    assert names == {"top_left_a", "bottom_right"}


# =========================================================================
# Mocked higher-level pipeline tests
# =========================================================================


def _planner_response(regions):
    """Helper: build a parsed-VLM dict with the given candidate_regions."""
    return {"candidate_regions": regions}


def _grounder_response(
    confidence=0.95,
    label_match=0.95,
    visual_match=0.95,
    x1=0.4, y1=0.4, x2=0.6, y2=0.6,
    click_x=0.5, click_y=0.5,
):
    """Helper: build a parsed-VLM dict with a single grounding proposal."""
    return {
        "proposals": [
            {
                "found": True,
                "confidence": confidence,
                "label_match": label_match,
                "visual_match": visual_match,
                "x1_pct": x1, "y1_pct": y1, "x2_pct": x2, "y2_pct": y2,
                "click_x_pct": click_x, "click_y_pct": click_y,
            }
        ]
    }


def test_propose_candidate_regions_parses_valid_json(fake_screenshot):
    """Three valid, non-overlapping, high-score regions should round-trip
    through propose_candidate_regions with correct pixel coordinates."""
    parsed = _planner_response([
        {"name": "top_left", "score": 0.95, "x1_pct": 0.0, "y1_pct": 0.0, "x2_pct": 0.2, "y2_pct": 0.2},
        {"name": "center",   "score": 0.80, "x1_pct": 0.4, "y1_pct": 0.4, "x2_pct": 0.6, "y2_pct": 0.6},
        {"name": "bot_right","score": 0.70, "x1_pct": 0.7, "y1_pct": 0.7, "x2_pct": 0.9, "y2_pct": 0.9},
    ])
    with patch("vision_desktop_automation.grounding.call_gemini_vision", return_value="{}"), \
         patch("vision_desktop_automation.grounding.parse_vlm_json", return_value=parsed):
        regions = propose_candidate_regions(fake_screenshot, "Test target")

    assert len(regions) == 3
    for r in regions:
        assert {"px1", "py1", "px2", "py2"} <= r.keys()
        assert r["px2"] > r["px1"] and r["py2"] > r["py1"]
    # Sorted descending by score after NMS
    scores = [r["score"] for r in regions]
    assert scores == sorted(scores, reverse=True)


def test_propose_candidate_regions_recovers_partial_planner_json(fake_screenshot):
    """When parse_vlm_json returns no regions, malformed planner JSON should still
    recover candidate regions from the raw response."""
    response = '''```json
    {
      "candidate_regions": [
        {
          "name": "notepad_desktop_icon",
          "reason": "Clearly visible Notepad shortcut icon on desktop.",
          "score": 0.99,
          "x1_pct": 0.346,
          "y1_pct": 0.346,
          "x2_pct": 0.395,
          "y2_pct": 0.431
        }
      ]
    }
    ```'''

    with patch("vision_desktop_automation.grounding.call_gemini_vision", return_value=response), \
         patch("vision_desktop_automation.grounding.parse_vlm_json", return_value={"reason": "Clearly visible Notepad shortcut icon on desktop."}):
        regions = propose_candidate_regions(fake_screenshot, "Test target")

    assert len(regions) == 1
    assert regions[0]["name"] == "notepad_desktop_icon"
    assert regions[0]["score"] == pytest.approx(0.99)


def test_propose_candidate_regions_filters_low_score_regions(fake_screenshot):
    """Regions below MIN_PLANNER_REGION_SCORE (0.50) must be dropped."""
    parsed = _planner_response([
        {"name": "good", "score": 0.95, "x1_pct": 0.0, "y1_pct": 0.0, "x2_pct": 0.2, "y2_pct": 0.2},
        {"name": "bad",  "score": 0.30, "x1_pct": 0.7, "y1_pct": 0.7, "x2_pct": 0.9, "y2_pct": 0.9},
    ])
    with patch("vision_desktop_automation.grounding.call_gemini_vision", return_value="{}"), \
         patch("vision_desktop_automation.grounding.parse_vlm_json", return_value=parsed):
        regions = propose_candidate_regions(fake_screenshot, "Test target")

    assert len(regions) == 1
    assert regions[0]["name"] == "good"
    assert regions[0]["score"] >= MIN_PLANNER_REGION_SCORE


def test_propose_candidate_regions_expands_tight_planner_regions(fake_screenshot):
    """A region tighter than MIN_REGION_WIDTH_PX must be expanded so the
    grounder gets enough icon + label context."""
    # 0.01 of 1920 == 19px wide → smaller than MIN_REGION_WIDTH_PX (80)
    parsed = _planner_response([
        {"name": "tiny", "score": 0.95, "x1_pct": 0.50, "y1_pct": 0.50, "x2_pct": 0.51, "y2_pct": 0.51},
    ])
    with patch("vision_desktop_automation.grounding.call_gemini_vision", return_value="{}"), \
         patch("vision_desktop_automation.grounding.parse_vlm_json", return_value=parsed):
        regions = propose_candidate_regions(fake_screenshot, "Test target")

    assert len(regions) == 1
    r = regions[0]
    assert (r["px2"] - r["px1"]) >= MIN_REGION_WIDTH_PX
    assert (r["py2"] - r["py1"]) >= MIN_REGION_HEIGHT_PX


def test_vlm_ground_icon_returns_4_tuple(fake_screenshot):
    """vlm_ground_icon must return (x:int, y:int, conf:float, prop_score:float)
    with click coordinates inside the image bounds."""
    grounder = _grounder_response(confidence=0.95, click_x=0.5, click_y=0.5)
    with patch("vision_desktop_automation.grounding.call_gemini_vision", return_value="{}"), \
         patch("vision_desktop_automation.grounding.parse_vlm_json", return_value=grounder):
        result = vlm_ground_icon(fake_screenshot, "Test target")

    assert isinstance(result, tuple) and len(result) == 4
    x, y, conf, prop_score = result
    assert isinstance(x, int) and isinstance(y, int)
    assert isinstance(conf, float) and isinstance(prop_score, float)
    assert 0 <= x <= fake_screenshot.width
    assert 0 <= y <= fake_screenshot.height
    assert 0.0 <= conf <= 1.0


def test_vlm_ground_icon_recurses_on_low_confidence(fake_screenshot):
    """When the grounder's confidence is below SEARCH_CONFIDENCE_THRESHOLD and
    the patch is large enough, vlm_ground_icon must recurse, producing more
    than one VLM call."""
    low = _grounder_response(confidence=0.50, x1=0.4, y1=0.4, x2=0.6, y2=0.6, click_x=0.5, click_y=0.5)
    high = _grounder_response(confidence=0.95, click_x=0.5, click_y=0.5)
    with patch("vision_desktop_automation.grounding.call_gemini_vision", return_value="{}") as mock_call, \
         patch("vision_desktop_automation.grounding.parse_vlm_json", side_effect=[low, high]):
        x, y, conf, _ = vlm_ground_icon(fake_screenshot, "Test target")

    # Recursion fired, so call_gemini_vision was called at least twice.
    assert mock_call.call_count >= 2
    assert conf == pytest.approx(0.95)


def test_combined_score_uses_three_distinct_signals(fake_screenshot):
    """The combined_score formula must be:
       GROUNDING_CONFIDENCE_WEIGHT * conf + REGION_SCORE_WEIGHT * region_score + BOX_SCORE_WEIGHT * prop_score.
    Guards against the regression where conf was used twice (BOX_SCORE_WEIGHT * conf)
    instead of being a third independent signal weighted by proposal_score."""
    region = {
        "name": "test", "score": 0.8, "reason": "",
        "px1": 0, "py1": 0, "px2": 1920, "py2": 1080,
        "x1_pct": 0.0, "y1_pct": 0.0, "x2_pct": 1.0, "y2_pct": 1.0,
    }
    expected = (
        GROUNDING_CONFIDENCE_WEIGHT * 0.9
        + REGION_SCORE_WEIGHT * 0.8
        + BOX_SCORE_WEIGHT * 0.85
    )
    with patch(
        "vision_desktop_automation.grounding.vlm_ground_icon",
        return_value=(100, 100, 0.9, 0.85),
    ), patch(
        "vision_desktop_automation.grounding.verify_icon_identity",
        return_value=True,
    ):
        results = search_region_recursive(fake_screenshot, "Test target", region, depth=0)

    assert results, "expected at least one verified result"
    assert results[0]["combined_score"] == pytest.approx(expected, abs=1e-9)


def test_planner_guided_ground_icon_falls_back_to_direct(fake_screenshot):
    """If propose_candidate_regions raises a non-API error and
    ALLOW_DIRECT_GROUNDING_FALLBACK is True, the function must fall back to
    direct vlm_ground_icon and return the 3-tuple public API contract."""
    with patch(
        "vision_desktop_automation.grounding.propose_candidate_regions",
        side_effect=ValueError("planner returned no usable candidate regions"),
    ), patch(
        "vision_desktop_automation.grounding.vlm_ground_icon",
        return_value=(500, 500, 0.9, 0.85),
    ) as mock_ground:
        result = planner_guided_ground_icon(fake_screenshot, "Test target")

    assert isinstance(result, tuple) and len(result) == 3
    x, y, conf = result
    assert (x, y, conf) == (500, 500, 0.9)
    mock_ground.assert_called_once_with(fake_screenshot, "Test target")


# =========================================================================
# Disambiguation tests (multi-icon resolution — Notepad vs Notepad++ etc.)
# =========================================================================


def test_disambiguate_proposals_picks_verified_candidate(fake_screenshot):
    """When the verifier rejects the higher-scored proposal but accepts the
    lower-scored one, the lower-scored proposal must win because verified=True
    multiplies its score by 1.5x while unverified gets 0.5x."""
    first = {
        "proposal_score": 0.85,
        "confidence": 0.85,
        "label_match": 0.85,
        "visual_match": 0.85,
        "box": {"x1": 0.10, "y1": 0.10, "x2": 0.20, "y2": 0.20},
        "click_x_pct": 0.15,
        "click_y_pct": 0.15,
        "reason": "first",
    }
    second = {
        "proposal_score": 0.80,
        "confidence": 0.80,
        "label_match": 0.80,
        "visual_match": 0.80,
        "box": {"x1": 0.70, "y1": 0.70, "x2": 0.80, "y2": 0.80},
        "click_x_pct": 0.75,
        "click_y_pct": 0.75,
        "reason": "second",
    }

    # Verifier: first → False, second → True
    # → first adjusted = 0.85 * 0.5 = 0.425
    # → second adjusted = 0.80 * 1.5 = 1.20  (winner)
    verifier_results = [{"correct": False}, {"correct": True}]

    with patch("vision_desktop_automation.grounding.call_gemini_vision", return_value="{}"), \
         patch("vision_desktop_automation.grounding.parse_vlm_json", side_effect=verifier_results):
        result = disambiguate_proposals(
            fake_screenshot, [first, second], "Test target", 0, 0
        )

    assert result is second


def _disambig_grounder_response(score_a: float, score_b: float) -> dict:
    """Build a grounder response with two non-overlapping proposals so NMS keeps both.
    proposal_score = 0.50*conf + 0.30*label_match + 0.20*visual_match;
    setting all three equal makes proposal_score == score_a / score_b directly."""
    return {
        "proposals": [
            {
                "found": True,
                "confidence": score_a, "label_match": score_a, "visual_match": score_a,
                "x1_pct": 0.10, "y1_pct": 0.10, "x2_pct": 0.20, "y2_pct": 0.20,
                "click_x_pct": 0.15, "click_y_pct": 0.15,
            },
            {
                "found": True,
                "confidence": score_b, "label_match": score_b, "visual_match": score_b,
                "x1_pct": 0.70, "y1_pct": 0.70, "x2_pct": 0.80, "y2_pct": 0.80,
                "click_x_pct": 0.75, "click_y_pct": 0.75,
            },
        ]
    }


def test_disambiguation_triggers_when_proposals_within_margin(fake_screenshot):
    """If the top-2 grounder proposals are within DISAMBIGUATION_MARGIN of each
    other in proposal_score, vlm_ground_icon must run disambiguate_proposals
    instead of blindly picking the higher score."""
    # Top two scores 0.95 and 0.90 → margin 0.05 < DISAMBIGUATION_MARGIN (0.10)
    grounder = _disambig_grounder_response(0.95, 0.90)
    assert (0.95 - 0.90) < DISAMBIGUATION_MARGIN, "test setup must be within margin"

    # The mocked disambiguator returns a parsed-proposal-shaped dict with high
    # confidence so vlm_ground_icon doesn't recurse.
    fake_winner = {
        "proposal_score": 0.95,
        "confidence": 0.95,
        "label_match": 0.95,
        "visual_match": 0.95,
        "box": {"x1": 0.10, "y1": 0.10, "x2": 0.20, "y2": 0.20},
        "click_x_pct": 0.15,
        "click_y_pct": 0.15,
        "reason": "mocked winner",
    }

    with patch("vision_desktop_automation.grounding.call_gemini_vision", return_value="{}"), \
         patch("vision_desktop_automation.grounding.parse_vlm_json", return_value=grounder), \
         patch("vision_desktop_automation.grounding.disambiguate_proposals",
               return_value=fake_winner) as mock_disambig:
        vlm_ground_icon(fake_screenshot, "Test target")

    mock_disambig.assert_called_once()


def test_disambiguation_skipped_when_clear_winner(fake_screenshot):
    """When the top-2 proposals are well outside DISAMBIGUATION_MARGIN, the
    disambiguator must NOT be invoked — the higher-scored proposal wins
    directly (saves an extra round of VLM calls)."""
    # 0.95 vs 0.50 → margin 0.45 > DISAMBIGUATION_MARGIN
    grounder = _disambig_grounder_response(0.95, 0.50)
    assert (0.95 - 0.50) > DISAMBIGUATION_MARGIN, "test setup must exceed margin"

    with patch("vision_desktop_automation.grounding.call_gemini_vision", return_value="{}"), \
         patch("vision_desktop_automation.grounding.parse_vlm_json", return_value=grounder), \
         patch("vision_desktop_automation.grounding.disambiguate_proposals") as mock_disambig:
        x, y, conf, _ = vlm_ground_icon(fake_screenshot, "Test target")

    mock_disambig.assert_not_called()
    assert conf == pytest.approx(0.95)
