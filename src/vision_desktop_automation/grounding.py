import logging
from typing import Any

from PIL import Image

from vision_desktop_automation.config import (
    ALLOW_DIRECT_GROUNDING_FALLBACK,
    BOX_NMS_IOU_THRESHOLD,
    BOX_SCORE_WEIGHT,
    GROUNDING_CONFIDENCE_WEIGHT,
    MAX_CANDIDATE_REGIONS,
    MAX_GROUNDING_PROPOSALS,
    MAX_SEARCH_DEPTH,
    MIN_PLANNER_REGION_SCORE,
    MIN_REGION_HEIGHT_PX,
    MIN_REGION_WIDTH_PX,
    PATCH_THRESHOLD_PX,
    RECURSIVE_ACCEPT_CONFIDENCE,
    RECURSIVE_ACCEPT_SCORE,
    RECURSIVE_PLANNER_DEPTH,
    REGION_NMS_IOU_THRESHOLD,
    REGION_SCORE_WEIGHT,
    SEARCH_CONFIDENCE_THRESHOLD,
    SKIP_VERIFICATION_IF_CONFIDENT,
    USE_PLANNER_CANDIDATE_REGIONS,
    VERIFICATION_SKIP_CONFIDENCE,
    VERIFICATION_SKIP_REGION_SCORE,
)

from vision_desktop_automation.geometry import (
    box_iou,
    expand_region_pixels,
    normalize_pct,
)

from vision_desktop_automation.prompts import (
    GROUNDING_PROMPT,
    PLANNER_PROMPT,
    VERIFY_PROMPT,
)

from vision_desktop_automation.vlm_client import (
    call_gemini_vision,
    parse_vlm_json,
    recover_planner_regions_from_text,
)

def propose_candidate_regions(screenshot: Image.Image, target_description: str) -> list[dict[str, Any]]:
    prompt = PLANNER_PROMPT.format(
        target_description=target_description,
        max_regions=MAX_CANDIDATE_REGIONS,
    )
    response = call_gemini_vision(prompt, screenshot)
    logging.info(f"Planner: {response}")

    try:
        data = parse_vlm_json(response)
    except ValueError:
        data = recover_planner_regions_from_text(response)

    raw_regions = data.get("candidate_regions", [])
    if not isinstance(raw_regions, list):
        raise ValueError("Planner returned invalid candidate_regions")

    img_w, img_h = screenshot.size
    regions: list[dict[str, Any]] = []

    for idx, raw in enumerate(raw_regions[:MAX_CANDIDATE_REGIONS], start=1):
        if not isinstance(raw, dict):
            continue

        x1 = normalize_pct(raw.get("x1_pct"), 0.0)
        y1 = normalize_pct(raw.get("y1_pct"), 0.0)
        x2 = normalize_pct(raw.get("x2_pct"), 1.0)
        y2 = normalize_pct(raw.get("y2_pct"), 1.0)

        if x2 <= x1 or y2 <= y1:
            logging.warning(f"Skipping invalid region #{idx}: {raw}")
            continue

        px1 = int(x1 * img_w)
        py1 = int(y1 * img_h)
        px2 = int(x2 * img_w)
        py2 = int(y2 * img_h)

        if (px2 - px1) < MIN_REGION_WIDTH_PX or (py2 - py1) < MIN_REGION_HEIGHT_PX:
            old_box = (px1, py1, px2, py2)
            px1, py1, px2, py2 = expand_region_pixels(
                px1,
                py1,
                px2,
                py2,
                img_w,
                img_h,
            )

            x1 = px1 / img_w
            y1 = py1 / img_h
            x2 = px2 / img_w
            y2 = py2 / img_h

            logging.info(
                f"Expanded small planner region #{idx}: "
                f"{old_box} -> ({px1}, {py1}, {px2}, {py2})"
            )

        score = normalize_pct(raw.get("score"), 0.5)

        if score < MIN_PLANNER_REGION_SCORE:
            logging.info(
                f"Skipping low-score planner region #{idx}: "
                f"score={score:.2f}, name={raw.get('name', f'region_{idx}')}"
            )
            continue

        regions.append(
            {
                "name": str(raw.get("name", f"region_{idx}")),
                "reason": str(raw.get("reason", "")),
                "score": score,
                "x1_pct": x1,
                "y1_pct": y1,
                "x2_pct": x2,
                "y2_pct": y2,
                "px1": px1,
                "py1": py1,
                "px2": px2,
                "py2": py2,
            }
        )

    if not regions:
        raise ValueError("Planner returned no usable candidate regions")

    regions = nms_regions(regions)
    regions.sort(key=lambda r: r["score"], reverse=True)
    logging.info(f"Planner produced {len(regions)} usable candidate region(s) after NMS")
    for r in regions:
        logging.info(
            f"Candidate: {r['name']} score={r['score']:.2f} "
            f"box=({r['px1']},{r['py1']})-({r['px2']},{r['py2']}) reason={r['reason']}"
        )
    return regions


def crop_region(screenshot: Image.Image, region: dict[str, Any]) -> Image.Image:
    return screenshot.crop((region["px1"], region["py1"], region["px2"], region["py2"]))



def nms_regions(regions: list[dict[str, Any]], iou_threshold: float = REGION_NMS_IOU_THRESHOLD) -> list[dict[str, Any]]:
    """Remove heavily overlapping planner regions, keeping the highest planner score."""
    sorted_regions = sorted(regions, key=lambda r: float(r.get("score", 0.0)), reverse=True)
    kept: list[dict[str, Any]] = []

    for region in sorted_regions:
        rb = {
            "x1": float(region["x1_pct"]),
            "y1": float(region["y1_pct"]),
            "x2": float(region["x2_pct"]),
            "y2": float(region["y2_pct"]),
        }
        should_keep = True
        for kept_region in kept:
            kb = {
                "x1": float(kept_region["x1_pct"]),
                "y1": float(kept_region["y1_pct"]),
                "x2": float(kept_region["x2_pct"]),
                "y2": float(kept_region["y2_pct"]),
            }
            if box_iou(rb, kb) > iou_threshold:
                should_keep = False
                logging.info(f"NMS removed overlapping region: {region['name']}")
                break
        if should_keep:
            kept.append(region)

    return kept


def nms_proposals(proposals: list[dict[str, Any]], iou_threshold: float = BOX_NMS_IOU_THRESHOLD) -> list[dict[str, Any]]:
    """Remove duplicate target boxes, keeping the highest proposal score."""
    sorted_props = sorted(proposals, key=lambda p: float(p.get("proposal_score", 0.0)), reverse=True)
    kept: list[dict[str, Any]] = []

    for prop in sorted_props:
        pb = prop["box"]
        if all(box_iou(pb, k["box"]) <= iou_threshold for k in kept):
            kept.append(prop)

    return kept


def parse_grounding_proposals(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse Gemini grounding JSON into normalized proposal dictionaries."""
    raw_proposals = data.get("proposals", [])

    # Backward compatibility if the model accidentally returns the old single-point schema.
    if not raw_proposals and data.get("found") is True:
        raw_proposals = [data]

    if not isinstance(raw_proposals, list):
        return []

    proposals: list[dict[str, Any]] = []
    for raw in raw_proposals[:MAX_GROUNDING_PROPOSALS]:
        if not isinstance(raw, dict):
            continue
        if raw.get("found", True) is False:
            continue

        x1 = normalize_pct(raw.get("x1_pct", raw.get("box", {}).get("x1_pct") if isinstance(raw.get("box"), dict) else None), 0.0)
        y1 = normalize_pct(raw.get("y1_pct", raw.get("box", {}).get("y1_pct") if isinstance(raw.get("box"), dict) else None), 0.0)
        x2 = normalize_pct(raw.get("x2_pct", raw.get("box", {}).get("x2_pct") if isinstance(raw.get("box"), dict) else None), 0.0)
        y2 = normalize_pct(raw.get("y2_pct", raw.get("box", {}).get("y2_pct") if isinstance(raw.get("box"), dict) else None), 0.0)

        click_x = normalize_pct(raw.get("click_x_pct"), (x1 + x2) / 2 if x2 > x1 else 0.0)
        click_y = normalize_pct(raw.get("click_y_pct"), (y1 + y2) / 2 if y2 > y1 else 0.0)

        # If the model only returned click coordinates, synthesize a small box around the point.
        if x2 <= x1 or y2 <= y1:
            if 0.01 <= click_x <= 0.99 and 0.01 <= click_y <= 0.99:
                half_w = 0.045
                half_h = 0.060
                x1 = max(0.0, click_x - half_w)
                y1 = max(0.0, click_y - half_h)
                x2 = min(1.0, click_x + half_w)
                y2 = min(1.0, click_y + half_h)
            else:
                continue

        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            continue
        if not (0.01 <= click_x <= 0.99 and 0.01 <= click_y <= 0.99):
            continue

        confidence = normalize_pct(raw.get("confidence"), 0.0)
        label_match = normalize_pct(raw.get("label_match"), confidence)
        visual_match = normalize_pct(raw.get("visual_match"), confidence)
        proposal_score = 0.50 * confidence + 0.30 * label_match + 0.20 * visual_match

        proposals.append(
            {
                "confidence": confidence,
                "label_match": label_match,
                "visual_match": visual_match,
                "proposal_score": proposal_score,
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "click_x_pct": click_x,
                "click_y_pct": click_y,
                "reason": str(raw.get("reason", "")),
            }
        )

    return nms_proposals(proposals)


def verify_icon_identity(screenshot: Image.Image, x_pct: float, y_pct: float) -> bool:
    try:
        img_w, img_h = screenshot.size
        cx = int(x_pct * img_w)
        cy = int(y_pct * img_h)
        pad = 48
        x1 = max(0, cx - pad)
        y1 = max(0, cy - pad)
        x2 = min(img_w, cx + pad)
        y2 = min(img_h, cy + pad)
        crop = screenshot.crop((x1, y1, x2, y2))

        response = call_gemini_vision(VERIFY_PROMPT, crop)
        result = parse_vlm_json(response)
        correct = result.get("correct", True)
        if correct:
            logging.info("Icon verified: correct Windows Notepad icon")
        else:
            logging.warning(f"Icon verification failed: {result.get('reason', 'unknown')}")
        return bool(correct)
    except Exception as e:
        logging.warning(f"Icon verification failed unexpectedly: {e} — proceeding anyway")
        return True


# =========================
# VLM GROUNDING
# =========================
def vlm_ground_icon(
    screenshot: Image.Image,
    target_description: str,
    depth: int = 0,
    offset_x: int = 0,
    offset_y: int = 0,
) -> tuple[int, int, float]:
    """
    Box-based Gemini grounding.
    Returns absolute click coordinates plus confidence.
    Still includes a local confidence-zoom fallback inside the current crop.
    """
    img_w, img_h = screenshot.size
    logging.info(f"[Depth {depth}] Box grounding {img_w}x{img_h} at offset ({offset_x}, {offset_y})")

    prompt = GROUNDING_PROMPT.format(
        target_description=target_description,
        max_proposals=MAX_GROUNDING_PROPOSALS,
    )
    response = call_gemini_vision(prompt, screenshot)
    logging.info(f"Grounder: {response}")

    data = parse_vlm_json(response)
    proposals = parse_grounding_proposals(data)
    if not proposals:
        raise ValueError(f"VLM could not locate '{target_description}'")

    best = max(proposals, key=lambda p: p["proposal_score"])
    confidence = float(best["confidence"])
    click_x_pct = float(best["click_x_pct"])
    click_y_pct = float(best["click_y_pct"])

    if confidence < SEARCH_CONFIDENCE_THRESHOLD and depth < MAX_SEARCH_DEPTH:
        box = best["box"]
        pad_x = max(0.08, (box["x2"] - box["x1"]) * 1.5)
        pad_y = max(0.08, (box["y2"] - box["y1"]) * 1.5)
        cx = (box["x1"] + box["x2"]) / 2
        cy = (box["y1"] + box["y2"]) / 2

        x1 = max(0, int((cx - pad_x) * img_w))
        y1 = max(0, int((cy - pad_y) * img_h))
        x2 = min(img_w, int((cx + pad_x) * img_w))
        y2 = min(img_h, int((cy + pad_y) * img_h))

        if (x2 - x1) > PATCH_THRESHOLD_PX and (y2 - y1) > PATCH_THRESHOLD_PX:
            logging.info(f"Confidence {confidence:.2f} low — zooming around best proposal box")
            crop = screenshot.crop((x1, y1, x2, y2))
            return vlm_ground_icon(
                crop,
                target_description,
                depth + 1,
                offset_x + x1,
                offset_y + y1,
            )

    abs_x = int(offset_x + click_x_pct * img_w)
    abs_y = int(offset_y + click_y_pct * img_h)
    logging.info(
        f"Grounded box proposal: ({abs_x}, {abs_y}), "
        f"confidence={confidence:.2f}, proposal_score={best['proposal_score']:.2f}"
    )
    return abs_x, abs_y, confidence


def search_region_recursive(
    screenshot: Image.Image,
    target_description: str,
    region: dict[str, Any],
    depth: int = 0,
) -> list[dict[str, Any]]:
    """
    Gemini-only approximation of ScreenSeekeR's recursive planner/grounder search.
    It asks the planner for subregions inside the current region, grounds inside each crop,
    verifies the best candidate, and recurses if confidence is still not strong enough.
    """
    results: list[dict[str, Any]] = []
    crop = crop_region(screenshot, region)

    # First try direct box grounding inside this region.
    try:
        x, y, conf = vlm_ground_icon(
            crop,
            target_description,
            depth=0,
            offset_x=region["px1"],
            offset_y=region["py1"],
        )
        x_pct_full = x / screenshot.width
        y_pct_full = y / screenshot.height

        should_skip_verification = (
            SKIP_VERIFICATION_IF_CONFIDENT
            and conf >= VERIFICATION_SKIP_CONFIDENCE
            and float(region["score"]) >= VERIFICATION_SKIP_REGION_SCORE
        )

        if should_skip_verification:
            logging.info(
                "Skipping verifier because planner and grounder are both highly confident"
            )
            verified = True
        else:
            verified = verify_icon_identity(screenshot, x_pct_full, y_pct_full)

        if verified:
            combined_score = (
                GROUNDING_CONFIDENCE_WEIGHT * conf
                + REGION_SCORE_WEIGHT * float(region["score"])
                + BOX_SCORE_WEIGHT * conf
            )
            results.append(
                {
                    "x": x,
                    "y": y,
                    "confidence": conf,
                    "region_score": float(region["score"]),
                    "combined_score": combined_score,
                    "region_name": region["name"],
                    "depth": depth,
                }
            )
            if combined_score >= RECURSIVE_ACCEPT_SCORE and conf >= RECURSIVE_ACCEPT_CONFIDENCE:
                return results
    except Exception as e:
        logging.warning(f"Direct grounding in region {region['name']} failed: {e}")

    # If uncertain, recursively ask the planner for subregions inside this crop.
    if depth >= RECURSIVE_PLANNER_DEPTH:
        return results

    try:
        subregions = propose_candidate_regions(crop, target_description)
    except Exception as e:
        logging.warning(f"Recursive planner failed inside {region['name']}: {e}")
        return results

    for sub in subregions:
        mapped = dict(sub)
        mapped["name"] = f"{region['name']} > {sub['name']}"
        mapped["score"] = float(region["score"]) * float(sub["score"])
        mapped["px1"] = region["px1"] + sub["px1"]
        mapped["py1"] = region["py1"] + sub["py1"]
        mapped["px2"] = region["px1"] + sub["px2"]
        mapped["py2"] = region["py1"] + sub["py2"]
        mapped["x1_pct"] = mapped["px1"] / screenshot.width
        mapped["y1_pct"] = mapped["py1"] / screenshot.height
        mapped["x2_pct"] = mapped["px2"] / screenshot.width
        mapped["y2_pct"] = mapped["py2"] / screenshot.height
        results.extend(search_region_recursive(screenshot, target_description, mapped, depth + 1))

    return results


def planner_guided_ground_icon(screenshot: Image.Image, target_description: str) -> tuple[int, int, float]:
    """
    Gemini-only ScreenSeekeR-style search:
    1. Planner proposes candidate regions.
    2. Region NMS removes redundant overlaps.
    3. Grounder returns box proposals inside candidate crops.
    4. Verifier confirms icon identity.
    5. Recursive planner subregions are searched when uncertainty remains.
    6. Best verified candidate wins.
    7. Direct full-screen box grounding is used as fallback.
    """
    if not USE_PLANNER_CANDIDATE_REGIONS:
        return vlm_ground_icon(screenshot, target_description)

    try:
        regions = propose_candidate_regions(screenshot, target_description)
    except Exception as e:
        error_text = str(e)

        if "Gemini API failed" in error_text or "timeout" in error_text or "503" in error_text:
            raise RuntimeError(f"Primary planner API call failed, not falling back: {e}")

        if ALLOW_DIRECT_GROUNDING_FALLBACK:
            logging.warning(f"Planner failed: {e} — falling back to direct grounding")
            return vlm_ground_icon(screenshot, target_description)

        raise RuntimeError(f"Primary planner-guided flow failed before grounding: {e}")

    candidates: list[dict[str, Any]] = []
    for region in regions:
        logging.info(f"Recursive search in candidate region: {region['name']}")

        region_results = search_region_recursive(
            screenshot,
            target_description,
            region,
            depth=0,
        )

        candidates.extend(region_results)

        if region_results:
            best_region_result = max(
                region_results,
                key=lambda c: float(c["combined_score"]),
            )

            if (
                best_region_result["combined_score"] >= RECURSIVE_ACCEPT_SCORE
                and best_region_result["confidence"] >= RECURSIVE_ACCEPT_CONFIDENCE
            ):
                logging.info(
                    f"Early accepted verified candidate from {region['name']} "
                    f"with confidence={best_region_result['confidence']:.2f}, "
                    f"score={best_region_result['combined_score']:.2f}"
                )
                return (
                    int(best_region_result["x"]),
                    int(best_region_result["y"]),
                    float(best_region_result["confidence"]),
                )

    if candidates:
        candidates = sorted(candidates, key=lambda c: float(c["combined_score"]), reverse=True)
        best = candidates[0]
        logging.info(
            f"Best verified candidate: {best['region_name']} at ({best['x']},{best['y']}) "
            f"confidence={best['confidence']:.2f} score={best['combined_score']:.2f} depth={best['depth']}"
        )
        return int(best["x"]), int(best["y"]), float(best["confidence"])

    if ALLOW_DIRECT_GROUNDING_FALLBACK:
        logging.warning("No planner region produced a verified result — falling back to direct full-screen grounding")
        return vlm_ground_icon(screenshot, target_description)

    raise RuntimeError("Primary planner-guided flow failed: no verified candidate from planner regions")