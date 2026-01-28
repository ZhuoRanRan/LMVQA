"""
Backfill 7-way diagram categories into an existing descriptions.json (no re-captioning).

Why:
- diagram_router.py now supports second-stage 7-way classification for diagram-like frames.
- Older outputs/frames/<video>_frames/descriptions.json may not include router.diagram_category.

What this script does:
- For entries where router.decision == "diagram", locate the corresponding frame image in the same
  directory (by matching the timestamp token like "12s-34s" in the filename).
- Run the lightweight model (LIGHTGPT_MODEL) to classify into one of 7 categories.
- Write router.diagram_category and router.diagram_category_conf back into descriptions.json.

Usage (Windows / PowerShell):
  python -m VideoQA_Pipeline.backfill_diagram_categories --frame_dir "outputs\\frames\\Ciena1_frames"
"""

import argparse
import json
import os
import re
from typing import Dict, Optional, Tuple

from PIL import Image

from openai_client import get_openai_client
from VideoQA_Pipeline import diagram_router


def _extract_timestamp_token_from_frame_filename(filename: str) -> Optional[str]:
    # Expected: frame_XXX_12s-34s.png  -> "12s-34s"
    m = re.search(r"_(\d+)s-(\d+)s\.png$", filename, re.I)
    if not m:
        return None
    return f"{m.group(1)}s-{m.group(2)}s"


def _build_token_to_image_path(frame_dir: str) -> Dict[str, str]:
    token_map: Dict[str, str] = {}
    for name in os.listdir(frame_dir):
        if not name.lower().endswith(".png"):
            continue
        tok = _extract_timestamp_token_from_frame_filename(name)
        if not tok:
            continue
        # Keep the first match; filenames should be unique per token anyway.
        token_map.setdefault(tok, os.path.join(frame_dir, name))
    return token_map


def _classify_7way_for_image(client, image_path: str) -> Tuple[str, float]:
    """Return (category, confidence). Quiet fallback happens inside router classifier."""
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    image_b64 = diagram_router._encode_image_to_b64(img, "PNG")  # type: ignore[attr-defined]
    det = diagram_router._classify_diagram_category(client, image_b64)  # type: ignore[attr-defined]
    cat = (det.get("category") or "OTHER").upper()
    conf = float(det.get("confidence") or 0.0)
    return cat, conf


def backfill(frame_dir: str, *, in_place: bool = True) -> str:
    desc_path = os.path.join(frame_dir, "descriptions.json")
    if not os.path.exists(desc_path):
        raise FileNotFoundError(f"descriptions.json not found: {desc_path}")

    with open(desc_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Deep copy for a correct backup of the original file contents
    original_data = json.loads(json.dumps(data))
    if not isinstance(data, list):
        raise ValueError("descriptions.json must be a JSON array")

    token_to_img = _build_token_to_image_path(frame_dir)
    client = get_openai_client()

    updated = 0
    skipped = 0
    missing_img = 0

    for entry in data:
        router = entry.get("router") or {}
        decision = router.get("decision")
        if decision != "diagram":
            skipped += 1
            continue

        # If already backfilled, skip unless value missing
        if router.get("diagram_category") and router.get("diagram_category_conf") is not None:
            skipped += 1
            continue

        token = entry.get("timestamp")
        if not isinstance(token, str) or token not in token_to_img:
            missing_img += 1
            continue

        cat, conf = _classify_7way_for_image(client, token_to_img[token])
        router["diagram_category"] = cat
        router["diagram_category_conf"] = conf
        entry["router"] = router
        updated += 1

    # Backup then write
    out_path = desc_path
    if in_place:
        bak = desc_path + ".bak"
        if not os.path.exists(bak):
            with open(bak, "w", encoding="utf-8") as f:
                json.dump(original_data, f, indent=4, ensure_ascii=False)
        # Note: .bak is created only if not present; user can delete manually.

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Backfill complete. updated={updated}, skipped={skipped}, missing_image={missing_img}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_dir", required=True, help="Path to outputs/frames/<video>_frames")
    args = ap.parse_args()
    backfill(args.frame_dir, in_place=True)


if __name__ == "__main__":
    main()

