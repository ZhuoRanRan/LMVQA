# VideoQA_Pipeline/GPT4_generate_description.py
# Use company gateway (LiteLLM-compatible) OpenAI client and pass it into the router.
# Restores per-frame logging: prints which model was used and the routing decision.
# All other logic remains unchanged.

import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
from VideoQA_Pipeline.diagram_router import route_and_caption_image_path

# Load environment and build OpenAI client (LiteLLM-compatible)
load_dotenv()
_BASE_URL = os.getenv("LITELLM_API_BASE") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
_API_KEY  = os.getenv("LITELLM_API_KEY")  or os.getenv("OPENAI_API_KEY")
if not _API_KEY:
    raise RuntimeError("Missing API key. Set LITELLM_API_KEY or OPENAI_API_KEY in .env")
client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL) if _BASE_URL else OpenAI(api_key=_API_KEY)

def _extract_timestamp_token(filename: str) -> str:
    """Return token like '12s-34s' from a frame filename 'frame_XXX_12s-34s.png'."""
    m = re.search(r'_(\d+)s-(\d+)s\.png$', filename)
    return f"{m.group(1)}s-{m.group(2)}s" if m else "unknown"

def _frame_dir_upper(video_name: str) -> str:
    """Path to outputs/Frames/<video>_frames (Windows-friendly)."""
    return os.path.join("outputs", "Frames", f"{video_name}_frames")

def _frame_dir_lower(video_name: str) -> str:
    """Path to outputs/frames/<video>_frames (lowercase variant)."""
    return os.path.join("outputs", "frames", f"{video_name}_frames")

def _resolve_frame_dir(video_name: str) -> str:
    """Pick existing frame directory (prefer uppercase variant if present)."""
    up = _frame_dir_upper(video_name)
    lo = _frame_dir_lower(video_name)
    if os.path.isdir(up):
        return up
    if os.path.isdir(lo):
        return lo
    return up  # default to upper-style path

def generate_description(image_path: str) -> dict:
    """Generate one frame description via router (passes the prebuilt client)."""
    filename = os.path.basename(image_path)
    token = _extract_timestamp_token(filename)
    frame_dir = os.path.dirname(image_path)
    return route_and_caption_image_path(
        client=client,                      # pass company-gateway client into router
        image_path=image_path,
        timestamp_token=token,
        router_log_dir=frame_dir,          # keep audit log beside frames
    )

def process_video_frames(video_name: str):
    """Iterate frames, describe each, print which model was used, write descriptions.json."""
    frame_dir   = _resolve_frame_dir(video_name)
    output_json = os.path.join(frame_dir, "descriptions.json")

    if not os.path.exists(frame_dir):
        print(f"❌ No frames found for video: {video_name}")
        return []

    def sort_key(fname: str):
        m = re.search(r'_(\d+)s-(\d+)s\.png$', fname)
        return int(m.group(1)) if m else 10**12

    frame_files = sorted(
        [f for f in os.listdir(frame_dir) if f.lower().endswith(".png")],
        key=sort_key
    )

    descriptions = []
    for f in frame_files:
        path = os.path.join(frame_dir, f)
        desc = generate_description(path)
        descriptions.append(desc)

        # >>> Restore concise per-frame model / route logging (no error noise)
        model_used = desc.get("caption_model", "")
        decision   = desc.get("router", {}).get("decision")
        print(f"📝 {f} → {model_used} | {decision}")

    os.makedirs(frame_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as out:
        json.dump(descriptions, out, indent=4, ensure_ascii=False)

    print(f"✅ Descriptions saved to {output_json}")
    return descriptions
