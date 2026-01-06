# scripts/test_gpt5_responses.py
# Feat-style GPT-5 test (company gateway):
# - Try Responses API with file upload (same as feat).
# - If uploads disabled on gateway, silently fallback to chat.completions + base64.
# - GPT-5 chat fallback uses strict policy: no temperature, no token params.

import os
import re
import json
import base64
from io import BytesIO
from typing import Optional
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
import openai  # for chat fallback

# Prompt import (same as feat)
try:
    from VideoQA_constants.prompt import GPT5_DIAGRAM_PROMPT_TEMPLATE
except Exception:
    from VideoQA_constants.prompts import GPT5_DIAGRAM_PROMPT_TEMPLATE  # type: ignore

load_dotenv()
BASE_URL = os.getenv("LITELLM_API_BASE") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
API_KEY  = os.getenv("LITELLM_API_KEY")  or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing API key. Set LITELLM_API_KEY or OPENAI_API_KEY in .env")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL) if BASE_URL else OpenAI(api_key=API_KEY)
openai.api_key = API_KEY
if BASE_URL:
    openai.base_url = BASE_URL

def resolve_frame_dir(video_name: str) -> str:
    upper = os.path.join("outputs", "Frames", f"{video_name}_frames")
    lower = os.path.join("outputs", "frames", f"{video_name}_frames")
    if os.path.isdir(upper): return upper
    if os.path.isdir(lower): return lower
    raise FileNotFoundError(f"No frame directory found for '{video_name}'")

def pick_test_frame(frame_dir: str) -> str:
    flags = os.path.join(frame_dir, "diagram_flags.jsonl")
    if os.path.exists(flags):
        with open(flags, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get("router", {}).get("decision") == "diagram":
                        cand = os.path.join(frame_dir, obj.get("frame_file"))
                        if os.path.exists(cand): return cand
                except Exception:
                    continue
    pngs = [f for f in os.listdir(frame_dir) if f.lower().endswith(".png")]
    if not pngs: raise RuntimeError(f"No PNG frames in {frame_dir}")
    def sort_key(name: str) -> int:
        m = re.search(r'_(\d+)s-(\d+)s\.png$', name)
        return int(m.group(1)) if m else 10**12
    pngs.sort(key=sort_key)
    return os.path.join(frame_dir, pngs[0])

def extract_timestamp_token(filename: str) -> str:
    m = re.search(r'_(\d+)s-(\d+)s\.png$', filename)
    return f"{m.group(1)}s-{m.group(2)}s" if m else "unknown"

def responses_output_text(resp) -> str:
    out = getattr(resp, "output_text", None)
    if isinstance(out, str) and out.strip():
        return out.strip()
    chunks = []
    seq = getattr(resp, "output", None)
    if isinstance(seq, list):
        for item in seq:
            if isinstance(item, dict) and item.get("type") in ("output_text", "message"):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
    return "\n".join(chunks).strip()

def try_responses_caption(image_path: str, token: str) -> Optional[str]:
    with open(image_path, "rb") as fh:
        up = client.files.create(file=fh, purpose="user_data")
    prompt_text = GPT5_DIAGRAM_PROMPT_TEMPLATE.format(timestamp=token)
    input_messages = [{
        "role": "user",
        "content": [
            {"type": "input_text",  "text": prompt_text},
            {"type": "input_image", "image_file_id": up.id},
        ],
    }]
    try:
        resp = client.responses.create(
            model="gpt-5",
            instructions="You are a precise diagram/flowchart describer. Output plain English text only.",
            input=input_messages,
        )
    except Exception:
        input_messages[0]["content"][1] = {"type": "input_image", "file_id": up.id}
        resp = client.responses.create(
            model="gpt-5",
            instructions="You are a precise diagram/flowchart describer. Output plain English text only.",
            input=input_messages,
        )
    return responses_output_text(resp) or None

def to_b64_png(path: str, max_side: int = 1024) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = BytesIO(); img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

def chat_fallback_caption(image_path: str, token: str) -> str:
    """STRICT GPT-5 chat: no temperature, no token params."""
    prompt_text = GPT5_DIAGRAM_PROMPT_TEMPLATE.format(timestamp=token)
    b64 = to_b64_png(image_path)
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        ]
    }]
    resp = openai.chat.completions.create(
        model="gpt-5",
        messages=messages,
    )
    return (resp.choices[0].message.content or "").strip()

def main():
    video = os.environ.get("VIDEO_NAME", "Lecture3")
    frame_dir = resolve_frame_dir(video)
    frame_path = pick_test_frame(frame_dir)
    token = extract_timestamp_token(os.path.basename(frame_path))

    # Try feat-style Responses first; if it returns empty or fails due to uploads being disabled, use chat fallback.
    text = None
    try:
        text = try_responses_caption(frame_path, token)
    except Exception as e:
        if "files_settings is not set" in str(e):
            text = None  # force fallback quietly
        else:
            text = None  # unexpected errors: still fallback quietly

    if not text:
        text = chat_fallback_caption(frame_path, token)

    print((text or "[EMPTY]")[:400])
    # Optionally: print full length or write to a temp file if you need
    # print("\nLength:", len(text or ""))

if __name__ == "__main__":
    main()
