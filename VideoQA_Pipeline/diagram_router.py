# VideoQA_Pipeline/diagram_router.py
# Feat-style diagram router (company gateway compatible, quiet fallbacks).
# - Classify a frame (diagram vs. normal) with a lightweight model (chat + base64).
# - If diagram → caption with GPT-5; otherwise → caption with NON_DIAGRAM_CAPTION_MODEL.
# - Primary GPT-5 path uses Responses API + file upload (feat-style).
# - If uploads are disabled, silently fallback to chat.completions + base64.
# - Param policy:
#     * GPT-5 family: DO NOT send temperature or token params (strict gateway mode).
#     * Others      : send max_tokens and temperature.

import os
import re
import json
import time
import base64
from io import BytesIO
from typing import Dict, Any, Optional, Tuple, List

from PIL import Image
from openai import OpenAI  # client is provided by the caller

from VideoQA_constants.prompts import (  # type: ignore
    GPT4_PROMPT_TEMPLATE,
    GPT5_DIAGRAM_PROMPT_TEMPLATE,
    DIAGRAM_ROUTER_CLASSIFY_PROMPT,
)

# ---------- Config (env-overridable, identical to feat) ----------
LIGHTGPT_MODEL = os.getenv("LIGHTGPT_MODEL", "gpt-4o-mini-model")
NON_DIAGRAM_CAPTION_MODEL = os.getenv("NON_DIAGRAM_CAPTION_MODEL", "gpt-4o-model")
DIAGRAM_CAPTION_MODEL     = os.getenv("DIAGRAM_CAPTION_MODEL", "gpt-5-model")
DIAGRAM_THRESHOLD = float(os.getenv("DIAGRAM_THRESHOLD", "0.65"))
CAPTION_MAX_TOKENS = int(os.getenv("CAPTION_MAX_TOKENS", "1200"))  # for non-gpt-5
ROUTER_LOG_NAME = os.getenv("ROUTER_LOG_NAME", "diagram_flags.jsonl")

ALLOWED_TYPES = {"diagram", "flowchart"}  # same categories as feat

# ---------- helpers ----------
def _encode_image_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _is_gpt5_family(model: str) -> bool:
    return model.lower().startswith("gpt-5")

def _chat_complete(
    client: OpenAI,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    token_budget: Optional[int] = None,
    temperature: Optional[float] = None,
    response_format: Optional[dict] = None,
):
    """
    Gateway-safe chat wrapper:
      - GPT-5: no temperature, no token params.
      - Others: max_tokens (=token_budget) and temperature are allowed.
    """
    params: Dict[str, Any] = {"model": model, "messages": messages}
    if response_format is not None:
        params["response_format"] = response_format

    if _is_gpt5_family(model):
        # Strict mode for GPT-5: do not send temperature or token params.
        pass
    else:
        if token_budget is not None:
            params["max_tokens"] = token_budget
        if temperature is not None:
            params["temperature"] = temperature

    return client.chat.completions.create(**params)

# ---------- classifier ----------
def _classify_diagram(client: OpenAI, image_b64: str) -> Dict[str, Any]:
    """
    One-line classifier, quiet on failure (defaults to NOT_DIAGRAM).
    Expected output format:
      LABEL: <DIAGRAM|NOT_DIAGRAM>; TYPES: <...>; CONF: <0.00-1.00>
    """
    messages = [
        {"role": "system", "content": "Answer with ONE LINE only."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": DIAGRAM_ROUTER_CLASSIFY_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}" }},
            ],
        },
    ]
    try:
        resp = _chat_complete(
            client,
            model=LIGHTGPT_MODEL,
            messages=messages,
            token_budget=80,      # non-gpt-5 → max_tokens=80
            temperature=0.0,      # non-gpt-5 → allowed
        )
        line = (resp.choices[0].message.content or "").strip().replace("\n", " ")
    except Exception:
        # Quiet fallback: treat as NOT_DIAGRAM
        return {"is_diagram": False, "confidence": 0.0, "types": []}

    label = "NOT_DIAGRAM"
    m_label = re.search(r"LABEL:\s*(DIAGRAM|NOT_DIAGRAM)", line, re.I)
    if m_label:
        label = m_label.group(1).upper()

    types: List[str] = []
    m_types = re.search(r"TYPES:\s*([^;]+)", line, re.I)
    if m_types:
        types = [t.strip().lower() for t in m_types.group(1).split(",") if t.strip()]

    conf = 0.0
    m_conf = re.search(r"CONF:\s*([01](?:\.\d+)?)", line, re.I)
    if m_conf:
        try:
            conf = float(m_conf.group(1))
        except ValueError:
            conf = 0.0
    conf = max(0.0, min(1.0, conf))

    return {"is_diagram": (label == "DIAGRAM"), "confidence": conf, "types": types}

# ---------- GPT-5 via Responses API ----------
def _responses_output_text(resp: Any) -> str:
    out = getattr(resp, "output_text", None)
    if isinstance(out, str) and out.strip():
        return out.strip()
    chunks: List[str] = []
    seq = getattr(resp, "output", None)
    if isinstance(seq, list):
        for item in seq:
            if isinstance(item, dict) and item.get("type") in ("output_text", "message"):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
    return "\n".join(chunks).strip()

def _gpt5_responses_caption(client: OpenAI, image_path: str, prompt_text: str) -> Tuple[str, Optional[str]]:
    """
    Upload image → Responses API (image_file_id; retry file_id). Return (text, finish_reason).
    """
    with open(image_path, "rb") as fh:
        up = client.files.create(file=fh, purpose="user_data")

    input_msg = [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": prompt_text},
            {"type": "input_image", "image_file_id": up.id},
        ],
    }]

    try:
        resp = client.responses.create(
            model=DIAGRAM_CAPTION_MODEL,  # use variable, not string literal
            instructions="You are a precise diagram/flowchart describer. Output plain English text only.",
            input=input_msg,
        )
    except Exception:
        input_msg[0]["content"][1] = {"type": "input_image", "file_id": up.id}
        resp = client.responses.create(
            model=DIAGRAM_CAPTION_MODEL,
            instructions="You are a precise diagram/flowchart describer. Output plain English text only.",
            input=input_msg,
        )

    txt = _responses_output_text(resp)
    finish_reason = getattr(resp, "finish_reason", None)
    return txt, finish_reason

# ---------- caption selection ----------
def _caption_with_model(
    client: OpenAI,
    *,
    image_b64: str,
    image_path: str,
    prompt: str,
    model: str
) -> Tuple[str, str, Optional[str]]:
    """
    Caption with selected model, quiet fallbacks.
    - GPT-5: Responses preferred; if uploads disabled, fallback to chat (no temp, no tokens).
    - Non-GPT-5: chat (max_tokens + temperature).
    """
    if _is_gpt5_family(model):
        try:
            text, finish = _gpt5_responses_caption(client, image_path, prompt)
            if text:
                return text, model, finish
            # empty → continue to fallback
        except Exception as e:
            msg = str(e)
            if "files_settings is not set" in msg:
                # Silent fallback to chat on GPT-5
                messages = [
                    {"role": "system", "content": "You are a precise, concise vision captioner. Output plain English text only."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}]},
                ]
                r = _chat_complete(
                    client,
                    model=model,
                    messages=messages,
                    token_budget=None,
                    temperature=None,
                )
                txt = (r.choices[0].message.content or "").strip()
                finish_reason = getattr(r.choices[0], "finish_reason", None)
                return (txt if txt else "[No textual output returned.]"), model, finish_reason
            # else: fall through to generic fallback

        # Generic fallback to non-gpt-5 (quiet)
        messages = [
            {"role": "system", "content": "You are a precise, concise vision captioner. Output plain English text only."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}]},
        ]
        resp = _chat_complete(
            client,
            model=NON_DIAGRAM_CAPTION_MODEL,
            messages=messages,
            token_budget=CAPTION_MAX_TOKENS,
            temperature=0.2,
        )
        txt = (resp.choices[0].message.content or "").strip()
        finish_reason = getattr(resp.choices[0], "finish_reason", None)
        used = f"{model}→{NON_DIAGRAM_CAPTION_MODEL}"
        return txt if txt else "[No textual output returned by either model.]", used, finish_reason

    # Non-GPT-5: standard chat path
    messages = [
        {"role": "system", "content": "You are a precise, concise vision captioner. Output plain English text only."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}]},
    ]
    resp = _chat_complete(
        client,
        model=model,
        messages=messages,
        token_budget=CAPTION_MAX_TOKENS,
        temperature=0.2,
    )
    txt = (resp.choices[0].message.content or "").strip()
    finish_reason = getattr(resp.choices[0], "finish_reason", None)
    return txt, model, finish_reason

# ---------- public API ----------
def route_and_caption_image_path(
    client: OpenAI,
    image_path: str,
    timestamp_token: str,
    router_log_dir: Optional[str],
) -> Dict[str, Any]:
    """Single-frame routing: classify → choose model → caption. Quiet fallbacks; structured result."""
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    image_b64 = _encode_image_to_b64(img, "PNG")

    det = _classify_diagram(client, image_b64)
    types = set(t.lower() for t in det.get("types", []))
    is_diag = det.get("is_diagram", False) and det.get("confidence", 0.0) >= DIAGRAM_THRESHOLD and bool(types & ALLOWED_TYPES)

    if is_diag:
        model = DIAGRAM_CAPTION_MODEL
        prompt = GPT5_DIAGRAM_PROMPT_TEMPLATE.format(timestamp=timestamp_token)
        decision = "diagram"
    else:
        model = NON_DIAGRAM_CAPTION_MODEL
        prompt = GPT4_PROMPT_TEMPLATE.format(timestamp=timestamp_token)
        decision = "normal"

    caption_text, model_used_label, finish_reason = _caption_with_model(
        client=client,
        image_b64=image_b64,
        image_path=image_path,
        prompt=prompt,
        model=model,
    )
    result = {
        "timestamp": timestamp_token,
        "description": f"This video, during the timestamp {timestamp_token}, shows: {caption_text}",
        "caption_model": model_used_label,
        "router": {
            "light_model": LIGHTGPT_MODEL,
            "decision": decision,
            "is_diagram": det.get("is_diagram", False),
            "confidence": det.get("confidence", 0.0),
            "threshold": DIAGRAM_THRESHOLD,
            "types": list(types),
            "finish_reason": finish_reason,
        },
    }

    # Quiet audit log (optional)
    if router_log_dir:
        os.makedirs(router_log_dir, exist_ok=True)
        with open(os.path.join(router_log_dir, ROUTER_LOG_NAME), "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "frame_file": os.path.basename(image_path),
                "timestamp_token": timestamp_token,
                "router": result["router"],
                "caption_model": model_used_label,
                "ts": time.time(),
            }, ensure_ascii=False) + "\n")

    return result
