# VideoQA_Pipeline/askVideoQA_gpt4o.py
import os
import logging
import openai
from dotenv import load_dotenv
from VideoQA_Pipeline.utils import (
    get_video_duration,
    classify_video_type,
)
from VideoQA_Pipeline.build_chunks import build_chunks
from VideoQA_Pipeline.align_multimodal_data import align_multimodal_data
from RAG_Pipeline.RagRetriever_Milvus import RagRetrieverMilvus

try:
    import tiktoken
    _HAS_TIKTOKEN = True
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _HAS_TIKTOKEN = False
    _ENC = None

def _approx_token_len(text: str) -> int:
    if not text:
        return 0
    if _HAS_TIKTOKEN:
        return len(_ENC.encode(text))
    return max(1, len(text) // 4) 

def _head_tail_cut_by_tokens(s: str, keep_tokens: int) -> str:
    if not s:
        return s
    if _HAS_TIKTOKEN:
        ids = _ENC.encode(s)
        if len(ids) <= keep_tokens:
            return s
        half = max(1, keep_tokens // 2)
        sep_ids = _ENC.encode("\n...\n")
        cut = ids[:half] + sep_ids + ids[-(keep_tokens - half):]
        return _ENC.decode(cut)
    ratio = max(1, len(s) // max(1, _approx_token_len(s)))
    keep_chars = max(64, keep_tokens * ratio)
    if len(s) <= keep_chars:
        return s
    half = keep_chars // 2
    return s[:half] + "\n...\n" + s[-half:]

def enforce_prompt_token_limit(prompt: str,
                               max_input_tokens: int = 120_000,
                               reserve_for_output: int = 2_048) -> str:
    hard_cap = max(1024, max_input_tokens - reserve_for_output)
    cur = _approx_token_len(prompt)
    if cur <= hard_cap:
        return prompt

    logging.warning(f"[TOK] Prompt too long: {cur} tokens; trimming to {hard_cap} tokens (reserve {reserve_for_output}).")

    lo, hi = 64, hard_cap
    best = _head_tail_cut_by_tokens(prompt, keep_tokens=hard_cap)
    for _ in range(16):
        t = _approx_token_len(best)
        if t <= hard_cap:
            lo = (lo + hi) // 2
            cand = _head_tail_cut_by_tokens(prompt, keep_tokens=lo)
            if _approx_token_len(cand) <= hard_cap:
                best = cand
            else:
                hi = lo
        else:
            hi = max(64, (lo + hi) // 2)
            best = _head_tail_cut_by_tokens(prompt, keep_tokens=hi)

    logging.info(f"[TOK] Prompt tokens: {cur} -> {_approx_token_len(best)} (cap={hard_cap}).")
    return best

load_dotenv()
openai.api_key = os.getenv("LITELLM_API_KEY")
openai.base_url = os.getenv("LITELLM_API_BASE")

MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "120000"))
RESERVE_TOKENS   = int(os.getenv("RESERVE_TOKENS", "2048"))

class AskVideoQAGPT4o:
    def __init__(self, model_name="gpt-4o", max_tokens=5000):
        self.model = model_name
        self.max_tokens = max_tokens

    def load_video_data(self, video_path):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_duration = get_video_duration(video_path)

        frame_dir = os.path.join("outputs", "frames", f"{video_name}_frames")
        frame_files = [f for f in os.listdir(frame_dir) if f.endswith(".png")] if os.path.exists(frame_dir) else []
        video_type = classify_video_type(frame_files, video_duration)

        chunk_path = os.path.join("outputs", "chunks", video_name, "all_chunks.jsonl")
        if not os.path.exists(chunk_path):
            print(f"🧱 Chunks not found for video '{video_name}', building...")
            build_chunks(video_name)

        retriever = RagRetrieverMilvus(video_name)
        if retriever.collection.num_entities == 0:
            print(f"🔍 Embeddings missing for '{video_name}', rebuilding...")
            retriever.build_from_chunks()

        return {
            "video_name": video_name,
            "video_duration": video_duration,
            "video_type": video_type
        }

    def answer_question(self, multimodal_context, user_question):
        prompt = align_multimodal_data(multimodal_context, user_question)

        prompt = enforce_prompt_token_limit(
            prompt,
            max_input_tokens=MAX_INPUT_TOKENS,
            reserve_for_output=RESERVE_TOKENS
        )

        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content.strip()
