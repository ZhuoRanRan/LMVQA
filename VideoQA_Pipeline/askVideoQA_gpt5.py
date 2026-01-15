# VideoQA_Pipeline/askVideoQA_gpt5.py
# -*- coding: utf-8 -*-
"""
AskVideoQAGPT5: same pipeline as askVideoQA_gpt4o, but uses GPT-5 as the core LLM.
IMPORTANT:
- Do NOT send 'temperature' or any token params (max_tokens / max_completion_tokens) to GPT-5.
- Client uses direct OpenAI (OPENAI_API_KEY).
"""

import os

from openai_client import get_openai_client

from VideoQA_Pipeline.utils import (
    get_video_duration,
    classify_video_type,
)
from VideoQA_Pipeline.build_chunks import build_chunks
from VideoQA_Pipeline.align_multimodal_data import align_multimodal_data
from RAG_Pipeline.RagRetriever_Milvus import RagRetrieverMilvus

# ------------------------- client wiring (direct OpenAI) -------------------------
client = get_openai_client()


class AskVideoQAGPT5:
    """
    Lightweight QA runner:
      - loads existing video artifacts,
      - ensures chunks & embeddings are present,
      - asks GPT-5 for an answer (NO temperature / NO token params).
    """

    def __init__(self, model_name: str = "gpt-5-model", max_tokens: int = 5000):
        # NOTE: max_tokens is kept for signature parity but NOT sent to GPT-5.
        self.model = model_name
        self.max_tokens = max_tokens  # intentionally unused for GPT-5

    def load_video_data(self, video_path: str) -> dict:
        """
        Ensure chunks and embeddings exist for the given video.
        Returns a minimal context dict used to build the prompt.
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_duration = get_video_duration(video_path)

        # Infer video type from extracted frames (if present)
        frame_dir = os.path.join("outputs", "frames", f"{video_name}_frames")
        frame_files = [f for f in os.listdir(frame_dir) if f.endswith(".png")] if os.path.exists(frame_dir) else []
        video_type = classify_video_type(frame_files, video_duration)

        # Ensure chunks are built
        chunk_path = os.path.join("outputs", "chunks", video_name, "all_chunks.jsonl")
        if not os.path.exists(chunk_path):
            print(f"🧱 Chunks not found for video '{video_name}', building...")
            build_chunks(video_name)

        # Ensure embeddings are present in Milvus; build if empty
        retriever = RagRetrieverMilvus(video_name)
        if getattr(retriever.collection, "num_entities", 0) == 0:
            print(f"🔍 Embeddings missing for '{video_name}', rebuilding...")
            retriever.build_from_chunks()

        return {
            "video_name": video_name,
            "video_duration": video_duration,
            "video_type": video_type,
        }

    def answer_question(self, multimodal_context: dict, user_question: str) -> str:
        """
        Compose an aligned prompt and query GPT-5.
        STRICT: do NOT send temperature / token params to GPT-5.
        """
        prompt = align_multimodal_data(multimodal_context, user_question)

        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for VideoQA."},
                {"role": "user", "content": prompt},
            ],
            # GPT-5 strict: do NOT pass temperature or any token params
            # (no temperature, no max_tokens, no max_completion_tokens)
        )
        return resp.choices[0].message.content.strip()

    def run(self, video_path: str, user_question: str) -> str:
        """
        High-level entry: ensure data is ready, then answer a single question.
        """
        context = self.load_video_data(video_path)
        return self.answer_question(context, user_question)
