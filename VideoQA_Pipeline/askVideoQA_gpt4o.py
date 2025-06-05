import os
import openai
from dotenv import load_dotenv
from VideoQA_Pipeline.utils import (
    get_video_duration,
    classify_video_type,
)
from VideoQA_Pipeline.build_chunks import build_chunks
from VideoQA_Pipeline.align_multimodal_data import align_multimodal_data
from RAG_Pipeline.RagRetriever_Milvus import RagRetrieverMilvus

load_dotenv()
openai.api_key = os.getenv("LITELLM_API_KEY")
openai.base_url = os.getenv("LITELLM_API_BASE")

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
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content.strip()
