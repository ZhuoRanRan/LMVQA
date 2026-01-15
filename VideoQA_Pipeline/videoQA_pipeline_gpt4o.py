import os

from openai_client import get_openai_client
from VideoQA_Pipeline.extract_audio import extract_audio
from VideoQA_Pipeline.audio_to_text import audio_to_text
from VideoQA_Pipeline.extracted_frames import extract_video_frames
from VideoQA_Pipeline.GPT4_generate_description import process_video_frames
from VideoQA_Pipeline.build_chunks import build_chunks
from VideoQA_Pipeline.utils import get_video_duration, classify_video_type
from VideoQA_Pipeline.align_multimodal_data import align_multimodal_data
from RAG_Pipeline.RagRetriever_Milvus import RagRetrieverMilvus

client = get_openai_client()

class VideoQAPipelineGPT4o:
    def __init__(self, model_name="gpt-4o-model", max_tokens=5000):
        self.model = model_name
        self.max_tokens = max_tokens

    def process_video(self, video_path):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_duration = get_video_duration(video_path)

        # Audio
        audio_path = extract_audio(video_path)
        if audio_path and os.path.exists(audio_path):
            audio_to_text(audio_path)

        # Visual frames + description
        frames, _, video_type = extract_video_frames(video_path)
        process_video_frames(video_name)

        # Merge
        build_chunks(video_name)

        # Embedding
        retriever = RagRetrieverMilvus(video_name)
        if retriever.collection.num_entities == 0:
            retriever.build_from_chunks()

        return {
            "video_name": video_name,
            "video_duration": video_duration,
            "video_type": video_type,
        }

    def answer_question(self, multimodal_context, user_question):
        prompt = align_multimodal_data(multimodal_context, user_question)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content.strip()

    def run(self, video_path, user_question):
        context = self.process_video(video_path)
        return self.answer_question(context, user_question)
