import os
import openai
from dotenv import load_dotenv
from VideoQA_Pipeline.align_multimodal_data import load_all_chunks
from VideoQA_Pipeline.utils import get_video_duration
from ActionVideoQA_Pipeline.prompts import QA_ANSWERING_PROMPT

load_dotenv()
openai.api_key = os.getenv("LITELLM_API_KEY")
openai.base_url = os.getenv("LITELLM_API_BASE")

def get_video_name(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]

def answer_video_question(video_path: str, query: str, model_name="gpt-4o", max_tokens=1000) -> str:
    """
    Generate an answer to the user's question based on visual and (optional) audio context
    extracted from the input video.
    """
    video_name = get_video_name(video_path)
    context = load_all_chunks(video_name)
    duration = get_video_duration(video_path)

    prompt = QA_ANSWERING_PROMPT.format(
        user_query=query.strip(),
        extracted_context=context.strip(),
        video_duration=round(duration, 2)
    )

    response = openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content.strip()
