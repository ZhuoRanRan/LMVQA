import os
import openai
from dotenv import load_dotenv
from VideoQA_Pipeline.utils import get_video_duration, classify_video_type
from VideoQA_Pipeline.align_multimodal_data import load_all_chunks
from ADS_Simulator.prompts import NARRATION_CONSTRUCTION_PROMPT
import json

load_dotenv()
openai.api_key = os.getenv("LITELLM_API_KEY")
openai.base_url = os.getenv("LITELLM_API_BASE")

def get_video_name(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]

def build_video_narration(
    video_path: str,
    model_name: str = "gpt-4o",
    max_tokens: int = 5000
) -> str:
    """
    Generate structured narration based on grouped image descriptions.
    Save final narration to outputs/narration/{video_name}.json.
    """
    video_name = get_video_name(video_path)
    duration = get_video_duration(video_path)

    # Load all grouped visual descriptions
    context = load_all_chunks(video_name)

    prompt = NARRATION_CONSTRUCTION_PROMPT.format(
        video_duration=round(duration, 2),
        retrieved_context=context.strip()
    )

    # Generate narration from GPT
    response = openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=max_tokens
    )

    narration_text = response.choices[0].message.content.strip()

    narration_path = os.path.join("outputs", "narration", f"{video_name}.json")
    os.makedirs(os.path.dirname(narration_path), exist_ok=True)

    if os.path.exists(narration_path):
        os.remove(narration_path)

    # Save to JSON
    with open(narration_path, "w", encoding="utf-8") as f:
        json.dump({"video_name": video_name, "narration": narration_text}, f, indent=2, ensure_ascii=False)

    print(f"✅ Narration saved to {narration_path}")
    return narration_text

