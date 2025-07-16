import os
import re
import json
from dotenv import load_dotenv
import openai
import base64
from PIL import Image
from ADS_Simulator.prompts import ADS_PROMPT_TEMPLATE_GROUPED

load_dotenv()
openai.api_key = os.getenv("LITELLM_API_KEY")
openai.base_url = os.getenv("LITELLM_API_BASE")

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def generate_ads_description_grouped(image_paths, timestamp_ranges):
    intro = (
        "You are analyzing a sequence of frames from an autonomous driving simulation.\n"
        "Each frame below corresponds to a specific time range in the video.\n"
        "Use this timing information to reason about driving behavior (e.g., stop, turn, acceleration).\n\n"
    )
    intro += "Frame timestamps:\n"
    for ts in timestamp_ranges:
        intro += f"- Frame from {ts} seconds\n"

    intro += (
        "\nBased on the images and their timestamps, describe what happened during this interval. "
        "Summarize driving actions, transitions, and detect any unusual or illegal behavior (e.g., failing to stop, sudden lane change, ignoring traffic signals)."
    )

    prompt = ADS_PROMPT_TEMPLATE_GROUPED + "\n\n" + intro

    image_contents = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        image_b64 = encode_image(path)
        image_contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"}
        })

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": prompt}] + image_contents
        }],
        temperature=0.3,
        max_tokens=2000
    )

    return response.choices[0].message.content.strip()


def process_ads_grouped_frames(video_name, group_size=10):
    frame_dir = os.path.join("outputs", "frames", f"{video_name}_frames")
    out_path = os.path.join(frame_dir, "descriptions.json")

    frames = [f for f in os.listdir(frame_dir) if f.endswith(".png")]
    frames.sort(key=lambda x: int(re.search(r"_(\d+)", x).group(1)))

    descriptions = []

    for i in range(0, len(frames), group_size):
        group = frames[i:i+group_size]
        image_paths = [os.path.join(frame_dir, f) for f in group]
        timestamp_ranges = [f.split("_")[-1].replace(".png", "") for f in group]
        timestamp_span = f"{timestamp_ranges[0]}–{timestamp_ranges[-1]}"
        try:
            desc = generate_ads_description_grouped(image_paths, timestamp_ranges)
            descriptions.append({
                "timestamp": timestamp_span,
                "description": desc
            })
            print(f"✅ Processed {timestamp_span}")
        except Exception as e:
            print(f"❌ Failed to process {timestamp_span}: {e}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)

    print(f"✅ All grouped frame descriptions saved to {out_path}")
    return descriptions
