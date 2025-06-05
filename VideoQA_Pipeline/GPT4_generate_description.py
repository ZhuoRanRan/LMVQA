import os
import json
import re
import base64
import openai
from PIL import Image
from dotenv import load_dotenv
from VideoQA_constants.prompts import GPT4_PROMPT_TEMPLATE

load_dotenv()
openai.api_key = os.getenv("LITELLM_API_KEY")
openai.base_url = os.getenv("LITELLM_API_BASE")

def encode_image(image_path):
    """Convert image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def generate_description(image_path):
    """
    Generate a detailed caption for a video frame using GPT-4.1 Vision via LiteLLM.
    Prepends a temporal anchor to the output.
    """
    try:
        # Load and resize image
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

        # Extract timestamp
        timestamp = image_path.split("_")[-1].replace(".png", "")
        prompt = GPT4_PROMPT_TEMPLATE.format(timestamp=timestamp)
        image_b64 = encode_image(image_path)

        # Send to GPT-4.1 Vision via LiteLLM
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]
                }
            ],
            max_tokens=4000
        )

        # Extract model output and prepend temporal anchor
        raw_caption = response.choices[0].message.content.strip()
        final_caption = f"This video, during the timestamp {timestamp}, shows: {raw_caption}"

        return {"timestamp": timestamp, "description": final_caption}

    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return {"timestamp": "unknown", "description": "Error processing image"}

def process_video_frames(video_name):
    """
    Process all extracted frames for a video and save structured captions to JSON.
    """
    frame_dir = os.path.join("outputs", "frames", f"{video_name}_frames")
    output_json = os.path.join(frame_dir, "descriptions.json")

    if not os.path.exists(frame_dir):
        print(f"❌ No frames found for video: {video_name}")
        return []

    descriptions = []

    def extract_timestamp(filename):
        match = re.search(r'_(\d+)-(\d+)s\.png$', filename)
        if match:
            return int(match.group(1))
        return float('inf')

    frame_files = sorted(os.listdir(frame_dir), key=extract_timestamp)

    for frame_file in frame_files:
        if frame_file.endswith(".png"):
            frame_path = os.path.join(frame_dir, frame_file)
            desc = generate_description(frame_path)
            descriptions.append(desc)
            print(f"📝 Processed: {frame_file} → {desc['description'][:100]}...")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, indent=4, ensure_ascii=False)

    print(f"✅ Descriptions saved to {output_json}")
    return descriptions
