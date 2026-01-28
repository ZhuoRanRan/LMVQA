import os
import json
import re

def parse_start_time(timestamp):
    """Extracts start time as float from '12.3-45.6s' or '297–435s'."""
    match = re.match(r"(\d+(?:\.\d+)?)", timestamp)
    return float(match.group(1)) if match else 0.0

def build_chunks(video_name):
    """
    Merge audio transcription and visual frame descriptions for a given video into a sorted `.jsonl` file.
    
    Args:
        video_name (str): The name of the video (e.g., 'Lecture1').

    Returns:
        str: Path to the saved `all_chunks.jsonl` file.
    """
    audio_path = os.path.join("outputs", "audio_transcriptions", f"{video_name}_transcription.json")
    visual_path = os.path.join("outputs", "frames", f"{video_name}_frames", "descriptions.json")
    output_dir = os.path.join("outputs", "chunks", video_name)
    output_path = os.path.join(output_dir, "all_chunks.jsonl")
    
    os.makedirs(output_dir, exist_ok=True)

    all_chunks = []

    # ✅ Load audio
    if os.path.exists(audio_path):
        with open(audio_path, "r", encoding="utf-8") as f:
            audio_data = json.load(f)
            for entry in audio_data:
                all_chunks.append({
                    "id": f"aud_{entry['timestamp'].replace('.', '_').replace('-', '_').replace('s','')}",
                    "source": "audio",
                    "timestamp": entry["timestamp"],
                    "text": entry["text"],
                    "start_time": parse_start_time(entry["timestamp"])
                })
    else:
        print(f"⚠️ Audio file not found: {audio_path}")

    # ✅ Load visual
    if os.path.exists(visual_path):
        with open(visual_path, "r", encoding="utf-8") as f:
            visual_data = json.load(f)
            for entry in visual_data:
                all_chunks.append({
                    "id": f"vis_{entry['timestamp'].replace('-', '_').replace('s','')}",
                    "source": "visual",
                    "timestamp": entry["timestamp"],
                    "text": entry["description"],
                    "start_time": parse_start_time(entry["timestamp"])
                })
    else:
        print(f"⚠️ Visual file not found: {visual_path}")

    # ✅ Sort and clean
    all_chunks.sort(key=lambda x: x["start_time"])
    for chunk in all_chunks:
        chunk.pop("start_time", None)

    # ✅ Save as JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"✅ Merged {len(all_chunks)} chunks saved to {output_path}")
    return output_path
