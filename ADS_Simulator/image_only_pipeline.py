import os
import shutil
from VideoQA_Pipeline.extracted_frames import extract_video_frames
from ADS_Simulator.image_grouped_captioner import process_ads_grouped_frames
from VideoQA_Pipeline.build_chunks import build_chunks
from VideoQA_Pipeline.utils import get_video_duration

def image_only_information_extraction(video_path: str, overwrite: bool = False) -> str:
    """
    Only use visual information to extract frame-level descriptions and chunk them.
    Skips audio transcription entirely. Stores output in `outputs/` directory.

    Args:
        video_path (str): Path to input video (.mp4)
        overwrite (bool): If True, force regeneration even if results already exist

    Returns:
        str: Path to final all_chunks.jsonl
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    duration = get_video_duration(video_path)

    # ==== Frames & Descriptions ====
    frame_dir = os.path.join("outputs", "frames", f"{video_name}_frames")
    visual_out_path = os.path.join(frame_dir, "descriptions.json")

    if overwrite and os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)

    if not os.path.exists(visual_out_path):
        extract_video_frames(video_path)
        process_ads_grouped_frames(video_name)
    else:
        print(f"✅ Skipping visual extraction (already exists): {visual_out_path}")

    chunk_out_dir = os.path.join("outputs", "chunks", video_name)
    chunk_out_path = os.path.join(chunk_out_dir, "all_chunks.jsonl")
    if overwrite and os.path.exists(chunk_out_path):
        os.remove(chunk_out_path)
    if not os.path.exists(chunk_out_path):
        output_path = build_chunks(video_name)
    else:
        print(f"✅ Skipping chunk building (already exists): {chunk_out_path}")
        output_path = chunk_out_path

    return output_path
