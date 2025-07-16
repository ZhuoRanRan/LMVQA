import os
import shutil
from VideoQA_Pipeline.extracted_frames import extract_video_frames
from ADS_Simulator.image_grouped_captioner import process_ads_grouped_frames
from VideoQA_Pipeline.extract_audio import extract_audio
from VideoQA_Pipeline.audio_to_text import audio_to_text
from VideoQA_Pipeline.build_chunks import build_chunks
from VideoQA_Pipeline.utils import get_video_duration

def videoqa_information_extraction(video_path: str, overwrite: bool = False, use_audio: bool = False) -> str:
    """
    Extract grouped image descriptions and optional audio transcription.
    Save as a unified chunked format for downstream QA.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    duration = get_video_duration(video_path)

    frame_dir = os.path.join("outputs", "frames", f"{video_name}_frames")
    visual_out_path = os.path.join(frame_dir, "descriptions.json")
    audio_json_out = os.path.join("outputs", "audio_transcriptions", f"{video_name}_transcription.json")

    if overwrite and os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    if not os.path.exists(visual_out_path):
        extract_video_frames(video_path)
        process_ads_grouped_frames(video_name)
    else:
        print(f"✅ Skipping visual extraction: {visual_out_path}")

    if use_audio:
        if overwrite and os.path.exists(audio_json_out):
            os.remove(audio_json_out)
        if not os.path.exists(audio_json_out):
            audio_path = extract_audio(video_path)
            if audio_path:
                audio_to_text(audio_path)
        else:
            print(f"✅ Skipping audio transcription: {audio_json_out}")

    chunk_out_dir = os.path.join("outputs", "chunks", video_name)
    chunk_out_path = os.path.join(chunk_out_dir, "all_chunks.jsonl")
    if overwrite and os.path.exists(chunk_out_path):
        os.remove(chunk_out_path)
    if not os.path.exists(chunk_out_path):
        output_path = build_chunks(video_name)
    else:
        print(f"✅ Skipping chunk building: {chunk_out_path}")
        output_path = chunk_out_path

    return output_path
