import os
import shutil
from VideoQA_Pipeline.extract_audio import extract_audio
from VideoQA_Pipeline.audio_to_text import audio_to_text
from VideoQA_Pipeline.extracted_frames import extract_video_frames
from VideoQA_Pipeline.GPT4_generate_description import process_video_frames
from VideoQA_Pipeline.build_chunks import build_chunks
from VideoQA_Pipeline.utils import get_video_duration

def information_extraction_builder(video_path: str, overwrite: bool = False) -> str:
    """
    Perform full information extraction: audio transcription, frame description, and chunk merging.
    Intermediate outputs are written under outputs/audio_transcriptions, outputs/frames, etc.

    Args:
        video_path (str): Path to the input video (.mp4)
        overwrite (bool): If True, force regeneration even if intermediate results exist

    Returns:
        str: Path to final all_chunks.jsonl file
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    duration = get_video_duration(video_path)

    # ==== Audio ====
    audio_out_path = os.path.join("outputs", "audio_transcriptions", f"{video_name}_transcription.json")
    if overwrite and os.path.exists(audio_out_path):
        os.remove(audio_out_path)
    if not os.path.exists(audio_out_path):
        audio_path = extract_audio(video_path)
        if audio_path:
            audio_to_text(audio_path)
    else:
        print(f"✅ Skipping audio transcription (already exists): {audio_out_path}")

    # ==== Frames & Visual ====
    frame_dir = os.path.join("outputs", "frames", f"{video_name}_frames")
    visual_out_path = os.path.join(frame_dir, "descriptions.json")
    if overwrite and os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    if not os.path.exists(visual_out_path):
        extract_video_frames(video_path)
        process_video_frames(video_name)
    else:
        print(f"✅ Skipping visual extraction (already exists): {visual_out_path}")

    # ==== Chunks ====
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
