import cv2
import subprocess
import os
import json

def get_ffmpeg_total_frames(video_path):
    """
    Use FFmpeg to get the correct total frame count to avoid OpenCV errors.
    
    Args:
        video_path (str): Path to the video file.

    Returns:
        int: Correct total number of frames.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0", video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0  # If FFmpeg fails, return 0

def get_video_duration(video_path):
    """
    Get the duration of a video in seconds using FFmpeg.

    Args:
        video_path (str): Path to the video file.

    Returns:
        float: Duration of the video in seconds.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # ✅ Use get_ffmpeg_total_frames() for accuracy
    total_frames = get_ffmpeg_total_frames(video_path)
    
    duration = total_frames / fps if fps > 0 else 0
    return round(duration, 2)  # Return duration in seconds (rounded)


def classify_video_type(frames, duration):
    """
    Classify video type based on extracted frames and video duration.
    - If video < 60s and only 1 frame → "low-action".
    - If video ≥ 60s and frames < duration / 60 (less than 1 frame per minute) → "low-action".
    - Otherwise → "normal-action".
    """
    if duration < 60 and len(frames) == 1:
        return "low-action"
    if duration >= 60 and len(frames) < (duration / 60):  
        return "low-action"
    return "normal-action"
   

def load_audio_transcription(video_name):
    """
    Loads the saved Whisper transcription JSON file.

    Args:
        video_name (str): Name of the video file (without extension).

    Returns:
        list: List of transcription segments formatted with timestamps.
    """
    json_path = os.path.join("Audio_Transcriptions", f"{video_name}_transcription.json")

    if not os.path.exists(json_path):
        print(f"❌ Transcription JSON file not found: {json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_frame_descriptions(video_name):
    """ 
    Loads Phi-4 generated frame descriptions and formats them with timestamps.

    Args:
        video_name (str): The name of the video file (without extension).

    Returns:
        str: A structured description of frames with timestamps.
    """
    desc_path = os.path.join("Frames", f"{video_name}_frames", "descriptions.json")
    if not os.path.exists(desc_path):
        print(f"Frame descriptions not found: {desc_path}")
        return "No frame descriptions available."
    
    with open(desc_path, "r", encoding="utf-8") as f:
        descriptions = json.load(f)

    video_summary = "\n".join(
        [f"Time {desc['timestamp']}: {desc['description']}" for desc in descriptions]
    )
    return video_summary



def format_audio_with_timestamps(audio_data):
    """
    Formats the transcription into a structured text output.

    Args:
        audio_data (list): List of dictionaries with "timestamp" and "text".

    Returns:
        str: A formatted string where each line contains timestamp and text.
    """
    if not audio_data:
        return "No detailed audio transcript available."

    return "\n".join([f"Time {seg['timestamp']}: {seg['text']}" for seg in audio_data])


def extract_answer_from_response(response: str) -> str:
    """
    Extract the actual AI answer from the full LLM output prompt+answer block.
    """
    if "**AI Answer:**" in response:
        return response.split("**AI Answer:**")[-1].strip()
    return response.strip()