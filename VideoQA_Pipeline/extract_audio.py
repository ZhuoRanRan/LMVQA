import os
import ffmpeg

def extract_audio(video_path):
    video_name = os.path.basename(video_path).split('.')[0]
    output_dir = os.path.join("outputs", "extracted_audios")
    os.makedirs(output_dir, exist_ok=True)
    output_audio_path = os.path.join(output_dir, f"{video_name}.wav")

    if os.path.exists(output_audio_path):
        print(f"⚠️ Audio already exists at {output_audio_path}, skipping.")
        return output_audio_path

    try:
        ffmpeg.input(video_path).output(output_audio_path, format="wav").run(
            overwrite_output=True, quiet=True, capture_stderr=True
        )
        print(f"✅ Audio extracted to {output_audio_path}")
        return output_audio_path
    except Exception as e:
        print(f"❌ Error extracting audio: {e}")
        return None
