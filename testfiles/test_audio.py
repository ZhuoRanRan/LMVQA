from VideoQA_Pipeline.extract_audio import extract_audio
from VideoQA_Pipeline.audio_to_text import audio_to_text

video_path = "input_videos/Lecture1.mp4"

# Step 1: Extract audio from video
audio_path = extract_audio(video_path)

# Step 2: Transcribe audio to text
if audio_path:
    json_path = audio_to_text(audio_path)
    print("\n📊 Transcription Complete!")
    print(f"Audio file: {audio_path}")
    print(f"Transcription JSON: {json_path}")
else:
    print("❌ Audio extraction failed.")
