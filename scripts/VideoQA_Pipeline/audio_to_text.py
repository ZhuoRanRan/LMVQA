import whisper
import json
import os

# model = whisper.load_model("small")
# model = whisper.load_model("medium")
# model = whisper.load_model("large-v3")
model = whisper.load_model("large-v3-turbo")


def audio_to_text(audio_path):
    try:
        result = model.transcribe(audio_path, word_timestamps=False)
        segments = result.get("segments", [])
        if not isinstance(segments, list):
            return None

        video_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_dir = os.path.join("outputs", "audio_transcriptions")
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{video_name}_transcription.json")

        data = [
            {"timestamp": f"{round(s['start'], 2)}-{round(s['end'], 2)}s", "text": s["text"]}
            for s in segments
        ]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"✅ Transcription saved to {json_path}")
        return json_path
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None
