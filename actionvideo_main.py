import argparse
from ActionVideoQA_Pipeline.videoqa_information_extraction import videoqa_information_extraction
from ActionVideoQA_Pipeline.videoqa_answering import answer_video_question

def run_pipeline(video_path: str, question: str, use_audio: bool = False, overwrite: bool = False):
    print("🔍 Extracting information from video...")
    chunk_path = videoqa_information_extraction(video_path, overwrite=overwrite, use_audio=use_audio)
    print(f"✅ Information extraction complete. Chunks saved at: {chunk_path}")

    print("\n💬 Generating answer to the question...")
    answer = answer_video_question(video_path, question)
    print("\n🎯 Answer:")
    print(answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Action VideoQA Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--question", type=str, required=True, help="User question about the video")
    parser.add_argument("--use_audio", action="store_true", default=False, help="Enable audio transcription (default: False)")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Force re-extraction (default: False)")

    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        question=args.question,
        use_audio=args.use_audio,
        overwrite=args.overwrite
    )
