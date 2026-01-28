# main.py
import os
import json
import argparse
from time import perf_counter
from VideoQA_Pipeline.videoQA_pipeline_gpt4o import VideoQAPipelineGPT4o


def main():
    parser = argparse.ArgumentParser(description="Run VideoQA pipeline with GPT-4o on a new video.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--question", type=str, default="Describe the video", help="Question to ask about the video.")
    parser.add_argument("--max_tokens", type=int, default=500, help="Max tokens for GPT-4o response.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode.")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return

    pipeline = VideoQAPipelineGPT4o(model_name="gpt-4o-model", max_tokens=args.max_tokens)

    # --- timing only the preprocessing (process_video), excluding QA ---
    t0 = perf_counter()
    context = pipeline.process_video(args.video)
    preprocess_seconds = perf_counter() - t0

    # --- save preprocess timing JSON ---
    results_dir = os.path.join("results", "Process_time")
    os.makedirs(results_dir, exist_ok=True)
    video_name = context.get("video_name") or os.path.splitext(os.path.basename(args.video))[0]
    timing_path = os.path.join(results_dir, f"{video_name}_preprocess_timing.json")
    payload = {
        "video_name": video_name,
        "preprocess_seconds": round(preprocess_seconds, 4),
        "note": "Timing covers process_video only (audio/video extraction, chunking, embeddings/RAG build). QA excluded.",
    }
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"⏱️  Preprocess timing saved to {timing_path}")

    if args.verbose:
        print("\n🔍 Video Info")
        print("────────────────────────────")
        print(f"🎥 Name: {context.get('video_name')}")
        print(f"⏱ Duration: {context.get('video_duration')} seconds")
        print(f"📊 Type: {context.get('video_type')}")
        print("────────────────────────────\n")

    # QA phase remains the same, but not counted in timing
    answer = pipeline.answer_question(context, args.question)

    print("\n📌 GPT-4o Answer")
    print("────────────────────────────────────────────")
    print(f"🎥 Video: {args.video}")
    print(f"❓ Question: {args.question}")
    print(f"📝 Answer: {answer}")
    print("────────────────────────────────────────────")


if __name__ == "__main__":
    main()

