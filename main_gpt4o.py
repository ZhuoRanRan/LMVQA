import os
import argparse
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

    pipeline = VideoQAPipelineGPT4o(model_name="gpt-4o", max_tokens=args.max_tokens)
    context = pipeline.process_video(args.video)

    if args.verbose:
        print("\n🔍 Video Info")
        print("────────────────────────────")
        print(f"🎥 Name: {context['video_name']}")
        print(f"⏱ Duration: {context['video_duration']} seconds")
        print(f"📊 Type: {context['video_type']}")
        print("────────────────────────────\n")

    answer = pipeline.answer_question(context, args.question)

    print("\n📌 GPT-4o Answer")
    print("────────────────────────────────────────────")
    print(f"🎥 Video: {args.video}")
    print(f"❓ Question: {args.question}")
    print(f"📝 Answer: {answer}")
    print("────────────────────────────────────────────")

if __name__ == "__main__":
    main()
