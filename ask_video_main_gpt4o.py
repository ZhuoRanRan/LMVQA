import os
import argparse
from VideoQA_Pipeline.askVideoQA_gpt4o import AskVideoQAGPT4o

def main():
    parser = argparse.ArgumentParser(description="Ask GPT-4o about a processed video.")
    parser.add_argument("--video", type=str, required=False, help="Path to the video file (used to derive video_name and duration). Optional if you use --video_name and only query Milvus.")
    parser.add_argument("--video_name", type=str, required=False, help="Video name used to locate Milvus collection `videoqa_<video_name>`. Useful on machines without the original .mp4.")
    parser.add_argument("--video_duration", type=float, required=False, help="Optional duration in seconds (only used for prompt metadata when --video is not provided).")
    parser.add_argument("--question", type=str, required=True, help="Question to ask.")
    parser.add_argument("--max_tokens", type=int, default=5000, help="Max tokens for GPT-4o response.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")

    args = parser.parse_args()

    if not args.video and not args.video_name:
        print("❌ You must provide either --video or --video_name.")
        return
    if args.video and (not os.path.exists(args.video)):
        if not args.video_name:
            print(f"❌ Video file not found: {args.video}")
            return
        print(f"⚠️ Video file not found: {args.video}. Proceeding with --video_name={args.video_name} (Milvus-only query).")

    qa = AskVideoQAGPT4o(model_name="gpt-4o-model", max_tokens=args.max_tokens)
    context = qa.load_video_data(video_path=args.video, video_name=args.video_name, video_duration=args.video_duration)

    if args.verbose:
        print("\n🔍 Video Info")
        print("────────────────────────────")
        print(f"🎥 Name: {context['video_name']}")
        print(f"⏱ Duration: {context['video_duration']} seconds")
        print(f"📊 Type: {context['video_type']}")
        print("────────────────────────────\n")

    answer = qa.answer_question(context, args.question)

    print("\n📌 GPT-4o Answer")
    print("────────────────────────────────────────────")
    if args.video:
        print(f"🎥 Video: {args.video}")
    else:
        print(f"🎥 Video name: {context['video_name']}")
    print(f"❓ Question: {args.question}")
    print(f"📝 Answer: {answer}")
    print("────────────────────────────────────────────")

if __name__ == "__main__":
    main()
