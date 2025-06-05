import os
import argparse
from VideoQA_Pipeline.askVideoQA_gpt4o import AskVideoQAGPT4o

def main():
    parser = argparse.ArgumentParser(description="Ask GPT-4o about a processed video.")
    parser.add_argument("--video", type=str, required=True, help="Path to the pre-processed video file.")
    parser.add_argument("--question", type=str, required=True, help="Question to ask.")
    parser.add_argument("--max_tokens", type=int, default=5000, help="Max tokens for GPT-4o response.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return

    qa = AskVideoQAGPT4o(model_name="gpt-4o", max_tokens=args.max_tokens)
    context = qa.load_video_data(args.video)

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
    print(f"🎥 Video: {args.video}")
    print(f"❓ Question: {args.question}")
    print(f"📝 Answer: {answer}")
    print("────────────────────────────────────────────")

if __name__ == "__main__":
    main()
