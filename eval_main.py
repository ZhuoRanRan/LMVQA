import argparse
import traceback
from evaluation_pipeline.videoqa_eval import evaluate_video_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VideoQA predictions with GPT-4o evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--video_name", 
        type=str, 
        required=True,
        help="Video name (e.g., Ciena_Video1)"
    )
    args = parser.parse_args()

    video_name = args.video_name

    print(f"📊 Starting evaluation process for {video_name}...")
    try:
        metrics = evaluate_video_pipeline(video_name)
        if metrics.get("error"):
            print(f"⚠️ Evaluation completed with errors: {metrics['error']}")
    except Exception as e:
        print(f"❌ Fatal error in evaluation: {str(e)}")
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    main()
