import os
import csv
import json
import pandas as pd
from evaluation_pipeline.geval_eval import run_geval_evaluation

def _save_summary_csv(result_dir: str, metrics: dict):
    """Write overall metrics to one CSV file."""
    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(result_dir, "summary_metrics.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])

def evaluate_video_pipeline(video_name: str):
    """
    Evaluate {video_name}_predictions.csv and save:
      - detailed_scores.csv (per item)
      - summary_metrics.csv (overall only)
    """
    prediction_file = os.path.join("Ciena_Eval_Datasets", f"{video_name}_predictions.csv")
    result_dir = os.path.join("Ciena_Eval_Results", f"{video_name}_results")
    os.makedirs(result_dir, exist_ok=True)

    log_path = os.path.join(result_dir, "geval_log.txt")
    detailed_csv_path = os.path.join(result_dir, "detailed_scores.csv")

    if not os.path.exists(prediction_file):
        raise FileNotFoundError(
            f"❌ Missing predictions: {prediction_file}\n"
            f"➡️ Run qa_generate_gpt4o.py first."
        )

    try:
        df = pd.read_csv(prediction_file)

        required_cols = ["question", "prediction", "ground_truth"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ Missing columns: {missing_cols}. Required: {required_cols}")

        questions = df["question"].astype(str).tolist()
        predictions = df["prediction"].astype(str).tolist()
        ground_truths = df["ground_truth"].astype(str).tolist()

        print(f"📋 Loaded {len(questions)} QA pairs from {prediction_file}")

        metrics = run_geval_evaluation(
            questions=questions,
            predictions=predictions,
            ground_truths=ground_truths,
            output_txt_path=log_path,
            detailed_csv_path=detailed_csv_path,
            threshold=float(os.getenv("GEVAL_THRESHOLD", "5.0")),
        )

        if metrics.get("status") == "success":
            _save_summary_csv(result_dir, metrics)
            print(f"✅ Detailed results: {detailed_csv_path}")
            print(f"📝 Summary CSV: {os.path.join(result_dir, 'summary_metrics.csv')}")
            print(
                f"📈 accuracy: {metrics['accuracy']:.3f}, "
                f"precision: {metrics['precision']:.3f}, "
                f"recall: {metrics['recall']:.3f}, "
                f"f1: {metrics['f1']:.3f} (threshold={metrics.get('threshold')})"
            )
        else:
            print(f"⚠️ Evaluation error: {metrics.get('error')}")

        return metrics

    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        return {"status": "failed", "error": str(e)}
