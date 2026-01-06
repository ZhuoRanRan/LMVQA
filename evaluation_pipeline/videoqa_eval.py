# -*- coding: utf-8 -*-
"""
Evaluate a video's predictions via GEval (0..1 scale).
Outputs:
  - detailed_scores.csv (per item)
  - summary_metrics.csv (correctness mean & pass rate)
If predictions CSV contains 'retrieval_context' column, split by '|||'.
"""

import os
import csv
import json
import argparse
import pandas as pd
from typing import Optional, Dict, Any, List

from evaluation_pipeline.geval_eval import run_geval_evaluation

# (keep your Ciena paths as provided)
# DEFAULT_PRED_DIR = "Ciena_Eval_Datasets"
# DEFAULT_RESULTS_BASE_DIR = "Ciena_Eval_Results"
DEFAULT_PRED_DIR = "Lecture_Eval_Datasets"
DEFAULT_RESULTS_BASE_DIR = "Lecture_Eval_Results"


def _save_summary_csv(result_dir: str, metrics: Dict[str, Any]) -> None:
    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(result_dir, "summary_metrics.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])


def _split_retrieval_context(col: pd.Series) -> List[List[str]]:
    def split_cell(cell) -> List[str]:
        if isinstance(cell, str) and cell.strip():
            return [c.strip() for c in cell.split("|||") if c.strip()]
        return []
    return col.map(split_cell).tolist()


def evaluate_video_pipeline(
    video_name: str,
    pred_dir: str = DEFAULT_PRED_DIR,
    results_base_dir: str = DEFAULT_RESULTS_BASE_DIR,
    correctness_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    prediction_file = os.path.join(pred_dir, f"{video_name}_predictions.csv")
    result_dir = os.path.join(results_base_dir, f"{video_name}_results")
    os.makedirs(result_dir, exist_ok=True)

    log_path = os.path.join(result_dir, "geval_log.txt")
    detailed_csv_path = os.path.join(result_dir, "detailed_scores.csv")

    if not os.path.exists(prediction_file):
        msg = (
            f"❌ Missing predictions: {prediction_file}\n"
            f"➡️ Generate predictions first (e.g., qa_generate_gpt4o.py)."
        )
        print(msg)
        return {"status": "failed", "error": msg}

    if correctness_threshold is None:
        correctness_threshold = float(os.getenv("GEVAL_CORRECTNESS_THRESHOLD", "0.5"))

    try:
        df = pd.read_csv(prediction_file)

        required_cols = ["question", "prediction", "ground_truth"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ Missing columns: {missing_cols}. Required: {required_cols}")

        questions = df["question"].astype(str).tolist()
        predictions = df["prediction"].astype(str).tolist()
        ground_truths = df["ground_truth"].astype(str).tolist()

        retrieval_contexts = None
        if "retrieval_context" in df.columns:
            retrieval_contexts = _split_retrieval_context(df["retrieval_context"])

        print(f"📋 Loaded {len(questions)} QA pairs from {prediction_file}")
        print(f"🧪 Using correctness_threshold (0..1): {correctness_threshold}")

        metrics = run_geval_evaluation(
            questions=questions,
            predictions=predictions,
            ground_truths=ground_truths,
            output_txt_path=log_path,
            detailed_csv_path=detailed_csv_path,
            correctness_threshold=correctness_threshold,
            retrieval_contexts=retrieval_contexts,
        )

        if metrics.get("status") == "success":
            _save_summary_csv(result_dir, metrics)
            print(f"✅ Detailed results: {detailed_csv_path}")
            print(f"📝 Summary CSV: {os.path.join(result_dir, 'summary_metrics.csv')}")
            print(
                "📈 correctness_mean: {m:.3f}, pass_rate(>=thr): {p:.3f} (thr={thr})".format(
                    m=metrics.get("correctness_mean", 0.0),
                    p=metrics.get("correctness_pass_rate(>=thr)", 0.0),
                    thr=metrics.get("correctness_threshold"),
                )
            )
            # --- context-related prints are no longer needed; keeping them commented for future use ---
            # if "context_precision_mean" in metrics:
            #     print("🔎 context_precision_mean: {:.3f}, pass_rate>=0.5: {:.3f}".format(
            #         metrics["context_precision_mean"],
            #         metrics["context_precision_pass_rate(>=0.5)"]
            #     ))
            # if "context_recall_mean" in metrics:
            #     print("🔎 context_recall_mean: {:.3f}, pass_rate>=0.5: {:.3f}".format(
            #         metrics["context_recall_mean"],
            #         metrics["context_recall_pass_rate(>=0.5)"]
            #     ))
            # if "truthfulness_mean" in metrics:
            #     print("🔎 truthfulness_mean: {:.3f}, pass_rate>=0.5: {:.3f}".format(
            #         metrics["truthfulness_mean"],
            #         metrics["truthfulness_pass_rate(>=0.5)"]
            #     ))
        else:
            print(f"⚠️ Evaluation error: {metrics.get('error')}")

        return metrics

    except Exception as e:
        msg = f"❌ Evaluation failed: {str(e)}"
        print(msg)
        return {"status": "failed", "error": str(e)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GEval evaluation for one or more videos (0..1 scale).")
    parser.add_argument(
        "--videos", nargs="+", required=False,
        default=["Lecture1", "Lecture2"],
        help="Video names (without extension). Default: Lecture1 Lecture2"
    )
    parser.add_argument(
        "--pred_dir", default=DEFAULT_PRED_DIR,
        help=f"Directory containing <video>_predictions.csv files (default: {DEFAULT_PRED_DIR})"
    )
    parser.add_argument(
        "--out_dir", default=DEFAULT_RESULTS_BASE_DIR,
        help=f"Base directory to write results (default: {DEFAULT_RESULTS_BASE_DIR})"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Correctness pass threshold on a 0–1 GEval scale (default from env GEVAL_CORRECTNESS_THRESHOLD or 0.5)"
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    overall = {}
    for v in args.videos:
        print(f"\n===================== {v} =====================")
        m = evaluate_video_pipeline(
            video_name=v,
            pred_dir=args.pred_dir,
            results_base_dir=args.out_dir,
            correctness_threshold=args.threshold,
        )
        overall[v] = m

    try:
        os.makedirs(args.out_dir, exist_ok=True)
        summ_path = os.path.join(args.out_dir, "evaluation_summary.json")
        with open(summ_path, "w", encoding="utf-8") as f:
            json.dump(overall, f, ensure_ascii=False, indent=2)
        print(f"\n🧾 Wrote combined summary: {summ_path}")
    except Exception as e:
        print(f"⚠️ Could not write combined summary JSON: {e}")


if __name__ == "__main__":
    main()
