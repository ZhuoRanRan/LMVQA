import os
import csv
import argparse
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
 
BASE_DIR = "Lecture_Eval_Results"
 
# Per-lecture histogram
def _load_scores(video: str) -> pd.Series:
    """Read detailed_scores.csv and return correctness scores normalized to 0–1."""
    path = os.path.join(BASE_DIR, f"{video}_results", "detailed_scores.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing detailed file: {path}")
    df = pd.read_csv(path)
    cmap = {c.lower(): c for c in df.columns}
    s_col = cmap.get("score")
    if s_col is None:
        raise ValueError(f"{path} must contain 'score' column. Got: {list(df.columns)}")
    s = pd.to_numeric(df[s_col], errors="coerce").fillna(0.0)
    # auto scale 0–10 -> 0–1
    if s.max() > 1.0:
        s = s / 10.0
    return s
 
def _plot_hist_for_video(video: str, threshold: float = 0.5):
    """Draw 0–1 histogram with dashed threshold and count labels; save to score_histogram.png."""
    scores = _load_scores(video)
    out_dir = os.path.join(BASE_DIR, f"{video}_results")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "score_histogram.png")
 
    bins = [i/10 for i in range(0, 11)]  # 0..1 step 0.1
    x_ticks = [i/10 for i in range(0, 11)]
 
    fig, ax = plt.subplots(figsize=(10, 6))
    n, b, _ = ax.hist(scores.values, bins=bins, edgecolor="black", alpha=0.8)
 
    # threshold
    ax.axvline(threshold, linestyle="--", linewidth=1.2, color="gray")
 
    # counts
    ymax = max(n) if len(n) else 1
    for count, left, right in zip(n, b[:-1], b[1:]):
        if count > 0:
            xpos = (left + right) / 2
            ax.text(xpos, count + ymax * 0.02, f"{int(count)}",
                    ha="center", va="bottom", fontsize=11)
 
    ax.set_xticks(x_ticks)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, max(ymax * 1.25, ymax + 1))
    ax.set_xlabel("Correctness Score (0–1)")
    ax.set_ylabel("Count")
    ax.set_title(f"{video} – Correctness Score Histogram")
 
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"🖼️ Saved histogram: {out_png}")
 
def make_all_histograms(videos: List[str], threshold: float):
    for v in videos:
        try:
            _plot_hist_for_video(v, threshold=threshold)
        except Exception as e:
            print(f"⚠️ Skip {v}: {e}")
 
# Overview charts (acc/rec/truth)
def _read_summary_csv(path: str) -> Dict[str, float]:
    data: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # header metric,value
        for row in reader:
            if len(row) < 2:
                continue
            k, v = row[0].strip(), row[1].strip()
            try:
                data[k] = float(v)
            except ValueError:
                pass
    return data
 
def load_summary_metrics(video_name: str) -> Dict[str, float]:
    """
    Map to accuracy/recall/truthfulness (0–1). Robust to key variations:
      accuracy     := correctness_pass_rate(>=thr) or correctness_mean
      recall       := context_recall_mean or context_recall_pass_rate(>=0.5)
      truthfulness := truthfulness_mean or truthfulness_pass_rate(>=0.5) or truthfulness
    """
    path = os.path.join(BASE_DIR, f"{video_name}_results", "summary_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing summary file: {path}")
    raw = _read_summary_csv(path)
 
    accuracy = raw.get("correctness_pass_rate(>=thr)", raw.get("correctness_mean", 0.0))
    recall = raw.get("context_recall_mean", raw.get("context_recall_pass_rate(>=0.5)", 0.0))
    truth = raw.get("truthfulness_mean",
                    raw.get("truthfulness_pass_rate(>=0.5)",
                            raw.get("truthfulness", 0.0)))
 
    # soft warnings
    if "correctness_pass_rate(>=thr)" not in raw and "correctness_mean" not in raw:
        print(f"⚠️  {video_name}: accuracy not found; using 0.0")
    if "context_recall_mean" not in raw and "context_recall_pass_rate(>=0.5)" not in raw:
        print(f"⚠️  {video_name}: recall not found; using 0.0")
    if ("truthfulness_mean" not in raw and
        "truthfulness_pass_rate(>=0.5)" not in raw and
        "truthfulness" not in raw):
        print(f"⚠️  {video_name}: truthfulness not found; using 0.0")
 
    return {"accuracy": float(accuracy), "recall": float(recall), "truthfulness": float(truth)}
 
def aggregate_summary(videos: List[str]) -> pd.DataFrame:
    rows = []
    for v in videos:
        try:
            rows.append({"video": v, **load_summary_metrics(v)})
        except FileNotFoundError as e:
            print(f"⚠️ {e}")
    if not rows:
        raise RuntimeError("No summary metrics found.")
    return pd.DataFrame(rows)
 
def plot_accuracy_bars(df: pd.DataFrame, out_png: str):
    x = range(len(df))
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bars = ax.bar(x, df["accuracy"].values, width=0.55)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["video"].tolist(), rotation=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy")
    ax.set_title("QA Evaluation – Accuracy")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    for rect, val in zip(bars, df["accuracy"].values):
        ax.text(rect.get_x() + rect.get_width()/2.0,
                min(val + 0.035, 1.06),
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
 
def plot_recall_truthfulness_bars(df: pd.DataFrame, out_png: str):
    metrics = ["recall", "truthfulness"]
    labels = {"recall": "Recall", "truthfulness": "Truthfulness"}
    x = range(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for i, m in enumerate(metrics):
        ax.bar([p + i*width for p in x], df[m].values, width, label=labels[m])
    ax.set_xticks([p + width/2 for p in x])
    ax.set_xticklabels(df["video"].tolist(), rotation=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("QA Evaluation – Recall & Truthfulness")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    for i, m in enumerate(metrics):
        for j, val in enumerate(df[m].values):
            ax.text(j + i*width,
                    min(val + 0.035, 1.06),
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
 
def make_overview_plots(videos: List[str]):
    df = aggregate_summary(videos)
    os.makedirs(BASE_DIR, exist_ok=True)
    csv_path = os.path.join(BASE_DIR, "metrics_overview_acc_rec_truth.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    plot_accuracy_bars(df, os.path.join(BASE_DIR, "metrics_overview_accuracy.png"))
    plot_recall_truthfulness_bars(df, os.path.join(BASE_DIR, "metrics_overview_recall_truthfulness.png"))
    print(f"✅ Saved CSV: {csv_path}")
    print(f"✅ Saved plot: {os.path.join(BASE_DIR, 'metrics_overview_accuracy.png')}")
    print(f"✅ Saved plot: {os.path.join(BASE_DIR, 'metrics_overview_recall_truthfulness.png')}")
 
# CLI
def main():
    global BASE_DIR
    parser = argparse.ArgumentParser(
        description="Make per-lecture correctness histograms + overview charts."
    )
    parser.add_argument("--videos", nargs="+",
                        default=["Lecture1"],
                        help="Video names without extension")
    parser.add_argument("--out_dir", default=BASE_DIR, help="Base results dir")
    parser.add_argument("--threshold", type=float, default=0.5, help="Histogram threshold (0–1)")
    args = parser.parse_args()
 
 
    BASE_DIR = args.out_dir
 
    make_all_histograms(args.videos, threshold=args.threshold)
    make_overview_plots(args.videos)
 
if __name__ == "__main__":
    main()