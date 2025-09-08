import os
import csv
import argparse
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

BASE_DIR = "Ciena_Eval_Results"

def load_accuracy(video_name: str) -> float:
    """Read summary_metrics.csv and return accuracy (0–1)."""
    path = os.path.join(BASE_DIR, f"{video_name}_results", "summary_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing summary file: {path}")
    acc = 0.0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # header: ["metric","value"]
        for row in reader:
            if len(row) >= 2 and row[0] == "accuracy":
                try:
                    acc = float(row[1])
                except ValueError:
                    acc = 0.0
                break
    return acc

def aggregate_accuracy(videos: List[str]) -> pd.DataFrame:
    """Collect accuracy for all videos into a DataFrame."""
    rows = []
    for v in videos:
        try:
            acc = load_accuracy(v)
            rows.append({"video": v, "accuracy": acc})
        except FileNotFoundError as e:
            print(f"⚠️ {e}")
    if not rows:
        raise RuntimeError("No summary metrics found.")
    return pd.DataFrame(rows)

def plot_accuracy_bars(df: pd.DataFrame, out_png: str):
    """Bar chart for per-video accuracy (percent axis)."""
    x = range(len(df))
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bars = ax.bar(x, df["accuracy"].values, width=0.6)
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

def main():
    parser = argparse.ArgumentParser(description="Plot per-video accuracy only.")
    parser.add_argument(
        "--videos", nargs="+",
        default=["Ciena_Video1", "Ciena_Video2", "Ciena_Video3", "Ciena_Video4"],
        help="Video names without extension"
    )
    parser.add_argument("--out_dir", default=BASE_DIR, help="Output directory")
    args = parser.parse_args()

    df_acc = aggregate_accuracy(args.videos)

    os.makedirs(args.out_dir, exist_ok=True)
    overview_csv = os.path.join(args.out_dir, "metrics_overview_accuracy.csv")
    df_acc.to_csv(overview_csv, index=False, encoding="utf-8")

    overview_png = os.path.join(args.out_dir, "metrics_overview_accuracy.png")
    plot_accuracy_bars(df_acc, overview_png)

    print(f"✅ Saved: {overview_csv}")
    print(f"✅ Saved: {overview_png}")

if __name__ == "__main__":
    main()
