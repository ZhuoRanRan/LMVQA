import os
import csv
import argparse
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

BASE_DIR = "Ciena_Eval_Results"

def load_summary_metrics(video_name: str) -> Dict[str, float]:
    """Read summary_metrics.csv and return accuracy/precision/recall (0–1)."""
    path = os.path.join(BASE_DIR, f"{video_name}_results", "summary_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing summary file: {path}")
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] in {"accuracy", "precision", "recall"}:
                try:
                    out[row[0]] = float(row[1])
                except ValueError:
                    out[row[0]] = 0.0
    for k in ("accuracy", "precision", "recall"):
        out.setdefault(k, 0.0)
    return out

def aggregate_summary(videos: List[str]) -> pd.DataFrame:
    rows = []
    for v in videos:
        try:
            m = load_summary_metrics(v)
            rows.append({"video": v, **m})
        except FileNotFoundError as e:
            print(f"⚠️ {e}")
    if not rows:
        raise RuntimeError("No summary metrics found.")
    return pd.DataFrame(rows)

def plot_summary_bars(df: pd.DataFrame, out_png: str):
    """Grouped bars for accuracy/precision/recall (percent axis)."""
    metrics = ["accuracy", "precision", "recall"]
    x = range(len(df))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for i, m in enumerate(metrics):
        ax.bar([p + i*width for p in x], df[m].values, width, label=m)
    ax.set_xticks([p + width for p in x])
    ax.set_xticklabels(df["video"].tolist(), rotation=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("QA Evaluation – Accuracy / Precision / Recall")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    for i, m in enumerate(metrics):
        for j, val in enumerate(df[m].values):
            ax.text(j + i*width, min(val + 0.035, 1.06), f"{val:.2f}",
                    ha="center", va="bottom", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

def load_detailed_scores(video_name: str) -> pd.DataFrame:
    """Read detailed_scores.csv -> columns: question, score, video."""
    path = os.path.join(BASE_DIR, f"{video_name}_results", "detailed_scores.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing detailed file: {path}")
    df = pd.read_csv(path)
    col_map = {c.lower(): c for c in df.columns}
    q_col = col_map.get("question")
    s_col = col_map.get("score")
    if q_col is None or s_col is None:
        raise ValueError(f"{path} must contain 'question' and 'score'. Got: {list(df.columns)}")
    out = df[[q_col, s_col]].rename(columns={q_col: "question", s_col: "score"})
    out["video"] = video_name
    return out

def aggregate_questions(videos: List[str], threshold: float) -> pd.DataFrame:
    """Collect per-question scores; add is_correct based on threshold (auto scale)."""
    frames = []
    for v in videos:
        try:
            d = load_detailed_scores(v)
            frames.append(d)
        except FileNotFoundError as e:
            print(f"⚠️ {e}")
    if not frames:
        raise RuntimeError("No detailed_scores.csv found.")
    df = pd.concat(frames, ignore_index=True)
    df["q_idx"] = df.groupby("video").cumcount() + 1

    # Auto-detect score scale and map threshold
    max_score = df["score"].max()
    thr = threshold / 10.0 if max_score <= 1.0 else threshold
    df["is_correct"] = (df["score"] >= thr).astype(int)
    df.attrs["scale"] = "0-1" if max_score <= 1.0 else "0-10"
    df.attrs["thr_used"] = thr
    return df

def plot_score_distribution(per_q_df: pd.DataFrame, threshold_used: float, out_png: str):
    """Overview violin+scatter; draw threshold line without legend/label."""
    videos = per_q_df["video"].unique().tolist()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    data = [per_q_df.loc[per_q_df["video"] == v, "score"].values for v in videos]
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.4)
    for i, scores in enumerate(data, start=1):
        x = [i + (j - len(scores)/2) * 0.006 for j in range(len(scores))]
        ax.scatter(x, scores, s=12, alpha=0.7)
    scale = per_q_df.attrs.get("scale", "0-10")
    if scale == "0-1":
        ax.set_ylim(0, 1.03)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    else:
        ax.set_ylim(0, 10.3)
    ax.axhline(threshold_used, linestyle="--", linewidth=1, color="gray", alpha=0.8)
    ax.set_xticks(range(1, len(videos) + 1))
    ax.set_xticklabels(videos, rotation=10)
    ax.set_ylabel("Per-question score")
    ax.set_title("Per-Video Question Score Distribution")
    # no legend for threshold
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_hist_per_video(video_name: str, threshold: float):
    """Create and save a histogram for one video under its results folder."""
    df = load_detailed_scores(video_name)

    # Scale detection & threshold mapping
    max_score = df["score"].max()
    if max_score <= 1.0:
        thr_used = threshold / 10.0
        bins = [i/10 for i in range(0, 11)]      # 0..1 step 0.1
        x_ticks = [i/10 for i in range(0, 11)]
        x_label = "Score (0–1)"
    else:
        thr_used = threshold
        bins = list(range(0, 11))                # 0..10
        x_ticks = list(range(0, 11))
        x_label = "Score (0–10)"

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    n, b, _ = ax.hist(df["score"].values, bins=bins, edgecolor="black", alpha=0.78)

    # Count labels above bars
    for count, left, right in zip(n, b[:-1], b[1:]):
        if count > 0:
            xpos = (left + right) / 2
            ax.text(xpos, count + max(n)*0.02, f"{int(count)}",
                    ha="center", va="bottom", fontsize=9)

    # Threshold line (no legend/label)
    ax.axvline(thr_used, linestyle="--", linewidth=1, color="gray", alpha=0.9)

    ax.set_xticks(x_ticks)
    ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax.set_xlabel(x_label)
    ax.set_ylabel("Count")
    ax.set_title(f"{video_name} – Score Histogram")

    # Headroom to avoid overlaps
    ymax = max(n) if len(n) else 1
    ax.set_ylim(0, ymax + max(1, int(ymax * 0.25)))

    plt.tight_layout()
    out_dir = os.path.join(BASE_DIR, f"{video_name}_results")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "score_histogram.png")
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"🖼️ Saved histogram: {out_png}")

def main():
    parser = argparse.ArgumentParser(description="Aggregate metrics and per-question scores, then plot.")
    parser.add_argument("--videos", nargs="+",
                        default=["Ciena_Video1", "Ciena_Video2", "Ciena_Video3", "Ciena_Video4"],
                        help="Video names without extension")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Pass threshold on a 0–10 scale (default: 5.0)")
    parser.add_argument("--out_dir", default=BASE_DIR, help="Output directory")
    args = parser.parse_args()

    # Summary bars
    df_summary = aggregate_summary(args.videos)
    overview_csv = os.path.join(args.out_dir, "metrics_overview.csv")
    overview_png = os.path.join(args.out_dir, "metrics_overview.png")
    os.makedirs(args.out_dir, exist_ok=True)
    df_summary.to_csv(overview_csv, index=False, encoding="utf-8")
    plot_summary_bars(df_summary, overview_png)

    # Per-question distribution overview (kept)
    df_questions = aggregate_questions(args.videos, args.threshold)
    q_overview_csv = os.path.join(args.out_dir, "question_scores_overview.csv")
    df_questions.to_csv(q_overview_csv, index=False, encoding="utf-8")
    thr_used = df_questions.attrs["thr_used"]
    dist_png = os.path.join(args.out_dir, "per_video_score_distribution.png")
    plot_score_distribution(df_questions, thr_used, dist_png)

    # Histogram for each video (saved into each video's folder)
    for v in args.videos:
        try:
            plot_hist_per_video(v, threshold=args.threshold)
        except Exception as e:
            print(f"⚠️ Skip histogram for {v}: {e}")

    print(f"✅ Saved: {overview_csv}")
    print(f"✅ Saved: {overview_png}")
    print(f"✅ Saved: {q_overview_csv}")
    print(f"✅ Saved: {dist_png}")

if __name__ == "__main__":
    main()
