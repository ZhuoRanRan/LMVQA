# qa_generate_gpt4o.py
# -*- coding: utf-8 -*-
"""
Generate predictions for a given video & dataset, and write a predictions CSV
with columns: question, prediction, ground_truth, retrieval_context, latency_seconds.

Notes:
- retrieval_context is built by querying Milvus via RagRetrieverMilvus for each question,
  taking top-K short chunks (default K=5), trimming each to a max number of chars,
  and joining them with " ||| ".
- Keep the answering path exactly as before (AskVideoQAGPT4o).
"""

import os
import json
import pandas as pd
from time import perf_counter
from typing import Tuple, List
from statistics import mean, median, pstdev
import argparse
from VideoQA_Pipeline.askVideoQA_gpt4o import AskVideoQAGPT4o
from RAG_Pipeline.RagRetriever_Milvus import RagRetrieverMilvus


# ------------------------- config -------------------------

TOP_K = int(os.getenv("EVAL_CTX_TOP_K", "5"))                 # recommended 3~8
MAX_CHARS_PER_CHUNK = int(os.getenv("EVAL_CTX_MAX_CHARS", "800"))
CTX_JOIN = " ||| "


# ------------------------- helpers -------------------------

def _read_dataset(path: str) -> Tuple[pd.DataFrame, str, str]:
    """
    Read dataset CSV/XLSX with 'question' and 'gt' columns (case-insensitive).
    Returns (df, question_col, gt_col).
    """
    ext = os.path.splitext(path)[1].lower()

    def _read_csv_robust(p):
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        last_err = None
        for enc in encodings:
            try:
                return pd.read_csv(p, sep=None, engine="python", encoding=enc)
            except UnicodeDecodeError as e:
                last_err = e
                continue
        raise last_err or UnicodeDecodeError("read_csv", b"", 0, 1, "encoding fallback failed")

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif ext in [".csv", ".tsv"]:
        df = _read_csv_robust(path)
    else:
        raise ValueError(f"Unsupported dataset format: {ext}")

    df.columns = [str(c).strip().lower() for c in df.columns]
    if "question" not in df.columns:
        raise KeyError("Dataset must contain a 'question' column.")
    if "gt" not in df.columns:
        raise KeyError("Dataset must contain a 'gt' column.")
    return df, "question", "gt"


def _join_context(chunks, k: int = TOP_K, max_chars: int = MAX_CHARS_PER_CHUNK) -> str:
    """
    Convert a list of chunk dicts (with 'text' key) to a single string:
    - take top-k
    - sanitize NUL and strip
    - truncate to max_chars
    - join by CTX_JOIN
    """
    pieces: List[str] = []
    for c in chunks[:max(0, k)]:
        t = str(c.get("text", "") or "")
        t = t.replace("\u0000", " ").strip()
        if len(t) > max_chars:
            t = t[:max_chars]
        if t:
            pieces.append(t)
    return CTX_JOIN.join(pieces)


# ------------------------- main -------------------------

def main(video_path):
    # -------- paths / params --------
    #video_path = "Ciena_Video/Ciena4.mp4"
    dataset_path = None
    eval_dir = "Lecture_Eval_Datasets"
    model_name = "gpt-4o-model"
    max_tokens = 5000

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # resolve dataset path
    if dataset_path is None:
        cand_csv = os.path.join("Lecture_Datasets", f"{video_name}.csv")
        cand_xlsx = os.path.join("Lecture_Datasets", f"{video_name}.xlsx")
        if os.path.exists(cand_csv):
            dataset_path = cand_csv
        elif os.path.exists(cand_xlsx):
            dataset_path = cand_xlsx
        else:
            raise FileNotFoundError(
                f"Dataset not found for {video_name}. "
                f"Tried: {cand_csv} and {cand_xlsx}"
            )

    # -------- load dataset --------
    df, q_col, gt_col = _read_dataset(dataset_path)
    questions = df[q_col].astype(str).tolist()
    ground_truths = df[gt_col].astype(str).tolist()

    # -------- load video artifacts / ensure chunks & embeddings exist --------
    qa = AskVideoQAGPT4o(model_name=model_name, max_tokens=max_tokens)
    context = qa.load_video_data(video_path)  # will ensure chunks and Milvus embeddings exist

    # Create retriever bound to current video_name for retrieving context
    retriever = RagRetrieverMilvus(video_name)

    # -------- answer & build retrieval_context --------
    print(f"📊 Starting GPT-4o QA generation for {video_name} ...")
    answers: List[str] = []
    ctx_strings: List[str] = []
    per_q_seconds: List[float] = []

    # --- timing start (wall clock for entire loop) ---
    t_wall0 = perf_counter()

    for i, question in enumerate(questions):
        print(f"🎯 Asking ({i+1}/{len(questions)}): {question}")

        # --- per-question timing (answer + retrieval) ---
        t_q0 = perf_counter()

        # 1) generate answer
        answer = qa.answer_question(context, question)
        answers.append(answer)

        # 2) retrieve top-k chunks for eval context
        try:
            hits = retriever.query(question, top_k=TOP_K)
        except Exception as e:
            print(f"⚠️ Retrieval failed at q#{i+1}: {e}")
            hits = []

        ctx = _join_context(hits, k=TOP_K, max_chars=MAX_CHARS_PER_CHUNK)
        ctx_strings.append(ctx)

        # --- per-question timing end ---
        per_q_seconds.append(perf_counter() - t_q0)

    # --- timing end ---
    total_seconds_wall = perf_counter() - t_wall0
    num_questions = len(questions)

    # Summary stats on per-question latencies
    if num_questions > 0:
        avg_seconds_per_question = mean(per_q_seconds)
        med_seconds_per_question = median(per_q_seconds)
        min_seconds_per_question = min(per_q_seconds)
        max_seconds_per_question = max(per_q_seconds)
        std_seconds_per_question = pstdev(per_q_seconds) if num_questions > 1 else 0.0
        total_seconds_sum_questions = sum(per_q_seconds)
    else:
        avg_seconds_per_question = med_seconds_per_question = min_seconds_per_question = max_seconds_per_question = std_seconds_per_question = 0.0
        total_seconds_sum_questions = 0.0

    # -------- write predictions CSV --------
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, f"{video_name}_predictions.csv")
    df_out = pd.DataFrame({
        "question": questions,
        "prediction": answers,
        "ground_truth": ground_truths,
        "retrieval_context": ctx_strings,          # existing column
        "latency_seconds": per_q_seconds,          # NEW per-question latency (answer + retrieval)
    })
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n✅ Predictions saved to {out_path}")
    print(f"   - retrieval_context: top_k={TOP_K}, max_chars_per_chunk={MAX_CHARS_PER_CHUNK}")

    # -------- write timing JSON (expanded) --------
    results_dir = os.path.join("Lecture_Eval_Results", "Run_time")
    os.makedirs(results_dir, exist_ok=True)
    timing_path = os.path.join(results_dir, f"{video_name}_timing.json")

    timing_payload = {
        "video_name": video_name,
        "num_questions": num_questions,

        # Per-question latencies
        "per_question_seconds": [round(x, 4) for x in per_q_seconds],

        # Aggregates from per-question latencies
        "total_seconds_sum_questions": round(total_seconds_sum_questions, 4),
        "avg_seconds_per_question": round(avg_seconds_per_question, 4),
        "median_seconds_per_question": round(med_seconds_per_question, 4),
        "min_seconds_per_question": round(min_seconds_per_question, 4),
        "max_seconds_per_question": round(max_seconds_per_question, 4),
        "std_seconds_per_question": round(std_seconds_per_question, 4),

        # Whole-loop wall clock (kept for backward-compat)
        "total_seconds_wall": round(total_seconds_wall, 4),
    }

    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing_payload, f, ensure_ascii=False, indent=2)

    print(f"⏱️  Timing saved to {timing_path}")
    print(
        "   - total_wall={tw:.3f}s | sum_q={ts:.3f}s | avg={avg:.3f}s | median={med:.3f}s | "
        "min={mn:.3f}s | max={mx:.3f}s | std={sd:.3f}s".format(
            tw=total_seconds_wall,
            ts=total_seconds_sum_questions,
            avg=avg_seconds_per_question,
            med=med_seconds_per_question,
            mn=min_seconds_per_question,
            mx=max_seconds_per_question,
            sd=std_seconds_per_question,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT-4o QA generation with video input.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")

    args = parser.parse_args()

    # Pass the video_path argument to the main function
    main(video_path=args.video_path)
