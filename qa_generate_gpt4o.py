import os
import pandas as pd
from VideoQA_Pipeline.askVideoQA_gpt4o import AskVideoQAGPT4o


def _read_dataset(path: str) -> tuple[pd.DataFrame, str, str]:
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


def main():
    video_path = "Ciena_Video/Ciena_Video4.mp4"  
    dataset_path = None                           
    eval_dir = "Ciena_Eval_Datasets"
    model_name = "gpt-4o"
    max_tokens = 5000

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if dataset_path is None:
        cand_csv = os.path.join("Ciena_Datasets", f"{video_name}.csv")
        cand_xlsx = os.path.join("Ciena_Datasets", f"{video_name}.xlsx")
        if os.path.exists(cand_csv):
            dataset_path = cand_csv
        elif os.path.exists(cand_xlsx):
            dataset_path = cand_xlsx
        else:
            raise FileNotFoundError(
                f"Dataset not found for {video_name}. "
                f"Tried: {cand_csv} and {cand_xlsx}"
            )

    df, q_col, gt_col = _read_dataset(dataset_path)
    questions = df[q_col].astype(str).tolist()
    ground_truths = df[gt_col].astype(str).tolist()

    qa = AskVideoQAGPT4o(model_name=model_name, max_tokens=max_tokens)
    context = qa.load_video_data(video_path)

    print(f"📊 Starting GPT-4o QA generation for {video_name} ...")
    answers = []
    for i, question in enumerate(questions):
        print(f"🎯 Asking ({i+1}/{len(questions)}): {question}")
        answer = qa.answer_question(context, question)
        answers.append(answer)

    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, f"{video_name}_predictions.csv")
    df_out = pd.DataFrame({
        "question": questions,
        "prediction": answers,
        "ground_truth": ground_truths,
    })
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n✅ Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
