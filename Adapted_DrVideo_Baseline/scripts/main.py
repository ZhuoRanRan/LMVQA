# main.py — Open-ended QA pipeline (single-lecture, CPU-friendly, robust).
# ASR -> per-second DB -> retrieval (Embeddings with fallback) -> BLIP captions ->
# judge/find loop (robust) -> open-ended answer (robust) -> CSV.

import os
import cv2
import ast
import json
import re
import math
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv

from util import parse_args, load_json
from prompts import PromptFactory
from model import get_model
from models.blip2_model import ImageCaptioner
from models.whisper_model import AudioTranslator

import pandas as pd
from tqdm import tqdm

# --- OpenAI SDK (points to LiteLLM base_url) for embeddings ---
from openai import OpenAI

PROMPT_TEMPLATES = {
    'A': "What is shown in this image? Answer briefly.",
    'B': "Given the question: {question} Answer concisely using only visible evidence.",
    'C': "Describe this image in <=100 words; capture key objects, actions, interactions, scenes."
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===== Prompt size guardrails =====
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "12000"))  

def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)

def _trim_context(s: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    if not s or len(s) <= max_chars:
        return s
    half = max_chars // 2
    return s[:half] + "\n...\n" + s[-half:]

# ===== Robust JSON parser =====
def safe_llm_output_parse(output_str: str) -> Any:
    """
    Robustly parse:
    - fenced code blocks such as ```json ... ```
    - single quotes and smart quotes
    - JSON-style and Python-literal-style outputs
    """
    if not output_str:
        return None
    s = str(output_str).strip()

    # Strip fenced code blocks.
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()

    # Try strict JSON first.
    try:
        return json.loads(s)
    except Exception:
        pass

    # Normalize smart quotes and fall back to literal_eval.
    trans = {
        "\u2019": "'",  # ’
        "\u2018": "'",  # ‘
        "\u201c": '"',  # “
        "\u201d": '"',  # ”
    }
    s2 = s.translate(str.maketrans(trans))
    try:
        return ast.literal_eval(s2)
    except Exception:
        logging.error(f"Failed to parse LLM output: {output_str}")
        return None


def build_database_from_asr(video_path: Path, out_json_path: Path, whisper_model: AudioTranslator):
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"[ASR] Generating database via Whisper for {video_path} ...")
    messages = []
    try:
        segments = whisper_model(str(video_path))
        per_sec = {}
        for seg in segments:
            s, e = int(seg.get("start", 0)), int(seg.get("end", 0))
            text = seg.get("text", "").strip()
            for t in range(s, max(s + 1, e + 1)):
                per_sec.setdefault(t, []).append(text)
        for t in sorted(per_sec.keys()):
            text = " ".join(per_sec[t]).strip()
            messages.append({"time": t, "content": f"{t} {text}"})
    except Exception as e:
        logging.error(f"[ASR] Whisper failed ({e}). Writing stub DB.")
        messages = [{"time": 1, "content": "1 (no ASR available)."}]
    if not messages:
        messages = [{"time": 1, "content": "1 (empty ASR)."}]
    with open(out_json_path, "w") as f:
        json.dump({"messages": messages}, f, indent=2)
    logging.info(f"[ASR] Database saved to {out_json_path} with {len(messages)} messages")


# ---- simple cosine helpers to avoid numpy dependency ----
def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def _norm(a: List[float]) -> float:
    return math.sqrt(sum(x * x for x in a)) or 1.0

def _cosine(a: List[float], b: List[float]) -> float:
    return _dot(a, b) / (_norm(a) * _norm(b))


def get_relevant_frames(json_file_path: Path, question: str, k: int = 20) -> Dict[str, str]:
    """
    Embedding-based retrieval:
    - OpenAI SDK client(base_url=LITELLM_API_BASE, api_key=LITELLM_API_KEY)
    - model = EMBEDDING_MODEL (default: text-embedding-3-large)
    Falls back to keyword scoring when embeddings are unavailable.
    """
    with open(json_file_path, "r") as f:
        db = json.load(f)
    messages = db.get("messages", [])
    raw: List[tuple] = []
    for m in messages:
        sec = str(m.get("time", ""))
        content = m.get("content", "")
        try:
            _, text = content.split(" ", 1)
        except ValueError:
            text = content
        raw.append((sec, text))
    if not raw:
        return {}

    # Embedding path with batching and truncation.
    try:
        load_dotenv()
        base_url = os.getenv("LITELLM_API_BASE") or os.getenv("LITELLM_BASE_URL")
        api_key  = os.getenv("LITELLM_API_KEY") or os.getenv("COMPANY_LLM_API_KEY")
        embed_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        max_chars = int(os.getenv("EMBEDDING_MAX_CHARS", "4000"))
        batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

        if not api_key or not base_url:
            raise RuntimeError("Missing LITELLM_API_BASE or LITELLM_API_KEY")

        client = OpenAI(api_key=api_key, base_url=base_url)

        texts = [t if len(t) <= max_chars else t[:max_chars] for _, t in raw]

        vectors: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            emb_resp = client.embeddings.create(model=embed_model, input=chunk, timeout=60)
            vectors.extend([d.embedding for d in emb_resp.data])

        q_vec = client.embeddings.create(model=embed_model, input=[question], timeout=60).data[0].embedding

        scored = []
        for (sec, text), vec in zip(raw, vectors):
            sim = _cosine(q_vec, vec)
            scored.append((sim, sec, text))
        scored.sort(key=lambda x: (-x[0], int(x[1]) if x[1].isdigit() else 0))
        top = scored[:k]
        frame_dict = {sec: text for _, sec, text in top}
        if frame_dict:
            logging.info(f"[RAG] Embedding retrieval succeeded with {embed_model}.")
            return frame_dict

    except Exception as e:
        logging.warning(f"[RAG] Embedding retrieval failed; fallback to keyword scoring. Reason: {e}")

    # Fallback: keyword scoring.
    q = question.lower()
    toks = [t for t in q.replace("?", " ").replace(",", " ").split() if t]
    scored = []
    for sec, text in raw:
        t = text.lower()
        score = sum(t.count(tok) for tok in toks) + sum(1 for tok in set(toks) if tok in t)
        scored.append((score, sec, text))
    scored.sort(key=lambda x: (-x[0], int(x[1]) if x[1].isdigit() else 0))
    top = [s for s in scored if s[0] > 0][:k] or scored[:k]
    return {sec: text for _, sec, text in top}


def generate_captions(video_path: str, frames_to_caption: List[Dict], question: str,
                      image_captioner: ImageCaptioner) -> Dict[str, str]:
    captions = {}
    if not frames_to_caption:
        return captions
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video file: {video_path}")
        return captions
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    for frame_info in frames_to_caption:
        try:
            key_frame_sec = int(frame_info["frame"])
            frame_type = frame_info.get("type", 'C')
            frame_idx = max(0, int((key_frame_sec - 0.5) * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Could not read frame at second {key_frame_sec} from {video_path}")
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tpl = PROMPT_TEMPLATES.get(frame_type, PROMPT_TEMPLATES['C'])
            text_prompt = tpl.format(question=question) if "{question}" in tpl else tpl
            caption_output = image_captioner.image_caption(frame_rgb, text_prompt, frame_type)
            caption = (caption_output or "").strip()
            if caption:
                captions[str(key_frame_sec)] = caption
        except Exception as e:
            logging.error(f"Error processing frame_info {frame_info}: {e}")
    cap.release()
    return captions


def process_item(item: Dict, models: Dict, prompters: Dict, image_captioner: ImageCaptioner,
                 db_dir: Path, args: Any) -> Dict:
    ukey = item['uid']
    video_path = Path(item['video_path'])
    video_id = item['video_id']
    question = item['question']

    db_json = db_dir / f"{video_id}.json"
    if not db_json.exists():
        whisper = AudioTranslator(model=args.audio_translator, device=args.audio_translator_device)
        build_database_from_asr(video_path, db_json, whisper_model=whisper)

    raw_doc = load_json(str(db_json))
    all_captions = {
        str(m["time"]): m["content"].split(" ", 1)[1] if " " in m["content"] else m["content"]
        for m in raw_doc.get("messages", [])
    }
    relevant_frames_dict = get_relevant_frames(db_json, question, k=12)
    key_frame_indices = sorted([int(k) for k in relevant_frames_dict.keys()])

    frames_for_initial_captioning = [{'frame': str(idx), 'type': 'B'} for idx in key_frame_indices]
    initial_new_captions = generate_captions(str(video_path), frames_for_initial_captioning, question, image_captioner)
    if initial_new_captions:
        for frame_idx, caption in initial_new_captions.items():
            base = all_captions.get(frame_idx, f"(no base caption for frame {frame_idx})")
            if caption:
                all_captions[frame_idx] = f"{base} [Q-specific: {caption}]"

    max_turns = 2
    type_A_frames, type_B_frames = [], key_frame_indices.copy()
    gpt_feedback_history = ""
    for turn in range(max_turns):
        current_context_str = "\n".join(
            [f"frame {k}: {v}" for k, v in sorted(all_captions.items(), key=lambda x: int(x[0]))]
        )
        if len(current_context_str) > MAX_PROMPT_CHARS:
            logging.warning(f"[CTX] Judge/Find context too long: {len(current_context_str)} chars (~{_approx_tokens(current_context_str)} tok); trimming to {MAX_PROMPT_CHARS}.")
        current_context_str = _trim_context(current_context_str)

        # --- Judge (robust) ---
        try:
            prompt_judge = prompters['judge'].fill(
                captions=current_context_str, question_context=question, gpt_prompt=gpt_feedback_history
            )
            pred_judge_str, _ = models['judge'].forward(prompters['judge'].head, prompt_judge)
            pred_judge = safe_llm_output_parse(pred_judge_str)
            logging.info(f"[{ukey}] Judge: {pred_judge}")
        except Exception as e:
            logging.warning(f"[{ukey}] Judge failed ({e}); skipping loop to final reasoning.")
            pred_judge = {'confidence': '1', 'explanation': ["judge failed, skip loop"]}

        if pred_judge and str(pred_judge.get('confidence', '0')) == '1':
            break

        # --- Find (robust) ---
        try:
            judge_explanation = f"In round {turn + 1}, the reasoning was: {pred_judge.get('explanation', 'N/A')}\n" if pred_judge else ""
            prompt_find = prompters['find'].fill(
                captions=current_context_str, question_context=question,
                explanation=judge_explanation, type_A=str(type_A_frames), type_B=str(type_B_frames)
            )
            pred_find_str, _ = models['find'].forward(prompters['find'].head, prompt_find)
            frames_to_add = safe_llm_output_parse(pred_find_str)
            logging.info(f"[{ukey}] Find: {frames_to_add}")
        except Exception as e:
            logging.warning(f"[{ukey}] Find failed ({e}); ending loop early.")
            frames_to_add = None

        if not frames_to_add:
            break

        new_captions = generate_captions(str(video_path), frames_to_add, question, image_captioner)
        if new_captions:
            for f, cap in new_captions.items():
                base = all_captions.get(f, f"(no base caption for frame {f})")
                if cap:
                    all_captions[f] = f"{base} [Added: {cap}]"
            new_info_str = "\n".join([f"frame {k}: {v}" for k, v in new_captions.items() if v])
            if new_info_str:
                gpt_feedback_history += f"In round {turn + 1}, we added:\n{new_info_str}\n"
            for f_info in frames_to_add:
                try:
                    t = int(f_info['frame'])
                    if f_info.get('type') == 'A':
                        type_A_frames.append(t)
                    elif f_info.get('type') == 'B':
                        type_B_frames.append(t)
                except Exception:
                    pass

    # 4) Final open-ended reasoning (robust)
    final_context_str = "\n".join(
        [f"frame {k}: {v}" for k, v in sorted(all_captions.items(), key=lambda x: int(x[0]))]
    )
    if len(final_context_str) > MAX_PROMPT_CHARS:
        logging.warning(f"[CTX] Reasoning context too long: {len(final_context_str)} chars (~{_approx_tokens(final_context_str)} tok); trimming to {MAX_PROMPT_CHARS}.")
    final_context_str = _trim_context(final_context_str)

    try:
        prompt_reasoning = prompters['open_reasoning'].fill(
            context=final_context_str, question_text=question
        )
        pred_reasoning_str, info = models['reasoning'].forward(prompters['open_reasoning'].head, prompt_reasoning)
        pred_reasoning = safe_llm_output_parse(pred_reasoning_str)
        answer_text = (pred_reasoning or {}).get('final_answer', '').strip() or 'ERROR'
        rationale_text = (pred_reasoning or {}).get('rationale', '').strip()
    except Exception as e:
        logging.error(f"[{ukey}] Reasoning failed ({e}); returning fallback answer.")
        answer_text = 'API_QUOTA_EXCEEDED'
        rationale_text = 'returned by fallback to avoid pipeline crash'

    return {
        'uid': ukey,
        'video_id': video_id,
        'question': question,
        'answer': answer_text,
        'rationale': rationale_text,
        'video_path': str(video_path),
        'duration': 0  # Filled with the measured runtime in run_single_mode.
    }


def dump_csv_rows(rows: List[Dict], csv_path: Path, include_gt: bool):
    cols = ['uid', 'video_id', 'question'] + (['gt'] if include_gt else []) + ['answer', 'rationale', 'video_path', 'duration']
    df = pd.DataFrame(rows)[cols]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logging.info(f"[CSV] Written: {csv_path}")


def read_csv_robust(csv_path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gb18030", "cp1252", "latin-1"]
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc, engine="python")
            break
        except Exception:
            df = None
    if df is None:
        df = pd.read_csv(csv_path, encoding="latin-1", engine="python", on_bad_lines="skip", quoting=3)
    cols_norm = {c.strip().lower(): c for c in df.columns}
    if 'question' not in df.columns:
        for cand in ['question', 'q', 'prompt', 'question_text']:
            low = cand.lower()
            if low in cols_norm:
                df.rename(columns={cols_norm[low]: 'question'}, inplace=True)
                break
    if 'question' not in df.columns:
        df['question'] = ''
    if 'gt' not in df.columns:
        df['gt'] = ''
    df = df[df['question'].astype(str).str.strip() != '']
    df.reset_index(drop=True, inplace=True)
    return df


def _append_jsonl(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run_single_mode(args):
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY") and args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    csv_path = Path(args.single_csv_path)
    video_path = Path(args.single_video_path)
    assert csv_path.is_file(), f"CSV not found: {csv_path}"
    assert video_path.is_file(), f"Video not found: {video_path}"

    video_id = video_path.stem
    out_csv = Path(args.single_output_csv) if getattr(args, "single_output_csv", "") else Path(args.lecture_output_dir) / f"{video_id}_predictions.csv"
    db_dir = Path("data/transcript_database")
    db_dir.mkdir(parents=True, exist_ok=True)

    proc_dir = Path("Lecture_Baseline_Processtime")
    proc_log = proc_dir / f"{video_id}.jsonl"

    logging.info("Initializing models/prompters for SINGLE mode ...")
    image_captioner = ImageCaptioner(model_name=args.captioner_base_model, device=args.image_captioner_device)
    pf = PromptFactory()
    prompters = {
        'judge': pf.get('judge'),
        'find': pf.get('find'),
        'open_reasoning': pf.get('open_reasoning')
    }
    models = {
        'judge': get_model(args, override_model_name=args.judge_model),
        'find': get_model(args, override_model_name=args.find_model),
        'reasoning': get_model(args, override_model_name=args.reasoning_model)
    }
    models['judge'].set_post_process_fn(prompters['judge'].post_process_fn)
    models['find'].set_post_process_fn(prompters['find'].post_process_fn)
    models['reasoning'].set_post_process_fn(prompters['open_reasoning'].post_process_fn)

    df = read_csv_robust(csv_path)
    has_gt = 'gt' in df.columns
    rows_out = []

    # Total runtime timer.
    t_all0 = time.time()
    # Clear previous timing logs for this lecture.
    if proc_log.exists():
        proc_log.unlink(missing_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{video_id}"):
        t0 = time.time()
        q = str(row['question'])
        gt = str(row['gt']) if has_gt else ''
        item = {
            'uid': f"{video_id}_q{idx+1}",
            'video_id': video_id,
            'video_path': str(video_path),
            'question': q
        }
        try:
            result = process_item(item, models, prompters, image_captioner, db_dir=db_dir, args=args)
            duration = round(time.time() - t0, 3)
            result['duration'] = duration

            # Record per-question runtime.
            _append_jsonl(proc_log, {
                "uid": result['uid'],
                "video_id": result['video_id'],
                "question_index": idx + 1,
                "seconds": duration
            })

            out = {
                'uid': result['uid'],
                'video_id': result['video_id'],
                'question': q,
                'answer': result['answer'],
                'rationale': result['rationale'],
                'video_path': result['video_path'],
                'duration': result['duration']
            }
            if has_gt:
                out['gt'] = gt
            rows_out.append(out)
        except Exception as e:
            duration = round(time.time() - t0, 3)
            logging.critical(f"Error on {video_id} row {idx+1}: {e}", exc_info=True)
            # Record runtime even when the question fails.
            _append_jsonl(proc_log, {
                "uid": f"{video_id}_q{idx+1}",
                "video_id": video_id,
                "question_index": idx + 1,
                "seconds": duration,
                "error": str(e)
            })
            out = {
                'uid': f"{video_id}_q{idx+1}",
                'video_id': video_id,
                'question': q,
                'answer': "PROCESSING_ERROR",
                'rationale': "",
                'video_path': str(video_path),
                'duration': duration
            }
            if has_gt:
                out['gt'] = gt
            rows_out.append(out)

    # Record total and average runtime.
    total_secs = round(time.time() - t_all0, 3)
    avg_secs = round(total_secs / max(1, len(rows_out)), 3)
    _append_jsonl(proc_log, {
        "summary": {
            "video_id": video_id,
            "n_questions": len(rows_out),
            "total_seconds": total_secs,
            "avg_seconds": avg_secs
        }
    })

    dump_csv_rows(rows_out, out_csv, include_gt=has_gt)


def main():
    args = parse_args()
    if args.single_csv_path and args.single_video_path:
        run_single_mode(args)
    else:
        raise SystemExit("Use --single_csv_path and --single_video_path to run single-lecture mode.")


if __name__ == '__main__':
    main()
