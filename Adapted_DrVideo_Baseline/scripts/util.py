# util.py — Open-ended QA utilities with single-video mode & stage-wise model args.
# All comments in English.

import pickle
import json
from pathlib import Path
import argparse
import pandas as pd


def load_pkl(fn):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(data, fn):
    with open(fn, 'wb') as f:
        pickle.dump(data, f)

def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)

def makedir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser("Open-ended VideoQA runner (single/multi lecture)")

    # Generic dataset paths (kept for compatibility with 'database' style)
    parser.add_argument("--dataset", default='egoschema', type=str)
    parser.add_argument("--data_path", default='data/egoschema/lavila_subset_add_time.json', type=str)
    parser.add_argument("--anno_path", default='data/egoschema/subset_anno.json', type=str)
    parser.add_argument("--duration_path", default='data/egoschema/duration.json', type=str)
    parser.add_argument("--fps", default=1.0, type=float)
    parser.add_argument("--num_examples_to_run", default=-1, type=int)
    parser.add_argument("--backup_pred_path", default="", type=str)

    # Outputs
    parser.add_argument("--output_base_path", required=True, type=str)
    parser.add_argument("--output_filename", required=True, type=str)
    parser.add_argument("--csv_output_path", default="output/preds_open.csv", type=str)

    # Prompting / Task (stage-wise models + fallback)
    parser.add_argument("--model", default="gpt-4o", type=str, help="Legacy global model (kept for compat).")
    parser.add_argument("--judge_model", default="gpt-4o", type=str,
                        help="LLM for Judge stage.")
    parser.add_argument("--find_model", default="gpt-4o", type=str,
                        help="LLM for Find stage.")
    parser.add_argument("--reasoning_model", default="gpt-4o", type=str,
                        help="LLM for final open-ended answer.")
    parser.add_argument("--fallback_model", default="gpt-4o", type=str,
                        help="Fallback model when rate limited or API errors occur.")
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--prompt_type", default="open_reasoning", type=str)  # open-ended
    parser.add_argument("--task", default="open", type=str)  # open-ended flag

    # Misc
    parser.add_argument("--disable_eval", action='store_true')
    parser.add_argument("--start_from_scratch", action='store_true')
    parser.add_argument("--save_info", action='store_true')
    parser.add_argument("--save_every", default=10, type=int)

    # Video & cache dirs
    parser.add_argument('--video_root', default='~/Videos/drvideo', type=str,
                        help='Directory containing {uid}.mp4 files')
    parser.add_argument('--alpha', default=20, type=int)
    parser.add_argument('--beta', default=1, type=int)
    parser.add_argument('--data_dir', default='./data/egoschema/database-log', type=str)
    parser.add_argument('--tmp_dir', default='./data/egoschema/database-pkl', type=str)

    # API / devices
    parser.add_argument('--openai_api_key', default='', type=str, help='OpenAI API key (also read from .env)')
    parser.add_argument('--feature_extractor', default='openai/clip-vit-base-patch32')
    parser.add_argument('--feature_extractor_device', choices=['cuda', 'cpu'], default='cpu')
    parser.add_argument('--image_captioner', choices=['blip2-opt', 'blip2-flan-t5', 'blip'],
                        dest='captioner_base_model', default='blip',
                        help='blip is lighter and CPU-friendly; blip2 requires strong GPU')
    parser.add_argument('--image_captioner_device', choices=['cuda', 'cpu'], default='cpu')
    parser.add_argument('--dense_caption', action='store_true', dest='dense_caption', default=True)
    parser.add_argument('--dense_captioner_device', choices=['cuda', 'cpu'], default='cpu')
    parser.add_argument('--audio_translator', default='base', type=str,
                        help='whisper model: tiny/base/small/medium/large')
    parser.add_argument('--audio_translator_device', choices=['cuda', 'cpu'], default='cpu')

    # Multi-lecture (batch) mode
    parser.add_argument('--lecture_csv_dir', default='Lecture_Datasets', type=str,
                        help='Directory with LectureX.csv (columns: question, gt)')
    parser.add_argument('--lecture_video_dir', default='Lecture_Video', type=str,
                        help='Directory with LectureX.mp4 files')
    parser.add_argument('--lecture_output_dir', default='Lecture_Baseline_Eval_Datasets', type=str,
                        help='Directory to write LectureX_predictions.csv')
    parser.add_argument('--only_video_id', default='', type=str,
                        help='If set (e.g., Lecture3), only process this lecture in batch mode.')

    # Single-lecture (smoke-test) mode
    parser.add_argument('--single_csv_path', default='', type=str,
                        help='Path to a single CSV (columns: question, gt). e.g., Lecture_Datasets/Lecture0.csv')
    parser.add_argument('--single_video_path', default='', type=str,
                        help='Path to the single MP4. e.g., Lecture_Video/Lecture0.mp4')
    parser.add_argument('--single_output_csv', default='', type=str,
                        help='Optional explicit output CSV path. If empty, will write to lecture_output_dir/<video_id>_predictions.csv')

    return parser.parse_args()
