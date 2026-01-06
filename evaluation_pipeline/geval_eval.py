# -*- coding: utf-8 -*-
"""
Run GEval metrics via a DeepEval-compatible OpenAI wrapper.

- Ask the judge to return scores in 0..10 (GEval convention).
- After parsing, normalize all scores to 0..1 for internal use.
- Single-threshold 0..1 aggregation for correctness (mean, pass rate).
- (Context Precision / Context Recall / Truthfulness have been commented out per request.)
"""

import os
import json
import math
from typing import List, Tuple, Optional, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate

from evaluation_pipeline.log_parser import extract_from_logfile
from evaluation_pipeline.save_to_csv import save_to_csv
from VideoQA_constants.prompts import (
    GEVAL_CORRECTNESS_STEPS,
    # GEVAL_CONTEXT_PRECISION_STEPS,
    # GEVAL_CONTEXT_RECALL_STEPS,
    # GEVAL_TRUTHFULNESS_STEPS,
)

# ------------------------- proxy / client wiring ONLY -------------------------

load_dotenv()

# Prefer company LiteLLM gateway; fall back to plain OpenAI if not provided.
_LITELLM_KEY  = os.getenv("LITELLM_API_KEY")
_LITELLM_BASE = os.getenv("LITELLM_API_BASE")

_OPENAI_KEY   = os.getenv("OPENAI_API_KEY")
_OPENAI_BASE  = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")

_api_key  = _LITELLM_KEY  or _OPENAI_KEY
_base_url = _LITELLM_BASE or _OPENAI_BASE or None
if not _api_key:
    raise RuntimeError("No API key found. Set LITELLM_API_KEY (preferred) or OPENAI_API_KEY in .env")

client = OpenAI(api_key=_api_key, base_url=_base_url) if _base_url else OpenAI(api_key=_api_key)

# ----------------------------- LLM wrapper -----------------------------

class GPT4oEvalLLM(DeepEvalBaseLLM):
    """
    Minimal DeepEval-compatible LLM wrapper using OpenAI Chat Completions.
    Return JSON: {"score": <0..10>, "reason": "<one-line>"}.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("GEVAL_MODEL", "gpt-4o-mini")

    def load_model(self):
        return self.model_name

    def _build_prompt(self, input_text: str) -> str:
        # IMPORTANT: ask for 0..10 (GEval convention). We will normalize to 0..1 after parsing.
        return (
            "Return a compact JSON object: "
            '{"score": <float between 0 and 10>, "reason": "<one-line explanation>"}.\n\n'
            f"{input_text}"
        )

    def _postprocess_to_json(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return json.dumps({"score": 0.0, "reason": "Empty response"})
        if not text.startswith("{"):
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start: end + 1]
        try:
            json.loads(text)
            return text
        except Exception:
            return json.dumps({"score": 0.0, "reason": f"Non-JSON response: {text[:120]}"})

    def generate(self, prompt: str) -> str:
        try:
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a rigorous but fair evaluator."},
                    {"role": "user", "content": self._build_prompt(prompt)},
                ],
                temperature=0,
                max_tokens=300,
            )
            raw = resp.choices[0].message.content
            return self._postprocess_to_json(raw)
        except Exception as e:
            return json.dumps({"score": 0.0, "reason": f"Generation error: {type(e).__name__}: {str(e)}"})

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


# -------------------------- helpers (0..1) -----------------------------

def _normalize_scores_to_01(scores: List[float]) -> List[float]:
    """If GEval returned 0..10, normalize to 0..1; otherwise return as-is."""
    if not scores:
        return scores
    max_s = max(scores)
    if max_s > 1.0:
        return [s / 10.0 for s in scores]
    return scores

def _mean_and_pass(scores_01: List[float], pass_thr: float = 0.5) -> Tuple[Optional[float], Optional[float]]:
    if not scores_01:
        return None, None
    m = sum(scores_01) / len(scores_01)
    p = sum(1 for s in scores_01 if s >= pass_thr) / len(scores_01)
    return float(m), float(p)

def _stats(scores_01: List[float]) -> Dict[str, float]:
    if not scores_01:
        return {"min": 0.0, "max": 0.0, "median": 0.0, "mean": 0.0, "std": 0.0}
    srt = sorted(scores_01)
    n = len(srt)
    median = (srt[n // 2] if n % 2 == 1 else (srt[n // 2 - 1] + srt[n // 2]) / 2.0)
    mean = sum(srt) / n
    var = sum((x - mean) ** 2 for x in srt) / n if n > 1 else 0.0
    import math
    return {
        "min": float(srt[0]),
        "max": float(srt[-1]),
        "median": float(median),
        "mean": float(mean),
        "std": float(math.sqrt(var)),
    }


# ------------------------------ main eval ------------------------------

def run_geval_evaluation(
    questions: List[str],
    predictions: List[str],
    ground_truths: List[str],
    output_txt_path: str,
    detailed_csv_path: str,
    correctness_threshold: float = 0.5,          # 0..1
    retrieval_contexts: Optional[List[List[str]]] = None,
):
    """
    Evaluate with GEval and return aggregated results (0..1 scale).
    Only correctness is computed; context-based metrics are commented out.
    """
    print("📊 Running GEval evaluation with OpenAI...")
    custom_llm = GPT4oEvalLLM()

    # ---------- Correctness ----------
    correctness_metric = GEval(
        name="Correctness",
        model=custom_llm,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        evaluation_steps=GEVAL_CORRECTNESS_STEPS,
    )

    test_cases = [
        LLMTestCase(input=q, actual_output=pred, expected_output=gt, retrieval_context=[])
        for q, pred, gt in zip(questions, predictions, ground_truths)
    ]

    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

    try:
        # Run only correctness, but keep print_results behavior as before
        try:
            evaluation_results = evaluate(test_cases, [correctness_metric], print_results=True)
        except TypeError:
            evaluation_results = evaluate(test_cases, [correctness_metric])

        if isinstance(evaluation_results, list):
            results_list = evaluation_results
        elif hasattr(evaluation_results, "results"):
            results_list = evaluation_results.results
        else:
            results_list = [evaluation_results]

        # Persist raw log
        with open(output_txt_path, "w", encoding="utf-8") as f:
            for r in results_list:
                f.write(str(r))
                f.write("\n\n")

        # Parse scores (0..10 from judge) then normalize to 0..1
        parsed_results = extract_from_logfile(output_txt_path)
        cor_scores = _normalize_scores_to_01([r.get("score", 0.0) for r in parsed_results])

        # Save detailed CSV using a single 0..1 threshold
        save_to_csv(
            test_cases=test_cases,
            parsed_results=[{"score": s, "reason": parsed_results[i].get("reason", "N/A")} for i, s in enumerate(cor_scores)],
            csv_path=detailed_csv_path,
            threshold=correctness_threshold,
        )

        # Aggregate correctness
        overall: Dict[str, Any] = {}
        cor_mean, cor_pass = _mean_and_pass(cor_scores, pass_thr=correctness_threshold)
        stats = _stats(cor_scores)

        overall["correctness_threshold"] = float(correctness_threshold)
        overall["correctness_mean"] = float(cor_mean) if cor_mean is not None else 0.0
        overall["correctness_pass_rate(>=thr)"] = float(cor_pass) if cor_pass is not None else 0.0
        overall["correctness_min"] = stats["min"]
        overall["correctness_max"] = stats["max"]
        overall["correctness_median"] = stats["median"]
        overall["correctness_std"] = stats["std"]

        # ---------- Context metrics (commented out) ----------
        # if retrieval_contexts is not None:
        #     # 1) Context Precision
        #     ctx_prec_metric = GEval(
        #         name="ContextPrecision",
        #         model=custom_llm,
        #         evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
        #         evaluation_steps=GEVAL_CONTEXT_PRECISION_STEPS,
        #     )
        #     ...
        #
        #     # 2) Context Recall
        #     ctx_rec_metric = GEval(
        #         name="ContextRecall",
        #         model=custom_llm,
        #         evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT, LLMTestCaseParams.EXPECTED_OUTPUT],
        #         evaluation_steps=GEVAL_CONTEXT_RECALL_STEPS,
        #     )
        #     ...
        #
        #     # 3) Truthfulness
        #     truth_metric = GEval(
        #         name="Truthfulness",
        #         model=custom_llm,
        #         evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
        #         evaluation_steps=GEVAL_TRUTHFULNESS_STEPS,
        #     )
        #     ...

        return {"status": "success", "num_cases": len(test_cases), **overall}
    except Exception as e:
        return {"status": "failed", "error": str(e)}
