import os
import json
from dotenv import load_dotenv
import openai

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate

from evaluation_pipeline.log_parser import extract_from_logfile
from evaluation_pipeline.save_to_csv import save_to_csv
from evaluation_pipeline.metrics_helper import compute_metrics
from VideoQA_constants.prompts import GEVAL_CORRECTNESS_STEPS

load_dotenv()
openai.api_key = os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY")
openai.base_url = os.getenv("LITELLM_API_BASE") or os.getenv("OPENAI_BASE_URL")


class GPT4oEvalLLM(DeepEvalBaseLLM):
    """Uses LiteLLM/OpenAI chat.completions; returns JSON {score 0..10, reason}"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("GEVAL_MODEL", "gpt-4o-mini")
        if not openai.api_key:
            raise RuntimeError("Set LITELLM_API_KEY or OPENAI_API_KEY")

    def load_model(self):
        return self.model_name

    def _build_prompt(self, input_text: str) -> str:
        return (
            "Return a compact JSON object: "
            '{"score": <float between 0 and 10>, "reason": "<one-line explanation>"}.\n\n'
            "Evaluation instructions:\n- " + "\n- ".join(GEVAL_CORRECTNESS_STEPS) + "\n\n"
            f"{input_text}"
        )

    def generate(self, prompt: str) -> str:
        try:
            resp = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a rigorous but fair evaluator."},
                    {"role": "user", "content": self._build_prompt(prompt)},
                ],
                temperature=0,
                max_tokens=300,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text.startswith("{"):
                start, end = text.find("{"), text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    text = text[start : end + 1]
            try:
                json.loads(text)
            except Exception:
                text = json.dumps({"score": 0.0, "reason": f"Non-JSON response: {text[:120]}"})
            return text
        except Exception as e:
            return json.dumps({"score": 0.0, "reason": f"Generation error: {str(e)}"})

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


def run_geval_evaluation(
    questions,
    predictions,
    ground_truths,
    output_txt_path,
    detailed_csv_path,
    threshold: float = 5.0,  # threshold on 0..10 scale
):
    """Run GEval, save per-item CSV, return overall metrics dict."""
    print("📊 Running GEval evaluation with GPT-4o (via LiteLLM/OpenAI)...")
    custom_llm = GPT4oEvalLLM()

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
        # Compat with different deepeval versions
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

        with open(output_txt_path, "w", encoding="utf-8") as f:
            for r in results_list:
                f.write(str(r))
                f.write("\n\n")

        parsed_results = extract_from_logfile(output_txt_path)

        if parsed_results and max(r.get("score", 0.0) for r in parsed_results) <= 1.0:
            for r in parsed_results:
                r["score"] = r.get("score", 0.0) * 10.0

        save_to_csv(test_cases, parsed_results, detailed_csv_path)

        overall = compute_metrics(parsed_results, threshold=threshold)

        return {"status": "success", "num_cases": len(test_cases), **overall}
    except Exception as e:
        return {"status": "failed", "error": str(e)}
