# model.py — LiteLLM-backed chat with rate-limit fallback & backoff.
# Returns a safe stub JSON on repeated failure, so the pipeline never raises and CSV always gets a string.

import time, os, transformers, torch
from transformers import AutoTokenizer
from prompts import identity

# --- NEW: LiteLLM ---
from litellm import completion

LITELLM_BASE_URL = os.getenv("LITELLM_API_BASE") or os.getenv("LITELLM_BASE_URL")
LITELLM_API_KEY  = os.getenv("LITELLM_API_KEY") or os.getenv("COMPANY_LLM_API_KEY")

def get_model(args, override_model_name: str = None):

    model_name = override_model_name or getattr(args, "model", "gpt-4o")
    temperature = args.temperature

    if "llama" in model_name.lower() or "meta-llama" in model_name.lower():
        return LLaMA(model_name, temperature)
    else:
        return GPT(
            model_name=model_name,
            fallback_model=getattr(args, "fallback_model", "gpt-4o"),
            temperature=temperature
        )

class Model(object):
    def __init__(self):
        self.post_process_fn = identity
    def set_post_process_fn(self, post_process_fn):
        self.post_process_fn = post_process_fn

class GPT(Model):
    def __init__(self, model_name, fallback_model, temperature):
        super().__init__()
        self.primary_model = model_name
        self.fallback_model = fallback_model
        self.temperature = temperature

        if not LITELLM_API_KEY:
            raise RuntimeError("Missing LITELLM_API_KEY in environment (.env).")

    def _request(self, model, messages, temperature):

        return completion(
            model=model,
            messages=messages,
            temperature=temperature,
            base_url=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
            timeout=120,
            seed=42  
        )

    def get_response(self, model, messages, temperature, max_retries=6):
        current_model = model
        for i in range(max_retries):
            try:
                return self._request(current_model, messages, temperature)
            except Exception as e:
                es = str(e)
                is_retryable = any(s in es for s in [
                    "RateLimitError", "insufficient_quota", "You exceeded your current quota",
                    "timeout", "ServiceUnavailable", "429", "502", "503"
                ])
                if is_retryable and i < max_retries - 1:
                    current_model = self.fallback_model
                    time.sleep(min(20, 2 ** i))
                    continue
                # Final stub to keep pipeline alive
                return {
                    "choices": [{"message": {"content": "{'final_answer': 'API_QUOTA_EXCEEDED', 'rationale': 'fallback stub'}"}}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }

    def forward(self, head, prompts):
        messages = [{"role": "system", "content": head}]
        info = {}
        current_model = self.primary_model
        for prompt in prompts:
            messages.append({"role": "user", "content": prompt})
            res = self.get_response(current_model, messages, self.temperature)
            try:
                content = res.choices[0].message["content"]  # LiteLLM(OpenAI-style)
                usage = getattr(res, "usage", None)
                usage_dict = dict(usage) if usage else {}
            except Exception:
                content = res["choices"][0]["message"]["content"]
                usage_dict = dict(res.get("usage", {}))
            messages.append({"role": "assistant", "content": content})
            info = dict(usage_dict)
            info["response"] = content
            info["message"] = messages
        return self.post_process_fn(info["response"]), info

class LLaMA(Model):
    def __init__(self, model_name, temperature):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.pad_token or tok.eos_token or "[PAD]"
        tok.padding_side = "left"
        self.tokenizer = tok
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            tokenizer=tok,
            temperature=temperature
        )

    def forward(self, head, prompts):
        prompt = prompts[0] if prompts else ""
        full_prompt = (head + "\n\n" + prompt) if head else prompt
        seq = self.pipeline(
            full_prompt, do_sample=False, top_k=1, num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id, max_new_tokens=512
        )[0]["generated_text"]
        info = {"message": full_prompt, "response": seq}
        return self.post_process_fn(info["response"]), info
