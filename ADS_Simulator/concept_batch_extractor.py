import os
import json
import time
import openai
from dotenv import load_dotenv
from ADS_Simulator.prompts import CONCEPT_EXTRACTION_BATCH_PROMPT

load_dotenv()
openai.api_key = os.getenv("LITELLM_API_KEY")
openai.base_url = os.getenv("LITELLM_API_BASE")

def extract_json_content(markdown_text):
    lines = markdown_text.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()

def try_parse_gpt_output(raw_content):
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        try:
            return [json.loads(line) for line in raw_content.splitlines() if line.strip()]
        except Exception as e:
            print("❌ Both array and line-by-line parsing failed.")
            return []

def extract_concepts_batch(narrations, model_name="gpt-4o", max_tokens=2000, retry=3):
    formatted_batch = ""
    for i, item in enumerate(narrations):
        formatted_batch += f"[{i+1}] {item['timestamp']} {item['narration']}\n"

    prompt = CONCEPT_EXTRACTION_BATCH_PROMPT.format(narration_batch=formatted_batch.strip())

    for attempt in range(retry):
        try:
            response = openai.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens
            )

            raw_content = response.choices[0].message.content.strip()
            cleaned = extract_json_content(raw_content)

            print(f"\n🧾 GPT Raw Output (batch attempt {attempt+1}):\n{cleaned[:300]}...\n")
            parsed = try_parse_gpt_output(cleaned)
            if not parsed:
                raise ValueError("Parsed result is empty.")
            return parsed

        except Exception as e:
            print(f"❌ Attempt {attempt+1} failed to parse: {e}")
            time.sleep(2)

    return []

def process_all_batches(
    combined_file="outputs/narration/all_narrations.jsonl",
    output_path="outputs/concepts/all_concepts.json",
    batch_size=10
):
    if not os.path.exists(combined_file):
        print(f"❌ Missing input file: {combined_file}")
        return

    with open(combined_file, "r", encoding="utf-8") as f:
        narrations = [json.loads(line) for line in f]

    all_concepts = []
    for i in range(0, len(narrations), batch_size):
        batch = narrations[i:i+batch_size]
        print(f"🔍 Processing batch {i} to {i+len(batch)-1}")
        concepts = extract_concepts_batch(batch)
        if not concepts:
            print(f"⚠️ Skipping batch {i} due to parsing failure.")
            continue
        for j, concept in enumerate(concepts):
            concept["timestamp"] = batch[j]["timestamp"]
            concept["video_name"] = batch[j].get("video_name", "unknown")
            all_concepts.append(concept)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_concepts, f, indent=2, ensure_ascii=False)

    print(f"✅ All concepts saved to: {output_path}")

if __name__ == "__main__":
    process_all_batches(batch_size=10)
