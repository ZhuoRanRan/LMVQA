import os
import json
import re

NARRATION_DIR = "outputs/narration"
COMBINED_JSONL_PATH = "outputs/narration/all_narrations.jsonl"

TIMESTAMP_LINE_PATTERN = re.compile(r".*?\[([0-9]+[\-–][0-9\.]+s)\].*?:\**\s*(.*)", re.UNICODE)

def combine_all_narrations():
    all_entries = []

    for fname in os.listdir(NARRATION_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(NARRATION_DIR, fname)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        video_name = data.get("video_name", fname.replace(".json", ""))
        narration_text = data.get("narration", "")
        lines = narration_text.splitlines()

        matched_lines = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = TIMESTAMP_LINE_PATTERN.match(line)
            if match:
                timestamp_raw = match.group(1).replace("–", "-").strip()
                content = match.group(2).strip()
                all_entries.append({
                    "video_name": video_name,
                    "timestamp": timestamp_raw,
                    "narration": content
                })
                matched_lines += 1

        if matched_lines == 0:
            print(f"⚠️ No valid narration lines found in: {fname}")

    os.makedirs(os.path.dirname(COMBINED_JSONL_PATH), exist_ok=True)
    with open(COMBINED_JSONL_PATH, "w", encoding="utf-8") as out:
        for entry in all_entries:
            json.dump(entry, out, ensure_ascii=False)
            out.write("\n")

    print(f"Combined {len(all_entries)} narration entries into {COMBINED_JSONL_PATH}")

