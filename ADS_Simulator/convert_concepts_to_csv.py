import os
import json
import pandas as pd

def convert_json_to_csv(input_path="outputs/concepts/all_concepts.json", output_csv="outputs/concepts/all_concepts.csv"):
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to load JSON: {e}")
            return

    if not data:
        print("⚠️ No data to write.")
        return

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"✅ CSV saved to: {output_csv}")

if __name__ == "__main__":
    convert_json_to_csv()
