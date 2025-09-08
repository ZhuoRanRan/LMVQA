import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ADS_Simulator"))

from ADS_Simulator.combine_all_narrations import combine_all_narrations
from ADS_Simulator.concept_batch_extractor import process_all_batches
from ADS_Simulator.convert_concepts_to_csv import convert_json_to_csv

if __name__ == "__main__":
    # print("📦 Step 1: Combining all narrations...")
    # combine_all_narrations()

    # print("\n🧠 Step 2: Extracting concepts using GPT...")
    # process_all_batches(batch_size=10)

    print("\n📊 Step 3: Saving concepts to CSV...")
    convert_json_to_csv()

    # print("\n✅ Concept pipeline completed!")
