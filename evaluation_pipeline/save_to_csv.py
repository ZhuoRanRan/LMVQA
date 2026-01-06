import csv
 
def save_to_csv(test_cases, parsed_results, csv_path, threshold: float):
    """
    Save per-item detailed results.
    Columns:
      - score (0..1)
      - is_correct: score >= threshold
      - threshold (correctness)
    """
    with open(csv_path, "w", encoding="utf-8", newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow([
            "question", "prediction", "ground_truth",
            "score", "is_correct",
            "threshold",
            "reason"
        ])
 
        for i, tc in enumerate(test_cases):
            score = parsed_results[i]["score"] if i < len(parsed_results) else 0.0
            reason = parsed_results[i]["reason"] if i < len(parsed_results) else "N/A"
            reason = reason.replace('"', '""').replace('\n', '\\n').replace("'", "''")
            is_correct = 1 if score >= threshold else 0
            writer.writerow([
                tc.input,
                tc.actual_output,
                tc.expected_output,
                score, is_correct,
                threshold,
                reason
            ])
 