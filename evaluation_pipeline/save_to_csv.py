import csv

def save_to_csv(test_cases, parsed_results, csv_path):
    with open(csv_path, "w", encoding="utf-8", newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(["question", "prediction", "ground_truth", "score", "reason"])

        for i, test_case in enumerate(test_cases):
            score = parsed_results[i]["score"] if i < len(parsed_results) else 0.0
            reason = parsed_results[i]["reason"] if i < len(parsed_results) else "N/A"
            reason = reason.replace('"', '""').replace('\n', '\\n').replace("'", "''")
            writer.writerow([
                test_case.input,
                test_case.actual_output,
                test_case.expected_output,
                score,
                reason
            ])
