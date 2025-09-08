from statistics import mean, median, pstdev

def compute_metrics(parsed_results, threshold=5.0):
    """
    Convert GEval scores to pass/fail by threshold (0..10).
    Dataset has only positive examples: TP = passes, FN = fails.
    """
    n = len(parsed_results)
    if n == 0:
        return {
            "threshold": threshold,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "mean_score": 0.0, "median_score": 0.0, "std_score": 0.0,
            "min_score": 0.0, "max_score": 0.0, "pass_rate": 0.0,
        }

    scores = [float(r.get("score", 0.0)) for r in parsed_results]
    passes = [1 if s >= threshold else 0 for s in scores]

    tp = sum(passes)
    fn = n - tp
    fp = 0
    tn = 0

    accuracy = tp / n
    precision = 1.0 if tp > 0 else 0.0
    recall = tp / n
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "mean_score": float(mean(scores)),
        "median_score": float(median(scores)),
        "std_score": float(pstdev(scores)) if n > 1 else 0.0,
        "min_score": float(min(scores)),
        "max_score": float(max(scores)),
        "pass_rate": float(tp / n),
    }
