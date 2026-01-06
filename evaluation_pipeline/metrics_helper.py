from statistics import mean, median, pstdev
 
def compute_metrics(
    parsed_results,
    decision_threshold: float = 5.0,
    label_threshold: float | None = None,
):
    """
    Compute metrics from GEval scores using a two-threshold protocol.
 
    - decision_threshold:   defines predicted positives (y_pred = score >= decision_threshold)
    - label_threshold:      defines ground-truth positives (y_true = score >= label_threshold)
      If label_threshold is None, fall back to decision_threshold for backward-compat
      (note: in这种退化情形 precision≈1/未能反映FP，建议总是传入label_threshold)
 
    Returns a dict with confusion matrix, accuracy/precision/recall/f1,
    plus score distribution stats and rates.
    """
    n = len(parsed_results)
    if n == 0:
        return {
            "decision_threshold": float(decision_threshold),
            "label_threshold": float(label_threshold) if label_threshold is not None else float(decision_threshold),
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "mean_score": 0.0, "median_score": 0.0, "std_score": 0.0,
            "min_score": 0.0, "max_score": 0.0,
            "pass_rate": 0.0,       # predicted positive rate
            "correct_rate": 0.0,    # label-positive rate
        }
 
    scores = [float(r.get("score", 0.0)) for r in parsed_results]
    if label_threshold is None:
        label_threshold = decision_threshold  # backward-compat (不建议长期使用)
 
    y_pred = [1 if s >= decision_threshold else 0 for s in scores]   # 系统判定（对外阈值）
    y_true = [1 if s >= label_threshold else 0 for s in scores]      # 参考真值（严格阈值）
 
    tp = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 1)
    tn = sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 0)
 
    accuracy  = (tp + tn) / n if n > 0 else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall    = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
 
    return {
        "decision_threshold": float(decision_threshold),
        "label_threshold": float(label_threshold),
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
        "pass_rate": float(sum(y_pred) / n),  
        "correct_rate": float(sum(y_true) / n),
    }
 
 