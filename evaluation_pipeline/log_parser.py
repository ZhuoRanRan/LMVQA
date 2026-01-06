import re
 
def extract_from_logfile(log_path):
    with open(log_path, "r", encoding="utf-8") as file:
        content = file.read()
 
    pattern = r'MetricData\(name=[\'"](.*?)[\'"].*?score=([\d\.]+),\s*reason=([\'"])((?:\\.|(?!\3).)*?)\3.*?evaluation_model=[\'"](.*?)[\'"].*?error=(None|[\'"](.*?)[\'"])'
    matches = re.findall(pattern, content, re.DOTALL)
 
    results = []
    for match in matches:
        metric_name, score, quote, reason, evaluation_model, error, error_str = match
        reason_clean = reason.replace("\n", "\\n").replace('"', '""').strip()
        results.append({
            "metric": metric_name,
            "score": float(score),
            "reason": reason_clean
        })
    return results