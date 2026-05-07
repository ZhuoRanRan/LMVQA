# RQ Results

This folder organizes the evaluation artifacts by the three research questions in the paper.

- `RQ1_Accuracy/`: answer-level accuracy results for LMVQA and DrVideo on the public Course dataset. The CSV files include predictions, ground-truth answers, expert annotations, and final correctness decisions. These files support the public Course results in Table VI. Ciena raw results are not included because the industrial dataset is confidential.
- `RQ2_Efficiency/`: timing and LLM API cost results. This includes LMVQA offline indexing time, LMVQA per-question answering latency, DrVideo per-question answering latency, and `llm_api_cost.csv`. These files support Table VII.
- `RQ3_User_Feedback/`: questionnaire and interview materials used for the qualitative user-feedback study with three Ciena engineers. `questionnaire.tex` is the LaTeX source and `questionnaire.pdf` is the rendered document.
