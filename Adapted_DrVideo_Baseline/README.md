# Adapted DrVideo Baseline

This directory contains the adapted DrVideo baseline materials used in the LMVQA evaluation for the Course dataset.

The original DrVideo implementation is available at [Upper9527/DrVideo](https://github.com/Upper9527/DrVideo.git). We include a copy of the upstream README for reference: [`docs/ORIGINAL_DRVIDEO_README.md`](docs/ORIGINAL_DRVIDEO_README.md).

## What Is Included

- [`scripts/`](scripts/): adapted DrVideo code for open-ended Course video QA.
- [`Lecture_dataset/`](Lecture_dataset/): Course dataset question and ground-truth CSV files for Lecture1-Lecture5.
- [`results/RQ1_Accuracy/DrVideo_Lecture_accuracy/`](results/RQ1_Accuracy/DrVideo_Lecture_accuracy/): DrVideo Course accuracy files with expert annotations.
- [`results/DrVideo_OpenEnded_Predictions/`](results/DrVideo_OpenEnded_Predictions/): raw adapted DrVideo open-ended predictions with rationales.
- [`results/RQ2_Efficiency/`](results/RQ2_Efficiency/): DrVideo per-question timing logs and API cost data.

Lecture videos are not duplicated here. They can be downloaded using the link in the main repository README.

## What Was Adapted

DrVideo was originally designed for multiple-choice long-video QA. For the LMVQA evaluation, we adapted it to support open-ended questions from the Course dataset while preserving the central DrVideo idea of retrieving question-relevant evidence and iteratively augmenting textual video descriptions.

The adapted pipeline in [`scripts/main.py`](scripts/main.py) performs:

1. **CSV-based open-ended QA input**: each lecture uses a CSV with `question` and `gt` columns.
2. **Audio transcript construction**: Whisper builds a per-second transcript database when no cached transcript exists.
3. **Question-conditioned retrieval**: the question is embedded and matched against transcript segments; keyword scoring is used as a fallback.
4. **Visual augmentation**: retrieved frames are captioned with BLIP using either generic captioning or question-directed prompts.
5. **Judge/find loop**: GPT-4o decides whether current evidence is sufficient and, when needed, requests additional frames to caption.
6. **Open-ended answer generation**: GPT-4o returns a compact JSON object with:

```json
{"final_answer": "<short textual answer>", "rationale": "<1-2 sentence reasoning>"}
```

This matches the paper description: we adapted DrVideo's prompts and post-processing pipeline to generate open-ended responses in compact JSON format with `final_answer` and `rationale`, using GPT-4o as the underlying LLM.

## Model Configuration

Default model settings are defined in [`scripts/util.py`](scripts/util.py):

- Judge model: `gpt-4o`
- Find model: `gpt-4o`
- Final reasoning model: `gpt-4o`
- Fallback model: `gpt-4o`
- Temperature: `0.0`
- Image captioner: BLIP base
- Audio transcription: Whisper base

The LLM wrapper in [`scripts/model.py`](scripts/model.py) uses LiteLLM-compatible OpenAI-style chat completion calls.

## Running the Adapted Baseline

Install dependencies in your preferred Python environment:

```bash
pip install -r scripts/requirements.txt
```

Set the required API variables in your shell or a local `.env` file. A template is provided at [`.env.example`](.env.example).

```bash
LITELLM_API_BASE=your_litellm_base_url
LITELLM_API_KEY=your_api_key
EMBEDDING_MODEL=text-embedding-3-large
```

Run one lecture at a time from this directory:

```bash
cd Adapted_DrVideo_Baseline/scripts

python main.py \
  --output_base_path ../output \
  --output_filename unused.json \
  --single_csv_path ../Lecture_dataset/Lecture1.csv \
  --single_video_path /path/to/Lecture1.mp4 \
  --single_output_csv ../output/Lecture1_predictions.csv \
  --lecture_output_dir ../output \
  --judge_model gpt-4o \
  --find_model gpt-4o \
  --reasoning_model gpt-4o \
  --temperature 0.0
```

The run writes open-ended predictions to the selected CSV path and per-question timing logs to `scripts/Lecture_Baseline_Processtime/`.

## Results

The Course dataset includes 295 open-ended questions across five lecture videos.

The timing files in [`results/RQ2_Efficiency/DrVideo_interactive_answering_time_per_question/`](results/RQ2_Efficiency/DrVideo_interactive_answering_time_per_question/) are the timing logs used for the paper's Course dataset DrVideo efficiency result. Across all 295 Course questions, they yield:

- mean answering time: 98.4 seconds per question;
- standard deviation: 25.0 seconds.

The prediction CSV files in [`results/DrVideo_OpenEnded_Predictions/`](results/DrVideo_OpenEnded_Predictions/) contain the adapted DrVideo answers and rationales. Timing should be read from the RQ2 JSONL files rather than from prediction CSV files.

## Directory Structure

```text
Adapted_DrVideo_Baseline/
├── README.md
├── .env.example
├── Lecture_dataset/
├── docs/
│   └── ORIGINAL_DRVIDEO_README.md
├── results/
│   ├── DrVideo_OpenEnded_Predictions/
│   ├── RQ1_Accuracy/
│   │   └── DrVideo_Lecture_accuracy/
│   └── RQ2_Efficiency/
│       ├── DrVideo_interactive_answering_time_per_question/
│       └── llm_api_cost.csv
└── scripts/
    ├── main.py
    ├── model.py
    ├── prompts.py
    ├── util.py
    ├── dataset.py
    ├── requirements.txt
    └── models/
        ├── blip2_model.py
        └── whisper_model.py
```

