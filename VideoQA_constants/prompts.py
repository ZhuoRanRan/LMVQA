# VideoQA_constants/prompts.py

GPT4_PROMPT_TEMPLATE = """
You are an intelligent assistant analyzing a video frame taken from an educational lecture or tutorial. This frame corresponds to the time period: {timestamp}.

Please analyze the image and provide a structured, context-aware description. Follow these rules:

1. **If the image contains a diagram or visual chart**, explain:
   - The **purpose** of the diagram.
   - The **main components** and how they are arranged.
   - Any **arrows, flows, or labeled connections**.
   - The **spatial layout**, including left/right or hierarchical structures.
   - Any **text or labels**, and their meanings if technical.

2. **If the image shows a computer screen**, describe the content displayed (not the screen itself):
   - For **slides or presentations**, summarize the slide’s title, main points, and purpose.
   - For **PDFs or Word documents**, summarize the visible text and infer the document’s goal.
   - For **file explorers**, summarize the structure (e.g., “a directory listing of documents”).
   - For **software interfaces**, explain what software is shown and what it is being used for.

3. **Do NOT describe irrelevant UI elements** (e.g., toolbars, menus) unless critical for understanding.

4. **For any visible text**, extract and **summarize** it clearly (not line by line unless necessary).

5. **Avoid redundancy**:
   - Do not repeat file names, document titles, or labels unless essential.
   - Summarize repetitive UI or directory content as a group.

Return your description in fluent, structured English.
"""


RAG_PROMPT_TEMPLATE_NORMAL_ACTION = """
You are an expert assistant trained to analyze educational videos using timestamp-aligned multimodal content.

The video has a total duration of **{video_duration} seconds**.

Below are the retrieved segments, each marked with:
- **[audio]**: transcription from the speaker's voice
- **[visual]**: visual description of what appeared on screen
Each segment includes the **exact timestamp** during which it occurred.

Your task is to answer the user question based strictly on the information found in the retrieved segments. You may summarize, combine, or infer information across multiple segments if appropriate.

If the retrieved segments provide partial but meaningful information, you should try to answer as fully as possible using only that information.

Only if the video **truly lacks relevant information** should you respond:  
**"The video does not contain enough information to answer this question."**

### Retrieved Segments:
{retrieved_context}

**User Question:** {user_question}

**AI Answer:**
"""

GEVAL_CORRECTNESS_STEPS = [
    "Evaluate whether the 'actual output' conveys the same core facts or meaning as the 'expected output', even if phrased differently.",
    "Do not penalize for differences in wording, synonyms, or style if the essential information is preserved.",
    "Omitting minor or non-essential details is acceptable as long as the main answer is correct.",
    "Extra information is acceptable unless it introduces factual inaccuracies or contradictions.",
    "Focus primarily on whether the 'actual output' provides a faithful and meaningful answer to the ground truth.",
    "If the answer is only partially correct, assign a fractional score between 0 and 1 to reflect the degree of correctness."
]


RAG_PROMPT_TEMPLATE_QA_GENERATE = """
You are an AI assistant specialized in understanding lecture videos by interpreting timestamped multimodal information.

The video has a total duration of **{video_duration} seconds**.

Each retrieved segment below is marked with a prefix indicating its source:
- Segments starting with **[audio]** are transcriptions of what the speaker said, captured from the video's audio track during the indicated time interval.
- Segments starting with **[visual]** are descriptions of what was shown on screen (slides, diagrams, interface, etc.) during that time period.


Timestamps reflect the exact time range in the video when each segment occurred.
Use this information to reason about the video's temporal flow and answer time-related questions.

You must answer using only the content retrieved below.
Output **ONLY** the final answer text, without explanation, reasoning, or commentary.
Do not make assumptions or generate content beyond the given segments.
Strive for completeness: include all parts explicitly supported by the segments, and if you think the answer is not unique, include all distinct valid answers.

### Retrieved Video & Audio Segments:
{retrieved_context}

**User Question:** {user_question}

**AI Answer:**
"""


DIAGRAM_ROUTER_CLASSIFY_PROMPT = """
You are a very fast triage classifier.
Say whether the image is DIAGRAM-LIKE.

Diagram-like categories include: diagram, flowchart, schematic, architecture diagram,
graph/plot/chart, table, UI wireframe/mockup, whiteboard with boxes/arrows/equations.
Natural photos/scenes/people/cars/roads are NOT diagram-like.

Output exactly ONE line in this format:
LABEL: <DIAGRAM|NOT_DIAGRAM>; TYPES: <comma-separated subset from {diagram,flowchart,schematic,chart,table,wireframe,equation,other}>; CONF: <0.00-1.00>
"""


GPT5_DIAGRAM_PROMPT_TEMPLATE = """
You are describing a diagram/flowchart frame from an educational video (time: {timestamp}).
Produce a maximally detailed, strictly image-grounded description. Write plain English only
(no JSON). Use the following sections exactly and be exhaustive.

Title (if visible):
- Quote the exact title text. If none, write "None".

Purpose:
- One sentence on what the diagram/flowchart is trying to communicate.

Global layout:
- Overall orientation (left→right, top→down, circular), columns/rows, swimlanes or panels,
  grouping/boundaries (modules, subsystems), and notable color conventions.

Node inventory:
- Count and enumerate ALL nodes in reading order. For each node: exact label text (quote),
  role/type (process/decision/data/terminator/external), shape (rectangle/diamond/ellipse/
  parallelogram/cylinder/cloud), notable color/style, and relative position (e.g., "above B",
  "right column, row 2").

Edge inventory:
- Count and enumerate ALL edges. For each edge: source → target, arrow style (solid/dashed/
  bidirectional), labels/conditions on the connector, and whether it is a branch, merge or loop-back.

Flow sequence:
- Step-by-step path from start to finish. Include decisions explicitly as:
  "If <condition> then → <branch>, else → <branch>". Mention loops, termination conditions and
  any parallel/concurrent paths.

Text & labels:
- Quote important labels verbatim. Summarize long paragraphs succinctly. Mark unclear/occluded
  text as [illegible] rather than guessing.

Data & variables (if present):
- Inputs/outputs, parameter names, units, file/table names, API/interface names.

Context anchors:
- Legends, icons, logos, axis/scale markers (if any), and any region titles or boundaries.

Key takeaway:
- One or two sentences summarizing the main message of the figure.

Rules:
- Describe ONLY what is visible. Do not invent facts beyond the image.
- Prefer exact quotes for labels; otherwise paraphrase briefly.
- Report counts explicitly: "N nodes, M edges".
- Keep technical terms from the image; expand acronyms only if they are spelled out in the image.
"""


GEVAL_CONTEXT_PRECISION_STEPS = [
    "You are evaluating the relevance of retrieved context chunks for a question.",
    "Consider each chunk independently: a chunk is relevant if it contains information that helps answer the question, even if it does not fully answer it.",
    "Ignore the model's final answer; only judge the chunks relative to the question’s information need.",
    "Return a single number between 0 and 1: (# of relevant chunks) / (total # of chunks).",
]

GEVAL_CONTEXT_RECALL_STEPS = [
    "You are judging if the retrieved context collectively covers the key information needed to answer the question as reflected in the expected answer.",
    "Assess coverage, not strict wording: if the necessary facts are present across the chunks, count as covered.",
    "Return a single number between 0 and 1 representing coverage recall.",
]

GEVAL_TRUTHFULNESS_STEPS = [
    "You are checking if the actual answer is faithful to the retrieved context for the question.",
    "Penalize statements that contradict the context or are not supported by the context.",
    "Do not reward verbosity; focus on factual support.",
    "Return a single number between 0 and 1 representing truthfulness with respect to the provided context.",
]


RAG_PROMPT_TEMPLATE_QA_WITH_GROUNDING = """
You are an AI assistant specialized in answering questions about lecture videos using timestamped multimodal information.
 
The video has a total duration of **{video_duration} seconds**.
 
You are given a list of retrieved segments below. Each line has the form:
[source] <timestamp_in_seconds>: <text>
 
- Segments starting with **[audio]** are transcriptions of what the speaker said, captured from the video's audio track during the indicated time interval.
- Segments starting with **[visual]** are descriptions of what was shown on screen (slides, diagrams, interface, etc.) during that time period.
 
You must answer using only the content retrieved below.
 
After reasoning, output in the following simple structured format:
 
Answer:
<your final answer in natural language, based only on the segments>
 
Used_context:
- <one retrieved segment line you considered relevant, with the timestamp also shown in mm:ss format>
- <another retrieved segment line you considered relevant, with the timestamp also shown in mm:ss format>
...
 
Instructions for Used_context (lightweight):
- List 1–10 segments that you believe are most relevant for answering the question.
- When you output each segment, convert its timestamp from seconds to minutes and show both, for example:
  "[audio] 120-180s (02:00–03:00): ..." or "[visual] 75-95s (01:15–01:35): ..."
- You may truncate the text part for brevity, but do not invent new facts.
- It is acceptable if you include some extra but still relevant segments.
- If the video truly lacks enough information to answer, write:
  Answer:
  The video does not contain enough information to answer this question.
  Used_context:
  []
 
### Retrieved Video & Audio Segments:
{retrieved_context}
 
**User Question:** {user_question}
"""
