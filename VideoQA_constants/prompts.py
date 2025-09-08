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

### Retrieved Video & Audio Segments:
{retrieved_context}

**User Question:** {user_question}

**AI Answer:**
"""
