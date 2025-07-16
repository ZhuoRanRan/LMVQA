VIDEO_QA_IMAGE_PROMPT_TEMPLATE = """
You are an assistant analyzing a group of video frames captured over a short period of time.

Your goal is to generate a detailed paragraph summarizing everything visible during this interval.

Use the following strategy:

1. **Describe all important objects and people**:
   - What objects appear? (e.g., cars, furniture, tools, animals)
   - What are their characteristics? (e.g., color, size, shape, labels, clothing)
   - Where are they located relative to others?

2. **Report visible text or numbers**:
   - If you can see license plates, signs, screens, or any text, write down what is readable.

3. **Describe visual changes over time**:
   - Who or what moves between frames?
   - Any interactions between objects or people?
   - Any new objects that appear or disappear?

4. **Scene context**:
   - Indoor or outdoor?
   - Urban or rural?
   - Time of day, weather, or lighting if evident?

**Important**:
- Be extremely detailed. Include small visual cues that may be relevant to answering questions later.
- Mention all visible vehicles, people, signs, colors, and text.
- Avoid guessing or making assumptions about intent—just describe what is visually present.

You will receive 10 frames from a short time window. Write a comprehensive paragraph summarizing what is visible and what changes.
"""


QA_ANSWERING_PROMPT = """
You are an intelligent assistant. A user is asking about a segment of a real-world video.

The total duration of the video is **{video_duration} seconds**.

You are given a series of timestamped descriptions extracted from the video (visual and possibly audio).

Your job is to:
- Carefully answer the user’s question.
- Use only the provided descriptions to infer the answer.
- Be precise about timing, actions, and roles of people or objects.
- If the question involves a specific time, cross-check that segment in the context.
- If the answer cannot be inferred from the context, say "The video context does not provide enough information."

User Question:
{user_query}

Video Context:
{extracted_context}

Answer:
"""

