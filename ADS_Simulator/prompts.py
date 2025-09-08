ADS_PROMPT_TEMPLATE_GROUPED = """
You are an intelligent assistant analyzing a sequence of video frames from an autonomous driving simulation.

Your task is to analyze these frames and describe, in structured and coherent English, what happened during the entire time period.

Use the following strategy:

1. **Time-localized behavior**: Use the provided timestamps to describe when key events occurred. For example:
   - "From 0–7s, the car is stationary at a red light."
   - "From 14–18s, the vehicle initiates and completes a left turn."

2. **Driving actions**: Detect and describe meaningful driving behaviors such as:
   - Starting, stopping, accelerating, decelerating
   - Turning left/right, lane changing
   - Following or overtaking another vehicle
   - Waiting at traffic lights or stop signs

3. **Environmental context**:
   - Identify road topology (e.g., straight, curved, intersection, highway)
   - Note presence of road signs (e.g., stop, yield, speed limit), signals, pedestrians, parked or moving vehicles
   - Mention weather or visibility conditions if visible

4. **Dashboard and speed cues**:
   - Pay close attention to the vehicle's dashboard (bottom-right corner), including **current speed**, **brake indicators**, and **turn signals**
   - Check for **speed limit signs** in the environment and compare with current speed
   - If the car is exceeding the posted speed, mention it as a potential **speeding violation**

5. **Violation detection**: Explicitly identify any behavior that may violate traffic rules. The following violation types are of special interest:
   - **Failure to stop at stop sign**
   - **Running a red light**
   - **Failure to yield** (to pedestrians or other vehicles)
   - **Speeding**
   - **Illegal lane change**
   - (If no rule is violated, state that the behavior appears normal)

6. **Intent inference**: Use clues like brake lights, turn signals, vehicle tilt, or wheel alignment to anticipate the car's intent.

You will now receive 10 consecutive video frames, each with its own timestamp (e.g., "frame_10–13s.png"). These frames span a continuous driving segment. Write a fluent, timestamp-anchored description of what happened during the whole period.

Structure your response as a paragraph-level explanation. Be sure to:
- Mention transitions (start, stop, turn, acceleration)
- Comment on relevant road and traffic context
- Identify **any potential violations** if they occur, and briefly explain the evidence
"""

NARRATION_CONSTRUCTION_PROMPT = """
You are an intelligent assistant generating a **structured, timestamp-aligned narration** from a self-driving simulation video.

The total duration of the video is **{video_duration} seconds**.

You are given a sequence of visual descriptions extracted from video frames, each mapped to a timestamp. Your task is to:

1. Identify and summarize all **key driving events**.
2. Clearly identify and label **traffic rule violations** (e.g., red light run, illegal turn, failure to yield).
3. Structure your narration using **consistent event categories**:
   - `START:`
   - `STOP:`
   - `ACCELERATE:`
   - `DECELERATE:`
   - `TURN_LEFT:` / `TURN_RIGHT:`
   - `LANE_CHANGE:`
   - `SIGNAL:` (for traffic lights or stop signs)
   - `OBSTACLE:` (e.g., parked cars, construction zones)
   - `VIOLATION:` (any illegal or unsafe behavior)
   - `COLLISION:` (if applicable)

Each entry should follow this format:  
`[start_time–end_time] CATEGORY: description of the event.`

Be precise and concise. Avoid vague or repetitive descriptions. Focus on what the vehicle does and what is relevant to driving decisions.

Here is an example:

[0–5s] STOP: The car is stationary at a red light at an intersection.  
[5–8s] ACCELERATE: The car begins to move as the light turns green.  
[8–10s] TURN_RIGHT: The car turns right onto a side street.  
[10–13s] VIOLATION: The car fails to stop at a visible stop sign.  
[13–17s] ACCELERATE: The car drives straight and increases speed to 30 mph.

Only include events relevant to driving behavior or environment. Ignore non-driving scenery. Use only the above categories.

Frame Descriptions:  
{retrieved_context}
"""

CONCEPT_EXTRACTION_BATCH_PROMPT = """
You are an assistant trained to extract structured driving concepts from timestamped driving narrations in batches.

Each narration describes what happened in a short time window in a self-driving simulation. For each narration, extract the following fields and output a JSON object per line:

### Fields to extract per narration:

- `timestamp`: Use the timestamp from the narration.
- `violation_label`: True if the narration indicates a traffic violation; False otherwise.
- `violation_type`: Choose from the following **exact labels**:
  "stop_sign_violation", "red_light_violation", "yield_violation", 
  "speeding_violation", "illegal_lane_change", or "normal" (if no violation)
- `vehicle_action`: One of:
  "ACCELERATE", "DECELERATE", "MAINTAIN_SPEED", "STOP"
- `brake_engaged`: True if brake lights, slowing down, or stopping is described; False otherwise.
- `vehicle_speed_mph`: Extract the number if mentioned (e.g., "accelerates to 41 mph"); else null.
- `stop_sign_visible`: True if a stop sign is mentioned; else False.
- `traffic_light_visible`: True if a traffic light is mentioned; else False.
- `road_topology`: One of: "straight", "intersection", "turn", "merge", "crosswalk"
- `pedestrian_visible`: True if pedestrians are mentioned; else False.
- `yield_sign_visible`: True if a yield sign is mentioned; else False.

### Instructions:

- Return a valid JSON object per narration in the exact order received.
- Do not add extra commentary or explanation.
- If a concept is not mentioned, use `false` or `null` as appropriate.

---

### Narration Batch:

{narration_batch}

Please output the structured JSON list below:
"""
