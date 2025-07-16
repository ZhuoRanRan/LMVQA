ADS_PROMPT_TEMPLATE_GROUPED = """
You are an intelligent assistant analyzing a sequence of video frames from an autonomous driving simulation.

Your task is to analyze these frames and describe, in structured and coherent English, what happened during the entire time period.

Use the following strategy:

1. **Time-localized behavior**: Use the provided timestamps to describe when key events occurred. For example:
   - "From 0–7s, the car is stationary at a red light."
   - "From 14–18s, the vehicle initiates and completes a left turn."

2. **Driving actions**: Focus on detecting meaningful driving behaviors such as:
   - Car starts, stops, turns left/right, changes lanes
   - Acceleration or deceleration
   - Following or overtaking another vehicle
   - Waiting at a stop sign or traffic light

3. **Environmental context**:
   - Identify road type (city street, highway, intersection, etc.)
   - Note presence of traffic signs, signals, parked cars, pedestrians, and nearby vehicles
   - Mention weather or visibility conditions if evident

4. **Event reasoning**: Use visual indicators to infer intent or upcoming behavior, such as:
   - Right/left turn signals
   - Brake lights
   - Vehicle body tilt
   - Steering wheel angle or mirror position
   - Speedometer or dashboard indicators

5. **Anomaly detection**: If anything appears to be a traffic rule violation or unsafe behavior (e.g., failing to stop, cutting off another car, ignoring signs), call it out clearly.

You will now receive a list of timestamped frames, followed by 10 visual frames. Each frame corresponds to a specific timestamp (e.g., "frame_0-7s.png"). Use these timestamps to associate image cues with driving behavior.

Write a coherent paragraph-level summary of what occurred during the entire range. Mention transitions (start, stop, turn), the reasoning for them (e.g., signal, traffic, blockage), and annotate any rule violations you detect.
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
