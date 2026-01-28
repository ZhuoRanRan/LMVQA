from VideoQA_Pipeline.GPT4_generate_description import process_video_frames

video_name = "Lecture1"

descriptions = process_video_frames(video_name)

print(f"\n✅ Total descriptions generated: {len(descriptions)}")
