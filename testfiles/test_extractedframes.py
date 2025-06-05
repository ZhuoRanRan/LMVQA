from VideoQA_Pipeline.extracted_frames import extract_video_frames

video_path = "input_videos/Lecture1.mp4"

frames, filenames, video_type = extract_video_frames(video_path)

print("\n📊 Test Summary:")
print(f"Total frames extracted: {len(frames)}")
print(f"Video classified as: {video_type}")
