from ADS_Simulator.image_only_pipeline import image_only_information_extraction
from ADS_Simulator.narration_builder import build_video_narration

def generate_narration_from_video(video_path: str, overwrite: bool = False) -> str:
    """
    Given a video file, extract frames, generate image descriptions, build narration.
    Args:
        video_path (str): Path to the input .mp4 video
        overwrite (bool): If True, forces reprocessing of frames and chunks
    Returns:
        str: Generated narration text
    """
    print(f"🎞️  Processing video: {video_path}")
    
    # Step 1: Extract visual information and build chunks
    image_only_information_extraction(video_path, overwrite=overwrite)

    # Step 2: Build narration using all visual chunks
    narration = build_video_narration(video_path)

    print("\n📝 Generated Narration:\n")
    print(narration)
    return narration

if __name__ == "__main__":
    video_path = "ADS_input_videos/ThirdPerson's_perspective.mp4"
    # video_path = "input_videos/Driver's_perspective.mp4"
    generate_narration_from_video(video_path, overwrite=False)
