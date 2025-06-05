import os
import cv2
import numpy as np
import shutil
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from VideoQA_Pipeline.utils import get_video_duration, classify_video_type

def extract_video_frames(video_path, ssim_threshold=0.85):
    """
    Extract key frames from a video based on SSIM & histogram similarity.
    
    Args:
        video_path (str): Path to the video file.
        ssim_threshold (float): SSIM similarity threshold (lower = stricter filtering).
        
    Returns:
        list: List of extracted frames (PIL.Image format).
        list: Frame file names (including timestamps).
        str: Video classification ("low-action" or "normal-action").
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps <= 0:
        print("❌ Invalid FPS detected. Exiting frame extraction.")
        cap.release()
        return [], [], "low-action"

    # ✅ Use get_video_duration() from utils.py
    duration = get_video_duration(video_path)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"🎥 Processing video: {video_path}")
    print(f"⏳ Duration: {duration:.2f} sec, 🎞 FPS: {fps:.2f}")

    if duration == 0:
        print("❌ No valid frames found.")
        cap.release()
        return [], [], "low-action"

    # ✅ Updated output directory path
    output_dir = os.path.join("outputs", "frames", f"{video_name}_frames")
    if os.path.exists(output_dir):
        print(f"🗑️ Deleting existing directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    frames = []
    frame_filenames = []
    prev_frame_gray = None
    last_saved_time = 0

    for sec in range(0, int(np.floor(duration))):
        frame_idx = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"⚠️ Skipping frame at {sec}s (index {frame_idx})")
            continue

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Check similarity with previous frame
        if prev_frame_gray is not None:
            similarity = ssim(prev_frame_gray, gray_frame)
            if similarity > ssim_threshold:
                print(f"⏩ Skipping frame at {sec}s (SSIM: {similarity:.2f})")
                continue

        prev_frame_gray = gray_frame
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Ensure correct timestamp in filename
        if frames:
            old_filename = frame_filenames[-1]
            new_filename = old_filename.split("_")[0] + f"_{last_saved_time}-{sec}s.png"
            os.rename(os.path.join(output_dir, old_filename), os.path.join(output_dir, new_filename))
            frame_filenames[-1] = new_filename

        frame_filename = f"frame_{len(frames)+1:03d}_{sec}-?.png"
        frame_path = os.path.join(output_dir, frame_filename)
        pil_image.save(frame_path)

        frames.append(pil_image)
        frame_filenames.append(frame_filename)
        last_saved_time = sec

    cap.release()

    # ✅ Update last frame timestamp to cover until the end of the video
    if frames:
        old_filename = frame_filenames[-1]
        new_filename = old_filename.split("_")[0] + f"_{last_saved_time}-{int(duration)}s.png"
        os.rename(os.path.join(output_dir, old_filename), os.path.join(output_dir, new_filename))
        frame_filenames[-1] = new_filename

    video_type = classify_video_type(frames, duration)
    print(f"✅ Extracted {len(frames)} key frames. Video classified as: {video_type}")

    return frames, frame_filenames, video_type
