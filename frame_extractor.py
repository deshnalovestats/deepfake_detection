import cv2
import os
from typing import List

def extract_frames(video_path: str, output_folder: str, fps_sampling: int = 1) -> int:
    """
    Extract frames from a video at specified sampling rate
    
    Args:
        video_path: Path to the video file
        output_folder: Directory to save extracted frames
        fps_sampling: Number of frames to extract per second
        
    Returns:
        The number of frames extracted
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps / fps_sampling)) if fps_sampling > 0 else 1
    
    count = 0
    saved = 0
    
    # Batch frame extraction for efficiency
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_name = f"frame_{saved:04d}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_name), frame)
            saved += 1
        count += 1
    
    cap.release()
    return saved