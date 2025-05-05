import os
import numpy as np
import cv2
import concurrent.futures
import shutil
from typing import List, Tuple, Optional, Dict

from frame_extractor import extract_frames
from face_feature_extractor import extract_face_feature

def process_frame_batch(batch_data: List[Tuple[str, str, str, int, bool]]) -> Dict[int, np.ndarray]:
    """
    Process a batch of frames to extract faces and features
    
    Args:
        batch_data: List of (frame_path, crop_save_dir, video_id, frame_idx, align) tuples
        
    Returns:
        Dictionary mapping frame indices to features
    """
    results = {}
    for frame_path, crop_save_dir, video_id, frame_idx, align in batch_data:
        feat = extract_face_feature(
            image_path=frame_path, 
            crop_save_dir=crop_save_dir, 
            video_id=video_id, 
            frame_idx=frame_idx,
            align=align,
        )
        if feat is not None:
            results[frame_idx] = feat
    return results

def process_single_video(video_path: str, 
                         save_dir: str, 
                         crop_save_dir: str = None,
                         temp_frame_dir: str = None, 
                         fps_sampling: int = 1, 
                         max_workers: int = 4, 
                         batch_size: int = 8,
                         align_faces: bool = True) -> bool:
    """
    Process a single video with multithreaded face extraction
    
    Args:
        video_path: Path to the video file
        save_dir: Directory to save feature embeddings
        crop_save_dir: Directory to save face crops
        temp_frame_dir: Directory for temporary frame storage
        fps_sampling: Frame sampling rate
        max_workers: Maximum number of worker threads
        batch_size: Number of frames to process in each batch
        
    Returns:
        True if processing was successful, False otherwise
    """
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create necessary directories
    os.makedirs(save_dir, exist_ok=True)
    if crop_save_dir:
        os.makedirs(crop_save_dir, exist_ok=True)
    
    # Use a unique temp directory for each video if not specified
    temp_dir_created = False
    if temp_frame_dir is None:
        temp_frame_dir = os.path.join('temp', f"frames_{video_id}")
        temp_dir_created = True
        
    os.makedirs(temp_frame_dir, exist_ok=True)
    
    # Extract frames
    try:
        frames_count = extract_frames(video_path, temp_frame_dir, fps_sampling)
        if frames_count == 0:
            print(f"No frames extracted from {video_id}")
            return False
    except Exception as e:
        print(f"Error extracting frames from {video_id}: {e}")
        return False
    
    # Get frame files
    frame_files = sorted(os.listdir(temp_frame_dir))
    
    # Prepare batches for processing
    batches = []
    current_batch = []
    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(temp_frame_dir, frame_file)
        current_batch.append((frame_path, crop_save_dir, video_id, i, align_faces))
        
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []
    
    if current_batch:
        batches.append(current_batch)
    
    # Process batches in parallel
    features_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_frame_batch, batch) for batch in batches]
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                batch_results = future.result()
                features_dict.update(batch_results)
            except Exception as e:
                print(f"Error processing batch in {video_id}: {e}")
    
    # Convert results to ordered list and save
    if features_dict:
        # Sort by frame index to maintain order
        ordered_features = [features_dict[idx] for idx in sorted(features_dict.keys())]
        features_array = np.stack(ordered_features)
        np.save(os.path.join(save_dir, f"{video_id}.npy"), features_array)
        print(f"Saved features for {video_id} - {features_array.shape[0]} faces")
    else:
        print(f"No faces found in video {video_id}")
        return False
    
    # Clean up temp frames
    if temp_dir_created or os.path.exists(temp_frame_dir):
        try:
            shutil.rmtree(temp_frame_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temp directory {temp_frame_dir}: {e}")
    
    return True

# # testing out
# if __name__ == "__main__":
#     video_path = 'data/Celeb-real/id0_0000.mp4'  # Replace with your video path
#     save_dir = 'features'  # Replace with your save directory
#     temp_frame_dir = 'temp/frames'  # Replace with your temp frame directory

#     process_single_video(video_path, save_dir, temp_frame_dir, fps_sampling=1)