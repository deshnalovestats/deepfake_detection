import os
import concurrent.futures
import shutil
from tqdm import tqdm

from utils import get_all_video_paths, ensure_dir
from process_videos import process_single_video

def process_videos_parallel(raw_data_dir: str, 
                           save_features_dir: str, 
                           save_crops_dir: str,
                           temp_frame_base_dir: str, 
                           max_video_workers: int = 2,
                           max_frame_workers: int = 4,
                           fps_sampling: int = 2,
                           align_faces: bool = True):
    """
    Process multiple videos in parallel
    
    Args:
        raw_data_dir: Directory containing videos
        save_features_dir: Directory to save feature embeddings
        save_crops_dir: Directory to save face crops
        temp_frame_base_dir: Base directory for temporary frame storage
        max_video_workers: Maximum videos to process in parallel
        max_frame_workers: Maximum threads per video for frame processing
        fps_sampling: Frame sampling rate
    """
    # Create necessary directories
    ensure_dir(save_features_dir)
    ensure_dir(save_crops_dir)
    ensure_dir(temp_frame_base_dir)
    
    video_paths = get_all_video_paths(raw_data_dir)
    total_videos = len(video_paths)
    print(f"Found {total_videos} videos to process")
    
    # Process videos in parallel
    with tqdm(total=total_videos, desc="Processing Videos", ncols=100) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_video_workers) as executor:
            futures = []
            
            for i, video_path in enumerate(video_paths):
                video_name = os.path.basename(video_path)
                relative_path = os.path.relpath(video_path, raw_data_dir)
                parent_folder = os.path.dirname(relative_path)
                
                # Create unique directories for this video
                feature_save_dir = os.path.join(save_features_dir, parent_folder)
                crop_save_dir = os.path.join(save_crops_dir, parent_folder)
                temp_frame_dir = os.path.join(temp_frame_base_dir, f"video_{i}")
                
                ensure_dir(feature_save_dir)
                ensure_dir(crop_save_dir)
                
                tqdm.write(f"Submitting: {video_name}")
                
                # Submit video for processing
                future = executor.submit(
                    process_single_video,
                    video_path=video_path,
                    save_dir=feature_save_dir,
                    crop_save_dir=crop_save_dir,
                    temp_frame_dir=temp_frame_dir,
                    fps_sampling=fps_sampling,
                    max_workers=max_frame_workers,
                    align_faces=align_faces
                )
                futures.append(future)
            
            # Update progress as videos complete
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)
                try:
                    future.result()
                except Exception as e:
                    tqdm.write(f"Error processing video: {e}")

def main():
    # Configuration
    raw_data_dir = 'data'
    save_features_dir = 'features'
    save_crops_dir = 'face_crops'
    temp_frame_dir = 'temp/frames'
    
    # Clean up temp directory if it exists
    if os.path.exists(temp_frame_dir):
        shutil.rmtree(temp_frame_dir)
    ensure_dir(temp_frame_dir)
    
    # Process videos with parallelization
    process_videos_parallel(
        raw_data_dir=raw_data_dir,
        save_features_dir=save_features_dir,
        save_crops_dir=save_crops_dir,
        temp_frame_base_dir=temp_frame_dir,
        max_video_workers=2,  # Process 2 videos in parallel (adjust based on your CPU)
        max_frame_workers=4,  # Use 4 threads per video for frame processing
        fps_sampling=2,       # Extract 2 frames per second
        align_faces=True      # Enable face alignment
    )

if __name__ == "__main__":
    main()