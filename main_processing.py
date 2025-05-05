import os
from tqdm import tqdm
from utils import get_all_video_paths
from process_videos import process_single_video

raw_data_dir = 'data'
save_features_dir = 'features'
save_crops_dir = 'face_crops'
temp_frame_dir = 'temp/frames'

os.makedirs(save_features_dir, exist_ok=True)
os.makedirs(save_crops_dir, exist_ok=True)
os.makedirs(temp_frame_dir, exist_ok=True)

video_paths = get_all_video_paths(raw_data_dir)

for video_path in tqdm(video_paths, desc="Processing Videos",ncols=100):
    video_name = os.path.basename(video_path)
    tqdm.write(f"Processing: {video_name}")

    relative_path = os.path.relpath(video_path, raw_data_dir)
    parent_folder = os.path.dirname(relative_path)  
    feature_save_dir = os.path.join(save_features_dir, parent_folder)
    os.makedirs(os.path.dirname(feature_save_dir), exist_ok=True)

    # Process each video and save features
    process_single_video(
        video_path=video_path,
        save_dir=feature_save_dir,
        temp_frame_dir=temp_frame_dir,
        fps_sampling=2  
    )

