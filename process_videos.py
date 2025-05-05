import os
import numpy as np
import cv2
from frame_extractor import extract_frames
from face_feature_extractor import extract_face_feature
from insightface.app import FaceAnalysis
from tqdm import tqdm

def align_face(img, landmarks, output_size=(112, 112)):
    src = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    dst = landmarks.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(dst, src, method=cv2.LMEDS)
    aligned_face = cv2.warpAffine(img, M, output_size, borderValue=0.0)
    return aligned_face

def process_single_video(video_path, save_dir, temp_frame_dir, fps_sampling=1,save_faces=False):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    
    os.makedirs(save_dir, exist_ok=True)

    extract_frames(video_path, temp_frame_dir, fps_sampling)

    features = []
    frame_files = sorted(os.listdir(temp_frame_dir))
    for frame_file in frame_files:
        frame_path = os.path.join(temp_frame_dir, frame_file)
        feat = extract_face_feature(frame_path,debug=0)
        if feat is not None:
            features.append(feat)

    if len(features) > 0:
        features = np.stack(features)
        np.save(os.path.join(save_dir, f"{video_id}.npy"), features)
    else:
        print(f"No faces found in video {video_id}")

    # Clean temp frames
    for file in frame_files:
        os.remove(os.path.join(temp_frame_dir, file))

# # testing out
# if __name__ == "__main__":
#     video_path = 'data/Celeb-real/id0_0000.mp4'  # Replace with your video path
#     save_dir = 'features'  # Replace with your save directory
#     temp_frame_dir = 'temp/frames'  # Replace with your temp frame directory

#     process_single_video(video_path, save_dir, temp_frame_dir, fps_sampling=1)