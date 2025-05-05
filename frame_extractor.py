import cv2
import os

def extract_frames(video_path, output_folder, fps_sampling=1):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / fps_sampling) if fps_sampling != 0 else 1

    count = 0
    saved = 0
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
