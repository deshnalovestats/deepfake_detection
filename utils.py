import os
import re

def get_all_video_paths(root_folder):
    video_paths = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_paths.append(os.path.join(subdir, file))
    video_paths.sort(key=natural_key)
    return video_paths

def natural_key(string):
    # Splits string into parts of digits and non-digits
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', string)]
