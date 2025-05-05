import os
import re
from typing import List

def natural_key(string):
    """
    Sort key for natural sorting of filenames
    Example: ['file1', 'file10', 'file2'] becomes ['file1', 'file2', 'file10']
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', string)]

def get_all_video_paths(root_folder: str) -> List[str]:
    """
    Get all video paths with natural sorting
    
    Args:
        root_folder: Root directory to search for videos
        
    Returns:
        List of video file paths sorted naturally
    """
    video_paths = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_paths.append(os.path.join(subdir, file))
    video_paths.sort(key=natural_key)
    return video_paths

def ensure_dir(directory: str) -> None:
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)