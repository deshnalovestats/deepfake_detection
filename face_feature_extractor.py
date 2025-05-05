import cv2
import numpy as np
import insightface # type: ignore[import]
import threading 
import os
from typing import Optional, Tuple

# Global model initialization with thread lock to ensure thread safety
model_lock = threading.Lock()
_model = None

def get_model():
    """Thread-safe model initialization"""
    global _model
    with model_lock:
        if _model is None:
            _model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            _model.prepare(ctx_id=-1, det_size=(640, 640))  # Maintain original detection size for quality
    return _model

def align_face(img, landmarks, output_size=(112, 112)):
    """
    Align face based on facial landmarks
    
    Args:
        img: Input image
        landmarks: Facial landmarks
        output_size: Size of aligned face output
        
    Returns:
        Aligned face image
    """
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



def extract_face_feature(image_path: str, crop_save_dir: str = None, video_id: str = None, 
                         frame_idx: int = None, debug: int = 0, align: bool = True) -> Optional[np.ndarray]:
    """
    Extract the largest face and its features from an image
    
    Args:
        image_path: Path to the image file
        crop_save_dir: Directory to save the face crop (optional)
        video_id: ID of the video being processed (optional)
        frame_idx: Frame index for naming the crop (optional)
        debug: Debug level (0=none, 1=print, 2=save debug images)
        align: Whether to align faces before saving crops
        
    Returns:
        Face embedding features or None if no face detected
    """
    if debug:
        print("Reading the image...")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot load image from {image_path}")
        return None
    
    if debug:
        print(f"Image shape: {img.shape}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if debug:
        print("Converted BGR to RGB.")
    
    # Get model in thread-safe way
    model = get_model()
    faces = model.get(img_rgb)
    
    if debug:
        print(f"Number of faces detected: {len(faces)}")
    
    if len(faces) == 0:
        if debug:
            print("No faces found in the image.")
        return None
    
    # Get the largest face
    largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    
    if debug:
        print(f"Largest face bounding box: {largest_face.bbox}")
    
    # Save face crop if requested
    if crop_save_dir and video_id is not None and frame_idx is not None:
        # For original crop with margin
        x1, y1, x2, y2 = map(int, largest_face.bbox)
        # Add margin
        h, w = img.shape[:2]
        margin = min(30, min(x1, y1, w-x2, h-y2))
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        face_crop = img[y1:y2, x1:x2]
        
        # Save aligned face if requested
        if align and hasattr(largest_face, 'landmark_2d_106') and largest_face.landmark_2d_106 is not None:
            landmarks = largest_face.landmark_2d_106
            # Use 5 key facial landmarks for alignment (eyes, nose, mouth corners)
            # Map from 106 points to 5 points used in alignment
            landmark_map = {
                "left_eye": 38,      # Left eye center
                "right_eye": 88,     # Right eye center
                "nose": 86,          # Nose tip
                "left_mouth": 52,    # Left mouth corner
                "right_mouth": 61    # Right mouth corner
            }
            
            alignment_landmarks = np.array([
                landmarks[landmark_map["left_eye"]],
                landmarks[landmark_map["right_eye"]],
                landmarks[landmark_map["nose"]],
                landmarks[landmark_map["left_mouth"]],
                landmarks[landmark_map["right_mouth"]]
            ])

            if debug >= 2:
                print(f"Alignment landmarks: {alignment_landmarks}")
                for (x, y) in alignment_landmarks:
                    cv2.circle(img_rgb, (int(x), int(y)), 1, (0, 255, 0), -1)
                # Save the debug image once after drawing all landmarks
                cv2.imwrite(image_path.replace('.jpg', '_landmarks.jpg'), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

                            
            # Align face using the 5 landmarks
            aligned_face = align_face(img_rgb, alignment_landmarks, output_size=(112, 112))

            
            
            # Save both original crop and aligned face
            video_folder = os.path.join(crop_save_dir, video_id)
            os.makedirs(video_folder, exist_ok=True)
            orig_crop_path = os.path.join(video_folder, f"frame{frame_idx:04d}_orig.jpg")
            aligned_crop_path = os.path.join(video_folder, f"frame{frame_idx:04d}.jpg")
            
            cv2.imwrite(orig_crop_path, face_crop)
            cv2.imwrite(aligned_crop_path, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
        else:
            # Just save the original crop if alignment isn't possible
            os.makedirs(crop_save_dir, exist_ok=True)
            crop_path = os.path.join(crop_save_dir, f"{video_id}_frame{frame_idx:04d}.jpg")
            cv2.imwrite(crop_path, face_crop)
    
    if debug >= 2:
        # Draw all detected faces
        debug_img = img.copy()
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Optionally draw facial landmarks
            if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                for (x, y) in face.landmark_2d_106:
                    cv2.circle(debug_img, (int(x), int(y)), 1, (0, 0, 255), -1)
        
        # Save debug image
        debug_path = image_path.replace('.jpg', '_debug.jpg')  # saves next to input image
        cv2.imwrite(debug_path, debug_img)
        print(f"Debug image saved at {debug_path}")
    
    feature = largest_face.normed_embedding
    
    if debug:
        print("Feature extraction successful.")
    
    return feature