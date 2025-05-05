import cv2
import numpy as np
import insightface # type: ignore


model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0, det_size=(640,640))

def extract_face_feature(image_path, debug=0):
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

    faces = model.get(img_rgb)

    if debug:
        print(f"Number of faces detected: {len(faces)}")

    if len(faces) == 0:
        if debug:
            print("No faces found in the image.")
        return None

    largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

    if debug:
        print(f"Largest face bounding box: {largest_face.bbox}")

        # Draw all detected faces
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Optionally draw facial landmarks
            if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                for (x, y) in face.landmark_2d_106:
                    cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)

        # Save debug image
        debug_path = image_path.replace('.jpg', '_debug.jpg')  # saves next to input image
        cv2.imwrite(debug_path, img)
        print(f"Debug image saved at {debug_path}")

    feature = largest_face.normed_embedding

    if debug:
        print("Feature extraction successful.")

    return feature
