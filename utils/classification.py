import os
import numpy as np
from collections import defaultdict
from .face_recognition import create_face_embeddings

CLASSIFICATION_SIMILARITY_THRESHOLD = 0.5  # Threshold for classifying images

def classify_images(root_folder, embeddings, model, threshold=CLASSIFICATION_SIMILARITY_THRESHOLD):
    results = {'images': defaultdict(list), 'videos': defaultdict(list)}
    for file in os.listdir(root_folder):
        file_path = os.path.join(root_folder, file)
        if file.lower().endswith(('png', 'jpg', 'jpeg')):
            face_encodings = create_face_embeddings(file_path, model)
            for face_encoding in face_encodings:
                distances = {path: np.linalg.norm(embedding - face_encoding) for path, embedding in embeddings.items()}
                if not distances:
                    print(f"No embeddings found for {file_path}")
                    continue
                closest_match = min(distances, key=distances.get)
                if distances[closest_match] < threshold:
                    results['images'][file_path].append(closest_match)
        elif file.lower().endswith(('mp4', 'avi', 'mov')):
            # Placeholder for video processing
            results['videos'][file_path] = "Video processing not implemented"
    return results