import os  # Add this import
import face_recognition
import numpy as np
from .image_processing import preprocess_image
from .file_operations import load_processed_files_log, update_processed_files_log, TARGET_EMBEDDINGS_FILENAME  # Add these imports
from .keras_model import train_or_load_model
import cv2  # Add this import
from sklearn.preprocessing import LabelEncoder  # Add this import

ENCODING_SIMILARITY_THRESHOLD = 0.2  # Threshold for filtering face encodings
NUM_JITTERS = 10  # Number of times to jitter the image for encoding
FACE_DETECTION_MODEL = 'cnn'  # Use 'cnn' for more accurate face detection
BATCH_SIZE = 32  # Adjust this value to control batch size

def create_face_embeddings(image_path, model):
    images = preprocess_image(image_path)
    face_embeddings = []
    for image in images:
        face_locations = face_recognition.face_locations(image, model=FACE_DETECTION_MODEL)
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = image[top:bottom, left:right]
            face_image = cv2.resize(face_image, (224, 224))  # Resize to match model input
            face_image = np.array(face_image) / 255.0  # Normalize the image
            embedding = model.create_embedding(face_image)
            face_embeddings.append(embedding)
    return face_embeddings

def batch_create_face_embeddings(image_paths, model):
    all_embeddings = []
    for image_path in image_paths:
        embeddings = create_face_embeddings(image_path, model)
        all_embeddings.extend(embeddings)
    return all_embeddings

def process_folder(folder_path, model):
    embeddings = {}
    for subdir, _, files in os.walk(folder_path):
        if subdir == folder_path:
            continue  # Skip the root folder itself
        processed_files_log = load_processed_files_log(subdir)
        all_encodings = []
        new_files = []
        image_paths = []
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')) and TARGET_EMBEDDINGS_FILENAME not in file:
                file_path = os.path.join(subdir, file)
                if file_path not in processed_files_log:
                    image_paths.append(file_path)
        if image_paths:
            all_encodings.extend(batch_create_face_embeddings(image_paths, model))
            new_files.extend(image_paths)
        if all_encodings:
            # Average the encodings to create a robust representation
            average_encoding = np.mean(all_encodings, axis=0)
            embeddings[subdir] = average_encoding
        if new_files:
            update_processed_files_log(subdir, new_files)
    return embeddings

def initialize_or_update_model(root_folder):
    model_path = os.path.join(root_folder, 'face_embedding_model.keras')  # Use .keras extension
    model = train_or_load_model(root_folder, model_path)
    embeddings = process_folder(root_folder, model)
    
    # Gather training data from the original images and their corresponding labels
    X, y = [], []
    for subdir, _, files in os.walk(root_folder):
        if subdir == root_folder:
            continue  # Skip the root folder itself
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')) and TARGET_EMBEDDINGS_FILENAME not in file:
                file_path = os.path.join(subdir, file)
                image = cv2.imread(file_path)
                image = cv2.resize(image, (224, 224))  # Resize to match model input
                image = np.array(image) / 255.0  # Normalize the image
                X.append(image)
                y.append(os.path.basename(subdir))
    
    # Convert string labels to numerical labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    X = np.array(X)
    y = np.array(y)
    
    model.train(X, y)
    model.save_model(model_path)
    return model, embeddings