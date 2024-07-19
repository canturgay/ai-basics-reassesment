import os
import time
import tkinter as tk
from tkinter import filedialog
import cv2
import face_recognition
import numpy as np
import pickle
import shutil
from collections import defaultdict
import imgaug.augmenters as iaa

LOG_FILENAME = 'processed_files.log'
TARGET_EMBEDDINGS_FILENAME = 'target_face.pkl'
ENCODING_SIMILARITY_THRESHOLD = 0.2  # Threshold for filtering face encodings
CLASSIFICATION_SIMILARITY_THRESHOLD = 0.5  # Threshold for classifying images
BATCH_SIZE = 32  # Adjust this value to control batch size
IMAGE_DIMENSIONS = (256, 256)  # Standard dimensions for resizing images
NUM_JITTERS = 10  # Number of times to jitter the image for encoding
FACE_DETECTION_MODEL = 'cnn'  # Use 'cnn' for more accurate face detection

def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    return folder_selected

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(rgb_image, IMAGE_DIMENSIONS)
    
    # Image augmentation
    seq = iaa.Sequential([
        iaa.GaussianBlur(sigma=(0, 1.0)),  # Apply Gaussian blur
        iaa.Fliplr(0.5),  # Horizontally flip 50% of the images
        iaa.Affine(rotate=(-20, 20))  # Rotate images
    ])
    
    augmented_images = seq(images=[resized_image])
    
    # Return the original and augmented images
    return [resized_image] + augmented_images

def create_face_embeddings(image_path, target_face_encoding=None):
    images = preprocess_image(image_path)
    face_encodings = []
    for image in images:
        face_locations = face_recognition.face_locations(image, model=FACE_DETECTION_MODEL)
        encodings = face_recognition.face_encodings(image, face_locations, num_jitters=NUM_JITTERS)
        if target_face_encoding is not None:
            encodings = [encoding for encoding in encodings if face_recognition.compare_faces([target_face_encoding], encoding, tolerance=ENCODING_SIMILARITY_THRESHOLD)[0]]
        face_encodings.extend(encodings)
    return face_encodings

def batch_create_face_embeddings(image_paths, target_face_encoding=None):
    images = []
    for image_path in image_paths:
        images.extend(preprocess_image(image_path))
    face_locations = face_recognition.batch_face_locations(images, number_of_times_to_upsample=1, batch_size=BATCH_SIZE)
    all_encodings = []
    for i, locations in enumerate(face_locations):
        encodings = face_recognition.face_encodings(images[i], locations, num_jitters=NUM_JITTERS)
        if target_face_encoding is not None:
            encodings = [encoding for encoding in encodings if face_recognition.compare_faces([target_face_encoding], encoding, tolerance=ENCODING_SIMILARITY_THRESHOLD)[0]]
        all_encodings.extend(encodings)
    return all_encodings

def load_embeddings(folder_path):
    embeddings = {}
    for subdir, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(TARGET_EMBEDDINGS_FILENAME):
                with open(os.path.join(subdir, file), 'rb') as f:
                    embeddings[subdir] = pickle.load(f)
    return embeddings

def save_embeddings(embeddings):
    for subdir, encoding in embeddings.items():
        with open(os.path.join(subdir, TARGET_EMBEDDINGS_FILENAME), 'wb') as f:
            pickle.dump(encoding, f)

def load_processed_files_log(subdir):
    log_path = os.path.join(subdir, LOG_FILENAME)
    if os.path.exists(log_path):
        with open(log_path, 'r') as log_file:
            return set(log_file.read().splitlines())
    return set()

def update_processed_files_log(subdir, processed_files):
    log_path = os.path.join(subdir, LOG_FILENAME)
    with open(log_path, 'a') as log_file:
        for file in processed_files:
            log_file.write(file + '\n')

def process_folder(folder_path):
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
            base_face_encodings = batch_create_face_embeddings(image_paths)
            if base_face_encodings:
                base_face_encoding = base_face_encodings[0]
                all_encodings.extend(batch_create_face_embeddings(image_paths, base_face_encoding))
                new_files.extend(image_paths)
        if all_encodings:
            # Average the encodings to create a robust representation
            average_encoding = np.mean(all_encodings, axis=0)
            embeddings[subdir] = average_encoding
        if new_files:
            update_processed_files_log(subdir, new_files)
    return embeddings

def classify_images_and_videos(root_folder, embeddings, threshold=CLASSIFICATION_SIMILARITY_THRESHOLD):
    results = {'images': defaultdict(list), 'videos': defaultdict(list)}
    for file in os.listdir(root_folder):
        file_path = os.path.join(root_folder, file)
        if file.lower().endswith(('png', 'jpg', 'jpeg')):
            face_encodings = create_face_embeddings(file_path)
            for face_encoding in face_encodings:
                distances = {path: np.linalg.norm(embedding - face_encoding) for path, embedding in embeddings.items()}
                closest_match = min(distances, key=distances.get)
                if distances[closest_match] < threshold:
                    results['images'][file_path].append(closest_match)
        elif file.lower().endswith(('mp4', 'avi', 'mov')):
            # Placeholder for video processing
            results['videos'][file_path] = "Video processing not implemented"
    return results

def move_files(root_folder, results):
    moved_files = defaultdict(set)
    for file_path, matches in results['images'].items():
        for match in matches:
            folder_name = os.path.basename(match)
            destination_folder = os.path.join(root_folder, folder_name)
            if os.path.exists(destination_folder):
                try:
                    shutil.copy(file_path, os.path.join(destination_folder, os.path.basename(file_path)))
                    moved_files[file_path].add(destination_folder)
                    os.remove(file_path)  # Remove the file from the root folder after copying
                except FileNotFoundError:
                    print(f"File not found: {file_path}, skipping.")
                except Exception as e:
                    print(f"Error moving file {file_path} to {destination_folder}: {e}")
    return moved_files

def main():
    print("Select the root folder:")
    root_folder = select_folder()
    print(f"Selected root folder: {root_folder}")

    while True:
        start_time = time.time()
        print("Loading existing embeddings...")
        embeddings = load_embeddings(root_folder)
        print(f"Loaded embeddings in {time.time() - start_time:.2f} seconds.")

        start_time = time.time()
        print("Processing subfolders to create/update embeddings...")
        new_embeddings = process_folder(root_folder)
        embeddings.update(new_embeddings)
        save_embeddings(embeddings)
        print(f"Created/Updated embeddings for {len(new_embeddings)} subfolders in {time.time() - start_time:.2f} seconds.")

        start_time = time.time()
        print("Classifying images and videos in the root folder...")
        results = classify_images_and_videos(root_folder, embeddings)
        print(f"Classified images and videos in {time.time() - start_time:.2f} seconds.")
        
        start_time = time.time()
        print("Moving files to corresponding directories...")
        moved_files = move_files(root_folder, results)
        print(f"Moved {len(moved_files)} files in {time.time() - start_time:.2f} seconds.")

        if not moved_files:
            break

        print("Classification and file moving results:")
        for file_path, destinations in moved_files.items():
            print(f" - {file_path} moved to:")
            for dest in destinations:
                print(f"   - {dest}")

if __name__ == "__main__":
    main()