import os
import tkinter as tk
from tkinter import filedialog
import pickle
import shutil
from collections import defaultdict

LOG_FILENAME = 'processed_files.log'
TARGET_EMBEDDINGS_FILENAME = 'target_face.pkl'

def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    return folder_selected

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