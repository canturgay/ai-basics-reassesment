from utils.file_operations import select_folder, load_embeddings, save_embeddings, move_files
from utils.classification import classify_images
from utils.face_recognition import initialize_or_update_model
import time

def main():
    print("Select the root folder:")
    root_folder = select_folder()
    print(f"Selected root folder: {root_folder}")

    while True:
        start_time = time.time()
        print("Initializing or updating the face embedding model...")
        model, embeddings = initialize_or_update_model(root_folder)
        print(f"Model initialized/updated with {len(embeddings)} embeddings in {time.time() - start_time:.2f} seconds.")

        save_embeddings(embeddings)

        start_time = time.time()
        print("Classifying images and videos in the root folder...")
        results = classify_images(root_folder, embeddings, model)
        print(f"Classified {len(results)} images and videos in {time.time() - start_time:.2f} seconds.")
        
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