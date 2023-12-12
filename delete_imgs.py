import os
import shutil

def remove_non_matching_images(source_folder, target_folder):
    # Get the list of files in both folders
    source_files = set(os.listdir(source_folder))
    target_files = set(os.listdir(target_folder))

    # Find the files in the target folder that are not in the source folder
    files_to_remove = target_files - source_files
    print("length",len(files_to_remove))

    # Remove the non-matching files from the target folder
    for file_to_remove in files_to_remove:
        file_path = os.path.join(target_folder, file_to_remove)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")

if __name__ == "__main__":
    # Replace these paths with your actual folder paths
    source_folder_path = "data/320_Patches"
    target_folder_path = "data/640_Patches"

    remove_non_matching_images(source_folder_path, target_folder_path)

