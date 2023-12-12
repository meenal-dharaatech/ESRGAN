import os

def rename_images(folder_path):
    for filename in os.listdir(folder_path):
    
        parts = filename.split('_')
        new_parts = [*parts[1:]]
        new_filename = '_'.join(new_parts)
        
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)

        try:
            os.rename(old_filepath, new_filepath)
            print(f"Renamed: {old_filepath} to {new_filepath}")
        except Exception as e:
            print(f"Error renaming {old_filepath}: {e}")

if __name__ == "__main__":
    # Replace these paths with your actual folder path
    folder_path = "/home/computervision/Desktop/ESRGAN/Real-ESRGAN-master/dataset/test_dataset/test_dataset_multiscale"

    rename_images(folder_path)
