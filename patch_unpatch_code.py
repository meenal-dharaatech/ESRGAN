import os
import cv2
import math
import argparse
import numpy as np
from PIL import Image


Image.MAX_IMAGE_PIXELS = None
def read_tif_file(image_file_path):
    image = Image.open(image_file_path)
    image_data = np.array(image)
    rgb_image = image_data[:, :, :3]
    img_array = np.moveaxis(rgb_image, 0, 2)
    img_array = np.uint8(rgb_image)
    return img_array


def patch_(image_file_path, patch_size, patch_files_dir_path):
    name = os.path.splitext(os.path.basename(image_file_path))[0]
    os.makedirs(patch_files_dir_path, exist_ok=True)
    image_array = read_tif_file(image_file_path)
    old_shape = image_array.shape
    height, width, _ = image_array.shape

    count = 1
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image_array[i : i + patch_size, j : j + patch_size, :]
            patch_image = Image.fromarray(patch)
            patch_image.save(os.path.join(patch_files_dir_path, f"{name}_{i}_{j}.png"))
            count=count+1
    return old_shape



def unpatch_(input_img, image_file_path, old_shape, new_shape, patch_size, patch_files_dir_path):
    name = os.path.splitext(os.path.basename(input_img))[0]
    old_image_height, old_image_width, o_c = old_shape
    new_image_height, new_image_width, n_c = new_shape

    output_image = np.zeros((new_image_height, new_image_width, n_c), dtype=np.uint8)

    for i in range(0, old_image_height, patch_size):
        for j in range(0, old_image_width, patch_size):
            # Construct the patch file path based on indices
            patch_file_path = os.path.join(patch_files_dir_path, f"{name}_{i}_{j}.png")

            if os.path.exists(patch_file_path):
                image = Image.open(patch_file_path)
                image_array = np.array(image)

                # Adjust indices for the upsampled image dimensions
                i_upsampled = int(i * (new_image_height / old_image_height))
                j_upsampled = int(j * (new_image_width / old_image_width))

                # Adjust the patch size for the upsampled image
                patch_size_upsampled = int(patch_size * (new_image_height / old_image_height))

                # Correct the indexing for patch placement in the upsampled image
                output_image[
                    i_upsampled : i_upsampled + patch_size_upsampled,
                    j_upsampled : j_upsampled + patch_size_upsampled,
                    :
                ] = image_array

    output_image = Image.fromarray(output_image)

    output_path = os.path.join(image_file_path, f"{name}.tif")
    output_image.save(output_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file-path",
        type=str,
        default="rgb.tif",
        help="Input TiF Image File Path!",
    )
    parser.add_argument(
        "--output-file-path",
        type=str,
        default="Output_Testing_Image.tif",
        help="Output TiF Image File Path!",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=640,
        help="Patch Size",
    )
    parser.add_argument(
        "--patch-images-dir",
        type=str,
        default="patch_files",
        help="Dir Path to save Patch Image",
    )
    args = parser.parse_args()
    # input_image_file_path = args.input_file_path
    # output_image_file_path = args.output_file_path
    # patch_files_dir = args.patch_images_dir
    # patch_size = (args.patch_size, args.patch_size)
    # new_size = (7680, 6080)
    # patch_(
    #     input_image_file_path,
    #     patch_size,
    #     patch_files_dir,
    #     new_size
    # )
    # unpatch_(
    #     output_image_file_path,
    #     old_shape,
    #     new_shape,
    #     patch_size,
    #     patch_files_dir,
    # )
