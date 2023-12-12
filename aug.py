import albumentations as augment
import os
import cv2
import pandas as pd
import glob
import natsort
import numpy as np
import traceback

import warnings

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.getcwd(), "data/320_augmented_imgs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def augmentation(image_path):
    """
    ColorJitter
    Downscale
    HueSaturationValue
    MotionBlur
    Posterize
    RandomFog
    Superpixels
    RandomRain
    Sharpen
    VerticalFlip
    ChannelShuffle
    """

    img_name = os.path.basename(image_path)[:-4]
    img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #center_crop = augment.Compose([augment.CenterCrop(width=160, height=160)])

    random_brightness = augment.Compose([augment.RandomBrightness(p=0.7)])

    #random_contrast = augment.Compose([augment.RandomContrast(limit=0.5, always_apply=False, p=0.5)])

    # random_hflip = augment.Compose([augment.HorizontalFlip(p=0.7)])

    random_brightness_contrast = augment.Compose([augment.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2)])

    sepia_augment = augment.Compose([augment.ToSepia(p=0.7)])

    gray_augment = augment.Compose([augment.ToGray(p=0.7)])

    # color_jitter = augment.Compose([augment.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5)])

    # downscale_pixel = augment.Compose([augment.Downscale(scale_min=0.20, scale_max=0.20, interpolation=None, always_apply=False, p=0.5)])

    # hsv_pixel = augment.Compose([augment.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5)])

    vertical_flip = augment.Compose([augment.VerticalFlip(p=0.5)])

    sharpen = augment.Compose([augment.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5)])

    motion_blur = augment.Compose([augment.MotionBlur(blur_limit=7, allow_shifted=True, always_apply=False, p=0.5)])

    # posterize = augment.Compose([augment.Posterize(num_bits=3, always_apply=False, p=0.5)])

    augment_types = {
        # "center_crop": center_crop,
        "random_brightness": random_brightness,
        # "random_contrast": random_contrast,
        # "random_horizontal_flip": random_hflip,
        "random_brightness_contrast": random_brightness_contrast,
        "sepia": sepia_augment,
        "gray": gray_augment,
        # "color_jitter": color_jitter,
        # "downscale_pixel": downscale_pixel,
        # "hsv_pixel": hsv_pixel,
        "vertical_flip": vertical_flip,
        "sharpen": sharpen,
        "motion_blur": motion_blur,
        # "posterize": posterize,
    }

    for augment_name, augment_type in augment_types.items():
        transformed = augment_type(image=img)
        print(f'{img_name} | {augment_name}')

        cv2.imwrite(
            os.path.join(OUTPUT_DIR, f"{img_name}_{augment_name}.png"),
            transformed["image"],
        )


def main(img_dir):
    """applies augmentations to image data"""
    if os.path.exists(img_dir):
        imgs = natsort.natsorted(glob.glob(img_dir + "/*.png"))
        print(f"Total Image Files: {len(imgs)}")
        for img in imgs:
            try:
                augmentation(img)
            except Exception as e:
                print(e)
                print(traceback.print_exc())


if __name__ == "__main__":
    images_dir = "data/320_Patches"
    main(images_dir)
