import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from patch_unpatch_code import *
import numpy as np


def main():
    """Inference demo for Real-ESRGAN."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, default="inputs", help="Input image or folder"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="results", help="Output folder"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="[Option] Model path. Usually, you do not need to specify it",
    )
    parser.add_argument("--patch_size",type=int,default=320)
    
    parser.add_argument(
        "-s",
        "--outscale",
        type=float,
        default=2,
        help="The final upsampling scale of the image",
    )

    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use fp32 precision during inference. Default: fp16 (half precision).",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="auto",
        help="Image extension. Options: auto | jpg | png, auto means using the same extension as inputs",
    )


    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    patch_folder = "/home/computervision/Desktop/ESRGAN/data/cropped_640_imgs/patch_folder"
    inhanced_patch_folder = "/home/computervision/Desktop/ESRGAN/data/cropped_640_imgs/inhanced_patch_folder"
    os.makedirs(patch_folder, exist_ok=True)
    os.makedirs(inhanced_patch_folder, exist_ok=True)


    # determine model paths
    if args.model_path is not None:
        model_path = args.model_path
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )

    dni_weight = None

    # restorer
    upsampler = RealESRGANer(
        scale=2,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id,
    )

    old_shape =  patch_(args.input,args.patch_size,patch_folder)
    old_height, old_width, _nc = old_shape
    new_height,new_width = args.outscale*old_height, args.outscale*old_width
    new_shape = (int(new_height),int(new_width),int(_nc))
    if os.path.isdir(patch_folder):
        folder = patch_folder
        for image in os.listdir(folder):
            paths = os.path.join(folder, image)
            # print(paths)
            imgname, extension = os.path.splitext(os.path.basename(paths))
            img = cv2.imread(paths, cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 3 and img.shape[2] == 4:
                img_mode = "RGBA"
            else:
                img_mode = None
            # print("image mode", img_mode)
            output, _ = upsampler.enhance(img, outscale=args.outscale)
            save_path = os.path.join(inhanced_patch_folder, f"{imgname}{extension}")
            cv2.imwrite(save_path, output)
    unpatch_(args.input,args.output,old_shape,new_shape,args.patch_size,inhanced_patch_folder)


if __name__ == "__main__":
    main()
