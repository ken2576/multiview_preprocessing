import os
import glob
import shutil
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Generate background data')
    parser.add_argument('root_dir', type=str,
                        help='data root directory')
    parser.add_argument('--bg_skip', type=int,
                        default=25,
                        help='background frame skip')

    return parser.parse_args()

def gen_background(folder):
    cams = sorted(glob.glob(os.path.join(folder, 'cam*', '')))
    bg_folder = os.path.join(folder, 'background')
    os.makedirs(bg_folder, exist_ok=True)

    png_count = len(glob.glob(os.path.join(cams[0], '*.png')))
    jpg_count = len(glob.glob(os.path.join(cams[0], '*.jpg')))

    if png_count == 0 and jpg_count != 0:
        extension = '.jpg'
        img_count = jpg_count
    elif png_count != 0 and jpg_count == 0:
        extension = '.png'
        img_count = png_count

    bg_frames = [x for x in range(img_count)]
    bg_frames = bg_frames[::args.bg_skip]

    for idx, cam in enumerate(cams):
        img_paths = sorted(glob.glob(os.path.join(cam, '*'+extension)))
        imgs_to_use = [img_paths[x] for x in bg_frames]
        imgs = [plt.imread(x) for x in imgs_to_use]
        imgs = np.stack(imgs)
        bg = np.median(imgs, 0)
        if extension == '.png':
            bg *= 255.
        bg = Image.fromarray(np.uint8(bg))
        bg.save(os.path.join(bg_folder, f'cam{idx:02d}'+extension))

if __name__ == '__main__':

    args = parse_args()

    scenes = sorted(glob.glob(os.path.join(args.root_dir, 'scene*')))
    print(scenes)

    for scene in scenes:
        gen_background(scene)