import os
import glob
import argparse

import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

def parse_args():
    # Parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_dir', type=str, metavar='PATH',
                        help='path to the scene folder')
    parser.add_argument('--out_dir', type=str, metavar='PATH',
                        help='path to the output folder')
    parser.add_argument('--img_wh', nargs='+', type=int, default=[640, 360],
                        help='width and height of the output image')
    parser.add_argument('--bg', action='store_true',
                        help='process background')
    parser.add_argument('--rgb', action='store_true',
                        help='process RGB')
    parser.add_argument('--fg', action='store_true',
                        help='process foreground')
    parser.add_argument('--scene_ids', nargs='+', type=int, default=[],
                        help='process only the entered ids')
    return parser.parse_args()

def read_data(path, scene_idx, cam_idx, img_idx):
    with h5py.File(path, 'r') as hf:
        im = hf[str(scene_idx)][cam_idx, img_idx]
    return im

def get_frame_count(folder, extension='.png'):
    return len(glob.glob(os.path.join(folder[0], f'*{extension}')))

def compress_data(folder, out_folder, width, height,
                bg, rgb, fg):

    comps = [
        bg,
        rgb,
        fg
    ]
    if any(comps) is False:
        comps = [True] * 3

    scene_name = os.path.split(folder)[-1]
    save_path = os.path.join(out_folder, scene_name + '.h5')
    num_cams = 10
    
    # Save data
    with h5py.File(save_path, 'w') as hf:
    
        if comps[0]:
            # Process background
            bg_paths = sorted(glob.glob(os.path.join(folder, 'background', '*.png'))) + \
                sorted(glob.glob(os.path.join(folder, 'background', '*.jpg')))
            assert len(bg_paths) == num_cams
            im_arr_shape = (num_cams, height, width, 3)    
            dataIn = hf.create_dataset(
                'bg_rgb',
                im_arr_shape,
                np.uint8,
                chunks=(1,) + im_arr_shape[1:],
                compression="lzf", shuffle=True
            )

            for cam_idx, bg_path in enumerate(bg_paths):
                bg = Image.open(bg_path)
                if width and height:
                    bg = bg.resize((width, height), Image.LANCZOS)
                bg = np.array(bg)
                dataIn[cam_idx] = bg

            print('Background processed')

        if comps[1]:
            # Process RGB
            png_count = len(glob.glob(os.path.join(folder, 'cam00', '*.png')))
            jpg_count = len(glob.glob(os.path.join(folder, 'cam00', '*.jpg')))

            if png_count == 0 and jpg_count != 0:
                extension = '.jpg'
                num_frames = jpg_count
            elif png_count != 0 and jpg_count == 0:
                extension = '.png'
                num_frames = png_count
            assert png_count == 0 or jpg_count == 0

            im_arr_shape = (num_cams, num_frames, height, width, 3)
            
            dataIn = hf.create_dataset(
                'rgb',
                im_arr_shape,
                np.uint8,
                chunks=(1,1) + im_arr_shape[2:],
                compression="lzf", shuffle=True
            )

            scene_name = folder.split(os.sep)[-1]
            scene = sorted(glob.glob(os.path.join(folder, 'cam*')))
            for cam_idx, cam in enumerate(scene):
                tqdm.write(f'Processing {scene_name}, camera {cam_idx}')
                frames = sorted(glob.glob(os.path.join(cam, f'*{extension}')))
                for frame_idx, frame in tqdm(enumerate(frames)):
                    im = Image.open(frame)
                    if width and height:
                        im = im.resize((width, height), Image.LANCZOS)
                    im = np.array(im)
                    dataIn[cam_idx, frame_idx] = im

        if comps[2]:
            # Process Foreground
            num_frames = len(glob.glob(os.path.join(folder, 'foreground', 'cam00', '*.png')))
            im_arr_shape = (num_cams, num_frames, height, width, 4)
            
            dataIn = hf.create_dataset(
                'fg_rgb',
                im_arr_shape,
                np.uint8,
                chunks=(1,1) + im_arr_shape[2:],
                compression="lzf", shuffle=True
            )

            scene_name = folder.split(os.sep)[-1]
            scene = sorted(glob.glob(os.path.join(folder, 'foreground', 'cam*')))
            for cam_idx, cam in enumerate(scene):
                tqdm.write(f'Processing {scene_name}, camera {cam_idx}')
                frames = sorted(glob.glob(os.path.join(cam, f'*.png')))
                for frame_idx, frame in tqdm(enumerate(frames)):
                    im = Image.open(frame)
                    if width and height:
                        im = im.resize((width, height), Image.LANCZOS)
                    im = np.array(im)
                    dataIn[cam_idx, frame_idx] = im

if __name__ == '__main__':
    args = parse_args()

    scenes = sorted(glob.glob(os.path.join(args.root_dir, 'scene*')))
    print(scenes)

    video_folders = []
    if len(args.scene_ids) != 0:
        for video_folder in scenes:
            scene_id = video_folder.split(os.sep)[-1][5:]
            scene_id = int(''.join(x for x in scene_id if x.isdigit()))
            if scene_id in args.scene_ids:
                video_folders.append(video_folder)
    else:
        video_folders = scenes
    print(video_folders)

    os.makedirs(args.out_dir, exist_ok=True)
    for scene in video_folders:
        compress_data(scene, args.out_dir, args.img_wh[0], args.img_wh[1],
                    args.bg, args.rgb, args.fg)