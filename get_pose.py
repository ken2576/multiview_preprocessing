import os
import glob
import argparse
import shutil
import numpy as np
from poses_util import gen_poses

def parse_args():
    # Parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('root_dir', type=str, metavar='PATH',
                        help='path to the scene folder')
    parser.add_argument('--scene_ids', nargs='+', type=int, default=[],
                        help='process only the entered ids')
    return parser.parse_args()

def get_same_frame(folder, extensions=['.png', '.jpg']):
    '''Get the same frames to calibrate multiple cameras
    Args:
        folder: path to folder containing camera folders
    '''
    out_folder = os.path.join(folder, 'images')
    os.makedirs(out_folder, exist_ok=True)

    img_id = 0
    cams = []
    for extension in extensions:
        cams += sorted(glob.glob(os.path.join(folder, 'cam*', f'{img_id:05d}{extension}')))
    assert len(cams) == 10
    for idx, cam in enumerate(cams):
        ext = cam[-4:]
        new_name = f'{idx:03d}{ext}'
        shutil.copy2(cam, os.path.join(out_folder, new_name))

def delete_calib_data(folder):
    db_path = os.path.join(folder, 'database.db')
    if os.path.exists(db_path):
        os.remove(db_path)

    sparse_path = os.path.join(folder, 'sparse')
    if os.path.exists(sparse_path):
        shutil.rmtree(sparse_path)

if __name__ == '__main__':
    args = parse_args()
    match_type = 'exhaustive_matcher'
    read_folders = sorted(glob.glob(os.path.join(args.root_dir, 'scene*', '')))
    
    print(read_folders)

    folders = []

    if len(args.scene_ids) != 0:
        for video_folder in read_folders:
            scene_id = video_folder.split(os.sep)[-2][5:]
            scene_id = int(''.join(x for x in scene_id if x.isdigit()))
            if scene_id in args.scene_ids:
                folders.append(video_folder)
    else:
        folders = read_folders

    print(folders)

    for folder in folders:
        print(folder)
        get_same_frame(folder)
        delete_calib_data(folder)
        gen_poses(folder, match_type)