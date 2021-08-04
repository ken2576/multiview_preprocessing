import os
import glob
import shutil
import argparse


def parse_args():
    # Parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_dir', type=str, metavar='PATH',
                        help='path to the scene folder')
    parser.add_argument('--out_dir', type=str, metavar='PATH',
                        help='path to the output folder')
    return parser.parse_args()

def proc_folder(folder, out_dir):
    src = os.path.join(folder, 'poses_bounds.npy')
    new_name = os.path.split(folder)[-1]
    dst = os.path.join(out_dir, new_name + '_pb.npy')
    shutil.copy2(src, dst)

if __name__ == '__main__':
    args = parse_args()

    scenes = sorted(glob.glob(os.path.join(args.root_dir, 'scene*')))
    print(scenes)

    for scene in scenes:
        proc_folder(scene, args.out_dir)