import os
import glob
import pickle
import argparse
import tqdm
import numpy as np
import cv2
from multiprocessing import Pool
from itertools import repeat
from threading import Thread
from PIL import Image

class ImageSequenceWriter:
    def __init__(self, path, extension):
        self.path = path
        self.extension = extension
        self.index = 0
        os.makedirs(path, exist_ok=True)
        
    def add_batch(self, frames):
        Thread(target=self._add_batch, args=(frames, self.index)).start()
        self.index += len(frames)
            
    def _add_batch(self, frames, index):
        for i in range(len(frames)):
            frame = frames[i]
            frame = Image.fromarray(frame)
            frame.save(os.path.join(self.path, str(index + i).zfill(5) + '.' + self.extension))

def parse_args():
    # Parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, metavar='PATH',
                        help='path to the scene folder')
    parser.add_argument('--out_folder', type=str, metavar='PATH',
                        help='path to the output folder')
    parser.add_argument('--param_path', type=str, metavar='PATH',
                        help='path to the calibrated camera parameters',
                        default='cam_params.pkl')
    parser.add_argument('--img_wh', nargs='+', type=int, default=[960, 540],
                        help='width and height of the output image')
    parser.add_argument('--scene_ids', nargs='+', type=int, default=[],
                        help='process only the entered ids')
    parser.add_argument('--margin', type=float,
                        default=0,
                        help='how many seconds to skip for the beginning and the end')
    parser.add_argument('--skip', type=int,
                        default=4,
                        help='frame skip')
    parser.add_argument('--fix_last', default=False, action="store_true",
                        help='extract last frame only (for fixing bugs)')
    parser.add_argument('--show_img', default=False, action="store_true",
                        help='whether to show image during processing')
    parser.add_argument('--n_jobs', default=4,
                        help='how many processes')
    return parser.parse_args()

def undistort(img, K, D, new_K, new_dim):    
    h,w = img.shape[:2]
    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, new_dim, cv2.CV_16SC2)
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)    

def proc_video(data, skip=4, fix_last=False, ext='jpg'):
    video = data['video']
    cam = data['cam']
    out_path = data['out_path']
    args = data['args']
    new_K = data['new_K']
    idx = data['idx']

    out_folder = os.path.join(out_path, f'cam{idx:02d}')
    os.makedirs(out_folder, exist_ok=True)

    K = cam['K']
    D = cam['D']

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - args.margin * fps
    start = 0 + args.margin * fps
    frames = [x for x in range(int(start), int(end))]
    frames = frames[::skip]

    if fix_last:
        last_idx = len(frames) - 1
        frames = frames[-1:]

    writer = ImageSequenceWriter(out_folder, ext)
    for frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, img = cap.read()
        ret = undistort(img, K, D, new_K, tuple(args.img_wh))
        ret = ret[..., ::-1]
        if fix_last:
            im = Image.fromarray(ret)
            im.save(os.path.join(out_folder, str(last_idx).zfill(5) + '.' + ext))
        else:
            writer.add_batch(ret[None, :])
        if args.show_img:
            cv2.imshow("undistorted", ret)
            cv2.waitKey(1)
    
def proc_folder(path, param_path, out_path, args):

    with open(param_path, 'rb') as handle:
        cams = pickle.load(handle)

    videos = sorted(glob.glob(os.path.join(path, '*.mp4')))
    new_K = np.eye(3)
    angle = np.deg2rad(90)
    new_K[0, 0] = args.img_wh[0] / 2 / np.tan(angle / 2) # FoV = 90
    new_K[1, 1] = args.img_wh[0] / 2 / np.tan(angle / 2)
    new_K[0, -1] = args.img_wh[0] / 2
    new_K[1, -1] = args.img_wh[1] / 2
    payloads = []

    for idx, (video, cam) in enumerate(zip(videos, cams)):
        payload = dict(video=video, cam=cam, out_path=out_path,
                args=args, new_K=new_K, idx=idx)
        payloads.append(payload)

    with Pool(processes=args.n_jobs) as pool:
        pool.starmap(proc_video, zip(payloads, repeat(args.skip), repeat(args.fix_last)))


if __name__ == '__main__':
    args = parse_args()
    read_video_folders = sorted(glob.glob(os.path.join(args.path, 'scene*', '')))
    print(read_video_folders)

    video_folders = []

    if len(args.scene_ids) != 0:
        for video_folder in read_video_folders:
            scene_id = video_folder.split(os.sep)[-2][5:]
            scene_id = int(''.join(x for x in scene_id if x.isdigit()))
            if scene_id in args.scene_ids:
                video_folders.append(video_folder)
    else:
        video_folders = read_video_folders

    print(video_folders)

    for video_folder in tqdm.tqdm(video_folders):
        if 'cannot_sync' in video_folder:
            continue
        else:
            name = os.path.split(video_folder[:-1])[-1]
            out_path = os.path.join(args.out_folder, name)
            proc_folder(video_folder, args.param_path, out_path, args)