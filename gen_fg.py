import os
import glob
import argparse

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import to_pil_image
from threading import Thread
from tqdm import tqdm
from PIL import Image

from model import MattingRefine


def parse_args():
    parser = argparse.ArgumentParser(description='Process foreground dataset')

    parser.add_argument('--model-backbone', type=str, required=True, choices=['resnet101', 'resnet50', 'mobilenetv2'])
    parser.add_argument('--model-backbone-scale', type=float, default=0.25)
    parser.add_argument('--model-checkpoint', type=str, required=True)
    parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
    parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
    parser.add_argument('--model-refine-threshold', type=float, default=0.7)
    parser.add_argument('--model-refine-kernel-size', type=int, default=3)

    parser.add_argument('--root_dir', type=str, required=True)

    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')

    return parser.parse_args()

# Worker function
def writer(img, path):
    img = to_pil_image(img[0].cpu())
    img.save(path)   

def proc_fg(root_dir):

    device = torch.device(args.device)

    # Load model
    model = MattingRefine(
        args.model_backbone,
        args.model_backbone_scale,
        args.model_refine_mode,
        args.model_refine_sample_pixels,
        args.model_refine_threshold,
        args.model_refine_kernel_size)

    model = model.to(device).eval()
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device), strict=False)

    extensions = ['.jpg', '.png']

    # Load images
    folders = sorted(glob.glob(os.path.join(root_dir, 'cam*')))
    for folder in folders:
        # Load background image
        bg_name = os.path.split(folder)[-1]
        bg_path = os.path.join(root_dir, 'background', bg_name + '*')
        bg_path = glob.glob(bg_path)[0]
        bg = np.array(Image.open(bg_path)) / 255.
        bgr = torch.FloatTensor(bg).permute([2, 0, 1])[None, :]
        bgr = bgr.to(device, non_blocking=True)

        frames = []
        for extension in extensions:
            frames += sorted(glob.glob(os.path.join(folder, '*' + extension)))

        out_dir = os.path.join(root_dir, 'foreground', bg_name)
        os.makedirs(out_dir, exist_ok=True)

        with torch.no_grad():
            for frame in tqdm(frames):
                new_name = os.path.split(frame)[-1][:-4]
                im = Image.open(frame)
                src = torch.FloatTensor(np.array(im) / 255.).permute([2, 0, 1])[None, :]
                src = src.to(device, non_blocking=True)
                
                pha, fgr, _, _, _, _ = model(src, bgr)
                com = torch.cat([fgr * pha.ne(0), pha], dim=1)
                Thread(target=writer, args=(com, os.path.join(out_dir, new_name + '.png'))).start()

if __name__ == '__main__':
    args = parse_args()

    scenes = sorted(glob.glob(os.path.join(args.root_dir, 'scene*')))
    print(scenes)

    for scene in scenes:
        proc_fg(scene)

