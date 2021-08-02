import os
import glob

import torch
import numpy as np
from tqdm import tqdm
import argparse
from PIL import Image
import matplotlib.pyplot as plt

from projector import (divide_safe, batch_inverse_warp)

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        help='root directory of dataset')

    parser.add_argument('--indices', nargs='+', type=int, default=[x for x in range(10)],
                        help='camera indices')
    parser.add_argument('--img_hw', nargs="+", type=int, default=[360, 640],
                        help='resolution (img_h, img_w) of the image')

    return parser.parse_args()

def pose2mat(pose):
    """Convert pose matrix (3x5) to extrinsic matrix (4x4) and
       intrinsic matrix (3x3)
    
    Args:
        pose: 3x5 pose matrix
    Returns:
        Extrinsic matrix (4x4 w2c) and intrinsic matrix (3x3)
    """
    extrinsic = np.eye(4)
    extrinsic[:3, :] = pose[:, :4]
    inv_extrinsic = np.linalg.inv(extrinsic)
    extrinsic = np.linalg.inv(inv_extrinsic)
    h, w, focal_length = pose[:, 4]
    intrinsic = np.array([[focal_length, 0, w/2],
                        [0, focal_length, h/2],
                        [0,            0,   1]])

    return extrinsic, intrinsic

def convert_llff(pose):
    """Convert LLFF poses to PyTorch convention (w2c extrinsic and hwf)
    """
    hwf = pose[:3, 4:]

    ext = np.eye(4)
    ext[:3, :4] = pose[:3, :4]
    mat = np.linalg.inv(ext)
    mat = mat[[1, 0, 2]]
    mat[2] = -mat[2]
    mat[:, 2] = -mat[:, 2]

    return np.concatenate([mat, hwf], -1)

def load_poses(filename):
    
    if filename.endswith('npy'):
        return np.load(filename)
    
    elif filename.endswith('txt'):
        with open(filename, 'r') as file:
            file.readline()
            x = np.loadtxt(file)
        x = np.transpose(np.reshape(x, [-1,5,3]), [0,2,1])
        x = np.concatenate([-x[...,1:2], x[...,0:1], x[...,2:]], -1)
        return x
    
    print('Incompatible pose file {}, must be .txt or .npy'.format(filename))
    return None

def read_llff_data(rootdir, img_wh=(640, 360)):
    poses_bounds = np.load(os.path.join(rootdir, 'poses_bounds.npy'))
    bds = poses_bounds[:, -2:]
    poses = poses_bounds[:, :-2].reshape([-1, 3, 5])
    new_poses = [convert_llff(x) for x in poses]
    src_w2cs = np.array([pose2mat(x)[0] for x in new_poses])
    src_ints = np.array([pose2mat(x)[1] for x in new_poses])

    img_paths = sorted(glob.glob(os.path.join(rootdir, 'images', '*')))
    imgs = [Image.open(x) for x in img_paths]
    imgs = [x.resize((img_wh[0], img_wh[1]), Image.LANCZOS) for x in imgs]
    src_imgs = np.stack([np.array(x) / 255. for x in imgs], 0)

    src_ints[:, :2] *= img_wh[0] / src_ints[:, 0:1, -1:] / 2 # Scale image accordingly

    src_imgs = torch.FloatTensor(src_imgs).permute([0, 3, 1, 2])
    src_w2cs = torch.FloatTensor(src_w2cs)
    src_ints = torch.FloatTensor(src_ints)
    out_bds = torch.FloatTensor([np.min(bds) *.9, np.max(bds) * 2])

    return src_imgs, src_w2cs, src_ints, out_bds

def get_depths(min_depth, max_depth, num_depths=32, device=None):
    depths = 1 / torch.linspace(1./max_depth, 1./min_depth,
            steps=num_depths, device=device)
    return depths

def create_psv(imgs, exts, ref_ext, ints, ref_int, depths):
    '''Create plane sweep volume from inputs

    Args:
        imgs: source images [#views, #channels, height, width]
        exts: source extrinsics [#views, 4, 4]
        ref_ext: reference extrinsics [4, 4]
        ints: source intrinsics [#views, 3, 3]
        ref_int: reference intrinsics [3, 3]
        depths: depth values [#planes]
    Returns:
        Plane sweep volume [#views, #channels, #depth_planes, height, width]
    '''
    num_views = imgs.shape[0]
    psv_depths = depths.unsqueeze(1).repeat([1, num_views])
    psv_input = imgs + 1.

    psv = batch_warp(psv_input, exts, ints,
        ref_ext[None, :], ref_int[None, :], psv_depths) - 1. # Move back to correct range
    return psv

def batch_warp(src_imgs, src_exts, src_ints, tgt_exts, tgt_ints, depths):
    '''Warp images to target pose

    Args:
        src_imgs: source images [batch, #channels, height, width]
        src_exts: source extrinsics [batch, 4, 4]
        src_ints: source intrinsics [batch, 3, 3]
        tgt_exts: target extrinsics [batch, 4, 4]
        tgt_ints: target intrinsics [batch, 3, 3]
        depths: depth values [#planes, batch]
    Returns:
        Warped images [batch, #channels, #depth_planes, height, width]
    '''
    trnfs = torch.matmul(src_exts, torch.inverse(tgt_exts))

    psv = batch_inverse_warp(src_imgs, depths,
            src_ints, torch.inverse(tgt_ints), trnfs)

    return psv

def run_psv(folder, device, num_depths=32):  
    src_imgs, src_exts, src_ints, bds = read_llff_data(folder)
    src_imgs = src_imgs[args.indices].to(device)
    src_exts = src_exts[args.indices].to(device)
    src_ints = src_ints[args.indices].to(device)
    bds = bds.to(device)

    depths = get_depths(bds[0], bds[1], device=device)

    v, c, h, w = src_imgs.shape
    imgs_to_warp = src_imgs
    exts = src_exts
    ref_ext = src_exts[0] # First camera is the reference camera
    ints = src_ints
    ref_int = src_ints[0] # First camera is the reference camera

    psv = create_psv(imgs_to_warp,
                    exts,
                    ref_ext,
                    ints,
                    ref_int,
                    depths)
    psv = psv.reshape([1, -1, num_depths, h, w])
    psv_folder = os.path.join(folder, 'psv')
    os.makedirs(psv_folder, exist_ok=True)
    save_psv(psv, psv_folder, v)
    return psv


def save_psv(volume, folder, n_views):
    '''Save plane sweep volume to images
    '''
    b, c, d, h, w = volume.shape
    psv = volume.permute([2, 0, 1, 3, 4])
    for idx, im in enumerate(psv):
        for v in range(n_views):
            tmp = im[:, 3*v:3*(v+1)] * 255.
            tmp = tmp.permute([0, 2, 3, 1])[0]
            tmp = tmp.cpu().detach().numpy().astype(np.uint8)
            plt.imsave(os.path.join(folder, str(v) + '_' + str(idx) + '.png'), tmp)
        
        combine = im.view([b, n_views, int(c/n_views), h, w])
        tmp = torch.clip(torch.mean(combine, 1), 0.0, 1.0) * 255.
        tmp = tmp.permute([0, 2, 3, 1])[0]
        tmp = tmp.cpu().detach().numpy().astype(np.uint8)
        plt.imsave(os.path.join(folder, 'combined_' + str(idx) + '.png'), tmp)

if __name__ == "__main__":
    device = 'cpu'

    args = get_opts()

    scenes = sorted(glob.glob(os.path.join(args.root_dir, 'scene*')))
    print(scenes)

    for scene in scenes:
        run_psv(scene, device)