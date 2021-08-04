import numpy as np
import torch
import torch.nn.functional as F

def divide_safe(num, denom):
    eps = 1e-8
    tmp = denom + eps * torch.le(denom, 1e-20).to(torch.float)
    return num / tmp

def meshgrid_pinhole(h, w,
                    is_homogenous=True, device=None):
    '''Create a meshgrid for image coordinate

    Args:
        h: grid height
        w: grid width
        is_homogenous: return homogenous or not
    Returns:
        Image coordinate meshgrid [height, width, 2 (3 if homogenous)]
    '''
    xs = torch.linspace(0, w-1, steps=w, device=device) + 0.5
    ys = torch.linspace(0, h-1, steps=h, device=device) + 0.5
    new_y, new_x = torch.meshgrid(ys, xs)
    grid = (new_x, new_y)

    if is_homogenous:
        ones = torch.ones_like(new_x)
        grid = torch.stack(grid + (ones, ), 2)
    else:
        grid = torch.stack(grid, 2)
    return grid

def batch_inverse_warp(src_im, depths, src_int, inv_tgt_int, trnsf, h=-1, w=-1):
    """Projective inverse warping for image

    Args:
        src_im: source image [batch, #channel, height, width]
        depths: depths of the image [#planes, batch]
        src_int: I_s matrix for source camera [batch, 3, 3]
        inv_tgt_int: I_t^-1 for target camera [batch, 3, 3]
        trnsf: E_s * E_t^-1, the transformation between cameras [batch, 4, 4]
        h: target height
        w: target width
    Returns:
        Warped image [batch #channel, #planes, height, width]
    """
    if h == -1 or w == -1:
        b, _, h, w = src_im.shape
        src_h, src_w = h, w
    else:
        b, _, src_h, src_w = src_im.shape

    # Generate image coordinates for target camera
    im_coord = meshgrid_pinhole(h, w, device=src_im.device)
    coord = im_coord.view([-1, 3])
    coord = coord.unsqueeze(0).repeat([b, 1, 1])

    # Convert to camera coordinates
    cam_coord = torch.matmul(inv_tgt_int.unsqueeze(1), coord[..., None])
    cam_coord = cam_coord * depths[:, :, None, None, None]
    ones = torch.ones_like(cam_coord[..., 0:1, :])

    cam_coord = torch.cat([cam_coord, ones], -2)

    # Convert to another camera's coordinates
    new_cam_coord = torch.matmul(trnsf.unsqueeze(1), cam_coord)

    # Convert to image coordinates at source camera
    im_coord = torch.matmul(src_int.unsqueeze(1), new_cam_coord[..., :3, :])
    im_coord = im_coord.squeeze(dim=-1).view(depths.shape + (h, w, 3))

    # Fix for NaN and backward projection
    zeros = torch.zeros_like(im_coord[..., 2:3])
    im_coord[..., 2:3] = torch.where(im_coord[..., 2:3] > 0, im_coord[..., 2:3], zeros)
    im_coord = divide_safe(im_coord[..., :2], im_coord[..., 2:3])
    im_coord[..., 0] = im_coord[..., 0] / src_w * 2 - 1.
    im_coord[..., 1] = im_coord[..., 1] / src_h * 2 - 1.

    # Sample from the source image
    warped = F.grid_sample(src_im.repeat(depths.shape[0], 1, 1, 1),
                           im_coord.view(-1, *(im_coord.shape[2:])), align_corners=True)
    return warped.view(depths.shape + (3, h, w)).permute([1, 2, 0, 3, 4])
