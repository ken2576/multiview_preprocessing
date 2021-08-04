import os
import glob
import shutil
import collections
import numpy as np
import matplotlib.pyplot as plt
from colmap_read_model import rotmat2qvec
from colmap_wrapper_with_prior import run_feature, run_triangulate, run_bundle_adjust
from db_utils import get_image_to_id


Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

def write_cameras_text(cameras, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = '# Camera list with one line of data per camera:\n'
    '#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n'
    '# Number of cameras: {}\n'.format(len(cameras))
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            # print(cam.params)
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")

def write_images_text(images, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    if len(images) == 0:
        mean_observations = 0
    else:
        mean_observations = sum((len(img.point3D_ids) for _, img in images.items()))/len(images)
    HEADER = '# Image list with two lines of data per image:\n'
    '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n'
    '#   POINTS2D[] as (X, Y, POINT3D_ID)\n'
    '# Number of images: {}, mean observations per image: {}\n'.format(len(images), mean_observations)

    with open(path, "w") as fid:
        fid.write(HEADER)
        for key in sorted(images):
            img = images[key]
            image_header = [img.id, *img.qvec, *img.tvec, img.camera_id, img.name]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                points_strings.append(" ".join(map(str, [*xy, point3D_id])))
            fid.write(" ".join(points_strings) + "\n")

def write_points3D_text(points3D, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    if len(points3D) == 0:
        mean_track_length = 0
    else:
        mean_track_length = sum((len(pt.image_ids) for _, pt in points3D.items()))/len(points3D)
    HEADER = '# 3D point list with one line of data per point:\n'
    '#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n'
    '# Number of points: {}, mean track length: {}\n'.format(len(points3D), mean_track_length)

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, pt in points3D.items():
            point_header = [pt.id, *pt.xyz, *pt.rgb, pt.error]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
                track_strings.append(" ".join(map(str, [image_id, point2D])))
            fid.write(" ".join(track_strings) + "\n")

def write_model(cameras, images, path):
    write_cameras_text(cameras, os.path.join(path, "cameras.txt"))
    write_images_text(images, os.path.join(path, "images.txt"))
    write_points3D_text({}, os.path.join(path, "points3D.txt"))
    return cameras, images


def create_colmap_cam(idx, width, height, focal_length, cx, cy, model='SIMPLE_PINHOLE'):
    params = np.array([focal_length, cx, cy])
    return Camera(id=idx, model=model,
        width=width, height=height,
        params=params)

def create_colmap_img(idx, name, ext, cam_id):
    R = ext[:3, :3]
    qvec = rotmat2qvec(R)
    tvec = np.asarray(ext[:3, 3]).squeeze()
    return BaseImage(id=idx, qvec=qvec, tvec=tvec, camera_id=cam_id, name=name, xys=[], point3D_ids=[])

def create_image_list(folder, extension='.png'):
    image_list = sorted(glob.glob(os.path.join(folder, 'images', f'*{extension}')))
    with open(os.path.join(folder, 'image_list.txt'), 'w') as f:
        for i in image_list:
            name = os.path.split(i)[-1]
            f.write(name + '\n')

def pose2mat(pose):
    """Convert pose matrix (3x5) to extrinsic matrix (4x4) and
    intrinsic matrix (3x3)
    
    Args:
        pose: 3x5 pose matrix
    Returns:
        Extrinsic matrix (4x4) and intrinsic matrix (3x3)
    """
    extrinsic = np.eye(4)
    extrinsic[:3, :] = pose[:, :4]
    h, w, focal_length = pose[:, 4]
    intrinsic = np.array([[focal_length, 0, w/2],
                            [0, focal_length, h/2],
                            [0,            0,   1]])

    return extrinsic, intrinsic

def convert_llff(pose):
    """Convert LLFF poses to OpenCV convention (w2c extrinsic and hwf)
    """
    hwf = pose[:3, 4:]

    ext = np.eye(4)
    ext[:3, :4] = pose[:3, :4]
    ext = ext[:, [1, 0, 2, 3]]
    ext[:, 2] *= -1
    mat = np.linalg.inv(ext)

    return np.concatenate([mat[:3], hwf], -1)

def proc_poses_bounds(input_poses):
    poses = input_poses[:, :-2].reshape([-1, 3, 5])
    poses = [pose2mat(convert_llff(x)) for x in poses]
    w2cs = np.stack([x[0] for x in poses])
    Ks = np.stack([x[1] for x in poses])

    params = []
    for K, w2c in zip(Ks, w2cs):
        focal_length = K[0, 0]
        cx = K[0, 2]
        cy = K[1, 2]
        ext = w2c

        params.append((focal_length, cx, cy, ext))
            
    return params

def get_shape(folder, extension='.png'):
    im_paths = sorted(glob.glob(os.path.join(folder, 'images', f'*{extension}')))
    im = plt.imread(im_paths[0])[..., :3]
    return im.shape[:2]

def proc_colmap(prior_path, scenedir, extension='.png'):
    run_feature(scenedir, 'exhaustive_matcher')

    out_folder = os.path.join(scenedir, 'sparse', '0')

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    params = proc_poses_bounds(np.load(prior_path))

    h, w = get_shape(scenedir, extension)

    colmap_cams = {}
    colmap_imgs = {}

    image_to_id = get_image_to_id(os.path.join(scenedir, 'database.db'))

    for idx, cam in enumerate(params):
        focal_length, cx, cy, ext = cam
        colmap_cam = create_colmap_cam(idx+1, w, h, focal_length, cx, cy)

        im_name = str(idx).zfill(3) + extension
        true_id = image_to_id[im_name]
        print(im_name, true_id)
        colmap_img = create_colmap_img(true_id, im_name, ext, idx+1)
        colmap_cams[idx+1] = colmap_cam
        colmap_imgs[true_id] = colmap_img

    write_model(colmap_cams, colmap_imgs, out_folder)
    run_triangulate(scenedir)
    run_bundle_adjust(scenedir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Processing data with COLMAP using prior poses.')
    parser.add_argument('--root_dir', type=str,
                        help='root directory of poses')
    parser.add_argument('--prior_path', type=str,
                        help='path to prior pose file')
    parser.add_argument('--extension', type=str,
                        default='.jpg',
                        help='image extension to use')

    args = parser.parse_args()
    create_image_list(args.root_dir, args.extension)
    db_file = glob.glob(os.path.join(args.root_dir, 'database.db'))
    if db_file:
        os.remove(db_file[0])
        shutil.rmtree(os.path.join(args.root_dir, 'sparse'))
        print('Old database file removed...')
    proc_colmap(args.prior_path, args.root_dir, args.extension)
    