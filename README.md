# Multiview Camera Preprocessing Scripts

### Requirements

1. Run ```pip install -r requirements.txt```

2. (Optional for checking camera pose) Install PyTorch (1.7.1, 1.9.0 tested)

3. (Optional for generating camera poses) [COLMAP](https://colmap.github.io/)

4. Foreground image generation [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2)

### Usage

1. Download data and extract

2. Run ```sh extract_frame.sh [data_root_directory] [output_directory] [intrinsics_folder]```

3.  Use the precalibrated camera poses or generate your own (see below).

    (Optional) Acquire camera pose with COLMAP using ```python get_pose.py [scene_directory] --scene_ids [desired scene ids]```

    For example, ```python get_pose.py [output_directory]/test/2_7k/2_8/ --scene_ids 18``` to process only scene018 or omit ```--scene_ids``` to process all scenes

    Might need to check COLMAP results to see if camera poses are reasonable (looks like a grid). If not, consider supplying a pose prior as follows.

    Run ```python get_pose_with_prior.py --root_dir [images_folder] --prior_path [poses_bounds_npy] --extension [jpg_or_png]```

    For example, ```python get_pose_with_prior.py --root_dir [output_directory]/test/2_7k/2_8/scene018/images --prior_path scene018_pb.npy --extension .jpg``` to acquire pose for scene018 using a pose prior stored in the npy file.

4. Generate background images with ```sh gen_bg.sh [output_directory]```

5. Generate foreground images (requires [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2))
    
    Drop ```gen_fg.py``` and ```gen_fg.sh``` into ```BackgroundMattingV2```'s folder and run ```sh gen_fg.sh [ckpt_path] [resnet101 | resnet50] [output_directory]```

6. Compress images with ```sh compress_data.sh [output_directory] [compressed_output_directory]```

7. Collect the poses with ```sh collect_poses.sh [output_directory] [compressed_output_directory]```

### Dataset usage

The camera poses are stored in [LLFF](https://github.com/Fyusion/LLFF) convention in ```npy``` format.

E.g. camera poses are stored in shape ```(#views, 17)``` where the camera parameters are flattened.

It can be converted to intrinsic and world-to-camera matrix with:

```python
import numpy as np

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
    mat = np.linalg.inv(ext)
    mat = mat[[1, 0, 2]]
    mat[2] = -mat[2]
    mat[:, 2] = -mat[:, 2]

    return np.concatenate([mat, hwf], -1)

input_poses = np.load('scene000_pb.npy')
pose = input_poses[0, :-2].reshape([3, 5])
bounds = input_poses[0, -2:]
w2c, k = pose2mat(convert_llff(pose))
print(w2c.shape)
print(k.shape)

```

The image data after compression is stored in ```h5``` format with keys:

```rgb```: original RGB frames in shape ```(#views, #frames, height, width, 3)```

```fg_rgb```: processed foreground RGB frames in shape ```(#views, #frames, height, width, 4)```

```bg_rgb```: median-filtered background RGB images in shape ```(#views, height, width, 3)```


Example script:
```python
import h5py
with h5py.File('scene000.h5', 'r') as hf:
    rgb = hf['rgb'] # (#views, #frames, height, width, 3)

```

### Troubleshooting

Sometimes the last frame could be corrupted when doing frame extraction. Rerun ```undistort_opencv.py``` with argument ```--fix_last``` to fix it.
