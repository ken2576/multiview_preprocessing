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


### Troubleshooting

Sometimes the last frame could be corrupted when doing frame extraction. Rerun ```undistort_opencv.py``` with argument ```--fix_last``` to fix it.