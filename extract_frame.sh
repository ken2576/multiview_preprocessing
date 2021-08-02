#bin/bash
echo "Data root directory: $1";
echo "Output to: $2";
echo "Intrinsics param folder: $3"

echo "Processing test data..."
python undistort_opencv.py \
    --path $1/test/1080p/2_8/ \
    --out_folder $2/test/1080p/2_8/ \
    --param_path calib_1080.pkl \
    --img_wh 960 540
python undistort_opencv.py \
    --path $1/test/2_7k/2_7/ \
    --out_folder $2/test/2_7k/2_7/ \
    --param_path calib_2k.pkl \
    --img_wh 1920 1080 
python undistort_opencv.py \
    --path $1/test/2_7k/2_8/ \
    --out_folder $2/test/2_7k/2_8/ \
    --param_path calib_2k.pkl \
    --img_wh 1920 1080 
python undistort_opencv.py \
    --path $1/test/2_7k/2_10/ \
    --out_folder $2/test/2_7k/2_10/ \
    --param_path calib_2k.pkl \
    --img_wh 1920 1080 

echo "Processing train data..."
python undistort_opencv.py \
    --path $1/test/1080p/2_7/ \
    --out_folder $2/test/1080p/2_7/ \
    --param_path calib_1080.pkl \
    --img_wh 960 540
python undistort_opencv.py \
    --path $1/test/1080p/2_8/ \
    --out_folder $2/test/1080p/2_8/ \
    --param_path calib_1080.pkl \
    --img_wh 960 540
python undistort_opencv.py \
    --path $1/test/1080p/2_10/ \
    --out_folder $2/test/1080p/2_10/ \
    --param_path calib_1080.pkl \
    --img_wh 960 540
python undistort_opencv.py \
    --path $1/test/1080p/2_24/ \
    --out_folder $2/test/1080p/2_24/ \
    --param_path calib_1080.pkl \
    --img_wh 960 540
python undistort_opencv.py \
    --path $1/train/2_7k/2_7/ \
    --out_folder $2/train/2_7k/2_7/ \
    --param_path calib_2k.pkl \
    --img_wh 1920 1080 
python undistort_opencv.py \
    --path $1/train/2_7k/2_8/ \
    --out_folder $2/train/2_7k/2_8/ \
    --param_path calib_2k.pkl \
    --img_wh 1920 1080 
python undistort_opencv.py \
    --path $1/train/2_7k/2_10/ \
    --out_folder $2/train/2_7k/2_10/ \
    --param_path calib_2k.pkl \
    --img_wh 1920 1080 
python undistort_opencv.py \
    --path $1/train/2_7k/3_22/ \
    --out_folder $2/train/2_7k/3_22/ \
    --param_path calib_2k.pkl \
    --img_wh 1920 1080 

echo "Processing unused data..."
python undistort_opencv.py \
    --path $1/unused/2_8/ \
    --out_folder $2/unused/2_8/ \
    --param_path calib_2k.pkl \
    --img_wh 1920 1080 
python undistort_opencv.py \
    --path $1/unused/2_10/ \
    --out_folder $2/unused/2_10/ \
    --param_path calib_2k.pkl \
    --img_wh 1920 1080 
