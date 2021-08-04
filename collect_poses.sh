#bin/bash
echo "Extracted image directory: $1";
echo "Output directory: $2";

echo "Processing test data...";
python collect_poses.py \
    --root_dir $1/test/2_7k/2_7/ \
    --out_dir $2/test/
python collect_poses.py \
    --root_dir $1/test/1080p/2_8/ \
    --out_dir $2/test/
python collect_poses.py \
    --root_dir $1/test/2_7k/2_8/ \
    --out_dir $2/test/
python collect_poses.py \
    --root_dir $1/test/2_7k/2_10/ \
    --out_dir $2/test/

echo "Processing train data..."
python collect_poses.py \
    --root_dir $1/train/1080p/2_7/ \
    --out_dir $2/train/
python collect_poses.py \
    --root_dir $1/train/1080p/2_8/ \
    --out_dir $2/train/
python collect_poses.py \
    --root_dir $1/train/1080p/2_10/ \
    --out_dir $2/train/
python collect_poses.py \
    --root_dir $1/train/1080p/2_24/ \
    --out_dir $2/train/
python collect_poses.py \
    --root_dir $1/train/2_7k/2_7/ \
    --out_dir $2/train/
python collect_poses.py \
    --root_dir $1/train/2_7k/2_8/ \
    --out_dir $2/train/
python collect_poses.py \
    --root_dir $1/train/2_7k/2_10/ \
    --out_dir $2/train/
python collect_poses.py \
    --root_dir $1/train/2_7k/3_22/ \
    --out_dir $2/train/
