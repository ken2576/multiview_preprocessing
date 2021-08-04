#bin/bash
echo "Extracted image directory: $1";

echo "Processing test data...";
python gen_bg.py \
    $1/test/1080p/2_8/
python gen_bg.py \
    $1/test/2_7k/2_7/
python gen_bg.py \
    $1/test/2_7k/2_8/
python gen_bg.py \
    $1/test/2_7k/2_10/

echo "Processing train data..."
python gen_bg.py \
    $1/train/1080p/2_7/
python gen_bg.py \
    $1/train/1080p/2_8/
python gen_bg.py \
    $1/train/1080p/2_10/
python gen_bg.py \
    $1/train/1080p/2_24/
python gen_bg.py \
    $1/train/2_7k/2_7/
python gen_bg.py \
    $1/train/2_7k/2_8/
python gen_bg.py \
    $1/train/2_7k/2_10/
python gen_bg.py \
    $1/train/2_7k/3_22/
