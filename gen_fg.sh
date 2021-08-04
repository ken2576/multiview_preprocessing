#bin/bash
echo "BackgroundMattingV2 checkpoint path: $1";
echo "BackgroundMattingV2 model type: $2";
echo "Extrated image directory: $3";

echo "Processing test data...";
python gen_fg.py \
    --model-checkpoint $1 \
    --model-backbone $2 \
    --root_dir $3/test/2_7k/2_7/
python gen_fg.py \
    --model-checkpoint $1 \
    --model-backbone $2 \
    --root_dir $3/test/1080p/2_8/ \
python gen_fg.py \
    --model-checkpoint $1 \
    --model-backbone $2 \
    --root_dir $3/test/2_7k/2_8/ \
python gen_fg.py \
    --model-checkpoint $1 \
    --model-backbone $2 \
    --root_dir $3/test/2_7k/2_10/ \

echo "Processing train data..."
python gen_fg.py \
    --model-checkpoint $1 \
    --model-backbone $2 \
    --root_dir $3/train/1080p/2_7/ \
python gen_fg.py \
    --model-checkpoint $1 \
    --model-backbone $2 \
    --root_dir $3/train/1080p/2_8/ \
python gen_fg.py \
    --model-checkpoint $1 \
    --model-backbone $2 \
    --root_dir $3/train/1080p/2_10/ \
python gen_fg.py \
    --model-checkpoint $1 \
    --model-backbone $2 \
    --root_dir $3/train/1080p/2_24/ \
python gen_fg.py \
    --model-checkpoint $1 \
    --model-backbone $2 \
    --root_dir $3/train/2_7k/2_7/ \
python gen_fg.py \
    --model-checkpoint $1 \
    --model-backbone $2 \
    --root_dir $3/train/2_7k/2_8/ \
python gen_fg.py \
    --model-checkpoint $1 \
    --model-backbone $2 \
    --root_dir $3/train/2_7k/2_10/ \
python gen_fg.py \
    --model-checkpoint $1 \
    --model-backbone $2 \
    --root_dir $3/train/2_7k/3_22/ \
