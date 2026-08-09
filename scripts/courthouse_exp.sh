#!/bin/bash
exec > ~/courthouse_exp.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lingbot-map; cd ~/lingbot-map
COMMON="--use_sdpa --kv_cache_sliding_window 32 --num_scale_frames 4 --model_path /home/bluekitty/models/lingbot-map/lingbot-map.pt --config demo_render/config/wsl16gb.yaml --save_predictions"
echo "===== SPARSE (stride 2, ~143 frames) ====="
python demo_render/batch_demo.py --input_folder example --scenes courthouse --image_stride 2 --output_folder ~/renders/courthouse_sparse $COMMON
echo "SPARSE_DONE $?"
echo "===== DENSE (stride 1, all ~286 frames) ====="
python demo_render/batch_demo.py --input_folder example --scenes courthouse --image_stride 1 --output_folder ~/renders/courthouse_dense $COMMON
echo "DENSE_DONE $?"
echo "COURTHOUSE_SENTINEL"
