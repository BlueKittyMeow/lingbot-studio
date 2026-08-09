#!/bin/bash
exec > ~/c2q48.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lingbot-map; cd ~/lingbot-map
echo "START c2q48: fps10 sw48 nsf4"
python demo_render/batch_demo.py --video_path ~/footage/catacombs2_crop.mp4 --fps 10 --first_k 300 \
  --use_sdpa --kv_cache_sliding_window 48 --num_scale_frames 4 \
  --output_folder ~/renders/c2q48 --model_path ~/models/lingbot-map/lingbot-map.pt \
  --config demo_render/config/wsl16gb.yaml --save_predictions
echo "PYEXIT=$?"; echo DONE_SENTINEL
