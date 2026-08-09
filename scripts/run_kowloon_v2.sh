#!/bin/bash
exec > ~/kowloon_v2.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lingbot-map; cd ~/lingbot-map
python demo_render/batch_demo.py --video_path ~/kowloon/kwc_walk_crop4.mp4 --fps 10 --first_k 300 \
  --use_sdpa --kv_cache_sliding_window 32 --num_scale_frames 4 \
  --output_folder ~/renders/kowloon_v2 --model_path ~/models/lingbot-map/lingbot-map.pt \
  --config demo_render/config/wsl16gb.yaml --save_predictions
echo "PYEXIT=$?"; echo KWLN_SENTINEL
