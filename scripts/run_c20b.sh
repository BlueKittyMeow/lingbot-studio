#!/bin/bash
exec > ~/c20b.log 2>&1
for i in $(seq 1 60); do [ "$(systemctl is-active c2qfuse 2>/dev/null)" != "active" ] && break; sleep 10; done
echo "START c20b: fps20 sw32 nsf4 (solo)"
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lingbot-map; cd ~/lingbot-map
python demo_render/batch_demo.py --video_path ~/footage/catacombs2_crop.mp4 --fps 20 --first_k 300 \
  --use_sdpa --kv_cache_sliding_window 32 --num_scale_frames 4 \
  --output_folder ~/renders/c20 --model_path ~/models/lingbot-map/lingbot-map.pt \
  --config demo_render/config/wsl16gb.yaml --save_predictions
echo "PYEXIT=$?"; echo DONE_SENTINEL
