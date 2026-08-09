#!/bin/bash
exec > ~/lingbot_demo.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lingbot-map
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/lingbot-map
echo "=== GPU state at launch:"
/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
python demo.py --model_path ~/models/lingbot-map/lingbot-map.pt --image_folder example/courthouse --mask_sky --offload_to_cpu --num_scale_frames 2 --first_k 120 --use_sdpa --port 8890
echo "DEMO_EXITED code=$?"
