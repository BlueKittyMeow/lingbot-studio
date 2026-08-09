#!/bin/bash
exec > ~/lingbot_render.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lingbot-map
export CUDA_HOME=$CONDA_PREFIX
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LIBRARY_PATH="/usr/lib/wsl/lib:$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH"
export OPEN3D_CPU_RENDERING=true
export EGL_PLATFORM=x11
export LIBGL_ALWAYS_SOFTWARE=1
export LINGBOT_UNMASK=1
export XDG_RUNTIME_DIR=/tmp/xdg-bluekitty
mkdir -p /tmp/xdg-bluekitty && chmod 700 /tmp/xdg-bluekitty
cd ~/lingbot-map
PYTHONFAULTHANDLER=1 xvfb-run -a python demo_render/batch_demo.py \
  --input_folder example/courthouse \
  --output_folder ~/renders/courthouse \
  --model_path ~/models/lingbot-map/lingbot-map.pt \
  --config demo_render/config/wsl16gb.yaml \
  --mask_sky --num_scale_frames 2 --first_k 120 --num_workers 4
echo "RENDER_EXITED code=$?"
