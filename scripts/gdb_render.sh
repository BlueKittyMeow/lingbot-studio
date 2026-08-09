#!/bin/bash
exec > ~/gdb_render.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lingbot-map
export CUDA_HOME=$CONDA_PREFIX
export LIBRARY_PATH="/usr/lib/wsl/lib:$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH"
cd ~/lingbot-map
gdb -batch -ex run -ex "bt 15" --args python demo_render/batch_demo.py \
  --input_folder example/courthouse --output_folder ~/renders/courthouse \
  --model_path ~/models/lingbot-map/lingbot-map.pt \
  --config demo_render/config/wsl16gb.yaml \
  --mask_sky --num_scale_frames 2 --first_k 120
echo "GDB_DONE"
