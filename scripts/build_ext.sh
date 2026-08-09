#!/bin/bash
exec > ~/ext_build.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lingbot-map
export CUDA_HOME=$CONDA_PREFIX
export LIBRARY_PATH="/usr/lib/wsl/lib:$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH"
cd ~/lingbot-map/demo_render/render_cuda_ext
pip install -e . -v --no-build-isolation 2>&1 | tail -5
python -c "import frustum_cull_ext; print('EXT_OK')" || python -c "import render_cuda_ext; print('EXT_OK via package')"
echo "EXT_BUILD_COMPLETE"
