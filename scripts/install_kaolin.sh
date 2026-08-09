#!/bin/bash
exec > ~/kaolin_install.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lingbot-map
export CUDA_HOME=$CONDA_PREFIX
export LIBRARY_PATH="/usr/lib/wsl/lib:$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH"
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html \
  || pip install kaolin --no-build-isolation
python -c "import kaolin; print('KAOLIN_OK', kaolin.__version__)"
echo "KAOLIN_INSTALL_COMPLETE"
