#!/bin/bash
exec > ~/o3d_swap.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lingbot-map
pip uninstall -y open3d
pip install open3d-cpu
python -c "import open3d; print('O3D_CPU_OK', open3d.__version__)"
echo "SWAP_DONE"
