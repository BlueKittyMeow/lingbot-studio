#!/bin/bash
exec > ~/open3d_install.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lingbot-map
pip install open3d
python -c "import open3d; print('OPEN3D_OK', open3d.__version__)"
echo "O3D_COMPLETE"
