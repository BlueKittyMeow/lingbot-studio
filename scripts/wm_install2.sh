#!/bin/bash
exec > ~/wm_install2.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hunyuanworld-mirror
pip install --no-input onnxruntime || { echo FAIL_ONNX; exit 1; }
python -c "import gsplat,torch,open3d,pycolmap,onnxruntime; print('IMPORTS_OK',torch.__version__,gsplat.__version__)" || { echo FAIL_IMPORTS; exit 1; }
huggingface-cli download tencent/HunyuanWorld-Mirror --local-dir ~/HunyuanWorld-Mirror/ckpts || { echo FAIL_WEIGHTS; exit 1; }
echo "WM_READY_SENTINEL"
