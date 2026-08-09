#!/bin/bash
exec > ~/wm_install.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh
echo "STAGE clone-env $(date +%T)"
conda env list | grep -q hunyuanworld-mirror || conda create --clone gsplat -n hunyuanworld-mirror -y || { echo FAIL_CLONE; exit 1; }
echo "STAGE git-clone $(date +%T)"
cd ~ && { [ -d HunyuanWorld-Mirror/.git ] || git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror; } || { echo FAIL_GIT; exit 1; }
conda activate hunyuanworld-mirror
cd ~/HunyuanWorld-Mirror
echo "STAGE pip-reqs $(date +%T)"
# preserve our working torch/torchvision/gsplat; install only the rest
grep -viE '^[[:space:]]*(torch|torchvision|gsplat)\b' requirements.txt > /tmp/req_notorch.txt
cat /tmp/req_notorch.txt
nice -n 15 pip install --no-input -r /tmp/req_notorch.txt || { echo FAIL_REQS; exit 1; }
echo "STAGE verify-imports $(date +%T)"
python -c "import gsplat, torch, open3d, pycolmap, onnxruntime; print('IMPORTS_OK', torch.__version__, gsplat.__version__)" || { echo FAIL_IMPORTS; exit 1; }
echo "STAGE hf-weights $(date +%T)"
pip install --no-input "huggingface_hub[cli]" || { echo FAIL_HFCLI; exit 1; }
huggingface-cli download tencent/HunyuanWorld-Mirror --local-dir ~/HunyuanWorld-Mirror/ckpts || { echo FAIL_WEIGHTS; exit 1; }
echo "INSTALL_DONE_SENTINEL $(date +%T)"
