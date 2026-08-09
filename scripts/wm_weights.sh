#!/bin/bash
exec > ~/wm_weights.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hunyuanworld-mirror
export HF_HUB_ENABLE_HF_TRANSFER=0
hf download tencent/HunyuanWorld-Mirror --local-dir ~/HunyuanWorld-Mirror/ckpts && echo "WM_WEIGHTS_OK" || echo "WM_WEIGHTS_FAIL"
ls -la ~/HunyuanWorld-Mirror/ckpts
echo "WM_WEIGHTS_SENTINEL"
