#!/bin/bash
exec > ~/wm_cond2.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hunyuanworld-mirror; cd ~/HunyuanWorld-Mirror
echo "START wm-cond2 (W2C) $(date +%T)"
python infer_cond.py --input_path ~/wm_courthouse_in --output_path ~/wm_cond2_out --target_size 518 2>&1
echo "PYEXIT=$?"
find ~/wm_cond2_out -name "gaussians.ply" -exec ls -la {} \;
echo WM_COND2_SENTINEL
