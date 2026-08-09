#!/bin/bash
exec > ~/wm_cond5.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hunyuanworld-mirror; cd ~/HunyuanWorld-Mirror
python infer_cond.py --input_path ~/wm_courthouse_in --output_path ~/wm_cond5_out --target_size 518 2>&1
echo "PYEXIT=$?"; find ~/wm_cond5_out -name gaussians.ply -exec ls -la {} \; ; echo WM_COND5_SENTINEL
