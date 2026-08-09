#!/bin/bash
exec > ~/wm_courthouse.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hunyuanworld-mirror; cd ~/HunyuanWorld-Mirror
echo "START wm-courthouse $(date +%T)"
python infer.py --input_path ~/wm_courthouse_in --output_path ~/wm_courthouse_out --target_size 518 2>&1
echo "PYEXIT=$?"
echo "=== plys ==="; find ~/wm_courthouse_out -name "*.ply" -exec ls -la {} \;
echo WM_CH_SENTINEL
