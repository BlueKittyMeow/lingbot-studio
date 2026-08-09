#!/bin/bash
exec > ~/wm_courthouse2.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hunyuanworld-mirror; cd ~/HunyuanWorld-Mirror
echo "START wm-courthouse2 (24f + sky mask) $(date +%T)"
python infer.py --input_path ~/wm_courthouse_in2 --output_path ~/wm_courthouse_out2 --target_size 518 --apply_sky_mask 2>&1
echo "PYEXIT=$?"
find ~/wm_courthouse_out2 -name "*.ply" -exec ls -la {} \;
echo WM_CH2_SENTINEL
