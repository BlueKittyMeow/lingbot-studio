#!/bin/bash
exec > ~/wm_smoke.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh; conda activate hunyuanworld-mirror; cd ~/HunyuanWorld-Mirror
echo "START smoke $(date +%T)"
python infer.py --output_path ~/wm_smoke_out 2>&1
echo "SMOKE_EXIT=$?"
echo "=== outputs ==="; find ~/wm_smoke_out -type f 2>/dev/null | head -20
echo WM_SMOKE_SENTINEL
