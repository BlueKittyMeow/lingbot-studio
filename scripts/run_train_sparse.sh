#!/bin/bash
exec > ~/train_sparse.log 2>&1
# wait for the WM GPU job to free the GPU (avoid collision)
for i in $(seq 1 60); do [ "$(systemctl is-active wm-courthouse2 2>/dev/null)" != "active" ] && break; sleep 10; done
source ~/miniforge3/etc/profile.d/conda.sh; conda activate gsplat
echo "START train on sparse courthouse $(date +%T) | gsplat env: $(python -c 'import gsplat;print(gsplat.__version__)')"
python ~/train_splat_sparse.py 6000 1500000
echo "TRAIN_EXIT=$?"
ls -la ~/renders/courthouse_sparse_splat.ply 2>/dev/null
echo TRAIN_SENTINEL
