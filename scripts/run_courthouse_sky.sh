#!/bin/bash
exec > ~/courthouse_sky.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh
echo "STAGE maskgen $(date +%T)"
conda activate hunyuanworld-mirror
python ~/gen_sky_masks.py || { echo FAIL_MASKGEN; exit 1; }
echo "STAGE train $(date +%T)"
conda activate gsplat
python ~/train_splat_sky.py 6000 1500000 || { echo FAIL_TRAIN; exit 1; }
echo SKY_SENTINEL
