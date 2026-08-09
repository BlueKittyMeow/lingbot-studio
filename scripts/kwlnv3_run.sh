#!/bin/bash
sed -e 's#kowloon_v2#kowloon_v3#g' -e 's#kwlnv2_#kwlnv3_#g' ~/kwlnv2_fuse.py > ~/kwlnv3_fuse.py
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lingbot-map
python ~/kwlnv3_fuse.py > ~/kwlnv3_fuse.log 2>&1
echo KWLNV3_FUSE_DONE >> ~/kwlnv3_fuse.log
