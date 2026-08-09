#!/bin/bash
exec > ~/courthouse_fuse.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lingbot-map
python ~/fuse_generic.py /home/bluekitty/renders/courthouse_sparse chsparse
python ~/fuse_generic.py /home/bluekitty/renders/courthouse_dense  chdense
echo FUSE_SENTINEL
