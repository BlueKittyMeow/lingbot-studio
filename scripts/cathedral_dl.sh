#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lingbot-map
cd ~/footage
yt-dlp -f "bv*[height<=1080]+ba/b[height<=1080]" -o ~/footage/cathedral_src.mp4 "https://www.youtube.com/watch?v=mzOwOSm2ubE" > ~/dl_cathedral.log 2>&1
echo "CATHEDRAL_DL_DONE exit=$?" >> ~/dl_cathedral.log
ls -la ~/footage/cathedral_src.mp4* >> ~/dl_cathedral.log 2>&1
