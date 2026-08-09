#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lingbot-map
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 > ~/torch_install.log 2>&1
echo "pip exit: $?" >> ~/torch_install.log
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))" >> ~/torch_install.log 2>&1
