#!/bin/bash
exec > ~/cuda_install.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lingbot-map
conda install -y -c nvidia/label/cuda-12.8.1 cuda-toolkit
echo "conda exit: $?"
export CUDA_HOME=$CONDA_PREFIX
which nvcc && nvcc --version | tail -1
echo "CUDA_INSTALL_COMPLETE"
