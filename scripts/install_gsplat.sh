#!/bin/bash
exec > ~/gsplat_install.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lingbot-map
export CUDA_HOME=$CONDA_PREFIX
export LIBRARY_PATH="/usr/lib/wsl/lib:$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH"
echo "=== installing gsplat + deps"
pip install gsplat imageio scikit-image 2>&1 | tail -4
echo "=== import + JIT-compile a tiny rasterization (first call compiles CUDA)"
python - <<'TEST'
import torch
from gsplat import rasterization
print("gsplat imported")
d='cuda'; N=200
means=torch.randn(N,3,device=d)
quats=torch.nn.functional.normalize(torch.randn(N,4,device=d),dim=-1)
scales=torch.rand(N,3,device=d)*0.1
opac=torch.rand(N,device=d)
colors=torch.rand(N,3,device=d)
viewmat=torch.eye(4,device=d)[None]
K=torch.tensor([[300.,0,256],[0,300.,256],[0,0,1]],device=d)[None]
out,alpha,meta=rasterization(means,quats,scales,opac,colors,viewmat,K,512,512)
print("RAST_OK", tuple(out.shape))
TEST
echo "GSPLAT_DONE"
