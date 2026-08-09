#!/bin/bash
exec > ~/gsplat_env.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh
conda create -n gsplat python=3.10 -y -q >/dev/null 2>&1
conda activate gsplat
echo "=== torch 2.4.0 cu124"
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2
echo "=== gsplat 1.5.3 PREBUILT wheel (no compile)"
pip install --no-deps gsplat==1.5.3 --index-url https://docs.gsplat.studio/whl/pt24cu124 2>&1 | tail -2
pip install -q numpy imageio jaxtyping typing_extensions 2>&1 | tail -1
echo "=== smoke test (uses precompiled kernel, no JIT):"
python - <<'TEST'
import torch
from gsplat import rasterization
d='cuda'; N=200
means=torch.randn(N,3,device=d); quats=torch.nn.functional.normalize(torch.randn(N,4,device=d),dim=-1)
scales=torch.rand(N,3,device=d)*0.1; opac=torch.rand(N,device=d); colors=torch.rand(N,3,device=d)
vm=torch.eye(4,device=d)[None]; K=torch.tensor([[300.,0,256],[0,300.,256],[0,0,1]],device=d)[None]
out,_,_=rasterization(means,quats,scales,opac,colors,vm,K,512,512)
print("RAST_OK", tuple(out.shape), torch.cuda.get_device_name(0))
TEST
echo "GSPLAT_ENV_DONE"
