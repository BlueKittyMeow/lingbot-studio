source ~/miniforge3/etc/profile.d/conda.sh; conda activate lingbot-map
export CUDA_HOME=$CONDA_PREFIX LIBRARY_PATH="/usr/lib/wsl/lib:$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=1
nice -n 15 python - <<'TEST' > ~/gsplat_smoke.log 2>&1
import torch
from gsplat import rasterization
print("importing done, compiling kernels...", flush=True)
d='cuda'; N=200
means=torch.randn(N,3,device=d); quats=torch.nn.functional.normalize(torch.randn(N,4,device=d),dim=-1)
scales=torch.rand(N,3,device=d)*0.1; opac=torch.rand(N,device=d); colors=torch.rand(N,3,device=d)
vm=torch.eye(4,device=d)[None]; K=torch.tensor([[300.,0,256],[0,300.,256],[0,0,1]],device=d)[None]
out,_,_=rasterization(means,quats,scales,opac,colors,vm,K,512,512)
print("RAST_OK", tuple(out.shape), flush=True)
TEST
echo "SMOKE_DONE" >> ~/gsplat_smoke.log
