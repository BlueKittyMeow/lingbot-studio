import numpy as np, torch, imageio, glob, os
from gsplat import rasterization
PLY='/home/bluekitty/renders/courthouse_splat.ply'; SRC='/home/bluekitty/renders/courthouse/courthouse'; dev='cuda'
with open(PLY,'rb') as f:
    f.readline(); f.readline(); n=int(f.readline().split()[-1]); props=[]
    while True:
        l=f.readline().strip()
        if l==b'end_header': break
        if l.startswith(b'property'): props.append(l.split()[-1].decode())
    data=np.fromfile(f,dtype=np.float32,count=n*len(props)).reshape(n,len(props))
d={p:data[:,i] for i,p in enumerate(props)}
means=torch.tensor(np.stack([d['x'],d['y'],d['z']],1),device=dev)
quats=torch.tensor(np.stack([d['rot_0'],d['rot_1'],d['rot_2'],d['rot_3']],1),device=dev)
scales=torch.tensor(np.exp(np.stack([d['scale_0'],d['scale_1'],d['scale_2']],1)),device=dev)
opac=torch.sigmoid(torch.tensor(d['opacity'],device=dev))
C0=0.28209479177387814
colors=torch.tensor(np.clip(np.stack([d['f_dc_0'],d['f_dc_1'],d['f_dc_2']],1)*C0+0.5,0,1),device=dev)
files=sorted(glob.glob(os.path.join(SRC,'frame_*.npz')))
for idx in [0,60]:
    dd=np.load(files[idx]); extr=dd['extrinsic']; intr=dd['intrinsic']; H,W=dd['depth'].shape[:2]
    vm=np.eye(4,dtype=np.float32); vm[:3,:4]=extr
    out,_,_=rasterization(means,quats/quats.norm(dim=-1,keepdim=True),scales,opac,colors,
        torch.tensor(vm[None],device=dev),torch.tensor(intr[None].astype(np.float32),device=dev),W,H)
    imageio.imwrite(f'/home/bluekitty/renders/splat_test_{idx}.png',(out[0].clamp(0,1).cpu().numpy()*255).astype(np.uint8))
    imageio.imwrite(f'/home/bluekitty/renders/splat_gt_{idx}.png',(np.transpose(dd['images'],(1,2,0))*255).astype(np.uint8))
    print("rendered",idx,flush=True)
print("DONE")
