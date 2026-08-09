import numpy as np, torch, imageio, glob, os
from gsplat import rasterization
IN='/home/bluekitty/renders/courthouse_splat.ply'; OUT='/home/bluekitty/renders/courthouse_splat_clean.ply'
SRC='/home/bluekitty/renders/courthouse/courthouse'; dev='cuda'
with open(IN,'rb') as f:
    f.readline(); f.readline(); n=int(f.readline().split()[-1]); props=[]
    while True:
        l=f.readline().strip()
        if l==b'end_header': break
        if l.startswith(b'property'): props.append(l.split()[-1].decode())
    data=np.fromfile(f,dtype=np.float32,count=n*len(props)).reshape(n,len(props))
d={p:i for i,p in enumerate(props)}
xyz=data[:,[d['x'],d['y'],d['z']]]
scales=np.exp(data[:,[d['scale_0'],d['scale_1'],d['scale_2']]]); maxs=scales.max(1)
opac=1/(1+np.exp(-data[:,d['opacity']]))
lo=np.percentile(xyz,0.5,0); hi=np.percentile(xyz,99.5,0); ext=float(np.linalg.norm(hi-lo)); marg=0.08*ext
inb=np.all((xyz>=lo-marg)&(xyz<=hi+marg),1)
keep=(opac>0.10)&(maxs<0.06*ext)&inb
print(f"total {n}  kept {keep.sum()} ({100*keep.mean():.0f}%)  ext {ext:.2f}",flush=True)
clean=data[keep]
with open(OUT,'wb') as fh:
    fh.write(b"ply\nformat binary_little_endian 1.0\n"); fh.write(("element vertex %d\n"%len(clean)).encode())
    for p in props: fh.write(("property float %s\n"%p).encode())
    fh.write(b"end_header\n"); clean.astype(np.float32).tofile(fh)
# render training view 0 + an off-axis view to check floaters
means=torch.tensor(clean[:,[d['x'],d['y'],d['z']]],device=dev)
quats=torch.tensor(clean[:,[d['rot_0'],d['rot_1'],d['rot_2'],d['rot_3']]],device=dev)
sc=torch.tensor(np.exp(clean[:,[d['scale_0'],d['scale_1'],d['scale_2']]]),device=dev)
op=torch.sigmoid(torch.tensor(clean[:,d['opacity']],device=dev))
C0=0.28209479177387814
col=torch.tensor(np.clip(clean[:,[d['f_dc_0'],d['f_dc_1'],d['f_dc_2']]]*C0+0.5,0,1),device=dev)
def render(vm,K,W,H):
    o,_,_=rasterization(means,quats/quats.norm(dim=-1,keepdim=True),sc,op,col,
        torch.tensor(vm[None].astype(np.float32),device=dev),torch.tensor(K[None].astype(np.float32),device=dev),W,H)
    return (o[0].clamp(0,1).cpu().numpy()*255).astype(np.uint8)
files=sorted(glob.glob(os.path.join(SRC,'frame_*.npz')))
dd=np.load(files[0]); extr=dd['extrinsic']; intr=dd['intrinsic']; H,W=dd['depth'].shape[:2]
vm0=np.eye(4); vm0[:3,:4]=extr
imageio.imwrite('/home/bluekitty/renders/clean_train0.png',render(vm0,intr,W,H))
# off-axis: pull back and up from scene center, look at center
cen=np.median(xyz,0)
def look_at(eye,tgt,up=np.array([0.,-1,0])):
    fwd=tgt-eye; fwd/=np.linalg.norm(fwd)
    r=np.cross(up,fwd); r/=np.linalg.norm(r); u=np.cross(fwd,r)
    R=np.stack([r,u,fwd],0); t=-R@eye; vm=np.eye(4); vm[:3,:3]=R; vm[:3,3]=t; return vm
eye=cen+np.array([ext*0.35,-ext*0.25,-ext*0.5])
imageio.imwrite('/home/bluekitty/renders/clean_offaxis.png',render(look_at(eye,cen),intr,W,H))
print("DONE",flush=True)
