import numpy as np, glob, torch, os, sys, time
from gsplat import rasterization
from PIL import Image
SRC='/home/bluekitty/renders/courthouse_sparse/courthouse'
MASKS='/home/bluekitty/renders/courthouse_sky143'
OUT='/home/bluekitty/renders/courthouse_sky_splat.ply'
ITERS=int(sys.argv[1]) if len(sys.argv)>1 else 6000
INIT_N=int(sys.argv[2]) if len(sys.argv)>2 else 1500000
dev='cuda'; np.random.seed(0); torch.manual_seed(0)
def unproject(depth,extr,intr):
    H,W=depth.shape; fu,fv,cu,cv=intr[0,0],intr[1,1],intr[0,2],intr[1,2]
    u,v=np.meshgrid(np.arange(W),np.arange(H))
    cam=np.stack([(u-cu)*depth/fu,(v-cv)*depth/fv,depth],-1).astype(np.float32)
    return (cam-extr[:3,3])@extr[:3,:3]
files=sorted(glob.glob(os.path.join(SRC,'frame_*.npz')))
imgs=[];vms=[];Ks=[];P=[];Cc=[];CF=[];Msk=[]
for i,f in enumerate(files):
    d=np.load(f); depth=d['depth'].squeeze(-1); extr=d['extrinsic']; intr=d['intrinsic']
    rgb=np.transpose(d['images'],(1,2,0)).astype(np.float32); imgs.append(rgb)
    mk=np.array(Image.open(os.path.join(MASKS,f'frame_{i:04d}.png')).convert('L'))
    keep=mk>127                                   # True = non-sky (train on it)
    Msk.append(keep.astype(np.float32))
    vm=np.eye(4,dtype=np.float32); vm[:3,:4]=extr; vms.append(vm); Ks.append(intr.astype(np.float32))
    w=unproject(depth,extr,intr); m=(depth>1e-6)&np.isfinite(w).all(-1)&keep
    P.append(w[m]); Cc.append(rgb[m]); CF.append(d['depth_conf'][m])
H,W=imgs[0].shape[:2]
imgs=torch.from_numpy(np.stack(imgs)).to(dev); vms=torch.from_numpy(np.stack(vms)).to(dev); Ks=torch.from_numpy(np.stack(Ks)).to(dev)
M=torch.from_numpy(np.stack(Msk)).to(dev)
F=imgs.shape[0]
pts=np.concatenate(P); col=np.concatenate(Cc); cf=np.concatenate(CF)
keep=cf>=np.percentile(cf,25); pts,col=pts[keep],col[keep]
if len(pts)>INIT_N:
    idx=np.random.choice(len(pts),INIT_N,replace=False); pts,col=pts[idx],col[idx]
N=len(pts); print(f"cams {F}  img {H}x{W}  init gaussians {N}  (sky excluded)",flush=True)
extent=float(np.linalg.norm(pts.max(0)-pts.min(0)))
means=torch.nn.Parameter(torch.from_numpy(pts.copy()).float().to(dev))
quats=torch.nn.Parameter(torch.tensor([1.,0,0,0],device=dev).repeat(N,1))
log_scales=torch.nn.Parameter(torch.full((N,3),float(np.log(0.002*extent)),device=dev))
opac_raw=torch.nn.Parameter(torch.full((N,),-2.2,device=dev))
c0=np.clip(col,1e-4,1-1e-4); color_raw=torch.nn.Parameter(torch.log(torch.from_numpy(c0/(1-c0)).float().to(dev)))
opt=torch.optim.Adam([{'params':[means],'lr':1.6e-4*extent},{'params':[log_scales],'lr':5e-3},
    {'params':[quats],'lr':1e-3},{'params':[opac_raw],'lr':5e-2},{'params':[color_raw],'lr':1e-2}])
def render(i):
    out,_,_=rasterization(means, quats/quats.norm(dim=-1,keepdim=True), torch.exp(log_scales),
        torch.sigmoid(opac_raw), torch.sigmoid(color_raw), vms[i:i+1], Ks[i:i+1], W, H)
    return out[0]
t0=time.time()
for it in range(ITERS):
    i=np.random.randint(F); mi=M[i]
    diff=torch.abs(render(i)-imgs[i])*mi[...,None]
    loss=diff.sum()/(mi.sum()*3+1)
    opt.zero_grad(); loss.backward(); opt.step()
    if it%500==0 or it==ITERS-1: print(f"it {it:5d}  loss {loss.item():.4f}  ({time.time()-t0:.0f}s)",flush=True)
print("TRAIN_DONE",flush=True)
C0=0.28209479177387814
with torch.no_grad():
    m=means.cpu().numpy(); q=(quats/quats.norm(dim=-1,keepdim=True)).cpu().numpy(); ls=log_scales.cpu().numpy()
    op=opac_raw.cpu().numpy().reshape(-1,1); fdc=(torch.sigmoid(color_raw).cpu().numpy()-0.5)/C0
data=np.concatenate([m,np.zeros((N,3),np.float32),fdc,op,ls,q],1).astype(np.float32)
names=['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
with open(OUT,'wb') as fh:
    fh.write(b"ply\nformat binary_little_endian 1.0\n"); fh.write(("element vertex %d\n"%N).encode())
    for nm in names: fh.write(("property float %s\n"%nm).encode())
    fh.write(b"end_header\n"); data.tofile(fh)
print("WROTE",OUT,os.path.getsize(OUT)//1024,"KB",N,"gaussians",flush=True)
