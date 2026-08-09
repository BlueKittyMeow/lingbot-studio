import numpy as np, glob, torch, os, sys, time
import torch.nn.functional as Fn
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
SRC='/home/bluekitty/renders/courthouse/courthouse'; OUT='/home/bluekitty/renders/courthouse_splat.ply'
ITERS=int(sys.argv[1]) if len(sys.argv)>1 else 20000
INIT_N=int(sys.argv[2]) if len(sys.argv)>2 else 500000
dev='cuda'; np.random.seed(0); torch.manual_seed(0)
def unproject(depth,extr,intr):
    H,W=depth.shape; fu,fv,cu,cv=intr[0,0],intr[1,1],intr[0,2],intr[1,2]
    u,v=np.meshgrid(np.arange(W),np.arange(H))
    cam=np.stack([(u-cu)*depth/fu,(v-cv)*depth/fv,depth],-1).astype(np.float32)
    return (cam-extr[:3,3])@extr[:3,:3]
files=sorted(glob.glob(os.path.join(SRC,'frame_*.npz')))
imgs=[];vms=[];Ks=[];P=[];Cc=[];CF=[]
for f in files:
    d=np.load(f); depth=d['depth'].squeeze(-1); extr=d['extrinsic']; intr=d['intrinsic']
    rgb=np.transpose(d['images'],(1,2,0)).astype(np.float32); imgs.append(rgb)
    vm=np.eye(4,dtype=np.float32); vm[:3,:4]=extr; vms.append(vm); Ks.append(intr.astype(np.float32))
    w=unproject(depth,extr,intr); m=(depth>1e-6)&np.isfinite(w).all(-1)
    P.append(w[m]); Cc.append(rgb[m]); CF.append(d['depth_conf'][m])
H,W=imgs[0].shape[:2]
imgs=torch.from_numpy(np.stack(imgs)).to(dev); vms=torch.from_numpy(np.stack(vms)).to(dev); Ks=torch.from_numpy(np.stack(Ks)).to(dev); F=imgs.shape[0]
pts=np.concatenate(P); col=np.concatenate(Cc); cf=np.concatenate(CF)
keep=cf>=np.percentile(cf,25); pts,col=pts[keep],col[keep]
if len(pts)>INIT_N:
    idx=np.random.choice(len(pts),INIT_N,replace=False); pts,col=pts[idx],col[idx]
N=len(pts); extent=float(np.linalg.norm(pts.max(0)-pts.min(0)))
print(f"cams {F} img {H}x{W} init {N} extent {extent:.2f}",flush=True)
c0=np.clip(col,1e-4,1-1e-4)
params=torch.nn.ParameterDict({
 'means':torch.nn.Parameter(torch.from_numpy(pts.copy()).float().to(dev)),
 'scales':torch.nn.Parameter(torch.full((N,3),float(np.log(0.0025*extent)),device=dev)),
 'quats':torch.nn.Parameter(torch.tensor([1.,0,0,0],device=dev).repeat(N,1)),
 'opacities':torch.nn.Parameter(torch.full((N,),-2.2,device=dev)),
 'colors':torch.nn.Parameter(torch.log(torch.from_numpy(c0/(1-c0)).float().to(dev))),
}).to(dev)
lrs={'means':1.6e-4*extent,'scales':5e-3,'quats':1e-3,'opacities':5e-2,'colors':1e-2}
optimizers={k:torch.optim.Adam([{'params':[params[k]],'lr':lrs[k]}],eps=1e-15) for k in params}
strat=DefaultStrategy(refine_start_iter=500, refine_stop_iter=int(ITERS*0.7), reset_every=3000, verbose=True)
state=strat.initialize_state(scene_scale=extent); strat.check_sanity(params,optimizers)
s=11; sig=1.5; cc=torch.arange(s,device=dev)-s//2; g=torch.exp(-(cc.float()**2)/(2*sig**2)); g/=g.sum()
WIN=(g[:,None]@g[None,:])[None,None].repeat(3,1,1,1)
def ssim(x,y):
    mx=Fn.conv2d(x,WIN,padding=5,groups=3); my=Fn.conv2d(y,WIN,padding=5,groups=3)
    mx2=mx*mx; my2=my*my; mxy=mx*my
    sx=Fn.conv2d(x*x,WIN,padding=5,groups=3)-mx2; sy=Fn.conv2d(y*y,WIN,padding=5,groups=3)-my2; sxy=Fn.conv2d(x*y,WIN,padding=5,groups=3)-mxy
    C1=.01**2; C2=.03**2
    return (((2*mxy+C1)*(2*sxy+C2))/((mx2+my2+C1)*(sx+sy+C2))).mean()
t0=time.time()
for step in range(ITERS):
    i=np.random.randint(F)
    out,_,info=rasterization(params['means'], params['quats'], torch.exp(params['scales']),
        torch.sigmoid(params['opacities']), torch.sigmoid(params['colors']), vms[i:i+1], Ks[i:i+1], W, H, packed=False)
    info['means2d'].retain_grad()
    strat.step_pre_backward(params,optimizers,state,step,info)
    pred=out[0]; gt=imgs[i]
    loss=0.8*(pred-gt).abs().mean()+0.2*(1-ssim(pred.permute(2,0,1)[None], gt.permute(2,0,1)[None]))
    loss.backward()
    for o in optimizers.values(): o.step(); o.zero_grad(set_to_none=True)
    strat.step_post_backward(params,optimizers,state,step,info,packed=False)
    if step%500==0 or step==ITERS-1: print(f"step {step:5d} loss {loss.item():.4f} N {params['means'].shape[0]} ({time.time()-t0:.0f}s)",flush=True)
print("TRAIN_DONE N",params['means'].shape[0],flush=True)
N=params['means'].shape[0]; C0=0.28209479177387814
with torch.no_grad():
    m=params['means'].cpu().numpy(); q=(params['quats']/params['quats'].norm(dim=-1,keepdim=True)).cpu().numpy()
    ls=params['scales'].cpu().numpy(); op=params['opacities'].cpu().numpy().reshape(-1,1)
    fdc=(torch.sigmoid(params['colors']).cpu().numpy()-0.5)/C0
data=np.concatenate([m,np.zeros((N,3),np.float32),fdc,op,ls,q],1).astype(np.float32)
nm=['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
with open(OUT,'wb') as fh:
    fh.write(b"ply\nformat binary_little_endian 1.0\n"); fh.write(("element vertex %d\n"%N).encode())
    for x in nm: fh.write(("property float %s\n"%x).encode())
    fh.write(b"end_header\n"); data.tofile(fh)
print("WROTE",OUT,os.path.getsize(OUT)//1024,"KB",N,"gaussians",flush=True)
