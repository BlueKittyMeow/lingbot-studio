import numpy as np, glob, torch, os, sys, time, imageio
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
SRC='/home/bluekitty/renders/courthouse/courthouse'; OUT='/home/bluekitty/renders/courthouse_splat.ply'
ITERS=7000; INIT_N=400000; dev='cuda'; np.random.seed(0); torch.manual_seed(0)
def unproject(depth,extr,intr):
    H,W=depth.shape; fu,fv,cu,cv=intr[0,0],intr[1,1],intr[0,2],intr[1,2]
    u,v=np.meshgrid(np.arange(W),np.arange(H))
    cam=np.stack([(u-cu)*depth/fu,(v-cv)*depth/fv,depth],-1).astype(np.float32)
    return (cam-extr[:3,3])@extr[:3,:3]
files=sorted(glob.glob(os.path.join(SRC,'frame_*.npz'))); imgs=[];vms=[];Ks=[];P=[];Cc=[];CF=[]
for f in files:
    d=np.load(f); depth=d['depth'].squeeze(-1); extr=d['extrinsic']; intr=d['intrinsic']
    rgb=np.transpose(d['images'],(1,2,0)).astype(np.float32); imgs.append(rgb)
    vm=np.eye(4,dtype=np.float32); vm[:3,:4]=extr; vms.append(vm); Ks.append(intr.astype(np.float32))
    w=unproject(depth,extr,intr); m=(depth>1e-6)&np.isfinite(w).all(-1); P.append(w[m]); Cc.append(rgb[m]); CF.append(d['depth_conf'][m])
H,W=imgs[0].shape[:2]
imgs=torch.from_numpy(np.stack(imgs)).to(dev); vms=torch.from_numpy(np.stack(vms)).to(dev); Ks=torch.from_numpy(np.stack(Ks)).to(dev); F=imgs.shape[0]
pts=np.concatenate(P); col=np.concatenate(Cc); cf=np.concatenate(CF); keep=cf>=np.percentile(cf,25); pts,col=pts[keep],col[keep]
idx=np.random.choice(len(pts),min(INIT_N,len(pts)),replace=False); pts,col=pts[idx],col[idx]
N=len(pts); extent=float(np.linalg.norm(pts.max(0)-pts.min(0))); print(f"init {N} extent {extent:.2f}",flush=True)
c0=np.clip(col,1e-4,1-1e-4)
params=torch.nn.ParameterDict({
 'means':torch.nn.Parameter(torch.from_numpy(pts.copy()).float().to(dev)),
 'scales':torch.nn.Parameter(torch.full((N,3),float(np.log(0.004*extent)),device=dev)),
 'quats':torch.nn.Parameter(torch.tensor([1.,0,0,0],device=dev).repeat(N,1)),
 'opacities':torch.nn.Parameter(torch.full((N,),-2.2,device=dev)),
 'colors':torch.nn.Parameter(torch.log(torch.from_numpy(c0/(1-c0)).float().to(dev)))}).to(dev)
lr={'means':1.6e-4*extent,'scales':5e-3,'quats':1e-3,'opacities':5e-2,'colors':1e-2}
opts={k:torch.optim.Adam([{'params':[params[k]],'lr':lr[k]}],eps=1e-15) for k in params}
strat=DefaultStrategy(grow_grad2d=0.00015, prune_opa=0.02, refine_start_iter=300, refine_stop_iter=5000, reset_every=3000, refine_every=100, verbose=True)
state=strat.initialize_state(scene_scale=extent); strat.check_sanity(params,opts)
smax=0.05*extent  # scale-reg threshold
t0=time.time()
for step in range(ITERS):
    i=np.random.randint(F)
    sc=torch.exp(params['scales'])
    out,_,info=rasterization(params['means'],params['quats'],sc,torch.sigmoid(params['opacities']),
        torch.sigmoid(params['colors']),vms[i:i+1],Ks[i:i+1],W,H,packed=False)
    info['means2d'].retain_grad(); strat.step_pre_backward(params,opts,state,step,info)
    l1=(out[0]-imgs[i]).abs().mean()
    sreg=torch.relu(sc.max(1).values - smax).mean()
    loss=l1 + 5.0*sreg
    loss.backward()
    for o in opts.values(): o.step(); o.zero_grad(set_to_none=True)
    strat.step_post_backward(params,opts,state,step,info,packed=False)
    if step%500==0 or step==ITERS-1: print(f"step {step} loss {loss.item():.4f} N {params['means'].shape[0]} ({time.time()-t0:.0f}s)",flush=True)
N=params['means'].shape[0]; print("TRAIN_DONE N",N,flush=True)
C0=0.28209479177387814
with torch.no_grad():
    m=params['means'].cpu().numpy(); q=(params['quats']/params['quats'].norm(dim=-1,keepdim=True)).cpu().numpy()
    ls=params['scales'].cpu().numpy(); op=params['opacities'].cpu().numpy().reshape(-1,1); fdc=(torch.sigmoid(params['colors']).cpu().numpy()-0.5)/C0
    data=np.concatenate([m,np.zeros((N,3),np.float32),fdc,op,ls,q],1).astype(np.float32)
nm=['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
with open(OUT,'wb') as fh:
    fh.write(b"ply\nformat binary_little_endian 1.0\n"); fh.write(("element vertex %d\n"%N).encode())
    for x in nm: fh.write(("property float %s\n"%x).encode())
    fh.write(b"end_header\n"); data.tofile(fh)
print("WROTE",OUT,N,flush=True)
# off-axis render to verify novel view
means=params['means'].detach(); quats=params['quats'].detach(); quats=quats/quats.norm(dim=-1,keepdim=True)
sc=torch.exp(params['scales']).detach(); op=torch.sigmoid(params['opacities']).detach(); cl=torch.sigmoid(params['colors']).detach()
cen=means.median(0).values.cpu().numpy()
def look_at(eye,tgt,up=np.array([0.,-1,0])):
    fwd=tgt-eye; fwd/=np.linalg.norm(fwd); r=np.cross(up,fwd); r/=np.linalg.norm(r); u=np.cross(fwd,r)
    R=np.stack([r,u,fwd],0); t=-R@eye; vm=np.eye(4); vm[:3,:3]=R; vm[:3,3]=t; return vm.astype(np.float32)
eye=cen+np.array([extent*0.35,-extent*0.25,-extent*0.5])
vm=torch.tensor(look_at(eye,cen)[None],device=dev); K=Ks[0:1]
o,_,_=rasterization(means,quats,sc,op,cl,vm,K,W,H)
imageio.imwrite('/home/bluekitty/renders/v5_offaxis.png',(o[0].clamp(0,1).cpu().numpy()*255).astype(np.uint8))
print("OFFAXIS_DONE",flush=True)
