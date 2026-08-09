import numpy as np, glob, torch, os, sys
os.chdir('/home/bluekitty/HunyuanWorld-Mirror'); sys.path.insert(0,'/home/bluekitty/HunyuanWorld-Mirror')
from src.models.models.worldmirror import WorldMirror
from scipy.spatial.transform import Rotation as Rot
SRC='/home/bluekitty/renders/courthouse_sparse/courthouse'
nf=sorted(glob.glob(SRC+'/frame_*.npz')); idx=list(range(0,len(nf),max(1,len(nf)//16)))[:16]
im=[]; lb_c2w=[]; lb_dep=[]
for i in idx:
    d=np.load(nf[i]); im.append(d['images'].astype('float32'))
    e=np.eye(4,dtype='float64'); e[:3,:4]=d['extrinsic']; lb_c2w.append(np.linalg.inv(e)); lb_dep.append(d['depth'].squeeze(-1).astype('float64'))
lb_c2w=np.stack(lb_c2w); lb_dep=np.stack(lb_dep)
model=WorldMirror.from_pretrained('tencent/HunyuanWorld-Mirror').to('cuda').eval()
views={'img':torch.from_numpy(np.stack(im)[None]).to('cuda')}
with torch.no_grad(), torch.amp.autocast('cuda',dtype=torch.bfloat16):
    p=model(views=views, cond_flags=[0,0,0])
sp=p['splats']; wm_c2w=p['camera_poses'][0].float().cpu().numpy().astype('float64')
wm_dep=p['depth'][0].float().cpu().numpy().astype('float64')   # [S,H,W] (or [S,H,W,1])
if wm_dep.ndim==4: wm_dep=wm_dep[...,0]
print('list len', len(sp['means']), 'elem0', tuple(sp['means'][0].shape), 'wm_dep', wm_dep.shape)
M=[];Q=[];Sc=[];Cl=[];Op=[]
for i in range(len(sp['means'])):
    m=sp['means'][i].reshape(-1,3).float().cpu().numpy().astype('float64')
    q=sp['quats'][i].reshape(-1,4).float().cpu().numpy().astype('float64')   # assume wxyz
    sc=sp['scales'][i].reshape(-1,3).float().cpu().numpy().astype('float64')
    col=sp['sh'][i].reshape(-1,3).float().cpu().numpy().astype('float64')
    op=sp['opacities'][i].reshape(-1).float().cpu().numpy().astype('float64')
    Rwm=wm_c2w[i,:3,:3]; twm=wm_c2w[i,:3,3]; Rlb=lb_c2w[i,:3,:3]; tlb=lb_c2w[i,:3,3]
    mk=(lb_dep[i]>1e-3)&(wm_dep[i]>1e-3)&np.isfinite(wm_dep[i]); s=float(np.median(lb_dep[i][mk]/wm_dep[i][mk]))
    Xcam=(m-twm)@Rwm                      # WM world -> WM cam
    mlb=(Xcam*s)@Rlb.T + tlb              # -> lingbot world (scaled)
    Rrel=Rlb@Rwm.T
    qx=q[:,[1,2,3,0]]; Rg=Rot.from_quat(qx).as_matrix(); Rgn=Rrel[None]@Rg
    qn=Rot.from_matrix(Rgn).as_quat(); qn=qn[:,[3,0,1,2]]
    M.append(mlb); Q.append(qn); Sc.append(sc*s); Cl.append(col); Op.append(op)
M=np.concatenate(M);Q=np.concatenate(Q);Sc=np.concatenate(Sc);Cl=np.concatenate(Cl);Op=np.concatenate(Op)
print('total gaussians',len(M))
# write splat PLY (same 3DGS layout as our viewer expects; DC color already in sh)
C0=0.28209479177387814
fdc=(Cl-0.5)/C0 if Cl.max()<=1.001 else Cl   # if sh already raw, keep
N=len(M); data=np.concatenate([M,np.zeros((N,3)),fdc,Op.reshape(-1,1),np.log(np.clip(Sc,1e-9,None)),Q],1).astype(np.float32)
# NOTE: our trained-gsplat writer stored log(scale) & raw opacity+sigmoid; WM stores activated scale/opacity.
# For a first look at CLOSURE, positions are what matter. Write means-only cloud too:
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
lo,hi=np.percentile(M,[2,98],0); k=np.all((M>=lo)&(M<=hi),1); P=M[k][::4]
fig,ax=plt.subplots(1,1,figsize=(9,9)); ax.scatter(P[:,0],P[:,2],s=0.4,alpha=0.3,c='purple',linewidths=0); ax.set_aspect('equal')
ax.set_title('CLOSED WM SPLAT (per-view Sim3 -> lingbot frame): gaussian means top-down')
plt.savefig('/mnt/e/wsl_deploy/closedsplat_td.png',dpi=90)
np.savez('/home/bluekitty/wm_closed_splat_raw.npz', M=M,Q=Q,Sc=Sc,Cl=Cl,Op=Op)
print('SAVED closure render + raw npz')
