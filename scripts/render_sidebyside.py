import numpy as np, glob, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,'/home/bluekitty/lingbot-map')
from lingbot_map.utils.geometry import depth_to_world_coords_points

def rot_to(a,b):
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b); v=np.cross(a,b); c=np.dot(a,b)
    if np.linalg.norm(v)<1e-8: return np.eye(3) if c>0 else -np.eye(3)
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+vx+vx@vx*(1/(1+c))

def load(scene_dir, cap=450000, per=16000):
    files=[f for f in sorted(glob.glob(os.path.join(scene_dir,'frame_*.npz'))) if os.path.getsize(f)>0]
    ups=[]
    for f in files:
        d=np.load(f); R=d['extrinsic'][:3,:3]; u=-R[1,:]; ups.append(u/np.linalg.norm(u))
    up=np.mean(ups,0); up/=np.linalg.norm(up)
    Rlevel=rot_to(up,np.array([0,1,0])); Rx=np.array([[1,0,0],[0,-1,0],[0,0,-1]],float); Rf=(Rx@Rlevel)
    P=[];C=[]
    for f in files:
        d=np.load(f); depth=d['depth'].squeeze(-1)
        world,_,mask=depth_to_world_coords_points(depth,d['extrinsic'],d['intrinsic'])
        rgb=np.transpose(d['images'],(1,2,0)); m=mask & np.isfinite(world).all(-1)
        pw=world[m]; pc=rgb[m]
        if len(pw)>per:
            idx=np.random.RandomState(0).choice(len(pw),per,replace=False); pw,pc=pw[idx],pc[idx]
        P.append(pw); C.append(pc)
    pts=np.concatenate(P)@Rf.T; cols=np.clip(np.concatenate(C),0,1)
    med=np.median(pts,0); pts-=med
    if len(pts)>cap:
        idx=np.random.RandomState(1).choice(len(pts),cap,replace=False); pts,cols=pts[idx],cols[idx]
    return pts, cols, len(files)

A=load('/home/bluekitty/renders/lowres648/catacombs2_crop')
B=load('/home/bluekitty/renders/catacombs2q/catacombs2_crop')
names=['CATACOMBS2-MAX  (sw64/nsf8 @448, full defaults)','CATACOMBS2  (sw32/nsf4 @518, shipped)']
scenes=[A,B]
# shared limits per view across both scenes
def lim(axis_pairs):
    allv=np.concatenate([np.stack([s[0][:,i] for i in axis_pairs],1) for s in scenes])
    lo=np.percentile(allv,0.3,0); hi=np.percentile(allv,99.7,0); pad=(hi-lo)*0.05
    return lo-pad, hi+pad
views=[("TOP-DOWN  X-Z",(0,2),False),("FRONT  X-Y",(0,1),False),("SIDE  Z-Y",(2,1),False),("3D  elev25 azim45",(0,2,1),True)]
lims=[lim(v[1][:2]) for v in views]

fig=plt.figure(figsize=(15,20))
fig.suptitle("Topography compare — MAX-KNOBS @448  vs  SHIPPED @518   ·   same ossuary-wall footage", fontsize=15, y=0.995)
for col,(pts,cols_,nf) in enumerate(scenes):
    for row,(vt,ax_idx,is3d) in enumerate(views):
        ax=fig.add_subplot(len(views),2,row*2+col+1, projection='3d' if is3d else None)
        if col==0: ax.set_ylabel(vt, fontsize=11, labelpad=10)
        if row==0: ax.set_title(f"{names[col]}\n{nf} frames · {len(pts):,} pts", fontsize=10)
        if is3d:
            ax.scatter(pts[:,0],pts[:,2],pts[:,1],s=0.3,c=cols_,alpha=0.28,edgecolors='none',rasterized=True)
            ax.view_init(elev=25,azim=45)
        else:
            i,j=ax_idx
            ax.scatter(pts[:,i],pts[:,j],s=0.4,c=cols_,alpha=0.35,edgecolors='none',rasterized=True)
            ax.set_aspect('equal'); ax.grid(alpha=0.25,lw=0.4)
            (lo,hi)=lims[row]; ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1])
plt.tight_layout(rect=[0,0,1,0.98])
out='/mnt/e/wsl_deploy/topography_sidebyside.png'
fig.savefig(out,dpi=115); print("WROTE",out,flush=True); print("DONE")
