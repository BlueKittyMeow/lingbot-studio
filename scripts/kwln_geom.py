import numpy as np, glob, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,'/home/bluekitty/lingbot-map')
from lingbot_map.utils.geometry import depth_to_world_coords_points

def rot_to(a,b):
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b); v=np.cross(a,b); c=np.dot(a,b)
    if np.linalg.norm(v)<1e-8: return np.eye(3) if c>0 else -np.eye(3)
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+vx+vx@vx*(1/(1+c))

def load(scene_dir, cap=500000, per=14000):
    files=[f for f in sorted(glob.glob(os.path.join(scene_dir,'frame_*.npz'))) if os.path.getsize(f)>0]
    ups=[]
    for f in files:
        d=np.load(f); R=d['extrinsic'][:3,:3]; u=-R[1,:]; ups.append(u/np.linalg.norm(u))
    up=np.mean(ups,0); up/=np.linalg.norm(up)
    Rlevel=rot_to(up,np.array([0,1,0])); Rx=np.array([[1,0,0],[0,-1,0],[0,0,-1]],float); Rf=(Rx@Rlevel)
    P=[];C=[];ts=[]
    for f in files:
        d=np.load(f); depth=d['depth'].squeeze(-1)
        world,_,mask=depth_to_world_coords_points(depth,d['extrinsic'],d['intrinsic'])
        rgb=np.transpose(d['images'],(1,2,0)); m=mask & np.isfinite(world).all(-1)
        pw=world[m]; pc=rgb[m]
        if len(pw)>per:
            idx=np.random.RandomState(0).choice(len(pw),per,replace=False); pw,pc=pw[idx],pc[idx]
        P.append(pw); C.append(pc)
        R=d['extrinsic'][:3,:3]; t=d['extrinsic'][:3,3]; ts.append(-R.T@t)
    pts=np.concatenate(P)@Rf.T; cols=np.clip(np.concatenate(C),0,1); traj=np.array(ts)@Rf.T
    med=np.median(pts,0); pts-=med; traj-=med
    if len(pts)>cap:
        idx=np.random.RandomState(1).choice(len(pts),cap,replace=False); pts,cols=pts[idx],cols[idx]
    return pts, cols, traj, len(files)

pts,cols,traj,nf=load('/home/bluekitty/renders/kowloon_full/kowloon_full')
fig=plt.figure(figsize=(20,11))
fig.suptitle(f"KOWLOON — FULL 84s WALK  ·  {nf} frames  ·  {len(pts):,} pts  ·  gold line = camera path", fontsize=13)
def sc2(ax,i,j):
    ax.scatter(pts[:,i],pts[:,j],s=0.4,c=cols,alpha=0.35,edgecolors='none',rasterized=True)
    ax.plot(traj[:,i],traj[:,j],'-',color='#c4941a',lw=1.4,alpha=0.9)
    ax.set_aspect('equal'); ax.grid(alpha=0.25,lw=0.4)
ax=fig.add_subplot(2,3,1); ax.set_title("TOP-DOWN  X-Z"); sc2(ax,0,2)
ax=fig.add_subplot(2,3,2); ax.set_title("FRONT  X-Y"); sc2(ax,0,1)
ax=fig.add_subplot(2,3,3); ax.set_title("SIDE  Z-Y"); sc2(ax,2,1)
for k,(el,az) in enumerate([(25,45),(60,20),(15,110)]):
    ax=fig.add_subplot(2,3,4+k,projection='3d'); ax.set_title(f"3D elev{el} azim{az}")
    ax.scatter(pts[:,0],pts[:,2],pts[:,1],s=0.3,c=cols,alpha=0.25,edgecolors='none',rasterized=True)
    ax.plot(traj[:,0],traj[:,2],traj[:,1],'-',color='#c4941a',lw=1.4,alpha=0.9)
    ax.view_init(elev=el,azim=az)
plt.tight_layout(rect=[0,0,1,0.96])
fig.savefig('/mnt/e/wsl_deploy/kwlnfull_geom.png',dpi=115); print("WROTE kwlnfull_geom.png"); print("DONE")
