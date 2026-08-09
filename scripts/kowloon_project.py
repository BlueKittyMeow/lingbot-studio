import numpy as np, glob, os, sys, imageio
sys.path.insert(0,'/home/bluekitty/lingbot-map')
from lingbot_map.utils.geometry import depth_to_world_coords_points
SRC='/home/bluekitty/renders/kowloon/kwc_walk'
files=[f for f in sorted(glob.glob(SRC+'/frame_*.npz')) if os.path.getsize(f)>0]
P=[];C=[]
for f in files[::3]:
    try: d=np.load(f); depth=d['depth'].squeeze(-1)
    except Exception: continue
    w,_,m=depth_to_world_coords_points(depth,d['extrinsic'],d['intrinsic'])
    mm=m&np.isfinite(w).all(-1); P.append(w[mm][::2].astype(np.float32)); C.append(np.transpose(d['images'],(1,2,0))[mm][::2].astype(np.float32))
pts=np.concatenate(P); col=np.concatenate(C)
for idx in [len(files)//4, len(files)//2, 3*len(files)//4]:
    d=np.load(files[idx]); extr=d['extrinsic']; intr=d['intrinsic']; H,W=d['depth'].squeeze(-1).shape
    cam=pts@extr[:3,:3].T + extr[:3,3]; z=cam[:,2]; val=z>0.05
    uvw=(intr@cam[val].T).T; uv=uvw[:,:2]/uvw[:,2:3]
    u=np.round(uv[:,0]).astype(int); v=np.round(uv[:,1]).astype(int)
    inb=(u>=0)&(u<W)&(v>=0)&(v<H)
    zz=z[val][inb]; cc=(np.clip(col[val][inb],0,1)*255).astype(np.uint8); uu=u[inb]; vv=v[inb]
    img=np.zeros((H,W,3),np.uint8); order=np.argsort(-zz)
    img[vv[order],uu[order]]=cc[order]
    imageio.imwrite(f'/home/bluekitty/renders/kowloon_proj_{idx}.png',img)
    print('rendered',idx,flush=True)
print('DONE')
