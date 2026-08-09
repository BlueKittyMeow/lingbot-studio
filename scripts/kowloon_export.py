import numpy as np, glob, sys, os
sys.path.insert(0,'/home/bluekitty/lingbot-map')
from lingbot_map.utils.geometry import depth_to_world_coords_points
SRC='/home/bluekitty/renders/kowloon/kwc_walk'; OUT='/home/bluekitty/renders'; PER_FRAME=45000
files=[f for f in sorted(glob.glob(os.path.join(SRC,'frame_*.npz'))) if os.path.getsize(f)>0]
print(len(files),'valid frames',flush=True)
P=[]; C=[]
for f in files:
    try: d=np.load(f); depth=d['depth'].squeeze(-1)
    except Exception as e: print('skip',os.path.basename(f),e,flush=True); continue
    world,_,mask=depth_to_world_coords_points(depth,d['extrinsic'],d['intrinsic'])
    rgb=np.transpose(d['images'],(1,2,0)); conf=d['depth_conf']
    m=mask & np.isfinite(world).all(-1)
    pw=world[m].astype(np.float32); pc=rgb[m].astype(np.float32); pcf=conf[m]
    if len(pw)>PER_FRAME:                       # keep highest-confidence points per frame
        top=np.argpartition(pcf,-PER_FRAME)[-PER_FRAME:]; pw,pc=pw[top],pc[top]
    P.append(pw); C.append(pc)
pts=np.concatenate(P); col=np.concatenate(C); del P,C
print('base points',len(pts),flush=True)
med=np.median(pts,0)
def save(p,c,name):
    p=(p-med).astype(np.float32)
    v=np.empty(len(p),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    v['x'],v['y'],v['z']=p[:,0],p[:,1],p[:,2]; cc=np.clip(c*255,0,255).astype(np.uint8)
    v['red'],v['green'],v['blue']=cc[:,0],cc[:,1],cc[:,2]
    path=os.path.join(OUT,name)
    with open(path,'wb') as fh:
        fh.write(b"ply\nformat binary_little_endian 1.0\n"); fh.write(("element vertex %d\n"%len(v)).encode())
        fh.write(b"property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        v.tofile(fh)
    print('WROTE',name,len(p),round(os.path.getsize(path)/1e6,1),'MB',flush=True)
def voxel(p,c,target):
    lo=p.min(0); diag=float(np.linalg.norm(p.max(0)-lo)); vx=diag/1600.0
    for _ in range(16):
        k=np.floor((p-lo)/vx).astype(np.int64); h=k[:,0]*73856093 ^ k[:,1]*19349663 ^ k[:,2]*83492791
        _,idx=np.unique(h,return_index=True)
        if len(idx)<=target: break
        vx*=1.20
    return p[idx],c[idx]
save(pts,col,'kowloon_max_fused.ply')     # full base (~15M) as max
for tgt,name in [(6_000_000,'kowloon_6m_fused.ply'),(3_000_000,'kowloon_3m_fused.ply'),(1_500_000,'kowloon_light_fused.ply')]:
    pp,cc=voxel(pts,col,tgt); save(pp,cc,name); del pp,cc
print('TIERS_DONE',flush=True)
