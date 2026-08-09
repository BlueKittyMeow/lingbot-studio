import numpy as np, glob, sys, os
sys.path.insert(0,'/home/bluekitty/lingbot-map')
from lingbot_map.utils.geometry import depth_to_world_coords_points
SRC='/home/bluekitty/renders/courthouse/courthouse'; OUT='/home/bluekitty/renders'; CONF_PCTL=25
files=sorted(glob.glob(os.path.join(SRC,'frame_*.npz')))
P,C,CF=[],[],[]
for f in files:
    d=np.load(f); depth=d['depth'].squeeze(-1)
    world,_,mask=depth_to_world_coords_points(depth,d['extrinsic'],d['intrinsic'])
    rgb=np.transpose(d['images'],(1,2,0)); conf=d['depth_conf']
    m=mask & np.isfinite(world).all(-1)
    P.append(world[m]); C.append(rgb[m]); CF.append(conf[m])
pts=np.concatenate(P).astype(np.float32); col=np.concatenate(C); cf=np.concatenate(CF)
thr=np.percentile(cf,CONF_PCTL); keep=cf>=thr; pts,col=pts[keep],col[keep]
med=np.median(pts,0); print('confident points:',len(pts),flush=True)
def save(p,c,name):
    p=(p-med).astype(np.float32)
    v=np.empty(len(p),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    v['x'],v['y'],v['z']=p[:,0],p[:,1],p[:,2]
    cc=np.clip(c*255,0,255).astype(np.uint8); v['red'],v['green'],v['blue']=cc[:,0],cc[:,1],cc[:,2]
    path=os.path.join(OUT,name)
    with open(path,'wb') as fh:
        fh.write(b"ply\nformat binary_little_endian 1.0\n"); fh.write(("element vertex %d\n"%len(v)).encode())
        fh.write(b"property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        v.tofile(fh)
    print('WROTE',name,len(p),'pts',round(os.path.getsize(path)/1e6,1),'MB',flush=True)
def voxel(p,c,target):
    lo=p.min(0); diag=float(np.linalg.norm(p.max(0)-lo)); vx=diag/1600.0
    for _ in range(14):
        k=np.floor((p-lo)/vx).astype(np.int64); h=k[:,0]*73856093 ^ k[:,1]*19349663 ^ k[:,2]*83492791
        _,idx=np.unique(h,return_index=True)
        if len(idx)<=target: break
        vx*=1.20
    return p[idx],c[idx]
save(pts,col,'courthouse_max_fused.ply')
for tgt,name in [(6_000_000,'courthouse_6m_fused.ply'),(3_000_000,'courthouse_3m_fused.ply'),(1_500_000,'courthouse_light_fused.ply')]:
    pp,cc=voxel(pts,col,tgt); save(pp,cc,name)
print('TIERS_DONE',flush=True)
