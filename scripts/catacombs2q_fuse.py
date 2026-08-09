import numpy as np, glob, sys, os
sys.path.insert(0,'/home/bluekitty/lingbot-map')
from lingbot_map.utils.geometry import depth_to_world_coords_points
SRC='/home/bluekitty/renders/catacombs2q/catacombs2_crop'; OUT='/home/bluekitty/renders'; PER=45000
files=[f for f in sorted(glob.glob(os.path.join(SRC,'frame_*.npz'))) if os.path.getsize(f)>0]
print(len(files),'frames',flush=True)
P=[];C=[]
for f in files:
    try: d=np.load(f); depth=d['depth'].squeeze(-1)
    except: continue
    world,_,mask=depth_to_world_coords_points(depth,d['extrinsic'],d['intrinsic'])
    rgb=np.transpose(d['images'],(1,2,0)); conf=d['depth_conf']
    m=mask & np.isfinite(world).all(-1)
    pw=world[m].astype(np.float32); pc=rgb[m].astype(np.float32); pcf=conf[m]
    if len(pw)>PER:
        top=np.argpartition(pcf,-PER)[-PER:]; pw,pc=pw[top],pc[top]
    P.append(pw); C.append(pc)
pts=np.concatenate(P); col=np.concatenate(C); del P,C
med=np.median(pts,0); print('base',len(pts),flush=True)
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
perm=np.random.RandomState(0).permutation(len(pts))
save(pts,col,'catacombs2q_max_fused.ply')
for tgt,name in [(6_000_000,'catacombs2q_6m_fused.ply'),(3_000_000,'catacombs2q_3m_fused.ply'),(1_500_000,'catacombs2q_light_fused.ply')]:
    idx=perm[:min(tgt,len(pts))]; save(pts[idx],col[idx],name)
print('DONE',flush=True)
