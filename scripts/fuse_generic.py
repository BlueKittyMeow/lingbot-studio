import numpy as np, glob, sys, os
sys.path.insert(0,'/home/bluekitty/lingbot-map')
from lingbot_map.utils.geometry import depth_to_world_coords_points
SRC=sys.argv[1]; PREFIX=sys.argv[2]; OUT='/home/bluekitty/renders'; PER=45000
files=[f for f in sorted(glob.glob(os.path.join(SRC,'**','frame_*.npz'),recursive=True)) if os.path.getsize(f)>0]
print(len(files),'frames from',SRC,flush=True)
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
save(pts,col,PREFIX+'_6m_fused.ply' if len(pts)>6_000_000 else PREFIX+'_max_fused.ply')
for tgt,suf in [(6_000_000,'6m'),(3_000_000,'3m'),(1_500_000,'light')]:
    idx=perm[:min(tgt,len(pts))]; save(pts[idx],col[idx],PREFIX+'_'+suf+'_fused.ply')
print('DONE',PREFIX,flush=True)
