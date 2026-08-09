import numpy as np, glob, sys, os, cv2
sys.path.insert(0,'/home/bluekitty/lingbot-map')
from lingbot_map.utils.geometry import depth_to_world_coords_points
SRC='/home/bluekitty/renders/courthouse_sparse/courthouse'
SKIN='/home/bluekitty/skins/waterlilies'
OUT='/home/bluekitty/renders'; PRE='courthouse_waterlilies'; PER=45000
files=[f for f in sorted(glob.glob(os.path.join(SRC,'frame_*.npz'))) if os.path.getsize(f)>0]
skins=sorted(glob.glob(os.path.join(SKIN,'*.png')))
print(len(files),'geom frames',len(skins),'skin frames',flush=True)
P=[];C=[]
for i,f in enumerate(files):
    d=np.load(f); depth=d['depth'].squeeze(-1)
    world,_,mask=depth_to_world_coords_points(depth,d['extrinsic'],d['intrinsic'])
    sk=cv2.cvtColor(cv2.imread(skins[i%len(skins)]),cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    if sk.shape[:2]!=depth.shape: sk=cv2.resize(sk,(depth.shape[1],depth.shape[0]))
    conf=d['depth_conf']; m=mask & np.isfinite(world).all(-1)
    pw=world[m].astype(np.float32); pc=sk[m].astype(np.float32); pcf=conf[m]
    if len(pw)>PER:
        top=np.argpartition(pcf,-PER)[-PER:]; pw,pc=pw[top],pc[top]
    P.append(pw); C.append(pc)
pts=np.concatenate(P); col=np.concatenate(C); med=np.median(pts,0)
print('base',len(pts),flush=True)
def save(p,c,name):
    p=(p-med).astype(np.float32)
    v=np.empty(len(p),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    v['x'],v['y'],v['z']=p[:,0],p[:,1],p[:,2]; cc=np.clip(c*255,0,255).astype(np.uint8)
    v['red'],v['green'],v['blue']=cc[:,0],cc[:,1],cc[:,2]
    path=os.path.join(OUT,name)
    with open(path,'wb') as fh:
        fh.write(b"ply\nformat binary_little_endian 1.0\n"); fh.write(("element vertex %d\n"%len(v)).encode())
        fh.write(b"property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"); v.tofile(fh)
    print('WROTE',name,len(p),flush=True)
perm=np.random.RandomState(0).permutation(len(pts))
for tgt,suf in [(6_000_000,'6m'),(3_000_000,'3m'),(1_500_000,'light')]:
    idx=perm[:min(tgt,len(pts))]; save(pts[idx],col[idx],PRE+'_'+suf+'_fused.ply')
print('DONE',flush=True)
