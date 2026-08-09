import numpy as np, os
OUT='/home/bluekitty/renders'
dt=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')]
def load(path):
    with open(path,'rb') as f:
        hdr=b''
        while b'end_header\n' not in hdr: hdr+=f.read(1)
        return np.fromfile(f,dtype=dt)
def save(v,name):
    path=os.path.join(OUT,name)
    with open(path,'wb') as fh:
        fh.write(b"ply\nformat binary_little_endian 1.0\n"); fh.write(("element vertex %d\n"%len(v)).encode())
        fh.write(b"property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        v.tofile(fh)
    print('WROTE',name,len(v),round(os.path.getsize(path)/1e6,1),'MB',flush=True)
def voxel(v,p,target):
    lo=p.min(0); diag=float(np.linalg.norm(p.max(0)-lo)); vx=diag/1600.0
    for _ in range(16):
        k=np.floor((p-lo)/vx).astype(np.int64); h=k[:,0]*73856093 ^ k[:,1]*19349663 ^ k[:,2]*83492791
        _,idx=np.unique(h,return_index=True)
        if len(idx)<=target: break
        vx*=1.20
    return v[idx]
v=load(os.path.join(OUT,'courthouse_6m_fused.ply'))
p=np.stack([v['x'],v['y'],v['z']],1)
save(voxel(v,p,3_000_000),'courthouse_3m_fused.ply')
save(voxel(v,p,1_500_000),'courthouse_light_fused.ply')
print('DONE2',flush=True)
