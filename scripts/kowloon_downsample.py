import numpy as np, os
OUT='/home/bluekitty/renders'; dt=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')]
def load(path):
    with open(path,'rb') as f:
        hdr=b''
        while b'end_header\n' not in hdr: hdr+=f.readline()
        return np.fromfile(f,dtype=dt)
def save(v,name):
    path=os.path.join(OUT,name)
    with open(path,'wb') as fh:
        fh.write(b"ply\nformat binary_little_endian 1.0\n"); fh.write(("element vertex %d\n"%len(v)).encode())
        fh.write(b"property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        v.tofile(fh)
    print('WROTE',name,len(v),round(os.path.getsize(path)/1e6,1),'MB',flush=True)
v=load(os.path.join(OUT,'kowloon_max_fused.ply')); print('loaded',len(v),flush=True)
perm=np.random.RandomState(0).permutation(len(v)); print('permuted',flush=True)
for tgt,name in [(6_000_000,'kowloon_6m_fused.ply'),(3_000_000,'kowloon_3m_fused.ply'),(1_500_000,'kowloon_light_fused.ply')]:
    save(v[perm[:tgt]],name)
print('DONE',flush=True)
