import numpy as np, open3d as o3d, glob, os
files = sorted(glob.glob('/home/bluekitty/renders/ch_max/courthouse/frame_*.npz'))
ups=[]
for f in files:
    d=np.load(f); R=d['extrinsic'][:3,:3]; u=-R[1,:]; ups.append(u/np.linalg.norm(u))
up=np.mean(ups,0); up/=np.linalg.norm(up)
print('mean camera up:', np.round(up,3), flush=True)
def rot_to(a,b):
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b); v=np.cross(a,b); c=float(np.dot(a,b))
    if np.linalg.norm(v)<1e-8: return np.eye(3) if c>0 else np.diag([1.,-1.,-1.])
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+vx+vx@vx*(1/(1+c))
Rlevel=rot_to(up, np.array([0,1,0]))   # map scene-up -> +Y (three.js up)
m=o3d.io.read_triangle_mesh('/home/bluekitty/renders/ch_max_mesh.ply')
m.rotate(Rlevel, center=(0,0,0))
# recenter to origin so the viewer frames it cleanly
m.translate(-m.get_center())
m.compute_vertex_normals()
o3d.io.write_triangle_mesh('/home/bluekitty/renders/ch_max_mesh_level.ply', m)
import trimesh
tm=trimesh.Trimesh(vertices=np.asarray(m.vertices), faces=np.asarray(m.triangles),
                   vertex_colors=(np.clip(np.asarray(m.vertex_colors),0,1)*255).astype(np.uint8), process=False)
tm.export('/mnt/e/wsl_deploy/ch_max_mesh.glb')   # overwrite the deployed name
print('WROTE leveled glb', round(os.path.getsize('/mnt/e/wsl_deploy/ch_max_mesh.glb')/1e6,1),'MB', flush=True)
print('LEVEL_DONE')
