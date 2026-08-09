import numpy as np, glob, os, sys
import open3d as o3d
sys.path.insert(0, '/home/bluekitty/lingbot-map')
from lingbot_map.utils.geometry import depth_to_world_coords_points

SRC = '/home/bluekitty/renders/ch_max/courthouse'
files = [f for f in sorted(glob.glob(os.path.join(SRC, 'frame_*.npz'))) if os.path.getsize(f) > 0]
print(len(files), 'frames', flush=True)

pts=[]; cols=[]; cams=[]
for f in files:
    d=np.load(f); depth=d['depth'].squeeze(-1); conf=d['depth_conf']
    world,_,mask=depth_to_world_coords_points(depth, d['extrinsic'], d['intrinsic'])
    R=d['extrinsic'][:3,:3]; t=d['extrinsic'][:3,3]; C=-R.T@t
    rgb=np.clip(np.transpose(d['images'],(1,2,0)),0,1)
    thr=np.percentile(conf,55)
    m=mask & np.isfinite(world).all(-1) & (conf>=thr)
    w=world[m]; c=rgb[m]
    if len(w)>9000:
        idx=np.random.RandomState(0).choice(len(w),9000,replace=False); w,c=w[idx],c[idx]
    pts.append(w); cols.append(c); cams.append(np.tile(C,(len(w),1)))
P=np.concatenate(pts); Cl=np.concatenate(cols); Cm=np.concatenate(cams)
print('accumulated', len(P), 'pts', flush=True)

pc=o3d.geometry.PointCloud(); pc.points=o3d.utility.Vector3dVector(P); pc.colors=o3d.utility.Vector3dVector(Cl)
pc,ind=pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0); Cm=Cm[np.asarray(ind)]
Q=np.asarray(pc.points)
lo,hi=np.percentile(Q,1.5,0),np.percentile(Q,98.5,0); pad=(hi-lo)*0.04; lo-=pad; hi+=pad
inbox=(Q>=lo).all(1)&(Q<=hi).all(1)
pc=pc.select_by_index(np.where(inbox)[0]); Cm=Cm[inbox]
diag=float(np.linalg.norm(hi-lo)); print('bbox diag', round(diag,2), 'pts', len(pc.points), flush=True)

pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=diag/150, max_nn=30))
Nrm=np.asarray(pc.normals); Pn=np.asarray(pc.points)
flip=((Cm-Pn)*Nrm).sum(1)<0; Nrm[flip]*=-1; pc.normals=o3d.utility.Vector3dVector(Nrm)
print('normals oriented, running Poisson...', flush=True)

mesh,dens=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=10, linear_fit=True)
dens=np.asarray(dens)
mesh.remove_vertices_by_mask(dens < np.quantile(dens,0.04))     # trim ballooned low-density
mesh=mesh.crop(o3d.geometry.AxisAlignedBoundingBox(lo,hi))
print('poisson raw:', len(mesh.vertices),'v', len(mesh.triangles),'f', flush=True)
if len(mesh.triangles)>900000:
    mesh=mesh.simplify_quadric_decimation(900000)
mesh.compute_vertex_normals()
# drop tiny clusters
tc,cn,_=mesh.cluster_connected_triangles(); tc=np.asarray(tc); cn=np.asarray(cn)
if len(cn):
    keepc=cn[tc]>=max(400,0.004*cn.sum()); mesh.remove_triangles_by_mask(~keepc); mesh.remove_unreferenced_vertices()
print('poisson clean:', len(mesh.vertices),'v', len(mesh.triangles),'f', flush=True)

# level + recenter
ups=[]
for f in files:
    d=np.load(f); Rr=d['extrinsic'][:3,:3]; u=-Rr[1,:]; ups.append(u/np.linalg.norm(u))
up=np.mean(ups,0); up/=np.linalg.norm(up)
def rot_to(a,b):
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b); v=np.cross(a,b); c=float(np.dot(a,b))
    if np.linalg.norm(v)<1e-8: return np.eye(3) if c>0 else np.diag([1.,-1.,-1.])
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]); return np.eye(3)+vx+vx@vx*(1/(1+c))
mesh.rotate(rot_to(up,np.array([0,1,0])), center=(0,0,0)); mesh.translate(-mesh.get_center())
mesh.compute_vertex_normals()

import trimesh
tm=trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles),
                   vertex_colors=(np.clip(np.asarray(mesh.vertex_colors),0,1)*255).astype(np.uint8), process=False)
tm.export('/mnt/e/wsl_deploy/ch_max_mesh_poisson.glb')
print('WROTE poisson glb', round(os.path.getsize('/mnt/e/wsl_deploy/ch_max_mesh_poisson.glb')/1e6,1),'MB', flush=True)
print('POISSON_DONE', flush=True)
