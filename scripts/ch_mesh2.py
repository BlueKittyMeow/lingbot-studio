import numpy as np, glob, os, sys
import open3d as o3d
sys.path.insert(0, '/home/bluekitty/lingbot-map')
from lingbot_map.utils.geometry import depth_to_world_coords_points

SRC = '/home/bluekitty/renders/ch_max/courthouse'
files = [f for f in sorted(glob.glob(os.path.join(SRC, 'frame_*.npz'))) if os.path.getsize(f) > 0]
print(len(files), 'frames', flush=True)

# --- pass 1: robust building bbox from high-confidence points, outliers removed ---
allpts = []
for f in files[::2]:
    d = np.load(f); depth = d['depth'].squeeze(-1); conf = d['depth_conf']
    world, _, mask = depth_to_world_coords_points(depth, d['extrinsic'], d['intrinsic'])
    thr = np.percentile(conf, 60)                      # keep top 40% confidence
    m = mask & np.isfinite(world).all(-1) & (conf >= thr)
    w = world[m]
    if len(w) > 30000: w = w[np.random.RandomState(0).choice(len(w), 30000, replace=False)]
    allpts.append(w)
P = np.concatenate(allpts)
pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(P)
pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)   # kill sparse floaters
Q = np.asarray(pc.points)
lo, hi = np.percentile(Q, 1.5, 0), np.percentile(Q, 98.5, 0)             # tight building box
pad = (hi - lo) * 0.04; lo -= pad; hi += pad
diag = float(np.linalg.norm(hi - lo))
voxel = diag / 700.0                    # finer than before (was /512 on an inflated bbox)
sdf_trunc = voxel * 3                   # less gap-bridging -> keeps column/wall separation
print(f'building bbox diag {diag:.2f} -> voxel {voxel:.4f}, sdf_trunc {sdf_trunc:.4f}', flush=True)

# --- pass 2: TSDF, masking each frame's depth to the building box (drops sky + floaters) ---
vol = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel, sdf_trunc=sdf_trunc,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
for i, f in enumerate(files):
    d = np.load(f)
    depth = d['depth'].squeeze(-1).astype(np.float32); conf = d['depth_conf'].astype(np.float32)
    rgb = (np.clip(np.transpose(d['images'], (1, 2, 0)), 0, 1) * 255).astype(np.uint8)
    H, W = depth.shape
    world, _, mask = depth_to_world_coords_points(depth, d['extrinsic'], d['intrinsic'])
    inbox = (world >= lo).all(-1) & (world <= hi).all(-1)          # only building-volume pixels
    thr = np.percentile(conf, 45)
    keep = mask & inbox & np.isfinite(depth) & (depth > 0) & (conf >= thr)
    dd = depth.copy(); dd[~keep] = 0.0
    K = d['intrinsic']
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2]))
    E = np.eye(4); E[:3, :4] = d['extrinsic']
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(rgb)), o3d.geometry.Image(np.ascontiguousarray(dd)),
        depth_scale=1.0, depth_trunc=diag*1.5, convert_rgb_to_intensity=False)
    vol.integrate(rgbd, intr, E)
    if i % 60 == 0: print(f'  integrated {i}/{len(files)}', flush=True)

mesh = vol.extract_triangle_mesh(); mesh.compute_vertex_normals()
print('raw:', len(mesh.vertices), 'v', len(mesh.triangles), 'f', flush=True)
# crop to building box + drop tiny clusters
mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(lo, hi))
tc, cn, _ = mesh.cluster_connected_triangles(); tc = np.asarray(tc); cn = np.asarray(cn)
if len(cn):
    keepc = cn[tc] >= max(400, 0.004 * cn.sum())
    mesh.remove_triangles_by_mask(~keepc); mesh.remove_unreferenced_vertices()
print('cleaned:', len(mesh.vertices), 'v', len(mesh.triangles), 'f', flush=True)

# level (up -> +Y) and recenter
ups = []
for f in files:
    d = np.load(f); R = d['extrinsic'][:3, :3]; u = -R[1, :]; ups.append(u/np.linalg.norm(u))
up = np.mean(ups, 0); up /= np.linalg.norm(up)
def rot_to(a, b):
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b); v=np.cross(a,b); c=float(np.dot(a,b))
    if np.linalg.norm(v)<1e-8: return np.eye(3) if c>0 else np.diag([1.,-1.,-1.])
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]); return np.eye(3)+vx+vx@vx*(1/(1+c))
mesh.rotate(rot_to(up, np.array([0,1,0])), center=(0,0,0)); mesh.translate(-mesh.get_center())
mesh.compute_vertex_normals()

o3d.io.write_triangle_mesh('/home/bluekitty/renders/ch_max_mesh_v3.ply', mesh)
import trimesh
tm = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles),
                     vertex_colors=(np.clip(np.asarray(mesh.vertex_colors),0,1)*255).astype(np.uint8), process=False)
tm.export('/mnt/e/wsl_deploy/ch_max_mesh_v3.glb')
print('WROTE v3 glb', round(os.path.getsize('/mnt/e/wsl_deploy/ch_max_mesh_v3.glb')/1e6,1),'MB', flush=True)
print('MESH2_DONE', flush=True)
