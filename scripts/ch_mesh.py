import numpy as np, glob, os, sys
import open3d as o3d
sys.path.insert(0, '/home/bluekitty/lingbot-map')
from lingbot_map.utils.geometry import depth_to_world_coords_points

SRC = '/home/bluekitty/renders/ch_max/courthouse'
OUT_PLY = '/home/bluekitty/renders/ch_max_mesh.ply'
OUT_GLB = '/home/bluekitty/renders/ch_max_mesh.glb'
CONF_PCT = 30.0      # drop the lowest-confidence 30% of depth pixels per frame

files = [f for f in sorted(glob.glob(os.path.join(SRC, 'frame_*.npz'))) if os.path.getsize(f) > 0]
print(len(files), 'frames', flush=True)

# --- pass 1: scene scale from world points (sample frames) -> adaptive voxel size ---
samp = files[::max(1, len(files)//30)]
pts = []
for f in samp:
    d = np.load(f); depth = d['depth'].squeeze(-1)
    world, _, mask = depth_to_world_coords_points(depth, d['extrinsic'], d['intrinsic'])
    m = mask & np.isfinite(world).all(-1)
    w = world[m]
    if len(w) > 20000:
        w = w[np.random.RandomState(0).choice(len(w), 20000, replace=False)]
    pts.append(w)
pts = np.concatenate(pts)
lo, hi = np.percentile(pts, 1, 0), np.percentile(pts, 99, 0)
diag = float(np.linalg.norm(hi - lo))
voxel = diag / 512.0
sdf_trunc = voxel * 5
depth_trunc = diag * 1.5
print(f'scene diag {diag:.2f} -> voxel {voxel:.4f}, sdf_trunc {sdf_trunc:.4f}, depth_trunc {depth_trunc:.2f}', flush=True)

# --- pass 2: TSDF integrate with confidence masking ---
vol = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel, sdf_trunc=sdf_trunc,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
for i, f in enumerate(files):
    d = np.load(f)
    depth = d['depth'].squeeze(-1).astype(np.float32)
    conf = d['depth_conf'].astype(np.float32)
    rgb = (np.clip(np.transpose(d['images'], (1, 2, 0)), 0, 1) * 255).astype(np.uint8)
    H, W = depth.shape
    thr = np.percentile(conf, CONF_PCT)
    depth = depth.copy()
    depth[(conf < thr) | ~np.isfinite(depth) | (depth <= 0)] = 0.0
    K = d['intrinsic']
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    E = np.eye(4); E[:3, :4] = d['extrinsic']  # world->camera (OpenCV W2C)
    color = o3d.geometry.Image(np.ascontiguousarray(rgb))
    dep = o3d.geometry.Image(np.ascontiguousarray(depth))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, dep, depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
    vol.integrate(rgbd, intr, E)
    if i % 50 == 0: print(f'  integrated {i}/{len(files)}', flush=True)

mesh = vol.extract_triangle_mesh()
mesh.compute_vertex_normals()
print('raw mesh:', len(mesh.vertices), 'verts', len(mesh.triangles), 'faces', flush=True)

# clean: drop tiny disconnected clusters (floaters)
tri_clusters, cluster_n, _ = mesh.cluster_connected_triangles()
tri_clusters = np.asarray(tri_clusters); cluster_n = np.asarray(cluster_n)
if len(cluster_n):
    keep = cluster_n[tri_clusters] >= max(200, 0.002 * cluster_n.sum())
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
print('cleaned mesh:', len(mesh.vertices), 'verts', len(mesh.triangles), 'faces', flush=True)

o3d.io.write_triangle_mesh(OUT_PLY, mesh)
import trimesh
tm = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles),
                     vertex_colors=(np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8),
                     process=False)
tm.export(OUT_GLB)
print('WROTE', OUT_PLY, 'and', OUT_GLB, flush=True)
print('GLB size MB', round(os.path.getsize(OUT_GLB)/1e6, 1), flush=True)
print('MESH_DONE', flush=True)
