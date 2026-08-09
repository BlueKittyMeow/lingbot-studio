import numpy as np, glob, os, sys
import open3d as o3d
from PIL import Image
sys.path.insert(0, '/home/bluekitty/lingbot-map')

MESH = '/home/bluekitty/renders/ch_max_mesh.ply'          # RAW, camera-aligned (with sky blobs)
SRC  = '/home/bluekitty/renders/ch_max/courthouse'         # courthouse cameras
SKIN = '/home/bluekitty/skins/waterlilies'                 # skin frames
OUT_GLB = '/mnt/e/wsl_deploy/ch_max_mesh_lily.glb'

mesh = o3d.io.read_triangle_mesh(MESH); mesh.compute_vertex_normals()
V = np.asarray(mesh.vertices); N = np.asarray(mesh.vertex_normals)
print('verts', len(V), flush=True)
files = [f for f in sorted(glob.glob(os.path.join(SRC, 'frame_*.npz'))) if os.path.getsize(f) > 0]
skins = sorted(glob.glob(os.path.join(SKIN, '*.png')) + glob.glob(os.path.join(SKIN, '*.jpg')))
print(len(files), 'cam frames', len(skins), 'skin frames', flush=True)

acc = np.zeros((len(V), 3)); wsum = np.zeros(len(V))
for i, f in enumerate(files):
    d = np.load(f)
    C, H, W = d['images'].shape
    E = d['extrinsic']; R = E[:3, :3]; t = E[:3, 3]; K = d['intrinsic']
    # spread the skin clip across the orbit
    sk = np.asarray(Image.open(skins[int(i * len(skins) / len(files)) % len(skins)]).resize((W, H))).astype(np.float32) / 255.0
    if sk.ndim == 2: sk = np.repeat(sk[..., None], 3, 2)
    sk = sk[..., :3]
    cam = (R @ V.T + t[:, None]).T                       # (nv,3) camera coords
    z = cam[:, 2]
    uvw = (K @ cam.T).T
    u = uvw[:, 0] / (uvw[:, 2] + 1e-9); v = uvw[:, 1] / (uvw[:, 2] + 1e-9)
    inframe = (z > 1e-4) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    Ccenter = -R.T @ t
    vd = Ccenter[None, :] - V; vd /= (np.linalg.norm(vd, axis=1, keepdims=True) + 1e-9)
    facing = np.clip((N * vd).sum(1), 0, 1)
    w = inframe.astype(np.float32) * (facing ** 2 + 0.05)
    ui = np.clip(u, 0, W - 1).astype(np.int32); vi = np.clip(v, 0, H - 1).astype(np.int32)
    col = sk[vi, ui]
    acc += w[:, None] * col; wsum += w
    if i % 60 == 0: print(f'  {i}/{len(files)}', flush=True)

colors = acc / (wsum[:, None] + 1e-9)
colors[wsum < 1e-6] = [0.15, 0.16, 0.2]                  # never-seen verts -> dim slate
mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))

# level (up -> +Y) + recenter for display
ups = []
for f in files:
    d = np.load(f); Rr = d['extrinsic'][:3, :3]; u = -Rr[1, :]; ups.append(u / np.linalg.norm(u))
up = np.mean(ups, 0); up /= np.linalg.norm(up)
def rot_to(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b); vv = np.cross(a, b); c = float(np.dot(a, b))
    if np.linalg.norm(vv) < 1e-8: return np.eye(3) if c > 0 else np.diag([1., -1., -1.])
    vx = np.array([[0, -vv[2], vv[1]], [vv[2], 0, -vv[0]], [-vv[1], vv[0], 0]]); return np.eye(3) + vx + vx @ vx * (1 / (1 + c))
mesh.rotate(rot_to(up, np.array([0, 1, 0])), center=(0, 0, 0)); mesh.translate(-mesh.get_center())

import trimesh
tm = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles),
                     vertex_colors=(np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8), process=False)
tm.export(OUT_GLB)
print('WROTE', OUT_GLB, round(os.path.getsize(OUT_GLB) / 1e6, 1), 'MB', flush=True)
print('SKIN_DONE', flush=True)
