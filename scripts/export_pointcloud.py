import numpy as np, glob, sys, os
sys.path.insert(0, '/home/bluekitty/lingbot-map')
from lingbot_map.utils.geometry import depth_to_world_coords_points

SRC = '/home/bluekitty/renders/courthouse/courthouse'
OUT = '/home/bluekitty/renders/courthouse_fused.ply'
CONF_PCTL = 30          # drop the lowest 30% confidence points
TARGET = 2_500_000      # voxel-downsample target

files = sorted(glob.glob(os.path.join(SRC, 'frame_*.npz')))
print(f"{len(files)} frames")

pts_all, col_all, conf_all = [], [], []
for f in files:
    d = np.load(f)
    depth = d['depth'].squeeze(-1)            # (H,W)
    world, _, mask = depth_to_world_coords_points(depth, d['extrinsic'], d['intrinsic'])
    rgb = np.transpose(d['images'], (1,2,0))  # (H,W,3) 0-1
    conf = d['depth_conf']                     # (H,W)
    m = mask & np.isfinite(world).all(-1)
    pts_all.append(world[m]); col_all.append(rgb[m]); conf_all.append(conf[m])

pts = np.concatenate(pts_all); col = np.concatenate(col_all); conf = np.concatenate(conf_all)
print(f"raw points: {len(pts):,}")

thr = np.percentile(conf, CONF_PCTL)
keep = conf >= thr
pts, col = pts[keep], col[keep]
print(f"after conf>={thr:.2f}: {len(pts):,}")

# voxel downsample: pick voxel size to hit ~TARGET
lo, hi = pts.min(0), pts.max(0)
diag = np.linalg.norm(hi - lo)
print(f"bbox extent: {hi-lo}, diag {diag:.3f}")
voxel = diag / 700.0
for _ in range(8):
    keys = np.floor((pts - lo) / voxel).astype(np.int64)
    h = keys[:,0] * 73856093 ^ keys[:,1] * 19349663 ^ keys[:,2] * 83492791
    _, idx = np.unique(h, return_index=True)
    if len(idx) <= TARGET: break
    voxel *= 1.25
pts_d, col_d = pts[idx], col[idx]
print(f"after voxel {voxel:.4f}: {len(pts_d):,} points")

# center the cloud on its median so the viewer starts sensibly
pts_d = pts_d - np.median(pts_d, 0)

verts = np.empty(len(pts_d), dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
verts['x'], verts['y'], verts['z'] = pts_d[:,0], pts_d[:,1], pts_d[:,2]
c = np.clip(col_d*255, 0, 255).astype(np.uint8)
verts['red'], verts['green'], verts['blue'] = c[:,0], c[:,1], c[:,2]
with open(OUT, 'wb') as fh:
    fh.write(b"ply\nformat binary_little_endian 1.0\n")
    fh.write(f"element vertex {len(verts)}\n".encode())
    fh.write(b"property float x\nproperty float y\nproperty float z\n")
    fh.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    verts.tofile(fh)
print(f"WROTE {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
