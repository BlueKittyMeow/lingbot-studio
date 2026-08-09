"""
Stitch overlapping lingbot-map reconstruction windows into ONE long walk.
VGGT-Long-style: per-window scale-normalize -> dense-point Umeyama Sim(3) on the
shared overlap frames (confidence-weighted Huber-IRLS) -> chain -> frame-ownership
merge -> leveled PLY tiers.

Usage:
  python stitch_windows.py OVERLAP OUT_PREFIX  DIR0 DIR1 DIR2 ...
    OVERLAP    = number of shared frames between consecutive windows (last O of k = first O of k+1)
    OUT_PREFIX = e.g. cathedral_full  (writes <prefix>_{light,3m,6m,max}_fused.ply to ~/renders)
    DIR*       = window frame dirs (each holds frame_000000.npz ...), IN ORDER
"""
import numpy as np, glob, os, sys
import open3d as o3d

OVERLAP   = int(sys.argv[1])
OUTPREF   = sys.argv[2]
WIN_DIRS  = sys.argv[3:]
OUT       = '/home/bluekitty/renders'
SUB_ALIGN = 4000     # points/frame used for Sim3 fitting
SUB_EMIT  = 24000    # points/frame emitted into the final cloud (pre-voxel)
PER_TILE  = [1_500_000, 3_000_000, 6_000_000]

def frames(d):
    return [f for f in sorted(glob.glob(os.path.join(d, 'frame_*.npz'))) if os.path.getsize(f) > 0]

def backproject(depth, K, E):
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth.reshape(-1)
    x = (u.reshape(-1) - K[0,2]) / K[0,0] * z
    y = (v.reshape(-1) - K[1,2]) / K[1,1] * z
    cam = np.stack([x, y, z], 0)                 # (3, HW) camera coords
    R, t = E[:, :3], E[:, 3]
    return R.T @ (cam - t[:, None])              # (3, HW) world (this window's frame)

def umeyama_sim3(X, Y, w=None):
    d, n = X.shape
    if w is None: w = np.ones(n)
    w = w / w.sum()
    mx = (X*w).sum(1); my = (Y*w).sum(1)
    Xc = X - mx[:,None]; Yc = Y - my[:,None]
    var_x = (w*(Xc**2).sum(0)).sum()
    Sig = (Yc*w) @ Xc.T
    U, D, Vt = np.linalg.svd(Sig)
    S = np.eye(d)
    if np.linalg.det(U)*np.linalg.det(Vt) < 0: S[-1,-1] = -1.0
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / var_x
    t = my - s*(R @ mx)
    return s, R, t

def sim3_irls(X, Y, conf, iters=10):
    keep = conf > 0.1*np.median(conf)
    X, Y, conf = X[:,keep], Y[:,keep], conf[keep]
    w = conf.copy()
    s=1.0; R=np.eye(3); t=np.zeros(3)
    for _ in range(iters):
        s, R, t = umeyama_sim3(X, Y, w)
        r = np.linalg.norm(Y - (s*(R@X)+t[:,None]), axis=0)
        delta = 1.345*np.median(r) + 1e-9
        huber = np.where(r <= delta, 1.0, delta/(r+1e-9))
        w = conf * huber
    return s, R, t

# ---------- load windows + normalize each to median-depth = 1 ----------
WINS = []
for d in WIN_DIRS:
    fs = frames(d)
    dep=[]; con=[]; rgb=[]; ext=[]; K=[]
    med_depths=[]
    for f in fs:
        z=np.load(f); depth=z['depth'].squeeze(-1).astype(np.float32)
        dep.append(depth); con.append(z['depth_conf'].astype(np.float32))
        rgb.append(np.clip(np.transpose(z['images'],(1,2,0)),0,1).astype(np.float32))
        ext.append(z['extrinsic'].astype(np.float64)); K.append(z['intrinsic'].astype(np.float64))
        m=np.isfinite(depth)&(depth>0)
        if m.any(): med_depths.append(np.median(depth[m]))
    scale = 1.0/ (np.median(med_depths)+1e-9)          # normalize -> median depth ~1
    ext = [e.copy() for e in ext]
    for e in ext: e[:,3] *= scale                       # scale translations with depths
    dep = [dd*scale for dd in dep]
    WINS.append(dict(fs=fs, dep=dep, con=con, rgb=rgb, ext=ext, K=K, n=len(fs)))
    print(f'window {d}: {len(fs)} frames, norm scale {scale:.4f}', flush=True)

# ---------- pairwise Sim(3): window k+1 -> window k (dense overlap points) ----------
def pair_sim3(A, B, O):
    # A owns overlap frames [nA-O, nA); B owns [0, O); pixel-aligned
    Xs=[]; Ys=[]; Ws=[]
    for i in range(O):
        ai = A['n']-O+i; bi = i
        PA = backproject(A['dep'][ai], A['K'][ai], A['ext'][ai])   # target frame
        PB = backproject(B['dep'][bi], B['K'][bi], B['ext'][bi])   # source frame
        cA = A['con'][ai].reshape(-1); cB = B['con'][bi].reshape(-1)
        depA=A['dep'][ai].reshape(-1); depB=B['dep'][bi].reshape(-1)
        good = np.isfinite(PA).all(0)&np.isfinite(PB).all(0)&(depA>0)&(depB>0)
        idx = np.where(good)[0]
        if len(idx) > SUB_ALIGN:
            idx = idx[np.random.RandomState(i).choice(len(idx), SUB_ALIGN, replace=False)]
        Ys.append(PA[:,idx]); Xs.append(PB[:,idx]); Ws.append(np.sqrt(cA[idx]*cB[idx]))
    X=np.concatenate(Xs,1); Y=np.concatenate(Ys,1); Wc=np.concatenate(Ws)
    s,R,t = sim3_irls(X, Y, Wc)
    res = np.linalg.norm(Y-(s*(R@X)+t[:,None]),axis=0)
    print(f'  pair fit: s={s:.4f} inlier-med-resid={np.median(res):.4f} ({X.shape[1]} corr)', flush=True)
    return s,R,t

# chain: G[0]=identity; G[k+1] = G[k] o S_{k+1->k}
G = [(1.0, np.eye(3), np.zeros(3))]
for k in range(len(WINS)-1):
    s,R,t = pair_sim3(WINS[k], WINS[k+1], OVERLAP)          # maps win(k+1)->win(k)
    sa,Ra,ta = G[k]
    G.append((sa*s, Ra@R, sa*(Ra@t)+ta))                    # compose

# ---------- bake + frame-ownership merge ----------
half = OVERLAP//2
allP=[]; allC=[]
for k, Wk in enumerate(WINS):
    s,R,t = G[k]
    lo = 0 if k==0 else half                                # drop first half-overlap (owned by k-1)
    hi = Wk['n'] if k==len(WINS)-1 else Wk['n']-half        # drop last half-overlap (owned by k+1)
    for fi in range(lo, hi):
        P = backproject(Wk['dep'][fi], Wk['K'][fi], Wk['ext'][fi])   # (3,HW) window frame
        c = Wk['con'][fi].reshape(-1); col = Wk['rgb'][fi].reshape(-1,3); dep=Wk['dep'][fi].reshape(-1)
        good = np.isfinite(P).all(0)&(dep>0)&(c > 0.5*np.median(c))
        idx = np.where(good)[0]
        if len(idx) > SUB_EMIT:
            top = np.argpartition(c[idx], -SUB_EMIT)[-SUB_EMIT:]; idx = idx[top]
        Pg = s*(R@P[:,idx]) + t[:,None]                      # -> global frame
        allP.append(Pg.T.astype(np.float32)); allC.append(col[idx].astype(np.float32))
pts=np.concatenate(allP); cols=np.concatenate(allC); del allP,allC
print(f'baked {len(pts):,} pts across {len(WINS)} windows', flush=True)

# voxel downsample (dedup overlap density) in normalized units
pc=o3d.geometry.PointCloud(); pc.points=o3d.utility.Vector3dVector(pts.astype(np.float64)); pc.colors=o3d.utility.Vector3dVector(cols.astype(np.float64))
diag=float(np.linalg.norm(pts.max(0)-pts.min(0)))
pc=pc.voxel_down_sample(diag/1400.0)
pts=np.asarray(pc.points); cols=np.asarray(pc.colors)
print(f'after voxel dedup: {len(pts):,} pts (diag {diag:.2f})', flush=True)

# ---------- level (mean camera up -> +Y, + Rx viewer flip) ----------
ups=[]
for Wk in WINS:
    for e in Wk['ext']:
        u=-e[:3,1]; ups.append(u/np.linalg.norm(u))
up=np.mean(ups,0); up/=np.linalg.norm(up)
def rot_to(a,b):
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b); v=np.cross(a,b); c=float(np.dot(a,b))
    if np.linalg.norm(v)<1e-8: return np.eye(3) if c>0 else np.diag([1.,-1.,-1.])
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]); return np.eye(3)+vx+vx@vx*(1/(1+c))
Rx=np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)
pts = pts @ (Rx@rot_to(up,np.array([0,1,0]))).T
med=np.median(pts,0); pts=(pts-med).astype(np.float32)

def save(p,c,name):
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
save(pts,cols,f'{OUTPREF}_max_fused.ply')
for tgt in PER_TILE:
    tag={1_500_000:'light',3_000_000:'3m',6_000_000:'6m'}[tgt]
    idx=perm[:min(tgt,len(pts))]; save(pts[idx],cols[idx],f'{OUTPREF}_{tag}_fused.ply')
print('STITCH_DONE', flush=True)
