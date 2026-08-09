import numpy as np, glob, os, re
dirs = sorted(glob.glob('/home/bluekitty/renders/cathprobe/p*'))
rows=[]
for d in dirs:
    fs=[f for f in sorted(glob.glob(os.path.join(d,'*','frame_*.npz'))) if os.path.getsize(f)>0]
    if len(fs)<20: continue
    ts=[];ups=[];pts=[]
    for f in fs:
        z=np.load(f); R=z['extrinsic'][:3,:3]; t=z['extrinsic'][:3,3]
        ts.append(-R.T@t); u=-R[1,:]; ups.append(u/np.linalg.norm(u))
    ts=np.array(ts); ups=np.array(ups)
    drift=float(np.degrees(np.arccos(np.clip(ups[-1]@ups[0],-1,1))))
    path=float(np.sum(np.linalg.norm(np.diff(ts,axis=0),axis=1)))
    net=float(np.linalg.norm(ts[-1]-ts[0]))
    straight=net/(path+1e-9)
    m=re.search(r'p(\d+)', os.path.basename(d)); start=int(m.group(1)) if m else -1
    rows.append((start,len(fs),drift,path,net,straight))
# clean walk = low drift + high straightness(net/path near 1) + real path length
rows.sort(key=lambda r: (r[2] - 35*r[5] - 3*min(r[3],2.0)))
print("rank by clean-walk score (low drift, straight, real motion):")
print(" start  frames  drift  path   net   straight")
for start,n,dr,pa,ne,st in rows:
    flag = "  <- clean" if (dr<9 and st>0.55 and pa>0.5) else ("  ~ok" if (dr<14 and st>0.4) else "  x messy(zoom/pan?)")
    print(f" {start:5d}  {n:4d}   {dr:5.1f}  {pa:5.2f} {ne:5.2f}  {st:.2f}{flag}")
print("SCORE_DONE")
