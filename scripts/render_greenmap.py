import numpy as np, glob, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,'/home/bluekitty/lingbot-map')
from lingbot_map.utils.geometry import depth_to_world_coords_points

GREEN="#39ff14"; DIMG="#0d3b0d"; CAM="#d4ff00"; BG="#000000"
def rot_to(a,b):
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b); v=np.cross(a,b); c=np.dot(a,b)
    if np.linalg.norm(v)<1e-8: return np.eye(3) if c>0 else -np.eye(3)
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+vx+vx@vx*(1/(1+c))

def load(scene_dir, cap=350000, per=12000):
    files=[f for f in sorted(glob.glob(os.path.join(scene_dir,'frame_*.npz'))) if os.path.getsize(f)>0]
    ups=[]; cams=[]
    for f in files:
        d=np.load(f); R=d['extrinsic'][:3,:3]; t=d['extrinsic'][:3,3]
        u=-R[1,:]; ups.append(u/np.linalg.norm(u)); cams.append(-R.T@t)
    up=np.mean(ups,0); up/=np.linalg.norm(up)
    Rlevel=rot_to(up,np.array([0,1,0])); Rx=np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)
    Rf=(Rx@Rlevel)
    P=[]
    for f in files:
        d=np.load(f); depth=d['depth'].squeeze(-1)
        world,_,mask=depth_to_world_coords_points(depth,d['extrinsic'],d['intrinsic'])
        m=mask & np.isfinite(world).all(-1); pw=world[m]
        if len(pw)>per: pw=pw[np.random.RandomState(0).choice(len(pw),per,replace=False)]
        P.append(pw)
    pts=np.concatenate(P)@Rf.T
    cams=np.array(cams)@Rf.T
    med=np.median(pts,0); pts-=med; cams-=med
    if len(pts)>cap: pts=pts[np.random.RandomState(1).choice(len(pts),cap,replace=False)]
    return pts, cams, len(files)

def style_ax(ax, d3=False):
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_color(DIMG)
    ax.tick_params(colors=DIMG, labelsize=6)
    if d3:
        ax.xaxis.set_pane_color((0,0,0,1)); ax.yaxis.set_pane_color((0,0,0,1)); ax.zaxis.set_pane_color((0,0,0,1))
        for a in (ax.xaxis,ax.yaxis,ax.zaxis): a._axinfo["grid"]["color"]=(0.05,0.23,0.05,1)

def render(scene_dir, title, out):
    pts,cams,nf=load(scene_dir)
    fig=plt.figure(figsize=(20,11)); fig.patch.set_facecolor(BG)
    fig.suptitle(f"{title}   ·   {nf} frames   ·   {len(pts):,} pts   ·   green fluoro topography map",
                 color=GREEN, fontfamily="monospace", fontsize=13, y=0.98)
    def sc2(ax,x,z,cx,cz):
        ax.scatter(x,z,s=0.35,c=GREEN,alpha=0.035,edgecolors='none',rasterized=True)
        ax.plot(cx,cz,'-',color=CAM,lw=0.8,alpha=0.9)
        ax.scatter(cx,cz,s=8,c=CAM,alpha=0.9,marker='^',edgecolors='none')
        ax.set_aspect('equal'); style_ax(ax)
    ax=fig.add_subplot(2,3,1); ax.set_title("TOP-DOWN  X-Z",color=GREEN,fontfamily="monospace",fontsize=9); sc2(ax,pts[:,0],pts[:,2],cams[:,0],cams[:,2])
    ax=fig.add_subplot(2,3,2); ax.set_title("FRONT  X-Y",color=GREEN,fontfamily="monospace",fontsize=9); sc2(ax,pts[:,0],pts[:,1],cams[:,0],cams[:,1])
    ax=fig.add_subplot(2,3,3); ax.set_title("SIDE  Z-Y",color=GREEN,fontfamily="monospace",fontsize=9); sc2(ax,pts[:,2],pts[:,1],cams[:,2],cams[:,1])
    for i,(el,az) in enumerate([(25,45),(60,20),(15,110)]):
        ax=fig.add_subplot(2,3,4+i,projection='3d')
        ax.set_title(f"3D elev{el} azim{az}",color=GREEN,fontfamily="monospace",fontsize=9)
        ax.scatter(pts[:,0],pts[:,2],pts[:,1],s=0.3,c=GREEN,alpha=0.03,edgecolors='none',rasterized=True)
        ax.plot(cams[:,0],cams[:,2],cams[:,1],'-',color=CAM,lw=0.8,alpha=0.9)
        ax.view_init(elev=el,azim=az); style_ax(ax,d3=True)
    plt.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(out,dpi=110,facecolor=BG); plt.close(fig)
    print("WROTE",out,flush=True)

render('/home/bluekitty/renders/lowres648/catacombs2_crop',
       'CATACOMBS2-MAX  (sw64/nsf8 @448, full defaults)',
       '/mnt/e/wsl_deploy/greenmap_cata2max.png')
render('/home/bluekitty/renders/catacombs2q/catacombs2_crop',
       'CATACOMBS2  (sw32/nsf4 @518, shipped)',
       '/mnt/e/wsl_deploy/greenmap_catacombs2q.png')
print("DONE")
