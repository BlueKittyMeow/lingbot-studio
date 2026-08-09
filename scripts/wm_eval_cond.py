import numpy as np, glob, torch, os, sys
os.chdir('/home/bluekitty/HunyuanWorld-Mirror'); sys.path.insert(0,'/home/bluekitty/HunyuanWorld-Mirror')
from src.models.models.worldmirror import WorldMirror
SRC='/home/bluekitty/renders/courthouse_sparse/courthouse'
nf=sorted(glob.glob(SRC+'/frame_*.npz')); idx=list(range(0,len(nf),max(1,len(nf)//16)))[:16]
im=[]; lb_w2c=[]
for i in idx:
    d=np.load(nf[i]); im.append(d['images'].astype('float32'))
    e=np.eye(4,dtype='float32'); e[:3,:4]=d['extrinsic']; lb_w2c.append(e)
lb_w2c=np.stack(lb_w2c).astype(np.float64); lb_c2w=np.linalg.inv(lb_w2c)
dev='cuda'
model=WorldMirror.from_pretrained('tencent/HunyuanWorld-Mirror').to(dev).eval()
imgs_t=torch.from_numpy(np.stack(im)[None]).to(dev)
def run(cf,poses=None):
    v={'img':imgs_t}
    if poses is not None: v['camera_poses']=torch.from_numpy(poses[None]).float().to(dev)
    with torch.no_grad(), torch.amp.autocast('cuda',dtype=torch.bfloat16):
        p=model(views=v,cond_flags=cf)
    return p['camera_poses'][0].float().cpu().numpy().astype(np.float64)
def umeyama(X,Y):
    mx,my=X.mean(0),Y.mean(0); Xc,Yc=X-mx,Y-my
    S=Yc.T@Xc/len(X); U,D,Vt=np.linalg.svd(S); dd=np.sign(np.linalg.det(U@Vt))
    R=U@np.diag([1,1,dd])@Vt; s=(D*np.array([1,1,dd])).sum()/((Xc**2).sum()/len(X)); t=my-s*R@mx
    return np.sqrt(((Y-(s*(R@X.T).T+t))**2).sum(1)).mean()
def resid(out,ref):
    e=np.linalg.norm(ref[:,:3,3].max(0)-ref[:,:3,3].min(0)); return umeyama(out[:,:3,3],ref[:,:3,3])/e
un=run([0,0,0]); print('UNCOND  out-vs-lingbot resid/extent:',round(resid(un,lb_c2w),4),flush=True)
c1=run([0,0,1],lb_c2w); print('COND C2W(pose) out-vs-lingbot:',round(resid(c1,lb_c2w),4),flush=True)
c2=run([0,0,1],un);     print('VALIDATE (feed WM-own back) out-vs-those:',round(resid(c2,un),4),flush=True)
c3=run([0,0,1],np.stack([np.linalg.inv(x) for x in lb_c2w]))  # feed W2C by mistake
print('COND W2C(pose) out-vs-lingbot:',round(resid(c3,lb_c2w),4),flush=True)
print('EVAL_DONE',flush=True)
