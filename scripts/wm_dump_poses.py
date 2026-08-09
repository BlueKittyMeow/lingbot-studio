import numpy as np, glob, torch, os, sys
os.chdir('/home/bluekitty/HunyuanWorld-Mirror'); sys.path.insert(0,'/home/bluekitty/HunyuanWorld-Mirror')
from src.models.models.worldmirror import WorldMirror
SRC='/home/bluekitty/renders/courthouse_sparse/courthouse'
nf=sorted(glob.glob(SRC+'/frame_*.npz'))
idx=list(range(0,len(nf),max(1,len(nf)//16)))[:16]
im=[]; lb_w2c=[]; lb_K=[]
for i in idx:
    d=np.load(nf[i]); im.append(d['images'].astype('float32'))
    e=np.eye(4,dtype='float32'); e[:3,:4]=d['extrinsic']; lb_w2c.append(e); lb_K.append(d['intrinsic'].astype('float32'))
dev='cuda'
model=WorldMirror.from_pretrained('tencent/HunyuanWorld-Mirror').to(dev).eval()
views={'img':torch.from_numpy(np.stack(im)[None]).to(dev)}
with torch.no_grad(), torch.amp.autocast('cuda',dtype=torch.bfloat16):
    preds=model(views=views, cond_flags=[0,0,0])
wm_c2w=preds['camera_poses'][0].float().cpu().numpy()
wm_K=preds['camera_intrs'][0].float().cpu().numpy()
np.savez('/home/bluekitty/wm_align.npz', wm_c2w=wm_c2w, wm_K=wm_K, lb_w2c=np.stack(lb_w2c), lb_K=np.stack(lb_K), idx=np.array(idx))
print('DUMPED wm_c2w',wm_c2w.shape,'| lb_w2c',np.stack(lb_w2c).shape,flush=True)
print('POSE_DUMP_DONE',flush=True)
