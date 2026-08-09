src=open('/home/bluekitty/HunyuanWorld-Mirror/infer.py').read()
marker="    cond_flags = [0, 0, 0]\n"
inject='''
    # ==== ALL-THREE PRIORS: pose(C2W) + intrinsics + DEPTH, all flags on ====
    import numpy as _np, glob as _glob, cv2 as _cv2
    _SRC='/home/bluekitty/renders/courthouse_sparse/courthouse'
    _nf=sorted(_glob.glob(os.path.join(_SRC,'frame_*.npz')))
    _idx=list(range(0,len(_nf),max(1,len(_nf)//16)))[:16]
    _pd='/home/bluekitty/wm_cond_pngs'; os.makedirs(_pd,exist_ok=True)
    _im=[]; _pose=[]; _K=[]; _dep=[]; img_paths=[]
    for _k,_i in enumerate(_idx):
        _d=_np.load(_nf[_i]); _rgb=_d['images'].astype('float32'); _im.append(_rgb)
        _e=_np.eye(4,dtype='float32'); _e[:3,:4]=_d['extrinsic']
        _pose.append(_np.linalg.inv(_e).astype('float32'))          # C2W
        _K.append(_d['intrinsic'].astype('float32'))
        _dep.append(_d['depth'].squeeze(-1).astype('float32'))       # [H,W] metric depth
        _p=os.path.join(_pd,f'{_k:03d}.png')
        _cv2.imwrite(_p,_cv2.cvtColor((_np.clip(_np.transpose(_rgb,(1,2,0)),0,1)*255).astype('uint8'),_cv2.COLOR_RGB2BGR)); img_paths.append(_p)
    imgs=torch.from_numpy(_np.stack(_im)[None]).to(device); views['img']=imgs
    views['camera_poses']=torch.from_numpy(_np.stack(_pose)[None]).to(device)
    views['camera_intrs']=torch.from_numpy(_np.stack(_K)[None]).to(device)
    views['depthmap']=torch.from_numpy(_np.stack(_dep)[None]).to(device)
    cond_flags=[1,1,1]
    B,S,C,H,W=imgs.shape
    print('>>> ALL-THREE priors (pose+intr+depth) cond_flags=[1,1,1]',S,'frames',flush=True)
    # ==== end injection ====
'''
open('/home/bluekitty/HunyuanWorld-Mirror/infer_cond.py','w').write(src.replace(marker,marker+inject))
print("rebuilt all-three")
