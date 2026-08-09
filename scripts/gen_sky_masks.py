import numpy as np, glob, os, sys, cv2
os.chdir('/home/bluekitty/HunyuanWorld-Mirror'); sys.path.insert(0,'/home/bluekitty/HunyuanWorld-Mirror')
import onnxruntime as ort
from src.utils.visual_util import segment_sky
sess=ort.InferenceSession('/home/bluekitty/HunyuanWorld-Mirror/skyseg.onnx', providers=['CPUExecutionProvider'])
OUT='/home/bluekitty/renders/courthouse_sky143'; TMP=OUT+'/_tmp'
os.makedirs(TMP,exist_ok=True)
nf=sorted(glob.glob('/home/bluekitty/renders/courthouse_sparse/courthouse/frame_*.npz'))
for i,f in enumerate(nf):
    d=np.load(f); img=np.transpose(d['images'],(1,2,0))
    bgr=cv2.cvtColor((np.clip(img,0,1)*255).astype(np.uint8),cv2.COLOR_RGB2BGR)
    tp=os.path.join(TMP,f'{i}.png'); cv2.imwrite(tp,bgr)
    mask=segment_sky(tp,sess)
    if mask.ndim==3: mask=mask[...,0]
    if mask.shape[:2]!=img.shape[:2]: mask=cv2.resize(mask,(img.shape[1],img.shape[0]))
    cv2.imwrite(os.path.join(OUT,f'frame_{i:04d}.png'),mask)
    os.remove(tp)
skyfrac=1-np.mean([ (cv2.imread(os.path.join(OUT,f'frame_{i:04d}.png'),0)>127).mean() for i in range(len(nf))])
print('WROTE',len(nf),'masks | avg sky fraction',round(float(skyfrac),3),flush=True)
