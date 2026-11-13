"""Inference on ONE new 5-band scene prefix, with GT optional.
Require files <prefix>_R.png, _G.png, _B.png, _NIR.png, _RE.png.
This is a diagnostic demo, NOT new independent TEST evaluation.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SRC))
from scripts.exp22_run import load_base, tta_pairs

PALETTE=np.array([[38,38,38],[32,180,75],[230,70,65]],dtype=np.uint8)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prefix',required=True)
    p.add_argument('--ckpt',required=True)
    p.add_argument('--out_dir',required=True)
    p.add_argument('--tta',choices=['none','d4'],default='d4')
    a=p.parse_args()
    prefix=Path(a.prefix)
    files=[Path(str(prefix)+'_'+b+'.png') for b in ['R','G','B','NIR','RE']]
    for f in files:
        if not f.is_file():raise FileNotFoundError(f)
    raw=[]
    for f in files:
        arr=np.asarray(Image.open(f))
        if arr.ndim!=2:raise ValueError(f'Expected 1-channel image: {f}')
        if arr.dtype==np.uint8:scale=255.
        elif arr.dtype==np.uint16:scale=65535.
        else:raise ValueError(f'Unrecognized band dtype {arr.dtype}: {f}')
        raw.append(arr.astype('float32')/scale)
    if len({x.shape for x in raw})!=1:raise ValueError('Bands must be spatially aligned')
    stack=np.stack(raw)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=load_base(a.ckpt,dev).eval()
    x=torch.from_numpy(stack[None]).to(dev)
    with torch.inference_mode():
        total=None;views=0
        for tx,undo in tta_pairs(x,a.tta):
            z=undo(model(tx).float())
            total=z if total is None else total+z;views+=1
        pred=(total/views).argmax(1)[0].cpu().numpy().astype(np.uint8)
    rgb=np.stack(raw[:3],axis=-1)
    lo=np.percentile(rgb,2,axis=(0,1));hi=np.percentile(rgb,98,axis=(0,1))
    rgb=np.clip((rgb-lo)/(hi-lo+1e-8),0,1)
    overlay=.56*rgb+.44*(PALETTE[pred]/255.)
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    Image.fromarray(pred).save(out/'prediction_ids.png')
    Image.fromarray(PALETTE[pred]).save(out/'prediction_colors.png')
    Image.fromarray(np.uint8(255*overlay)).save(out/'overlay.png')
    fig,ax=plt.subplots(1,3,figsize=(14,5),constrained_layout=True)
    for axt,img,title in zip(ax,[rgb,PALETTE[pred],overlay],['5-band input (RGB display)','Prediction (BG/Crop/Weed)','Overlay']):
        axt.imshow(img);axt.set_title(title);axt.axis('off')
    fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    (out/'demo_meta.json').write_text(json.dumps({'input_prefix':str(prefix),'checkpoint':a.ckpt,
         'tta':a.tta,'note':'demo prediction only; no GT accuracy claim'},indent=2))
    print('EXP28 SINGLE IMAGE PASS:',out)

if __name__=='__main__':main()
