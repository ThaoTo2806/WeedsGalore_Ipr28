"""WeedsGalore reproducible qualitative figures: input RGB, GT, prediction, overlay, errors.
Figures are selected by fixed dataset indices, NEVER by test scores. TEST requires explicit opt-in.
"""
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from datasets import WeedsGaloreDataset
from scripts.exp22_run import load_base, tta_pairs

COLORS=np.array([[50,50,50],[39,172,75],[229,72,62]],dtype=np.uint8)


def normalize_rgb(x, low=2, high=98):
    """Visualization ONLY; does not alter network input. Clip percentile per channel."""
    arr=np.asarray(x[:3], dtype=np.float32).transpose(1,2,0)
    lo=np.percentile(arr,(low,),axis=(0,1))[0]
    hi=np.percentile(arr,(high,),axis=(0,1))[0]
    return np.clip((arr-lo)/(hi-lo+1e-8),0,1)


def metrics(gt,pr):
    valid=(gt>=0)&(gt<3)
    cm=np.bincount(3*gt[valid].astype(np.int64)+pr[valid].astype(np.int64),minlength=9).reshape(3,3)
    tp=cm.diagonal(); d=cm.sum(0)+cm.sum(1)-tp
    vals=np.divide(tp,d,out=np.zeros(3),where=d>0)*100
    return float(vals.mean()), vals,cm


@torch.inference_mode()
def pred_d4(model,x,tta='d4'):
    out=None;n=0
    for xt,undo in tta_pairs(x,tta):
        z=model(xt).float();z=undo(z)
        out=z if out is None else out+z;n+=1
    return (out/n).argmax(dim=1)[0].cpu().numpy()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset_path',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out_dir',required=True)
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--allow_test_figures',action='store_true')
    p.add_argument('--indices',default='0,1,2,3,4')
    p.add_argument('--tta',choices=['none','d4'],default='d4')
    a=p.parse_args()
    if a.split=='test' and not a.allow_test_figures:
        p.error('TEST figures require explicit --allow_test_figures; never cherry-pick on test')
    indices=[int(i) for i in a.indices.split(',')]
    if len(indices)!=len(set(indices)) or any(i<0 or i>=26 for i in indices):
        p.error('Choose unique fixed indices in 0..25')
    output=Path(a.out_dir);output.mkdir(parents=True,exist_ok=True)
    ds=WeedsGaloreDataset(a.dataset_path,None,5,3,False,a.split,False)
    assert len(ds)==26
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=load_base(a.checkpoint,device).eval()
    details=[]
    for i in indices:
        image,_,gt=ds[i]
        x=torch.as_tensor(np.ascontiguousarray(image),dtype=torch.float32,device=device)[None]
        gt=np.asarray(gt)
        pred=pred_d4(model,x,a.tta)
        if pred.shape!=gt.shape:raise RuntimeError('Input/GT/output shape mismatch')
        score,ious,cm=metrics(gt,pred)
        valid=(gt>=0)&(gt<3)
        err=(gt!=pred)&valid
        rgb=normalize_rgb(image)
        overlay=.57*rgb+.43*(COLORS[pred]/255.)
        fig,axes=plt.subplots(1,5,figsize=(20,4.6),constrained_layout=True)
        for ax,title in zip(axes,['RGB input','GT (BG/Crop/Weed)','Prediction','Prediction overlay','Error map']):
            ax.set_title(title,fontsize=11);ax.axis('off')
        axes[0].imshow(rgb)
        axes[1].imshow(COLORS[np.clip(gt,0,2)])
        axes[2].imshow(COLORS[pred])
        axes[3].imshow(overlay)
        axes[4].imshow(err,cmap=ListedColormap(['#1d1d1d','#ffd23f']),vmin=0,vmax=1)
        fig.suptitle(f'{ds.img_list[i]} | mIoU={score:.2f}% | crop={ious[1]:.2f}% | weed={ious[2]:.2f}%')
        dest=output/f'{a.split}_fixed_index{i:02d}.png'
        fig.savefig(dest,dpi=170);plt.close(fig)
        np.savez_compressed(output/f'{a.split}_fixed_index{i:02d}_masks.npz',gt=gt,pred=pred,cm=cm)
        details.append({'index':i,'name':str(ds.img_list[i]),'mIoU':score,'crop':float(ious[1]),'weed':float(ious[2]),'figure':str(dest)})
        print('FIGURE:',dest,flush=True)
    manifest={'split':a.split,'selection_rule':'fixed_indices_ascending_predefined_not_best_images',
              'indices':indices,'tta':a.tta,'checkpoint':str(a.checkpoint),
              'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
              'figures':details}
    (output/'figure_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('EXP28 FIGURES PASS',output)

if __name__=='__main__':main()
