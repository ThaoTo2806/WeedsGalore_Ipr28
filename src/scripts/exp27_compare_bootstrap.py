"""Paired image-bootstrap CIs for Exp27 vs its same-run baseline, one seed.

Read CSVs written by exp27_run_csbrl.py. Requires NO torch or TEST access.
Example: python src/scripts/exp27_compare_bootstrap.py --out_dir=.../seed0 --split=val
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

COLS=[f'cm{i}{j}' for i in range(3) for j in range(3)]

def read(path):
    with open(path,encoding='utf-8') as f: rows=sorted(csv.DictReader(f),key=lambda r:r['name'])
    names=[r['name'] for r in rows]
    cms=np.array([[int(r[k]) for k in COLS] for r in rows],dtype=np.int64).reshape(-1,3,3)
    return names,cms

def miou(cm):
    cm=np.asarray(cm,dtype=np.float64)
    tp=np.diagonal(cm,axis1=-2,axis2=-1)
    den=cm.sum(-1)+cm.sum(-2)-tp
    iou=np.divide(tp,den,out=np.zeros_like(tp),where=den>0)
    return iou.mean(axis=-1)*100

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out_dir',required=True)
    p.add_argument('--split',choices=('val','test'),default='val')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--tta',default='d4')
    p.add_argument('--draws',type=int,default=10000)
    a=p.parse_args()
    d=Path(a.out_dir)
    files=sorted(d.glob(f'perimg_exp27_*_{a.split}_{a.tta}_seed{a.seed}.csv'))
    if len(files)<2: raise RuntimeError('Need base and at least one Exp27 arm')
    results={}
    for f in files:
        name,cms=read(f)
        if len(name)!=26: raise RuntimeError(f'{f}: expected 26 images')
        arm=f.stem[len('perimg_exp27_'):-len(f'_{a.split}_{a.tta}_seed{a.seed}')]
        results[arm]=(name,cms)
    if 'base' not in results:raise RuntimeError('Missing base per-image CSV')
    bn,bc=results['base']
    weights=np.random.default_rng(27027).multinomial(len(bn),np.ones(len(bn))/len(bn),size=a.draws)
    base_draw=miou(np.einsum('bn,nij->bij',weights,bc))
    out={'split':a.split,'seed':a.seed,'draws':a.draws,'base_miou':float(miou(bc.sum(0))),'arms':{}}
    for arm,(names,cm) in results.items():
        if names!=bn:raise RuntimeError(f'Image alignment mismatch for {arm}')
        if arm=='base':continue
        delta=float(miou(cm.sum(0))-miou(bc.sum(0)))
        draw=miou(np.einsum('bn,nij->bij',weights,cm))-base_draw
        lo,hi=np.quantile(draw,[.025,.975])
        out['arms'][arm]={'delta_miou_pp':delta,'paired_image_ci95':[float(lo),float(hi)],
                          'bootstrap_fraction_nonpositive':float((draw<=0).mean())}
        print(f'{arm:10s} delta={delta:+.4f} pp CI95 [{lo:+.4f},{hi:+.4f}] nonpositive={((draw<=0).mean()):.3f}')
    dest=d/f'bootstrap_exp27_{a.split}_seed{a.seed}.json'
    dest.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('SAVED:',dest)

if __name__=='__main__':main()
