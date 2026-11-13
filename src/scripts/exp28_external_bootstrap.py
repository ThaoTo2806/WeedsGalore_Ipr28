"""One-seed paired image bootstrap on WeedMap external custom scene split CSVs."""
import argparse,csv,json
from pathlib import Path
import numpy as np
COLS=[f'cm{i}{j}' for i in range(3) for j in range(3)]

def load(path):
 with open(path,newline='') as f:r=sorted(csv.DictReader(f),key=lambda z:z['name'])
 return [v['name'] for v in r],np.array([[int(v[k]) for k in COLS] for v in r]).reshape(-1,3,3)

def miou(cm):
 tp=np.diagonal(cm,axis1=-2,axis2=-1)
 den=cm.sum(-1)+cm.sum(-2)-tp
 return (np.divide(tp,den,out=np.zeros_like(tp,dtype=float),where=den>0)*100).mean(-1)

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--out_dir',required=True);p.add_argument('--split',choices=['val','test'],default='val')
 p.add_argument('--draws',type=int,default=10000)
 a=p.parse_args();out=Path(a.out_dir)
 files=sorted(out.glob(f'perimg_weedmap_{a.split}_*.csv'))
 if len(files)<3:raise RuntimeError('Need base + at least two arms')
 data={f.stem.split(f'perimg_weedmap_{a.split}_')[1]:load(f) for f in files}
 if 'base' not in data:raise RuntimeError('Missing base')
 n=len(data['base'][0]);rng=np.random.default_rng(28281)
 w=rng.multinomial(n,np.full(n,1/n),size=a.draws)
 rep={'dataset':'WeedMap RedEdge','split':a.split,'bootstrap_images':n,'draws':a.draws,'arms':{}}
 for arm in data:
  if arm=='base':continue
  names,cm=data[arm]
  for comparator in ('base','control','random'):
   if comparator==arm or comparator not in data:continue
   n2,cm2=data[comparator]
   if names!=n2:raise RuntimeError('Image alignment mismatch')
   delta=miou(cm.sum(0))-miou(cm2.sum(0))
   draws=miou(np.einsum('bn,nij->bij',w,cm))-miou(np.einsum('bn,nij->bij',w,cm2))
   key=f'{arm}-minus-{comparator}'
   rep['arms'][key]={'delta_pp':float(delta),'ci95':[float(q) for q in np.quantile(draws,[.025,.975])],
                     'nonpositive_fraction':float((draws<=0).mean())}
   print(key,rep['arms'][key])
 dest=out/f'weedmap_bootstrap_{a.split}.json';dest.write_text(json.dumps(rep,indent=2))
 print('EXP28 EXTERNAL BOOTSTRAP PASS',dest)
if __name__=='__main__':main()
