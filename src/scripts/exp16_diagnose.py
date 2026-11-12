"""Read-only diagnostics of train / validation resolutions and available error reports.
No inference or test access. Doesn't run training.
"""
from pathlib import Path
import argparse,json,csv,collections
import numpy as np
from datasets import WeedsGaloreDataset

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset_path',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    report={'split_shapes':{}, 'test_opened':False}
    for split in ('train','val'):
        ds=WeedsGaloreDataset(dataset_path=a.dataset_path,dataset_size=None,in_bands=5,
                              num_classes=3,is_training=False,split=split,augmentation=False)
        tally=collections.Counter();counts=np.zeros(3,dtype=np.int64)
        for i in range(len(ds)):
            im,_,lab=ds[i]
            tally[(tuple(im.shape),tuple(lab.shape))]+=1
            counts += np.bincount(np.asarray(lab)[np.asarray(lab)>=0].astype(int).ravel(),minlength=3)[:3]
        report['split_shapes'][split]={'n':len(ds),'shape_counts':[
            {'image_and_mask_shapes':str(k),'n':v} for k,v in tally.items()], 'label_pixels':counts.tolist()}
    prior=root/'results/exp14_ascd/diagnostics/b2_seed0_d4.json'
    if prior.is_file():
        d=json.loads(prior.read_text());report['baseline_error_breakdown']=d.get('breakdown',{})
    else:
        report['baseline_error_breakdown_note']='Use existing Exp14 diagnosis JSON if available; no fake numbers.'
    out=root/'results/exp16/diagnostics/shape_and_error.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));print('SAVED:',out)
if __name__=='__main__':main()
