"""Preflight official training label values before running Exp21. NO val or test reads."""
from pathlib import Path
from PIL import Image
import argparse,collections,json

def audit_dataset(dataset_root, split_file):
    root=Path(dataset_root)
    names=[x.strip() for x in Path(split_file).read_text().splitlines() if x.strip()]
    if len(names)!=104: raise RuntimeError(f'Expected 104 train names, got {len(names)}')
    count=collections.Counter()
    per_img=[]
    for name in names:
        p=root/name[:10]/'semantics'/(name+'.png')
        if not p.is_file(): raise FileNotFoundError(p)
        import numpy as np
        y=np.asarray(Image.open(p))
        if y.ndim != 2 or y.shape!=(600,600): raise RuntimeError(f'Unexpected label shape: {p} {y.shape}')
        u,n=np.unique(y,return_counts=True)
        for v,c in zip(u,n):count[int(v)]+=int(c)
        per_img.append({'name':name,'unique_ids':[int(v) for v in u]})
    ids=sorted(count)
    if ids != [0,1,2,3,4,5]:
        raise RuntimeError(f'Exp21 needs original six IDs 0..5, found {ids}. DO NOT fabricate missing classes.')
    return {'split':'train','images':len(names),'fine_label_pixel_counts':dict(sorted(count.items())),
            'weeds_grouped_as':[2,3,4,5],'test_opened':False,
            'fine_label_mapping_requires_annotation_confirmation':True, 'per_image':per_img}

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset_path',required=True)
    ap.add_argument('--project',required=True)
    a=ap.parse_args();R=Path(a.project)
    p=R/'src/datasets/weedsgalore/splits/train.txt'
    report=audit_dataset(a.dataset_path,p)
    out=R/'results/exp21/diagnostics/train_fine_labels.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print('EXP21 LABEL AUDIT PASS: 104 train images, fine IDs 0..5')
    print('PIXEL COUNTS:',report['fine_label_pixel_counts'])
    print('TEST SEALED; label IDs are not species names until confirmed in official documentation')
