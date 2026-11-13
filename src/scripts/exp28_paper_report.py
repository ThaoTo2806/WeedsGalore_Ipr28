"""Produce paper evidence summary from EXISTING experiment outputs (no model/test access)."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np

COLS=[f'cm{i}{j}' for i in range(3) for j in range(3)]


def load_csv(path):
    with path.open(encoding='utf-8') as f: rows=sorted(csv.DictReader(f),key=lambda r:r['name'])
    if not rows:raise ValueError(f'Empty file {path}')
    return [r['name'] for r in rows],np.array([[int(r[k]) for k in COLS] for r in rows]).reshape(-1,3,3)


def iou(cm):
    cm=np.asarray(cm,dtype=np.float64)
    tp=np.diagonal(cm,axis1=-2,axis2=-1)
    den=cm.sum(-1)+cm.sum(-2)-tp
    return 100*np.divide(tp,den,out=np.zeros_like(tp),where=den>0)


def load_exp27(seed_dir,split,arms):
    result={}
    for arm in ['base']+arms:
        f=seed_dir/f'perimg_exp27_{arm}_{split}_d4_seed0.csv'
        if f.exists():result[arm]=load_csv(f)
    return result


def paired_bootstrap(a,b,draws=10000,seed=2828):
    if a[0]!=b[0]:raise ValueError('Images do not align')
    _,ac=a;_,bc=b
    if len(ac)!=26:raise ValueError('Expected 26 images')
    w=np.random.default_rng(seed).multinomial(len(ac),np.full(len(ac),1/len(ac)),size=draws)
    av=iou(np.einsum('bn,nij->bij',w,ac)).mean(1)
    bv=iou(np.einsum('bn,nij->bij',w,bc)).mean(1)
    d=av-bv
    return {'delta':float(iou(ac.sum(0)).mean()-iou(bc.sum(0)).mean()),
            'ci95':[float(x) for x in np.quantile(d,[.025,.975])],
            'bootstrap_nonpositive_frequency':float((d<=0).mean())}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',default='/kaggle/working/WeedsGalore_Ipr28')
    p.add_argument('--out_dir',default='')
    a=p.parse_args()
    P=Path(a.project)
    out=Path(a.out_dir) if a.out_dir else P/'results/exp28/PAPER_EVIDENCE'
    out.mkdir(parents=True,exist_ok=True)
    e27=P/'results/exp27/E27_CSBRL/seed0'
    d12=P/'results/exp12/PER_IMAGE_AUDIT'
    arms=['control','sbcl','csbrl','stage','random','rgb','ms']
    report={'experiment':'EXP28_EVIDENCE_AUDIT','no_new_test_inference':True,
            'previous_test_exposure':True,'datasets':{},'comparisons':{},'missing':[]}
    for split in ('val','test'):
        xs=load_exp27(e27,split,arms)
        if len(xs)<2:
            report['missing'].append(f'Exp27 {split} per-image CSVs; run exp27 evaluation before report')
            continue
        section={'points':{},'paired_vs_base':{},'paired_vs_control':{},'paired_vs_random':{}}
        for arm,(names,cm) in xs.items():
            section['points'][arm]={'mIoU':float(iou(cm.sum(0)).mean()),
                                    'IoU_by_class':[float(v) for v in iou(cm.sum(0))]}
        for arm in arms:
            if arm not in xs:continue
            section['paired_vs_base'][arm]=paired_bootstrap(xs[arm],xs['base'])
            for comparator in ('control','random'):
                if arm!=comparator and comparator in xs:
                    section['paired_vs_'+comparator][arm]=paired_bootstrap(xs[arm],xs[comparator])
        report['comparisons'][split]=section
    for split,mode in [('val','prob'),('test','logit')]:
        v={}
        for arm in ['P_B2_LLR','P_B0_LLR']:
            records=[];names0=None
            for seed in range(5):
                f=d12/f'perimg_{arm}_{split}_{mode}_seed{seed}.csv'
                if not f.exists():break
                names,cm=load_csv(f)
                if names0 is not None and names!=names0:raise ValueError('Inconsistent image order')
                names0=names;records.append(cm)
            if len(records)==5:
                a5=np.stack(records)
                v[arm]={'mean_mIoU':float(np.mean([iou(c.sum(0)).mean() for c in a5])),
                        'seed_mIoU':[float(iou(c.sum(0)).mean()) for c in a5]}
            else:report['missing'].append(f'Exp12 {split}/{mode} {arm}: need five seed CSVs')
        report['datasets'][split]=v
    (out/'evidence_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    lines=['# Exp28 — Paper evidence audit (generated, NOT proof of novelty)','',
           '**Status:** WeedsGalore TEST was previously opened; all Exp27 TEST comparisons are exploratory.',
           '**External dataset:** WeedMap RedEdge evaluation must be run separately.','']
    for split,sec in report['comparisons'].items():
        lines += [f'## {split.upper()} / D4 / seed 0','',
                  '| Arm | mIoU % | Δ vs base pp | paired 95% CI pp | Δ vs control pp |',
                  '|---|---:|---:|---|---:|']
        for arm,m in sec['points'].items():
            if arm=='base':lines.append(f'| base | {m["mIoU"]:.5f} | — | — | — |');continue
            x=sec['paired_vs_base'][arm]
            y=sec['paired_vs_control'].get(arm,{})
            lines.append(f'| {arm} | {m["mIoU"]:.5f} | {x["delta"]:+.5f} | [{x["ci95"][0]:+.3f}, {x["ci95"][1]:+.3f}] | {y.get("delta",float("nan")):+.5f} |')
        lines += ['','Paired bootstrap resamples 26 images; seed variation not included for Exp27 seed 0.','']
    lines += ['## Scientific interpretation','',
              'No superiority claim for CSBRL unless matched ablations, paired effects, and independent external evaluation support it.',
              'Do not claim SOTA without same dataset, split, label mapping, modality, and evaluation protocol.',
              'Do not choose TEST examples or checkpoints based on TEST performance.','',
              '## Missing evidence','']
    lines += [f'- {x}' for x in report['missing']] or ['- None of the tracked local reports is missing; external validation still required.']
    (out/'PAPER_RESULTS_DRAFT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('EXP28 PAPER AUDIT PASS:',out)
    print('\n'.join(lines[:30]))

if __name__=='__main__':main()
