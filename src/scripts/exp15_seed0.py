"""Validated protocol: final_ema, val only, none and D4, all metrics and gate frozen."""
import sys,argparse,subprocess,json
from pathlib import Path
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'src'))
from exp12_config import load_json,build_cli,eval_cli,check_run_config

ARMS=('E15_A_CW_LOVASZ','E15_B_SPEC_SPATIAL')

def run(command):
    print('RUN:', ' '.join(map(str,command)),flush=True)
    subprocess.run([str(x) for x in command],cwd=R,check=True)

def run_arm(arm,dataset,common,arms,common_p,arms_p,bins):
    out=R/'results/exp15'/arm/'seed0'
    if out.exists():
        p=out/'seed0_screen.json'
        assert p.is_file(), f'Incomplete previous run: {out}. Inspect manually; no auto overwrite.'
        record=load_json(p)
        assert record['arm']==arm and record['split']=='val'
        print('REUSE COMPLETED RUN:',p,flush=True)
        return record
    out.mkdir(parents=True,exist_ok=False)
    cli=build_cli(common,arms,arm,0,str(dataset),str(out),arm+'_s0',str(common_p),str(arms_p))
    assert '--amp=True' in cli and '--batch_size=8' in cli and '--head=bilinear' in cli
    run([sys.executable,'src/train.py',*cli])
    ckpt=out/'final_ema.pth'
    assert ckpt.is_file(),'Missing final_ema.pth'
    errors=check_run_config(out/'run_config.json',common,arms)
    assert not errors,errors
    metrics={}
    for tta in ('none','d4'):
        prefix=out/('val_'+tta)
        args=eval_cli(arms,arm,str(ckpt),str(dataset),'val',str(bins),str(prefix),arm+'_s0',str(arms_p),tta=tta)
        assert '--split=val' in args and f'--tta={tta}' in args
        run([sys.executable,'src/evaluate_ccmc.py',*args])
        sm=load_json(str(prefix)+'_summary.json')
        assert sm['split']=='val' and sm['n_images']==26 and sm['GT_LEAKAGE'] is False
        metrics[tta]={k:sm['metrics'][k] for k in ('mIoU','IoU_bg','IoU_crop','IoU_weed','BIoU_crop','BIoU_weed')}
    d4=metrics['d4']
    gate=all((d4['mIoU']>86,d4['IoU_crop']>80,d4['IoU_weed']>80))
    result={'arm':arm,'seed':0,'split':'val','checkpoint':'final_ema.pth','tta':'d4','metrics':metrics,
            'gate_mIoU_gt86_crop_gt80_weed_gt80':gate,'may_open_test':False}
    (out/'seed0_screen.json').write_text(json.dumps(result,indent=2)+'\n')
    print('EXP15 SEED0 RESULT:',json.dumps(result,indent=2),flush=True)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset_path',required=True)
    ap.add_argument('--mode',choices=['auto','a','b','both'],default='auto')
    a=ap.parse_args()
    dataset=Path(a.dataset_path)
    common_p=R/'configs/exp12/exp12_common_config.json'
    arms_p=R/'configs/exp15/exp15_arms.json'
    bins=R/'doc/exp02_results/results/exp02/instance_bins.json'
    assert dataset.is_dir() and common_p.is_file() and arms_p.is_file() and bins.is_file()
    common, arms=load_json(common_p),load_json(arms_p)
    if a.mode=='a':todo=ARMS[:1]
    elif a.mode=='b':todo=ARMS[1:]
    else:todo=ARMS
    for arm in todo:
        result=run_arm(arm,dataset,common,arms,common_p,arms_p,bins)
        if a.mode=='auto' and result['gate_mIoU_gt86_crop_gt80_weed_gt80']:
            print('A PASSED PRE-REGISTERED SEED0 GATE; B SKIPPED; CONFIRM MORE SEEDS BEFORE TEST.')
            break
    print('EXP15 RUN FINISHED; TEST SEALED',flush=True)
if __name__=='__main__':main()
