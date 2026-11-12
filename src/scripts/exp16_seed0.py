"""Exp16 seed0 validation screen. NEVER evaluates test. No implicit multi-seed expansion."""
import argparse,json,sys,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'src'))
from exp12_config import load_json, build_cli, eval_cli, check_run_config

def run(args):
    print('RUN:',' '.join(map(str,args)),flush=True)
    subprocess.run([str(v) for v in args],cwd=R,check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset_path',required=True);a=p.parse_args()
    ds=Path(a.dataset_path);assert ds.is_dir(),ds
    common_p=R/'configs/exp12/exp12_common_config.json'
    arms_p=R/'configs/exp16/exp16_arms.json'
    bins=R/'doc/exp02_results/results/exp02/instance_bins.json'
    assert common_p.is_file() and arms_p.is_file() and bins.is_file()
    common,arms=load_json(common_p),load_json(arms_p)
    arm='E16_B2_SPECTRAL_POINT';out=R/'results/exp16'/arm/'seed0'
    if out.exists():
        assert (out/'seed0_screen.json').is_file(),('Existing incomplete run: '+str(out)+
               '; inspect / resume manually, not overwrite or retrain silently')
        print('REUSING:',(out/'seed0_screen.json').read_text());return
    out.mkdir(parents=True)
    cli=build_cli(common,arms,arm,0,str(ds),str(out),arm+'_s0',str(common_p),str(arms_p))
    assert '--exp16_point_refine=True' in cli and '--exp16_point_budget=4096' in cli
    assert '--batch_size=8' in cli and '--amp=True' in cli
    run([sys.executable,'src/train.py',*cli])
    ckpt=out/'final_ema.pth';assert ckpt.is_file(),'Missing final_ema.pth'
    errors=check_run_config(out/'run_config.json',common,arms)
    assert not errors,errors
    metrics={}
    for tta in ('none','d4'):
        prefix=out/('val_'+tta)
        evalargs=eval_cli(arms,arm,str(ckpt),str(ds),'val',str(bins),str(prefix),arm+'_s0',str(arms_p),tta=tta)
        assert '--split=val' in evalargs and '--ccmc=False' in evalargs
        evalargs.append('--exp16_point_refine=True')
        run([sys.executable,'src/evaluate_ccmc.py',*evalargs])
        summary=load_json(str(prefix)+'_summary.json')
        assert summary['split']=='val' and summary['n_images']==26 and summary['GT_LEAKAGE'] is False
        metrics[tta]={k:summary['metrics'][k] for k in ('mIoU','IoU_bg','IoU_crop','IoU_weed','BIoU_crop','BIoU_weed','recall50_small')}
    d4=metrics['d4']
    gate=bool(d4['mIoU']>87 and d4['IoU_crop']>80 and d4['IoU_weed']>80)
    record={'arm':arm,'seed':0,'checkpoint':'final_ema.pth','split':'val','tta':'d4',
            'metrics':metrics,'reserve_gate_mIoU_gt87_crop_gt80_weed_gt80':gate,
            'may_open_test':False,'split_hash':'675667edf3f35f16'}
    (out/'seed0_screen.json').write_text(json.dumps(record,indent=2)+'\n')
    print('EXP16 SEED0 SCREEN:',json.dumps(record,indent=2),flush=True)
    print('TEST SEALED. Even if gate passes, run seeds 1-4 and 5-9 first.',flush=True)
if __name__=='__main__':main()
