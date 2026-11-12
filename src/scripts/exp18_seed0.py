"""Run only seed0 on validation. Never open test. No silent overwrite."""
from pathlib import Path
import argparse,sys,subprocess,json
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'src'))
from exp12_config import load_json,build_cli,eval_cli,check_run_config

def run(cmd):
    print('RUN:', ' '.join(map(str,cmd)),flush=True)
    subprocess.run(list(map(str,cmd)),cwd=R,check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset_path',required=True);a=p.parse_args()
    ds=Path(a.dataset_path);assert ds.is_dir(),ds
    common_p=R/'configs/exp12/exp12_common_config.json'
    arms_p=R/'configs/exp18/exp18_arms.json'
    bins=R/'doc/exp02_results/results/exp02/instance_bins.json'
    assert common_p.is_file() and arms_p.is_file() and bins.is_file()
    common,arms=load_json(common_p),load_json(arms_p)
    arm='E18_B2_OBJECT_CURRICULUM';out=R/'results/exp18'/arm/'seed0'
    if out.exists():
        assert (out/'seed0_screen.json').is_file(), ('Existing incomplete run: '+str(out)+ '; inspect/archive, no silent retrain')
        print('EXISTING COMPLETED RUN:',(out/'seed0_screen.json').read_text());return
    out.mkdir(parents=True)
    cli=build_cli(common,arms,arm,0,str(ds),str(out),arm+'_s0',str(common_p),str(arms_p))
    for flag in ('--exp18_curriculum=True','--exp16_point_refine=False','--exp17_safety_gate=False','--batch_size=8','--amp=True'):
        assert flag in cli,flag
    run([sys.executable,'src/train.py',*cli])
    ckpt=out/'final_ema.pth';assert ckpt.is_file(),ckpt
    errors=check_run_config(out/'run_config.json',common,arms)
    assert not errors,errors
    metrics={}
    for tta in ('none','d4'):
        prefix=out/('val_'+tta)
        args=eval_cli(arms,arm,str(ckpt),str(ds),'val',str(bins),str(prefix),arm+'_s0',str(arms_p),tta=tta)
        assert '--split=val' in args and '--ccmc=False' in args
        args+=['--exp16_point_refine=False','--exp17_safety_gate=False']
        run([sys.executable,'src/evaluate_ccmc.py',*args])
        summary=load_json(str(prefix)+'_summary.json')
        assert summary['split']=='val' and summary['n_images']==26 and summary['GT_LEAKAGE'] is False
        metrics[tta]={k:summary['metrics'][k] for k in ('mIoU','IoU_bg','IoU_crop','IoU_weed','BIoU_crop','BIoU_weed','recall50_small')}
    d4=metrics['d4']
    gate=bool(d4['mIoU']>87 and d4['IoU_crop']>80 and d4['IoU_weed']>80)
    record={'arm':arm,'seed':0,'checkpoint':'final_ema.pth','split':'val','tta':'d4',
            'metrics':metrics,'reserve_gate_mIoU_gt87_crop_gt80_weed_gt80':gate,
            'may_open_test':False,'split_hash':'675667edf3f35f16'}
    (out/'seed0_screen.json').write_text(json.dumps(record,indent=2)+'\n')
    print('EXP18 SEED0 SCREEN:',json.dumps(record,indent=2),flush=True)
    print('TEST SEALED. Confirmation seeds 1-4 and 5-9 required before one-time test.',flush=True)
if __name__=='__main__': main()
