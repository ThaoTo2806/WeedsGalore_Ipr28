"""Exp21 only seed 0 train->val none/D4, never test."""
from pathlib import Path
import sys,json,argparse,subprocess
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'src'))
from exp12_config import load_json,build_cli,eval_cli,check_run_config

def run(cmd):
    print('RUN:', ' '.join(map(str,cmd)),flush=True)
    subprocess.run(list(map(str,cmd)),cwd=R,check=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset_path',required=True)
    args=ap.parse_args();ds=Path(args.dataset_path)
    assert ds.is_dir(),ds
    run([sys.executable,'src/scripts/exp21_label_audit.py',f'--dataset_path={ds}',f'--project={R}'])
    run([sys.executable,'src/scripts/exp21_arms.py'])
    commonp=R/'configs/exp12/exp12_common_config.json'
    armsp=R/'configs/exp21/exp21_arms.json'
    bins=R/'doc/exp02_results/results/exp02/instance_bins.json'
    assert bins.is_file(),bins
    common,arms=load_json(commonp),load_json(armsp)
    arm='E21_B2_HIER';out=R/'results/exp21'/arm/'seed0'
    if out.exists():
        if (out/'seed0_screen.json').is_file():
            print('ALREADY COMPLETED:',(out/'seed0_screen.json').read_text());return
        raise RuntimeError(f'Existing incomplete run: {out}; archive/resume manually, no silent overwrite')
    out.mkdir(parents=True)
    cli=build_cli(common,arms,arm,0,str(ds),str(out),arm+'_s0',str(commonp),str(armsp))
    for required in ('--num_classes=3','--batch_size=8','--amp=True','--exp21_fine_weight=0.2'):
        assert required in cli,required
    run([sys.executable,'src/train_exp21.py',*cli])
    ckpt=out/'final_ema.pth';assert ckpt.is_file(),ckpt
    problems=check_run_config(out/'run_config.json',common,arms)
    assert not problems,problems
    assert load_json(out/'run_config.json')['split_hash']=='675667edf3f35f16'
    results={}
    for tta in ('none','d4'):
        prefix=out/('val_'+tta)
        eval_args=eval_cli(arms,arm,str(ckpt),str(ds),'val',str(bins),str(prefix),arm+'_s0',str(armsp),tta=tta)
        assert '--split=val' in eval_args and '--ccmc=False' in eval_args
        run([sys.executable,'src/evaluate_exp21.py',*eval_args])
        report=load_json(str(prefix)+'_summary.json')
        assert report['split']=='val' and report['n_images']==26 and report['GT_LEAKAGE'] is False
        results[tta]={k:report['metrics'][k] for k in ('mIoU','IoU_bg','IoU_crop','IoU_weed','BIoU_crop','BIoU_weed','recall50_small')}
    d4=results['d4']
    gate=bool(d4['mIoU']>87 and d4['IoU_crop']>80 and d4['IoU_weed']>80)
    record={'arm':arm,'seed':0,'checkpoint':'final_ema.pth','split':'val',
            'tta':'d4','metrics':results,'reserve_gate_mIoU_gt87_crop_gt80_weed_gt80':gate,
            'test_opened':False,'split_hash':'675667edf3f35f16'}
    (out/'seed0_screen.json').write_text(json.dumps(record,indent=2)+'\n')
    print('EXP21 SEED0 SCREEN:',json.dumps(record,indent=2))
    print('TEST SEALED.')
if __name__=='__main__':main()
