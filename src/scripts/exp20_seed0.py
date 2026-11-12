"""Exp20 single-model seed0 training then validation none/D4. Test separate explicit command."""
from pathlib import Path
import argparse,json,subprocess,sys
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'src'))
from exp12_config import load_json,build_cli,eval_cli,check_run_config


def run(cmd):
    print('RUN:', ' '.join(map(str,cmd)),flush=True)
    subprocess.run(list(map(str,cmd)),cwd=R,check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset_path',required=True)
    a=p.parse_args();ds=Path(a.dataset_path)
    assert ds.is_dir(),ds
    assert (R/'src/train_exp20.py').is_file()
    assert (R/'src/nets/exp20_icr.py').is_file()
    run([sys.executable,'src/scripts/exp20_arms.py'])
    commonp=R/'configs/exp12/exp12_common_config.json'
    armsp=R/'configs/exp20/exp20_arms.json'
    bins=R/'doc/exp02_results/results/exp02/instance_bins.json'
    assert commonp.is_file() and bins.is_file()
    common,arms=load_json(commonp),load_json(armsp)
    arm='E20_B2_ICR';out=R/'results/exp20'/arm/'seed0'
    if out.exists():
        assert (out/'seed0_screen.json').is_file(),f'Incomplete Exp20 run {out}; archive/resume manually'
        print('EXISTING COMPLETED SCREEN:',(out/'seed0_screen.json').read_text());return
    out.mkdir(parents=True)
    cli=build_cli(common,arms,arm,0,str(ds),str(out),arm+'_s0',str(commonp),str(armsp))
    for flag in ('--batch_size=8','--amp=True','--exp16_point_refine=False',
                 '--exp17_safety_gate=False','--exp18_curriculum=False',
                 '--exp20_icr=True'):
        assert flag in cli,flag
    run([sys.executable,'src/train_exp20.py',*cli])
    ckpt=out/'final_ema.pth';assert ckpt.is_file(),ckpt
    problems=check_run_config(out/'run_config.json',common,arms)
    assert not problems,problems
    cfg=load_json(out/'run_config.json')
    assert cfg['split_hash']=='675667edf3f35f16',cfg['split_hash']
    results={}
    for tta in ('none','d4'):
        prefix=out/('val_'+tta)
        args=eval_cli(arms,arm,str(ckpt),str(ds),'val',str(bins),str(prefix),arm+'_s0',str(armsp),tta=tta)
        assert '--split=val' in args and '--ccmc=False' in args
        args+=['--exp16_point_refine=False','--exp17_safety_gate=False','--exp20_icr=True']
        run([sys.executable,'src/evaluate_exp20.py',*args])
        summary=load_json(str(prefix)+'_summary.json')
        assert summary['split']=='val' and summary['n_images']==26 and summary['GT_LEAKAGE'] is False
        results[tta]={k:summary['metrics'][k] for k in ('mIoU','IoU_bg','IoU_crop','IoU_weed','BIoU_crop','BIoU_weed','recall50_small')}
    d4=results['d4']
    gate=bool(d4['mIoU']>87 and d4['IoU_crop']>80 and d4['IoU_weed']>80)
    record={'arm':arm,'seed':0,'checkpoint':'final_ema.pth',
            'split':'val','tta':'d4','metrics':results,
            'reserve_gate_mIoU_gt87_crop_gt80_weed_gt80':gate,
            'test_not_run_by_this_script':True,'split_hash':'675667edf3f35f16'}
    (out/'seed0_screen.json').write_text(json.dumps(record,indent=2)+'\n')
    print('EXP20 SEED0 SCREEN:',json.dumps(record,indent=2),flush=True)
    print('TEST NOT RUN. Use explicit exp20_test_once.py separately if you intentionally open test.',flush=True)

if __name__=='__main__':main()
