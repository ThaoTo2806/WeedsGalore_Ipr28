"""Stage A SSL, then Stage B supervised 100 epochs, then val-only none/D4."""
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
    assert (R/'src/train_exp19.py').is_file()
    assert (R/'src/exp19_ssl.py').is_file()
    ssl=R/'results/exp19/ssl_seed0/encoder_ssl.pth'
    if not ssl.is_file():
        run([sys.executable,'src/scripts/exp19_pretrain.py',f'--dataset_path={ds}'])
    assert ssl.is_file(),ssl
    run([sys.executable,'src/scripts/exp19_arms.py'])
    commonp=R/'configs/exp12/exp12_common_config.json'
    armsp=R/'configs/exp19/exp19_arms.json'
    bins=R/'doc/exp02_results/results/exp02/instance_bins.json'
    assert commonp.is_file() and bins.is_file()
    common,arms=load_json(commonp),load_json(armsp)
    arm='E19_B2_S3A';out=R/'results/exp19'/arm/'seed0'
    if out.exists():
        assert (out/'seed0_screen.json').is_file(),f'Incomplete Exp19 run {out}; archive/resume manually'
        print('EXISTING COMPLETED SCREEN:',(out/'seed0_screen.json').read_text());return
    out.mkdir(parents=True)
    cli=build_cli(common,arms,arm,0,str(ds),str(out),arm+'_s0',str(commonp),str(armsp))
    for flag in ('--batch_size=8','--amp=True','--exp16_point_refine=False',
                 '--exp17_safety_gate=False','--exp18_curriculum=False',
                 f'--exp19_ssl_encoder_ckpt={ssl}'):
        assert flag in cli,flag
    run([sys.executable,'src/train_exp19.py',*cli])
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
        args+=['--exp16_point_refine=False','--exp17_safety_gate=False']
        run([sys.executable,'src/evaluate_ccmc.py',*args])
        summary=load_json(str(prefix)+'_summary.json')
        assert summary['split']=='val' and summary['n_images']==26 and summary['GT_LEAKAGE'] is False
        results[tta]={k:summary['metrics'][k] for k in ('mIoU','IoU_bg','IoU_crop','IoU_weed','BIoU_crop','BIoU_weed','recall50_small')}
    d4=results['d4']
    gate=bool(d4['mIoU']>87 and d4['IoU_crop']>80 and d4['IoU_weed']>80)
    record={'arm':arm,'seed':0,'ssl_checkpoint':'encoder_ssl.pth','checkpoint':'final_ema.pth',
            'split':'val','tta':'d4','metrics':results,
            'reserve_gate_mIoU_gt87_crop_gt80_weed_gt80':gate,
            'may_open_test':False,'split_hash':'675667edf3f35f16'}
    (out/'seed0_screen.json').write_text(json.dumps(record,indent=2)+'\n')
    print('EXP19 SEED0 SCREEN:',json.dumps(record,indent=2),flush=True)
    print('TEST SEALED. No additional seeds or test automatically.',flush=True)

if __name__=='__main__':main()
