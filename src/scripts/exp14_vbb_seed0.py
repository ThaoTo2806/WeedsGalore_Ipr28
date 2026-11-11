"""One-armed experiment. No test split. Use --arm E14_B2_AMP_REF_VBB first, then E14_B2_VBB."""
import argparse, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from exp12_config import load_json, build_cli, eval_cli, check_run_config

def run(args):
    print('RUN:', ' '.join(map(str,args)),flush=True)
    subprocess.run([str(s) for s in args],cwd=ROOT,check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset_path',required=True)
    ap.add_argument('--arm',choices=['E14_B2_AMP_REF_VBB','E14_B2_VBB'],required=True)
    a=ap.parse_args()
    cp=ROOT/'configs/exp12/exp12_common_config.json'
    apath=ROOT/'configs/exp14/exp14_vbb_arms.json'
    bp=ROOT/'doc/exp02_results/results/exp02/instance_bins.json'
    common,arms=load_json(cp),load_json(apath)
    assert Path(a.dataset_path).is_dir() and bp.is_file()
    out=ROOT/'results/exp14_vbb'/a.arm/'seed0'
    assert not out.exists(),f'Refusing overwrite: {out}'
    out.mkdir(parents=True)
    cli=build_cli(common,arms,a.arm,0,a.dataset_path,str(out),a.arm+'_s0',str(cp),str(apath))
    assert '--batch_size=8' in cli and '--amp=True' in cli and '--head=bilinear' in cli
    run([sys.executable,'src/train.py',*cli])
    ckpt=out/'final_ema.pth'
    assert ckpt.is_file()
    issues=check_run_config(out/'run_config.json',common,arms)
    assert not issues,issues
    result={}
    for tta in ('none','d4'):
        pref=out/('val_'+tta)
        ev=eval_cli(arms,a.arm,str(ckpt),a.dataset_path,'val',str(bp),str(pref),a.arm+'_s0',str(apath),tta=tta)
        assert '--split=val' in ev
        run([sys.executable,'src/evaluate_ccmc.py',*ev])
        sm=load_json(str(pref)+'_summary.json')
        assert sm['split']=='val' and sm['n_images']==26 and sm['GT_LEAKAGE'] is False
        result[tta]={k:sm['metrics'][k] for k in ('mIoU','IoU_bg','IoU_crop','IoU_weed','BIoU_crop','BIoU_weed')}
    verdict={'arm':a.arm,'seed':0,'checkpoint':'final_ema.pth','split':'val','evaluation':result,
             'screen_gt86':result['d4']['mIoU']>86.0,'may_open_test':False}
    (out/'seed0_screen.json').write_text(json.dumps(verdict,indent=2)+'\n')
    print('EXP14 VBB SEED0 RESULT:',json.dumps(verdict,indent=2),flush=True)
if __name__=='__main__':main()
