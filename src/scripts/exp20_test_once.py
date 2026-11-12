"""ONE-TIME exploratory test for Exp20. Irreversible loss of clean held-out status.
Requires explicit --accept_exploratory_test flag. Refuses overwrite and split mismatch.
"""
import argparse, json, subprocess, sys
from pathlib import Path
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'src'))
from exp12_config import load_json, eval_cli

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset_path',required=True)
    p.add_argument('--accept_exploratory_test',action='store_true')
    a=p.parse_args()
    if not a.accept_exploratory_test:
        raise SystemExit('Refusing TEST: explicitly pass --accept_exploratory_test; future test will no longer be independent.')
    ds=Path(a.dataset_path); assert ds.is_dir(),ds
    arm='E20_B2_ICR';out=R/'results/exp20'/arm/'seed0'
    screen=load_json(out/'seed0_screen.json')
    assert screen['split']=='val' and screen['split_hash']=='675667edf3f35f16'
    ckpt=out/'final_ema.pth';assert ckpt.is_file()
    prefix=out/'exploratory_test_d4'
    # Refuse if ANY files indicate a previous evaluation attempt.
    assert not list(out.glob('exploratory_test*')), 'Test already attempted; never repeat it.'
    assert not (out/'test_opened_marker.json').exists()
    arms_path=R/'configs/exp20/exp20_arms.json'; arms=load_json(arms_path)
    args=eval_cli(arms,arm,str(ckpt),str(ds),'test',str(R/'doc/exp02_results/results/exp02/instance_bins.json'),str(prefix),arm+'_s0',str(arms_path),tta='d4')
    args+=['--exp16_point_refine=False','--exp17_safety_gate=False','--exp20_icr=True','--confirm_test_locked=True']
    assert '--split=test' in args
    # Marker before evaluation, to prevent repeats even when the run crashes.
    (out/'test_opened_marker.json').write_text(json.dumps({'status':'exploratory_test_opened','seed':0,'tta':'d4','warning':'NOT AN INDEPENDENT FINAL TEST IF USED TO INFORM DEVELOPMENT'},indent=2)+'\n')
    cmd=[sys.executable,'src/evaluate_exp20.py',*args]
    print('ONE-TIME EXPLORATORY TEST:', ' '.join(cmd),flush=True)
    subprocess.run(cmd,cwd=R,check=True)
    summary=load_json(str(prefix)+'_summary.json')
    assert summary['split']=='test' and summary['n_images']==26 and summary['GT_LEAKAGE'] is False
    print('EXPLORATORY TEST RESULT (NOT FOR FUTURE MODEL SELECTION):',json.dumps(summary['metrics'],indent=2))
if __name__=='__main__':main()
