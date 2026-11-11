"""EXP14 seed0 only. VALIDATION ONLY. Never accesses test split."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from exp12_config import load_json, build_cli, eval_cli, check_run_config


def run(cmd):
    print('RUN:', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', required=True)
    ap.add_argument('--baseline_ckpt', required=True,
                    help='Exp12 B2 LLR seed0 final_ema.pth (for validation diagnosis)')
    a = ap.parse_args()
    common_file = ROOT/'configs/exp12/exp12_common_config.json'
    arms_file = ROOT/'configs/exp14/exp14_ascd_arms.json'
    bins = ROOT/'doc/exp02_results/results/exp02/instance_bins.json'
    common, arms = load_json(common_file), load_json(arms_file)
    arm, seed = 'E14_B2_ASCD', 0
    out = ROOT/'results/exp14_ascd'/arm/'seed0'
    diag = ROOT/'results/exp14_ascd/diagnostics/b2_seed0_d4.json'
    assert common['common']['batch_size'] == 8
    assert arms['arms'][arm]['flags']['amp'] is True
    assert arms['arms'][arm]['flags']['ascd_weight'] == .15
    assert Path(a.baseline_ckpt).is_file(), 'Baseline checkpoint missing: '+a.baseline_ckpt
    assert bins.is_file(), 'Instance bins missing: '+str(bins)
    assert not out.exists(), f'Refusing to overwrite existing run: {out}'
    diag.parent.mkdir(parents=True, exist_ok=True)
    run([sys.executable, 'src/scripts/exp14_diagnose.py',
        '--dataset_path='+a.dataset_path, '--ckpt='+a.baseline_ckpt,
        '--tta=d4', '--boundary_px=3', '--out='+str(diag)])
    report = load_json(diag)
    assert report['split']=='val' and report['n_images']==26
    print('DIAG CROP-WEED ERROR SHARE:', report['breakdown']['all']['crop_weed_share_of_errors_pct'], '%')
    print('DIAG BOUNDARY ERROR SHARE:', report['border_share_of_all_errors_pct'], '%')
    # Diagnose first. Do not use diagnosis to tune an arm; registered config stays frozen.
    out.mkdir(parents=True)
    cli = build_cli(common, arms, arm, seed, a.dataset_path, str(out),
                    arm+'_s0', str(common_file), str(arms_file))
    assert '--batch_size=8' in cli and '--amp=True' in cli
    assert '--ascd_weight=0.15' in cli
    assert '--head=bilinear' in cli and '--backbone=b2' in cli
    run([sys.executable, 'src/train.py', *cli])
    ckpt = out/'final_ema.pth'
    assert ckpt.is_file(), 'final_ema missing'
    issues = check_run_config(out/'run_config.json', common, arms)
    assert not issues, issues
    results = {}
    for tta in ('none','d4'):
        prefix = out/f'val_{tta}'
        cli = eval_cli(arms, arm, str(ckpt), a.dataset_path, 'val', str(bins),
                       str(prefix), arm+'_s0', str(arms_file), tta=tta)
        assert '--split=val' in cli and '--tta='+tta in cli
        run([sys.executable, 'src/evaluate_ccmc.py', *cli])
        summary = load_json(str(prefix)+'_summary.json')
        assert summary['split']=='val' and summary['n_images']==26
        assert summary['GT_LEAKAGE'] is False
        results[tta] = {k:summary['metrics'][k] for k in
                        ('mIoU','IoU_bg','IoU_crop','IoU_weed','BIoU_crop','BIoU_weed')}
    m = results['d4']
    verdict = {
        'arm':arm, 'seed':0, 'split':'val', 'checkpoint':'final_ema.pth',
        'tta':'d4', 'threshold_mIoU_exclusive':86.0,
        'seed0_val_mIoU':m['mIoU'], 'seed0_crop_IoU':m['IoU_crop'],
        'seed0_weed_IoU':m['IoU_weed'],
        'screen_pass':bool(m['mIoU']>86.0),
        'may_open_test':False,
        'next': ('run matched B2 AMP control then dev seeds1-4 and ablations; keep test sealed'
                 if m['mIoU']>86 else 'stop ASCD expansion; inspect validation error breakdown'),
        'all_validation_metrics':results,
        'registry_sha256':hashlib.sha256(arms_file.read_bytes()).hexdigest(),
    }
    (out/'seed0_screen.json').write_text(json.dumps(verdict,indent=2)+'\n')
    print('ASCD SEED0 RESULT:',json.dumps(verdict,indent=2),flush=True)

if __name__=='__main__':
    main()
