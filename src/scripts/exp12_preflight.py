"""Fast, torch-free Exp12 guard. Run before any Kaggle GPU training."""
import argparse, json, pathlib, sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from exp12_config import load_json, expected_flags, sha256_file

def inspect(common, arms):
    issues=[]
    if common.get('common',{}).get('num_classes')!=3: issues.append('Expected three classes')
    if common['common'].get('batch_size')!=8: issues.append('Exp10 effective batch is eight')
    a=arms.get('arms',{})
    if not a: issues.append('No arms defined')
    for arm_id,arm in a.items():
        flags=arm.get('flags',{})
        try: expected_flags(common,arms,arm_id)
        except Exception as e: issues.append(f'{arm_id}: {e}')
        if arm_id.startswith('S') and arm_id.split('_')[0] in ('S0','S1','S2'):
            if flags.get('use_low_level_refine') is not False:
                issues.append(f'{arm_id}: historical S* MUST be no-LLR; do not silently rename')
        if arm_id.startswith('L12_') and flags.get('use_low_level_refine') is not True:
            issues.append(f'{arm_id}: declared RAW LLR arm must enable LLR')
        if flags.get('head') in ('occ_conditioned2','mcsd2') and flags.get('spectral_guide',False):
            issues.append(f'{arm_id}: LLR subcell does not support spectral_guide=True')
    return issues

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--common',default='configs/exp12/exp12_common_config.json')
    p.add_argument('--arms',default='configs/exp12/exp12_arms.json')
    a=p.parse_args()
    c=load_json(a.common);r=load_json(a.arms)
    issues=inspect(c,r)
    print(json.dumps({'status':'PASS_PREFLIGHT' if not issues else 'FAIL', 'arm_count':len(r['arms']),
      'common_sha256':sha256_file(a.common),'arms_sha256':sha256_file(a.arms),'issues':issues},indent=2))
    raise SystemExit(1 if issues else 0)
if __name__=='__main__':main()
