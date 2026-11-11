"""Register ASCD seed-0 screen and later ablations without modifying Exp12."""
from pathlib import Path
import copy
import json

ROOT = Path(__file__).resolve().parents[2]
common = json.loads((ROOT/'configs/exp12/exp12_common_config.json').read_text())
original = json.loads((ROOT/'configs/exp12/exp12_arms.json').read_text())
assert common['common']['batch_size'] == 8
base = original['arms']['P_B2_LLR']
assert base['flags']['backbone'] == 'b2' and base['flags']['head'] == 'bilinear'
assert base['flags']['llr_source'] == 'raw'
assert 'amp' not in common['common']

def make(weight, hard=1.0, spectral=1.0, radius=3):
    arm = copy.deepcopy(base)
    arm['branch'] = 'EXP14-ASCD-EXPLORATORY'
    arm['stage'] = 'E14-SEED0-SCREEN'
    arm['flags'].update({
        'amp': True,
        'ascd_weight': weight,
        'ascd_hard_alpha': hard,
        'ascd_spectral_alpha': spectral,
        'ascd_spectral_temperature': 1.0,
        'ascd_interface_radius': radius,
    })
    arm['seeds'] = list(range(10))
    arm['allowed_tta'] = ['none', 'd4']
    arm['note'] = 'B2 RawLLR AMP batch8; same network at inference; test remains locked'
    return arm

arms = {
    'E14_B2_AMP_REF': make(0.0),
    'E14_B2_AC': make(0.15, spectral=0.0, radius=0),
    'E14_B2_AC_BI': make(0.15, spectral=0.0, radius=3),
    'E14_B2_ASCD': make(0.15, spectral=1.0, radius=3),
}
registry = {'version': 'exp14-ascd-v1', 'description':
    'ASCD seed0 validation screen, later multi-seed confirmation and ablation; never test for selection.',
    'rp_bias_grid': original.get('rp_bias_grid', []), 'arms': arms}
p = ROOT/'configs/exp14/exp14_ascd_arms.json'
p.parent.mkdir(parents=True, exist_ok=True)
s = json.dumps(registry, indent=2, ensure_ascii=False) + '\n'
if p.exists():
    assert p.read_text(encoding='utf-8') == s, 'Existing registry differs; refusing overwrite'
else:
    p.write_text(s, encoding='utf-8')
print('ASCD ARMS PASS:', p)
