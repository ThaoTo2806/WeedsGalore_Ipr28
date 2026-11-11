"""Create new Exp14 VBB registry while preserving all original Exp12/ASCD registries."""
from pathlib import Path
import copy, json
ROOT = Path(__file__).resolve().parents[2]
common = json.loads((ROOT/'configs/exp12/exp12_common_config.json').read_text())
ascd_registry = json.loads((ROOT/'configs/exp14/exp14_ascd_arms.json').read_text())
assert common['common']['batch_size'] == 8
ref = copy.deepcopy(ascd_registry['arms']['E14_B2_AMP_REF'])
assert ref['flags']['head'] == 'bilinear' and ref['flags']['backbone'] == 'b2'
assert ref['flags']['llr_source'] == 'raw' and ref['flags']['amp'] is True
assert ref['flags']['ascd_weight'] == 0.0
ref['seeds'] = [0]
ref['flags'].update({'vbb_weight': 0.0, 'vbb_radius': 3})
vbb = copy.deepcopy(ref)
vbb['flags']['vbb_weight'] = 0.1
vbb['stage'] = 'Exp14 VBB seed0 screen (not validated)'
vbb['note'] = 'Training-only vegetation/background boundary BCE; keep test sealed.'
ref['stage'] = 'Exp14 matched AMP control seed0'
reg = {'version':'exp14-vbb-v1', 'description': 'Single-network B2 Raw LLR matched AMP comparison, validation only',
       'arms':{'E14_B2_AMP_REF_VBB':ref, 'E14_B2_VBB':vbb},
       'rp_bias_grid':ascd_registry.get('rp_bias_grid',[])}
p = ROOT/'configs/exp14/exp14_vbb_arms.json'
p.parent.mkdir(exist_ok=True,parents=True)
content=json.dumps(reg,indent=2,ensure_ascii=False)+'\n'
if p.exists():
    assert p.read_text(encoding='utf-8') == content, 'Registry differs: refusing overwrite'
else:
    p.write_text(content,encoding='utf-8')
print('EXP14 VBB ARMS PASS:',p)
