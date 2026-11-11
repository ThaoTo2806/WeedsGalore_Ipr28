"""Freeze exactly two training-only Exp15 arms using existing Exp14 VBB AMP baseline."""
from pathlib import Path
import json,copy
R=Path(__file__).resolve().parents[2]
prior=R/'configs/exp14/exp14_vbb_arms.json'
assert prior.is_file(), 'Missing Exp14 VBB registry; restore from Exp14 ZIP first'
old=json.loads(prior.read_text())
ref=old['arms']['E14_B2_AMP_REF_VBB']
assert ref['flags']['amp'] is True and ref['flags']['backbone']=='b2'
assert ref['flags']['llr_source']=='raw' and ref['flags']['vbb_weight']==0
assert ref['flags']['ascd_weight']==0
arms={}
for name in ('E15_A_CW_LOVASZ','E15_B_SPEC_SPATIAL'):
    spec=copy.deepcopy(ref)
    spec['seeds']=[0]
    spec['stage']='Exp15 seed0 validation-only screen; no test'
    spec['flags'].update({
        'exp15_lovasz_weight':0.15 if name.endswith('LOVASZ') else 0.0,
        'exp15_spectral_weight':0.0 if name.endswith('LOVASZ') else 0.10,
        'exp15_spectral_scale':4,'exp15_spectral_temperature':0.8,
        'exp15_spectral_radius':2})
    arms[name]=spec
reg={'version':'exp15-seed0-frozen-v1','description':'Two independent training-only CW losses; B conditional upon A failure. One B2 model per arm.','arms':arms,'rp_bias_grid':old.get('rp_bias_grid',[])}
p=R/'configs/exp15/exp15_arms.json'
p.parent.mkdir(parents=True,exist_ok=True)
content=json.dumps(reg,indent=2,ensure_ascii=False)+'\n'
if p.exists(): assert p.read_text()==content, 'Registry differs; refusing overwrite'
else: p.write_text(content)
print('EXP15 ARMS PASS:',p)
