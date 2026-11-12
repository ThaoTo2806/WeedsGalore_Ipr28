"""Freeze seed0 PointRend-style arm using Exp15 A's original AMP baseline configuration."""
from pathlib import Path
import json,copy
R = Path(__file__).resolve().parents[2]
p15=R/'configs/exp15/exp15_arms.json'
assert p15.is_file(), 'Missing Exp15 registry; restore from Exp15 archive'
old=json.loads(p15.read_text())
ref=copy.deepcopy(old['arms']['E15_A_CW_LOVASZ'])
ref['seeds']=[0]
ref['stage']='exp16 seed0 val-only gate; test locked'
ref['flags']['exp15_lovasz_weight']=0.0
ref['flags']['exp15_spectral_weight']=0.0
ref['flags']['exp16_point_refine']=True
ref['flags']['exp16_point_budget']=4096
assert ref['flags']['amp'] and ref['flags']['backbone']=='b2'
assert ref['flags']['use_low_level_refine'] and ref['flags']['llr_source']=='raw'
reg={'version':'exp16-point-frozen-v1','description':'One-model B2 + raw LLR + 4096-point residual refine. No test.',
     'arms':{'E16_B2_SPECTRAL_POINT':ref},'rp_bias_grid':old.get('rp_bias_grid',[])}
p=R/'configs/exp16/exp16_arms.json';p.parent.mkdir(parents=True,exist_ok=True)
s=json.dumps(reg,indent=2,ensure_ascii=False)+'\n'
if p.exists():assert p.read_text()==s, 'Frozen Exp16 registry differs; refuse overwrite'
else:p.write_text(s)
print('EXP16 ARMS PASS:',p)
