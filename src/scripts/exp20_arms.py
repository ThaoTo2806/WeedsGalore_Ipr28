"""Create frozen Exp20 registry from Exp18 arm (all other experiment switches OFF)."""
import json, copy
from pathlib import Path
R=Path(__file__).resolve().parents[2]
refp=R/'configs/exp18/exp18_arms.json'
assert refp.is_file(), 'Missing Exp18 registry; restore previous Exp18 Input first'
prior=json.loads(refp.read_text())
ref=copy.deepcopy(prior['arms']['E18_B2_OBJECT_CURRICULUM'])
f=ref['flags']
for k in ('exp16_point_refine','exp17_safety_gate','exp18_curriculum'):
    f[k]=False
for k in ('ascd_weight','vbb_weight','exp15_lovasz_weight','exp15_spectral_weight','exp17_safety_weight'):
    if k in f:f[k]=0.0
f['exp20_icr']=True
f.pop('exp19_ssl_encoder_ckpt',None)
assert f['backbone']=='b2' and f['llr_source']=='raw' and f['use_low_level_refine'] is True and f['amp'] is True
ref['seeds']=[0]
ref['stage']='Exp20 one single SegFormer-B2 + ICR module, test exploratory if explicitly authorized'
ref['allowed_tta']=['none','d4']
p=R/'configs/exp20/exp20_arms.json';p.parent.mkdir(parents=True,exist_ok=True)
obj={'version':'exp20-icr-1','description':'Single B2 + ICR, train 104; validation 26; test only explicit separate run','arms':{'E20_B2_ICR':ref},'rp_bias_grid':[]}
s=json.dumps(obj,indent=2,ensure_ascii=False)+'\n'
if p.exists():assert p.read_text()==s,'Frozen registry differs; refusing overwrite'
else:p.write_text(s)
print('EXP20 ARMS PASS:',p)
