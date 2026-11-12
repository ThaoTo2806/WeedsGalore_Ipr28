"""Freeze Exp19 single-model arm, retaining the Exp12 shared baseline protocol."""
from pathlib import Path
import json, copy, hashlib
R=Path(__file__).resolve().parents[2]
refp=R/'configs/exp18/exp18_arms.json'
assert refp.is_file(),f'Missing Exp18 registry: {refp}'
prior=json.loads(refp.read_text())
ref=copy.deepcopy(prior['arms']['E18_B2_OBJECT_CURRICULUM'])
f=ref['flags']
assert f['backbone']=='b2' and f['amp'] is True and f['llr_source']=='raw'
assert f['use_low_level_refine'] is True
for name in ('exp16_point_refine','exp17_safety_gate','exp18_curriculum'):
    f[name]=False
for name in ('ascd_weight','vbb_weight','exp15_lovasz_weight','exp15_spectral_weight','exp17_safety_weight'):
    if name in f: f[name]=0.0
f['exp19_ssl_encoder_ckpt']=str(R/'results/exp19/ssl_seed0/encoder_ssl.pth')
ref['seeds']=[0]
ref['stage']='Exp19 pretrained single MiT-B2, 100ep supervised, val only'
ref['allowed_tta']=['none','d4']
p=R/'configs/exp19/exp19_arms.json';p.parent.mkdir(parents=True,exist_ok=True)
data={'version':'exp19-s3a-frozen-v1',
      'description':'One B2 raw LLR, SSL train-only 104 images then supervised 100ep; no teacher, no ensemble, test sealed',
      'arms':{'E19_B2_S3A':ref},'rp_bias_grid':[]}
s=json.dumps(data,indent=2,ensure_ascii=False)+'\n'
if p.exists():
    assert p.read_text()==s,'Different Exp19 frozen registry exists; refuse overwrite'
else:p.write_text(s)
print('EXP19 ARMS PASS:',p)
