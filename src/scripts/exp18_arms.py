"""Freeze a single baseline B2 arm, one model, train-only curriculum."""
from pathlib import Path
import copy,json
R=Path(__file__).resolve().parents[2]
p=R/'configs/exp16/exp16_arms.json'
assert p.is_file(),f'Missing frozen reference: {p}'
prior=json.loads(p.read_text())
ref=copy.deepcopy(prior['arms']['E16_B2_SPECTRAL_POINT'])
ref['seeds']=[0]
ref['stage']='Exp18 one-model train-only curriculum, seed0, val-only'
f=ref['flags']
for name in ('exp16_point_refine','exp17_safety_gate'):
    f.pop(name,None)
f['exp16_point_refine']=False
f['exp17_safety_gate']=False
f['exp18_curriculum']=True
for name in ('exp15_lovasz_weight','exp15_spectral_weight','ascd_weight','vbb_weight','exp17_safety_weight'):
    if name in f: f[name]=0.0
assert f['amp'] is True and f['backbone']=='b2' and f['llr_source']=='raw'
assert f['use_low_level_refine'] is True and f['batch_size']==8
q=R/'configs/exp18/exp18_arms.json';q.parent.mkdir(parents=True,exist_ok=True)
s=json.dumps({'version':'exp18-curriculum-frozen-v1',
  'description':'Single SegFormer B2 raw LLR, train-only object->hard negatives->full 600; no point head, no ensemble, test sealed',
  'arms':{'E18_B2_OBJECT_CURRICULUM':ref},'rp_bias_grid':prior.get('rp_bias_grid',[])},indent=2,ensure_ascii=False)+'\n'
if q.exists():assert q.read_text()==s,'Different frozen registry: refuse overwrite'
else:q.write_text(s)
print('EXP18 ARMS PASS:',q)
