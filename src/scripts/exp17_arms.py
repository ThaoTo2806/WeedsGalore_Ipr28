"""Create frozen Exp17 arm from Exp16, without modifying prior arm or results."""
from pathlib import Path
import json,copy
R=Path(__file__).resolve().parents[2]
p=R/'configs/exp16/exp16_arms.json'
assert p.is_file(),f'Restore Exp16 registry first: {p}'
prior=json.loads(p.read_text())
arm=copy.deepcopy(prior['arms']['E16_B2_SPECTRAL_POINT'])
arm['seeds']=[0]
arm['stage']='exp17 seed0 only, validation gate; test locked'
arm['flags'].update({'exp17_safety_gate':True,'exp17_safety_weight':0.10,
                     'exp16_point_refine':True,'exp16_point_budget':4096})
assert arm['flags']['amp'] is True
assert arm['flags']['backbone']=='b2'
assert arm['flags']['llr_source']=='raw'
new={'version':'exp17-safety-gate-v1','description':'Single 5-band B2, sparse counterfactual safety gate, no ensemble, test sealed',
     'arms':{'E17_B2_SAFETY_GATE':arm},'rp_bias_grid':prior.get('rp_bias_grid',[])}
q=R/'configs/exp17/exp17_arms.json';q.parent.mkdir(parents=True,exist_ok=True)
s=json.dumps(new,ensure_ascii=False,indent=2)+'\n'
if q.exists():assert q.read_text()==s,'Exp17 frozen registry differs: do not overwrite'
else:q.write_text(s)
print('EXP17 ARMS PASS:',q)
