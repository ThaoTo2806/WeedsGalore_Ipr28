"""Frozen exp21 registry with an explicit compact B2 + raw LLR arm."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[2]
common=json.loads((R/'configs/exp12/exp12_common_config.json').read_text())['common']
assert common['num_classes']==3 and common['epochs']==100 and common['batch_size']==8
flags={
 'backbone':'b2','head':'bilinear','use_low_level_refine':True,
 'llr_source':'raw','input_scale':1.0,'lr_schedule':'cosine',
 'warmup_epochs':5,'min_lr_ratio':0.01,'ema_decay':0.99,'amp':True,
 'exp21_fine_weight':0.2,
 'exp16_point_refine':False,'exp17_safety_gate':False,'exp18_curriculum':False,
 'ascd_weight':0.0,'vbb_weight':0.0,'exp15_lovasz_weight':0.0,
 'exp15_spectral_weight':0.0
}
assert not (set(flags)&set(common))
obj={'version':'exp21-hier-1','description':'Single MiT-B2 6-way fine auxiliary supervision; exact probability merge to 3 classes, train-only subtype GT, test sealed',
 'arms':{'E21_B2_HIER':{'flags':flags,'seeds':[0],'allowed_tta':['none','d4'],
                         'stage':'Seed0 validation screening only'}},'rp_bias_grid':[]}
p=R/'configs/exp21/exp21_arms.json';p.parent.mkdir(parents=True,exist_ok=True)
s=json.dumps(obj,indent=2,ensure_ascii=False)+'\n'
if p.exists():assert p.read_text()==s,'Different registry already exists; do not overwrite'
else:p.write_text(s)
print('EXP21 ARMS PASS:',p)
