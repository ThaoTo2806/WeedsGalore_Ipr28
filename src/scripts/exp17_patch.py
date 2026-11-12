"""Fail-closed, idempotent Exp17 patch of working Exp16 Kaggle project."""
from pathlib import Path
import os, shutil, tempfile
R=Path(__file__).resolve().parents[2]

def edit(rel, changes, required):
    p=R/rel; assert p.is_file(),p
    raw=p.read_text(encoding='utf8')
    if required in raw:
        print('EXP17 ALREADY PATCHED:',rel); return
    s=raw
    for old,new in changes:
        assert s.count(old)==1, f'{rel} expected exactly once: {old!r}'
        s=s.replace(old,new,1)
    compile(s,str(p),'exec')
    backup=p.with_name(p.name+'.pre_exp17')
    if not backup.exists():shutil.copy2(p,backup)
    with tempfile.NamedTemporaryFile('w',suffix='.py',dir=p.parent,encoding='utf8',delete=False) as f:
        f.write(s); t=f.name
    os.replace(t,p)
    print('EXP17 PATCH PASS:',rel)

p=R/'src/train.py'
s=p.read_text(encoding='utf8')
assert 'EXP16_POINT_MODEL_TRAIN' in s, 'Exp16 integration required'
assert 'flags.DEFINE_bool(\'exp16_point_refine\'' in s
# inject before Exp16 import / flags, model selection, safety auxiliary immediately before backward
edit('src/train.py', [
    ('from nets.exp16_point import SpectralPointRefineSegFormer\n',
     'from nets.exp16_point import SpectralPointRefineSegFormer\nfrom nets.exp17_gate import SpectralSafetyGateSegFormer, exp17_safety_loss\n'),
    ("flags.DEFINE_bool('exp16_point_refine',", 
     "flags.DEFINE_bool('exp17_safety_gate', False, 'Exp17 gated refinement, single model')\n"
     "flags.DEFINE_float('exp17_safety_weight', 0.10, 'Exp17 train-only safety regularizer')\n"
     "flags.DEFINE_bool('exp16_point_refine',"),
    ('(SpectralPointRefineSegFormer if FLAGS.exp16_point_refine else SegFormer5Band)',
     '(SpectralSafetyGateSegFormer if FLAGS.exp17_safety_gate else SpectralPointRefineSegFormer if FLAGS.exp16_point_refine else SegFormer5Band)'),
    ('    if FLAGS.exp16_point_refine:\n',
     '    if FLAGS.exp17_safety_gate and not FLAGS.exp16_point_refine:\n'
     '        raise ValueError("Exp17 needs exp16_point_refine=True")\n'
     '    if FLAGS.exp16_point_refine:\n'),
    ('            scaler.scale(loss).backward()',
     '            # EXP17_SAFETY_TRAIN_ONLY\n'
     '            if FLAGS.exp17_safety_gate and FLAGS.exp17_safety_weight > 0:\n'
     '                loss = loss + FLAGS.exp17_safety_weight * exp17_safety_loss(net, labels)\n'
     '            scaler.scale(loss).backward()'),
], 'EXP17_SAFETY_TRAIN_ONLY')

edit('src/ccmc_common.py', [
    ('from nets.exp16_point import SpectralPointRefineSegFormer\n',
     'from nets.exp16_point import SpectralPointRefineSegFormer\nfrom nets.exp17_gate import SpectralSafetyGateSegFormer\n'),
    ('model_cls = SpectralPointRefineSegFormer if getattr(F, "exp16_point_refine", False) else SegFormer5Band',
     'model_cls = SpectralSafetyGateSegFormer if getattr(F, "exp17_safety_gate", False) else SpectralPointRefineSegFormer if getattr(F, "exp16_point_refine", False) else SegFormer5Band'),
], 'SpectralSafetyGateSegFormer if getattr(F, "exp17_safety_gate", False)')

edit('src/evaluate_ccmc.py', [
    ("flags.DEFINE_bool('exp16_point_refine',", 
     "flags.DEFINE_bool('exp17_safety_gate', False, 'Exp17 gated refinement')\n"
     "flags.DEFINE_bool('exp16_point_refine',"),
], "flags.DEFINE_bool('exp17_safety_gate'")
print('EXP17 INTEGRATION PASS')
