"""Idempotent fail-closed source patch, train.py only. No model/eval changes."""
from pathlib import Path
import shutil,tempfile,os
R=Path(__file__).resolve().parents[2]
p=R/'src/train.py'; s=p.read_text(encoding='utf8')
marker='EXP18_CURRICULUM_HOOK'
if marker in s:
    print('EXP18 TRAIN PATCH ALREADY PRESENT');raise SystemExit(0)
assert 'from datasets import WeedsGaloreDataset' in s, 'Dataset import anchor absent'
assert 'EXP17_SAFETY_TRAIN_ONLY' in s and 'EXP16_POINT_MODEL_TRAIN' in s, 'Needs fully integrated Exp17 project before patching' 
assert s.count('    # Dataloader')==1, 'Dataloader anchor absent or repeated'
assert s.count('    for epoch in range(start_epoch, FLAGS.epochs):')==1, 'Epoch anchor absent'
assert 'flags.DEFINE_boolean' in s or 'flags.DEFINE_bool' in s
# Insert independent flag before existing flags; allow current Exp17 source, but Exp18 flag disables heads.
s=s.replace('from datasets import WeedsGaloreDataset',
            'from datasets import WeedsGaloreDataset\nfrom exp18_curriculum import CurriculumTrainDataset, stage_for_epoch',1)
anchor='flags.DEFINE_string(\'dataset_path\''
assert s.count(anchor)==1, 'Flag anchor absent'
s=s.replace(anchor,"flags.DEFINE_boolean('exp18_curriculum', False, 'Exp18 train-only 384 patch curriculum')\n"+anchor,1)
s=s.replace('    # Dataloader',
"    # EXP18_CURRICULUM_HOOK -- changes train only; validation remains 26 full images.\n"
"    if FLAGS.exp18_curriculum:\n"
"        assert FLAGS.in_channels == 5 and FLAGS.num_classes == 3\n"
"        assert not FLAGS.ccmc and not FLAGS.exp16_point_refine and not FLAGS.exp17_safety_gate\n"
"        train_dataset = CurriculumTrainDataset(\n"
"            FLAGS.dataset_path, FLAGS.dataset_size_train, FLAGS.in_channels,\n"
"            FLAGS.num_classes, seed=FLAGS.seed, patch_size=384)\n\n"
"    # Dataloader",1)
s=s.replace('    for epoch in range(start_epoch, FLAGS.epochs):',
"    for epoch in range(start_epoch, FLAGS.epochs):\n"
"        if FLAGS.exp18_curriculum:\n"
"            train_dataset.set_epoch(epoch)\n"
"            if stage_for_epoch(epoch) == 'hard_negative' and (epoch == 35 or epoch % 5 == 0):\n"
"                train_dataset.refresh_hard_negatives(net, device, epoch)\n",1)
compile(s,str(p),'exec')
bak=p.with_suffix('.py.pre_exp18')
if not bak.exists():shutil.copy2(p,bak)
with tempfile.NamedTemporaryFile('w',suffix='.py',dir=p.parent,encoding='utf8',delete=False) as f:
    f.write(s); tmp=f.name
os.replace(tmp,p)
print('EXP18 TRAIN PATCH PASS:',p,'backup:',bak)
