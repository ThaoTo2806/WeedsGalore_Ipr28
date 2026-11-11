"""Idempotently add a training-only VBB loss to the already ASCD-patched Kaggle train.py."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
p = ROOT / 'src/train.py'
s = p.read_text(encoding='utf-8')
mark = '            # EXP14_VBB_TRAIN_ONLY\n'
if mark in s:
    assert 'vbb_loss(out, labels' in s
    print('EXP14 VBB PATCH ALREADY INSTALLED')
else:
    assert s.count('import exp12_config\n') == 1
    assert s.count('            epoch_steps += 1\n') == 1
    assert s.count('            scaler.scale(loss).backward()') == 1, 'Require patched AMP training source'
    assert "flags.DEFINE_bool('amp'," in s
    assert 'flags.DEFINE_float(\'ascd_weight\'' in s, 'Install original ASCD patch first'
    backup = ROOT / 'src/train_pre_vbb.py'
    if not backup.exists():
        shutil.copy2(p, backup)
    s = s.replace('import exp12_config\n', 'import exp12_config\nfrom exp14_vbb import vbb_loss\n')
    anchor = "flags.DEFINE_float('ascd_weight',"
    pos = s.index('\n', s.index(anchor)) + 1
    added = (
        "flags.DEFINE_float('vbb_weight', 0.0, 'Training-only vegetation boundary loss weight')\n"
        "flags.DEFINE_integer('vbb_radius', 3, 'Vegetation boundary radius in pixels')\n"
    )
    s = s[:pos] + added + s[pos:]
    # Inject before backward; original ASCD block, scheduler, scaler, EMA stay untouched.
    injection = (
        mark +
        '            if FLAGS.vbb_weight > 0:\n'
        '                if FLAGS.num_classes != 3 or FLAGS.head != "bilinear" or not FLAGS.segformer:\n'
        '                    raise ValueError("VBB requires 3-class bilinear SegFormer")\n'
        '                loss = loss + FLAGS.vbb_weight * vbb_loss(\n'
        '                    out, labels, radius=FLAGS.vbb_radius, ignore_index=FLAGS.ignore_index)\n'
    )
    s = s.replace('            epoch_steps += 1\n', '            epoch_steps += 1\n' + injection)
    compile(s, str(p), 'exec')
    p.write_text(s, encoding='utf-8')
    print('EXP14 VBB PATCH PASS, backup:', backup)
