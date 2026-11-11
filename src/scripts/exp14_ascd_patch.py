"""Safe, idempotent patch of the user's working Kaggle AMP train.py.

Patch only the train-only auxiliary loss. Does not modify model/evaluator.
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
p = ROOT / 'src/train.py'
s = p.read_text(encoding='utf-8')
mark = '            # EXP14_ASCD_TRAIN_ONLY\n'
if mark in s:
    assert 'ascd_loss(out, labels, features' in s
    print('ASCD PATCH ALREADY INSTALLED')
else:
    assert 'cw_loss_weight' not in s, 'Detected previous CW-DBL patch; restore clean AMP train.py first'
    assert s.count('import exp12_config\n') == 1
    assert s.count("flags.DEFINE_integer('batch_size',") == 1
    assert s.count('            epoch_steps += 1\n') == 1
    assert s.count('            scaler.scale(loss).backward()') == 1
    assert s.count('            with torch.autocast(') == 1
    assert 'flags.DEFINE_bool(\'amp\',' in s
    assert 'flags.DEFINE_integer(\'grad_accum_steps\'' in s
    backup = ROOT / 'src/train_pre_ascd.py'
    if not backup.exists():
        shutil.copy2(p, backup)
    s = s.replace('import exp12_config\n', 'import exp12_config\nfrom exp14_ascd import ascd_loss\n')
    anchor = "flags.DEFINE_integer('batch_size',"
    i = s.index('\n', s.index(anchor)) + 1
    flags = (
        "flags.DEFINE_float('ascd_weight', 0.0, 'Train-only ASCD weight')\n"
        "flags.DEFINE_float('ascd_hard_alpha', 1.0, 'Wrong CW class probability weighting')\n"
        "flags.DEFINE_float('ascd_spectral_alpha', 1.0, 'Spectral confusion weighting')\n"
        "flags.DEFINE_float('ascd_spectral_temperature', 1.0, 'Spectral confusion temperature')\n"
        "flags.DEFINE_integer('ascd_interface_radius', 3, 'Crop-Weed interface radius')\n"
    )
    s = s[:i] + flags + s[i:]
    new = (mark
       + '            if FLAGS.ascd_weight > 0:\n'
       + '                if FLAGS.num_classes != 3 or FLAGS.head != "bilinear" or not FLAGS.segformer:\n'
       + '                    raise ValueError("ASCD requires 3-class bilinear SegFormer")\n'
       + '                aux_ascd = ascd_loss(out, labels, features,\n'
       + '                    hard_alpha=FLAGS.ascd_hard_alpha,\n'
       + '                    spectral_alpha=FLAGS.ascd_spectral_alpha,\n'
       + '                    spectral_temperature=FLAGS.ascd_spectral_temperature,\n'
       + '                    interface_radius=FLAGS.ascd_interface_radius,\n'
       + '                    ignore_index=FLAGS.ignore_index)\n'
       + '                loss = loss + FLAGS.ascd_weight * aux_ascd\n')
    s = s.replace('            epoch_steps += 1\n', '            epoch_steps += 1\n' + new)
    compile(s, str(p), 'exec')
    p.write_text(s, encoding='utf-8')
    print('ASCD PATCH PASS:', p)
    print('BACKUP:', backup)
